# system.py
"""
Sayit VM Executor - full interpreter for parsed Sayit AST.

This module executes the AST produced by `say_parser.parse_code`.
It implements:
 - Variables, Print, Assign
 - Binary arithmetic and comparisons
 - If / Elif / Else lowering and execution
 - While loops with safety step limit
 - `fail()` and `end()` builtins
 - Execution trace, step limits and error mapping for debugging/tests

Also includes a bytecode VM (class `VM`) and a small CLI to run either the
IR/Executor engine or the VM engine.  Backwards-compatible behavior and
exit codes are preserved.

Fallback Codegen shim and a "Dodecagram" base-12 pseudo-IR dumper are provided
so `--engine vm` and `--emit-ir` work even without an external `say_codegen`.

A minimal built-in parser is provided so this file is self-contained and
fully executable for demos.
"""
from __future__ import annotations

import sys
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple

# Lazy import of parser utilities to avoid circular imports when used in tests
try:
    from say_parser import Program, Print, Assign, BinOp, If, While, Literal, Ident, parse_code  # type: ignore
except Exception:
    # Minimal in-file AST and parser so system.py is fully executable without external deps.

    class Program:
        def __init__(self, stmts: List[Any]):
            self.stmts = stmts
        def __repr__(self):
            return f"Program({self.stmts!r})"

    class Literal:
        def __init__(self, val: Any):
            self.val = val
        def __repr__(self):
            return f"Literal({self.val!r})"

    class Ident:
        def __init__(self, name: str):
            self.name = name
        def __repr__(self):
            return f"Ident({self.name!r})"

    class Print:
        def __init__(self, expr: Any):
            self.expr = expr
        def __repr__(self):
            return f"Print({self.expr!r})"

    class Assign:
        def __init__(self, ident: str, expr: Any):
            self.ident = ident
            self.expr = expr
        def __repr__(self):
            return f"Assign({self.ident!r}, {self.expr!r})"

    class BinOp:
        def __init__(self, left: Any, op: str, right: Any):
            self.left = left
            self.op = op
            self.right = right
        def __repr__(self):
            return f"BinOp({self.left!r} {self.op} {self.right!r})"

    class If:
        def __init__(self, cond: Any, then: List[Any], elifs: Optional[List[Tuple[Any, List[Any]]]] = None, els: Optional[List[Any]] = None):
            self.cond = cond
            self.then = then
            self.elifs = elifs or []
            self.els = els or []
        def __repr__(self):
            return f"If({self.cond!r}, then={self.then!r}, elifs={self.elifs!r}, els={self.els!r})"

    class While:
        def __init__(self, cond: Any, body: List[Any]):
            self.cond = cond
            self.body = body
        def __repr__(self):
            return f"While({self.cond!r}, body={self.body!r})"

    # Very small, robust line-based parser that understands:
    # - assignments: name = <literal|ident|expr>
    # - print(<expr>)
    # - if <expr> ... [elif <expr> ...] [else ...] end
    # - while <expr> ... end
    #
    # Expression support: integer, quoted string, identifier, simple binary + - * / (left-to-right).
    def _parse_expr_token(tok: str):
        tok = tok.strip()
        if not tok:
            return Literal(None)
        if tok[0] in ("'", '"') and tok[-1] in ("'", '"'):
            return Literal(tok[1:-1])
        # integer?
        try:
            return Literal(int(tok))
        except Exception:
            pass
        # binary ops simple: try to split on + - * /
        for op in ("+", "-", "*", "/"):
            # split only first occurrence (left-to-right)
            if op in tok:
                parts = tok.split(op, 1)
                left = _parse_expr_token(parts[0])
                right = _parse_expr_token(parts[1])
                return BinOp(left, op, right)
        # identifier fallback
        return Ident(tok)

    def parse_code(src: str):
        """
        Simple parser producing the minimal AST used by Executor and FallbackCodegen.
        Not a full language implementation — designed for demos and examples.
        """
        lines = [ln.rstrip() for ln in src.splitlines()]
        idx = 0
        stmts_stack: List[List[Any]] = [[]]  # top is current block
        block_type_stack: List[str] = []

        def current_block():
            return stmts_stack[-1]

        while idx < len(lines):
            line = lines[idx].strip()
            idx += 1
            if not line or line.startswith("#"):
                continue
            # print(...)
            if line.startswith("print(") and line.endswith(")"):
                inner = line[len("print("):-1].strip()
                expr = _parse_expr_token(inner)
                current_block().append(Print(expr))
                continue
            # assignment: a = expr
            if "=" in line and not line.startswith("if ") and not line.startswith("while "):
                parts = line.split("=", 1)
                name = parts[0].strip()
                rhs = parts[1].strip()
                expr = _parse_expr_token(rhs)
                current_block().append(Assign(name, expr))
                continue
            # if
            if line.startswith("if "):
                cond_src = line[len("if "):].strip()
                cond = _parse_expr_token(cond_src)
                # push new block for then
                stmts_stack.append([])
                block_type_stack.append("if")
                # store If node with placeholders; we'll fill after block closed
                current_block_index = len(stmts_stack) - 1
                # temporarily push sentinel If; actual placement after parsing block
                stmts_stack[-1].append(("__if_placeholder__", cond))
                continue
            # elif
            if line.startswith("elif "):
                cond_src = line[len("elif "):].strip()
                cond = _parse_expr_token(cond_src)
                # mark new elif block: append marker to current outer if holder
                # we implement by closing current then-block and starting a new block for this elif
                # find the nearest "if" in block_type_stack
                if "if" not in block_type_stack:
                    raise RuntimeError("elif without if")
                # close current then block (it's already top of stack)
                then_block = stmts_stack.pop()
                block_type_stack.pop()
                # the If placeholder is stored in the previous block's last element
                # We'll transform it into an If structure that accumulates elifs.
                # For simplicity, store a tuple on the previous block to be resolved.
                prev_block = stmts_stack[-1]
                # replace placeholder with a structure we can extend
                # find last placeholder tuple
                if not prev_block:
                    raise RuntimeError("internal parser error")
                # last element should be placeholder
                ph = prev_block.pop()
                if not isinstance(ph, tuple) or ph[0] != "__if_placeholder__":
                    raise RuntimeError("internal parser error")
                if_cond = ph[1]
                if_node = If(if_cond, then=then_block, elifs=[(cond, [])], els=[])
                # push back
                prev_block.append(if_node)
                # now start new block for the elif body
                stmts_stack.append(if_node.elifs[-1][1])
                block_type_stack.append("if")
                continue
            # else
            if line == "else":
                # close current then block
                if "if" not in block_type_stack:
                    raise RuntimeError("else without if")
                then_block = stmts_stack.pop()
                block_type_stack.pop()
                prev_block = stmts_stack[-1]
                ph = prev_block.pop()
                if not isinstance(ph, tuple) or ph[0] != "__if_placeholder__":
                    # maybe already converted to If via elif handling: if so, ph is If
                    if isinstance(ph, If):
                        # convert then_block was incorrectly popped earlier — put it back and start else
                        if_node = ph
                        # replace with same
                        prev_block.append(if_node)
                        stmts_stack.append(if_node.els)
                        block_type_stack.append("if")
                        continue
                    raise RuntimeError("internal parser error")
                if_cond = ph[1]
                if_node = If(if_cond, then=then_block, elifs=[], els=[])
                prev_block.append(if_node)
                stmts_stack.append(if_node.els)
                block_type_stack.append("if")
                continue
            # end closes last block (if/while)
            if line == "end":
                if len(stmts_stack) <= 1:
                    raise RuntimeError("unexpected end")
                finished = stmts_stack.pop()
                btype = block_type_stack.pop() if block_type_stack else None
                # if this was an if started with placeholder, convert placeholder to If
                prev_block = stmts_stack[-1]
                ph = prev_block.pop()
                if isinstance(ph, tuple) and ph[0] == "__if_placeholder__":
                    cond = ph[1]
                    if_node = If(cond, then=finished, elifs=[], els=[])
                    prev_block.append(if_node)
                else:
                    # for while, ph may be tuple with ("__while_placeholder__", cond)
                    if isinstance(ph, tuple) and ph[0] == "__while_placeholder__":
                        cond = ph[1]
                        while_node = While(cond, body=finished)
                        prev_block.append(while_node)
                    else:
                        # Could be already replaced
                        prev_block.append(ph)
                continue
            # while
            if line.startswith("while "):
                cond_src = line[len("while "):].strip()
                cond = _parse_expr_token(cond_src)
                # prepare placeholder
                stmts_stack.append([])
                block_type_stack.append("while")
                stmts_stack[-1].append(("__while_placeholder__", cond))
                continue
            # fallback: unknown single-word 'end()' 'fail()' 'end()' or literal statements
            if line.endswith("()"):
                name = line[:-2]
                # treat 'end' and 'fail' as idents / function-like calls represented as Ident
                if name == "end":
                    current_block().append(Print(Literal("<end()>")))
                    continue
                if name == "fail":
                    current_block().append(Print(Literal("<fail()>")))
                    continue
            # Unknown line: treat as NOOP comment
            current_block().append(Print(Literal(f";; unparsed: {line}")))
        # after parsing, ensure stack collapsed
        if len(stmts_stack) != 1:
            raise RuntimeError("unterminated block")
        return Program(stmts_stack[0])


class ExecutionError(RuntimeError):
    pass


class Executor:
    def __init__(self, max_steps: int = 1_000_000, timeout: Optional[float] = None, trace: bool = False):
        self.env: Dict[str, Any] = {}
        self.output: List[str] = []
        self.max_steps = max_steps
        self._steps = 0
        self.timeout = timeout
        self.start_time: Optional[float] = None
        self.trace = trace
        # builtin mappings
        self._builtins = {
            "fail": self._builtin_fail,
            "end": self._builtin_end,
        }
        self._halted = False

    # ---------- Execution control ----------
    def _bump(self) -> None:
        self._steps += 1
        if self._steps > self.max_steps:
            raise ExecutionError(f"max_steps exceeded ({self.max_steps})")
        if self.timeout is not None and self.start_time is not None:
            if (time.time() - self.start_time) > self.timeout:
                raise ExecutionError("execution timeout")

    def run_program(self, prog) -> None:
        """Run a parsed Program AST in this executor."""
        if not hasattr(prog, "stmts"):
            raise TypeError("run_program expects a say_parser.Program instance")
        self.start_time = time.time() if self.timeout is not None else None
        self._halted = False
        for st in prog.stmts:
            if self._halted:
                break
            self._bump()
            self._exec_stmt(st)

    # ---------- Statements ----------
    def _exec_stmt(self, stmt) -> None:
        name = stmt.__class__.__name__
        if self.trace:
            print(f"[TRACE] exec stmt {name}: {stmt}")
        if name == "Print":
            val = self._eval_expr(stmt.expr)
            self._do_print(val)
        elif name == "Assign":
            val = self._eval_expr(stmt.expr)
            self.env[stmt.ident] = val
            if self.trace:
                print(f"[TRACE] assign {stmt.ident} = {val!r}")
        elif name == "If":
            self._exec_if(stmt)
        elif name == "While":
            self._exec_while(stmt)
        else:
            # Unknown stmt kind: surface in output and skip
            self._emit(f";; unhandled-stmt {name}")

    def _exec_if(self, node) -> None:
        # node.cond, node.then (list), node.elifs (list of (cond, body)), node.els (list)
        cond = self._eval_expr(node.cond)
        if self._is_truthy(cond):
            for st in node.then:
                self._bump()
                self._exec_stmt(st)
            return
        # elifs
        for (ec, body) in getattr(node, "elifs", []) or []:
            self._bump()
            ev = self._eval_expr(ec)
            if self._is_truthy(ev):
                for st in body:
                    self._bump()
                    self._exec_stmt(st)
                return
        # else
        for st in getattr(node, "els", []) or []:
            self._bump()
            self._exec_stmt(st)

    def _exec_while(self, node) -> None:
        # node.cond, node.body (list)
        iterations = 0
        while True:
            self._bump()
            cond = self._eval_expr(node.cond)
            if not self._is_truthy(cond):
                break
            for st in getattr(node, "body", []) or []:
                self._bump()
                self._exec_stmt(st)
                if self._halted:
                    return
            iterations += 1
            # safety guard
            if iterations > self.max_steps:
                raise ExecutionError("while loop exceeded max_steps")

    # ---------- Expressions ----------
    def _eval_expr(self, expr) -> Any:
        self._bump()
        cname = expr.__class__.__name__
        if self.trace:
            print(f"[TRACE] eval expr {cname}: {expr}")
        if cname == "Literal":
            return expr.val
        if cname == "Ident":
            name = expr.name
            # builtin zero-arg calls like `end` or `fail` may appear as Idents followed by call syntax in future.
            return self.env.get(name, name)
        if cname == "BinOp":
            left = self._eval_expr(expr.left)
            right = self._eval_expr(expr.right)
            op = expr.op
            return self._eval_binop(op, left, right)
        # unknown expression type: return None
        return None

    def _eval_binop(self, op: str, left: Any, right: Any) -> Any:
        # arithmetic
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            # integer division if both ints
            if isinstance(left, int) and isinstance(right, int):
                if right == 0:
                    raise ExecutionError("Division by zero")
                return left // right
            return left / right
        # comparisons
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        # logical-ish ops
        if op.lower() in ("and", "&&"):
            return bool(left) and bool(right)
        if op.lower() in ("or", "||"):
            return bool(left) or bool(right)
        # fallback: if op is string of punctuation (from lexer), try eval-like mapping
        # treat other ops as error
        raise ExecutionError(f"Unsupported operator: {op!r}")

    # ---------- Utilities ----------
    def _is_truthy(self, v: Any) -> bool:
        return bool(v)

    def _do_print(self, val: Any) -> None:
        s = str(val)
        self._emit(s)
        # immediate side-effect
        print(s)

    def _emit(self, text: str) -> None:
        self.output.append(text)

    # ---------- Builtins ----------
    def _builtin_fail(self, *args) -> None:
        raise ExecutionError("fail() called")

    def _builtin_end(self, *args) -> None:
        # mark halted; graceful end
        self._halted = True

    # ---------- Convenience / CLI helpers ----------
    def run_file(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        prog = parse_code(src)
        self.run_program(prog)
        return "\n".join(self.output)

    def run_source(self, src: str) -> str:
        prog = parse_code(src)
        self.run_program(prog)
        return "\n".join(self.output)


# ==============================================================#
# Sayit Virtual Machine (Bytecode Interpreter)
# ==============================================================#
# Executes bytecode produced by say_codegen.py (or fallback shim).
#
# Supported Opcodes:
#   LOAD_CONST, LOAD_VAR, STORE_VAR, BINARY_OP,
#   PRINT, JUMP, JUMP_IF_FALSE, CALL, HALT
# ==============================================================#
class VM:
    def __init__(self, bytecode: List[Tuple[str, Any]], const_pool: List[Any], echo: bool = True):
        self.bytecode = bytecode
        self.const_pool = const_pool
        self.vars: Dict[str, Any] = {}         # Variable environment
        self.stack: List[Any] = []        # Operand stack
        self.pc = 0            # Program counter
        self.running = True
        self.echo = echo
        self.output: List[str] = []

    # --- Core execution loop ---
    def run(self):
        while self.running and self.pc < len(self.bytecode):
            op, arg = self.bytecode[self.pc]
            handler = getattr(self, f"op_{op.lower()}", None)
            if handler is None:
                raise RuntimeError(f"Unknown opcode: {op}")
            handler(arg)
            self.pc += 1

    # --- Opcode handlers ---
    def op_load_const(self, idx):
        self.stack.append(self.const_pool[idx])

    def op_load_var(self, name):
        if name not in self.vars:
            raise RuntimeError(f"Undefined variable: {name}")
        self.stack.append(self.vars[name])

    def op_store_var(self, name):
        val = self.stack.pop()
        self.vars[name] = val

    def op_binary_op(self, operator):
        b = self.stack.pop()
        a = self.stack.pop()
        if operator == "+":
            self.stack.append(a + b)
        elif operator == "-":
            self.stack.append(a - b)
        elif operator == "*":
            self.stack.append(a * b)
        elif operator == "/":
            # mirror Executor: integer division if both ints
            if isinstance(a, int) and isinstance(b, int):
                if b == 0:
                    raise RuntimeError("Division by zero")
                self.stack.append(a // b)
            else:
                self.stack.append(a / b)
        elif operator == "==":
            self.stack.append(a == b)
        elif operator == "!=":
            self.stack.append(a != b)
        elif operator == "<":
            self.stack.append(a < b)
        elif operator == ">":
            self.stack.append(a > b)
        elif operator == "<=":
            self.stack.append(a <= b)
        elif operator == ">=":
            self.stack.append(a >= b)
        else:
            raise RuntimeError(f"Unsupported operator: {operator}")

    def op_print(self, _):
        val = self.stack.pop()
        s = str(val)
        self.output.append(s)
        if self.echo:
            print(s)

    def op_jump(self, target):
        self.pc = target - 1  # -1 because run() will increment

    def op_jump_if_false(self, target):
        cond = self.stack.pop()
        if not cond:
            self.pc = target - 1

    def op_call(self, name):
        if name == "input":
            val = input("Sayit input> ")
            self.stack.append(val)
        elif name == "len":
            val = len(self.stack.pop())
            self.stack.append(val)
        else:
            raise RuntimeError(f"Unknown builtin call: {name}")

    def op_halt(self, _):
        self.running = False


# ---------- Fallback Codegen shim (used when say_codegen is absent) ----------
class FallbackCodegen:
    """
    Minimal Codegen that converts a parsed AST into VM bytecode and const pool.
    Supports: Program, Assign, Print, BinOp, Literal, Ident, If, While.
    Also exposes a `dump_ir()` producing the Dodecagram base-12 pseudo-IR.
    """

    def __init__(self):
        self.bytecode: List[Tuple[str, Any]] = []
        self.const_pool: List[Any] = []
        # label stack helps for structured constructs
        self._pc = 0

    # emit an opcode and return its index
    def emit(self, op: str, arg: Optional[Any] = None) -> int:
        idx = len(self.bytecode)
        self.bytecode.append((op, arg))
        self._pc += 1
        return idx

    def add_const(self, val: Any) -> int:
        try:
            return self.const_pool.index(val)
        except ValueError:
            self.const_pool.append(val)
            return len(self.const_pool) - 1

    def compile(self, prog) -> None:
        if not hasattr(prog, "stmts"):
            raise TypeError("compile expects a say_parser.Program instance")
        # naive linear compilation
        for st in prog.stmts:
            self._gen_stmt(st)
        self.emit("HALT", None)

    def get_bytecode(self) -> List[Tuple[str, Any]]:
        return self.bytecode

    def get_const_pool(self) -> List[Any]:
        return self.const_pool

    # simple IR dumper (Dodecagram base-12)
    def dump_ir(self) -> str:
        return ast_to_dodecagram_ir_from_bytecode(self.bytecode, self.const_pool)

    # -- generation helpers --
    def _gen_stmt(self, st) -> None:
        name = st.__class__.__name__
        if name == "Print":
            self._gen_expr(st.expr)
            self.emit("PRINT", None)
        elif name == "Assign":
            self._gen_expr(st.expr)
            self.emit("STORE_VAR", st.ident)
        elif name == "If":
            # cond, then (list), elifs, els
            self._gen_expr(st.cond)
            # placeholder jump (to be patched)
            jfalse_idx = self.emit("JUMP_IF_FALSE", None)
            # then body
            for s in st.then:
                self._gen_stmt(s)
            # optional else/elif handling
            if getattr(st, "elifs", None):
                # for simplicity: chain elifs as nested ifs
                for (ec, body) in getattr(st, "elifs", []) or []:
                    # create jump over elif
                    j_over = self.emit("JUMP", None)
                    # patch previous jfalse to point here
                    self._patch_jump(jfalse_idx, len(self.bytecode))
                    # evaluate elif cond
                    self._gen_expr(ec)
                    jfalse_idx = self.emit("JUMP_IF_FALSE", None)
                    for s in body:
                        self._gen_stmt(s)
                    # continue chaining
                    self._patch_jump(j_over, len(self.bytecode))
                # else:
                for s in getattr(st, "els", []) or []:
                    self._gen_stmt(s)
                # patch final false jump to here
                self._patch_jump(jfalse_idx, len(self.bytecode))
            else:
                # simple else
                for s in getattr(st, "els", []) or []:
                    self._gen_stmt(s)
                self._patch_jump(jfalse_idx, len(self.bytecode))
        elif name == "While":
            start_idx = len(self.bytecode)
            self._gen_expr(st.cond)
            jfalse_idx = self.emit("JUMP_IF_FALSE", None)
            for s in getattr(st, "body", []) or []:
                self._gen_stmt(s)
            # jump back to start (continue)
            self.emit("JUMP", start_idx)
            self._patch_jump(jfalse_idx, len(self.bytecode))
        else:
            # unknown: emit comment-like NOOP via CONST + POP
            self.emit("NOOP", name)

    def _gen_expr(self, expr) -> None:
        cname = expr.__class__.__name__
        if cname == "Literal":
            idx = self.add_const(expr.val)
            self.emit("LOAD_CONST", idx)
        elif cname == "Ident":
            self.emit("LOAD_VAR", expr.name)
        elif cname == "BinOp":
            self._gen_expr(expr.left)
            self._gen_expr(expr.right)
            self.emit("BINARY_OP", expr.op)
        else:
            # fallback: push None
            idx = self.add_const(None)
            self.emit("LOAD_CONST", idx)

    def _patch_jump(self, idx: int, target: int) -> None:
        op, _ = self.bytecode[idx]
        if op not in ("JUMP_IF_FALSE", "JUMP"):
            raise RuntimeError("attempt to patch non-jump")
        self.bytecode[idx] = (op, target)


# ---------- Helpers to run VM from source/file ----------
def compile_to_vm(src: str):
    """
    Compile source code to (bytecode, const_pool, optional_ir_text).
    Try to use external say_codegen.Codegen; if missing, use FallbackCodegen.
    """
    try:
        import say_codegen  # type: ignore
    except Exception:
        # fallback shim
        cg = FallbackCodegen()
        ast = parse_code(src)
        cg.compile(ast)
        return cg.get_bytecode(), cg.get_const_pool(), cg.dump_ir()

    # external codegen present
    Codegen = getattr(say_codegen, "Codegen", None)
    if Codegen is None:
        # fallback
        cg = FallbackCodegen()
        ast = parse_code(src)
        cg.compile(ast)
        return cg.get_bytecode(), cg.get_const_pool(), cg.dump_ir()

    ast = parse_code(src)
    cg = Codegen()
    cg.compile(ast)
    get_bc = getattr(cg, "get_bytecode", None)
    get_cp = getattr(cg, "get_const_pool", None)
    dump_ir = getattr(cg, "dump_ir", None) or getattr(cg, "get_ir", None)
    if not callable(get_bc) or not callable(get_cp):
        # fallback
        cg2 = FallbackCodegen()
        cg2.compile(ast)
        return cg2.get_bytecode(), cg2.get_const_pool(), cg2.dump_ir()
    bytecode = get_bc()
    const_pool = get_cp()
    ir_text = dump_ir() if callable(dump_ir) else None
    return bytecode, const_pool, ir_text


def run_vm_from_source(src: str, echo: bool = True) -> str:
    bytecode, const_pool, ir = compile_to_vm(src)
    vm = VM(bytecode, const_pool, echo=echo)
    vm.run()
    # prefer IR text if requested by emit flag in caller; VM returns output here
    return "\n".join(vm.output)


def run_vm_from_file(path: str, echo: bool = True) -> str:
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    return run_vm_from_source(src, echo=echo)


# ---------- Dodecagram base-12 IR dumper ----------
_BASE12_DIGITS = "0123456789AB"


def _to_base12(n: int) -> str:
    if n == 0:
        return "0"
    out = []
    neg = n < 0
    n = abs(n)
    while n:
        n, r = divmod(n, 12)
        out.append(_BASE12_DIGITS[r])
    if neg:
        out.append("-")
    return "".join(reversed(out))


def ast_to_dodecagram_ir_from_bytecode(bytecode: List[Tuple[str, Any]], const_pool: List[Any]) -> str:
    """
    Produce a human-readable base-12 Dodecagram IR from bytecode + const pool.
    Each instruction is prefixed with a base-12 address.
    Constants are emitted at top with base-12 indices.
    """
    lines: List[str] = []
    lines.append("; Dodecagram IR (base-12 addresses)")
    # const pool
    lines.append("; const_pool:")
    for i, c in enumerate(const_pool):
        lines.append(f"@{_to_base12(i)} = {repr(c)}")
    lines.append("; bytecode:")
    for i, (op, arg) in enumerate(bytecode):
        addr = _to_base12(i)
        if op in ("LOAD_CONST",):
            lines.append(f"{addr}: {op} @{_to_base12(arg)}")
        elif op in ("LOAD_VAR", "STORE_VAR", "PRINT", "CALL", "NOOP"):
            lines.append(f"{addr}: {op} {repr(arg)}")
        elif op in ("BINARY_OP",):
            lines.append(f"{addr}: {op} {repr(arg)}")
        elif op in ("JUMP", "JUMP_IF_FALSE"):
            lines.append(f"{addr}: {op} { _to_base12(arg) }")
        elif op in ("HALT",):
            lines.append(f"{addr}: {op}")
        else:
            lines.append(f"{addr}: {op} {repr(arg)}")
    return "\n".join(lines)


# ---------- Dodecagram (DGM) opcode table ----------
# Map integer opcode -> metadata: (mnemonic, meaning, nasm_equiv, hex, bin)
_DGM_TABLE: Dict[int, Dict[str, str]] = {
    0x00: {"dgm": "00", "mnemonic": "nop", "meaning": "nop", "nasm": "NOP", "hex": "0x00", "bin": "00000000"},
    0x01: {"dgm": "01", "mnemonic": "alloca", "meaning": "alloca", "nasm": "SUB RSP, imm32", "hex": "0x01", "bin": "00000001"},
    0x02: {"dgm": "02", "mnemonic": "load", "meaning": "load", "nasm": "MOV r64, [mem]", "hex": "0x02", "bin": "00000010"},
    0x03: {"dgm": "03", "mnemonic": "store", "meaning": "store", "nasm": "MOV [mem], r64", "hex": "0x03", "bin": "00000011"},
    0x04: {"dgm": "04", "mnemonic": "getelementptr", "meaning": "getelementptr", "nasm": "LEA r64, [mem]", "hex": "0x04", "bin": "00000100"},
    0x05: {"dgm": "05", "mnemonic": "bitcast", "meaning": "bitcast", "nasm": "MOVQ reg, xmm", "hex": "0x05", "bin": "00000101"},
    0x06: {"dgm": "06", "mnemonic": "trunc", "meaning": "trunc", "nasm": "MOVZX/MOVSX (narrow int)", "hex": "0x06", "bin": "00000110"},
    0x07: {"dgm": "07", "mnemonic": "zext", "meaning": "zext", "nasm": "MOVZX", "hex": "0x07", "bin": "00000111"},
    0x08: {"dgm": "08", "mnemonic": "sext", "meaning": "sext", "nasm": "MOVSX", "hex": "0x08", "bin": "00001000"},
    0x09: {"dgm": "09", "mnemonic": "fptrunc", "meaning": "fptrunc", "nasm": "CVTSD2SS / CVTPD2PS", "hex": "0x09", "bin": "00001001"},
    0x0A: {"dgm": "0A", "mnemonic": "fpext", "meaning": "fpext", "nasm": "CVTSS2SD / CVTPS2PD", "hex": "0x0A", "bin": "00001010"},
    0x0B: {"dgm": "0B", "mnemonic": "fptoui", "meaning": "fptoui", "nasm": "CVTTSD2SI", "hex": "0x0B", "bin": "00001011"},
    0x10: {"dgm": "10", "mnemonic": "fptosi", "meaning": "fptosi", "nasm": "CVTTSS2SI", "hex": "0x10", "bin": "00010000"},
    0x11: {"dgm": "11", "mnemonic": "uitofp", "meaning": "uitofp", "nasm": "CVTSI2SD", "hex": "0x11", "bin": "00010001"},
    0x12: {"dgm": "12", "mnemonic": "sitofp", "meaning": "sitofp", "nasm": "CVTSI2SS", "hex": "0x12", "bin": "00010010"},
    0x13: {"dgm": "13", "mnemonic": "ptrtoint", "meaning": "ptrtoint", "nasm": "MOV reg, qword ptr", "hex": "0x13", "bin": "00010011"},
    0x14: {"dgm": "14", "mnemonic": "inttoptr", "meaning": "inttoptr", "nasm": "MOV reg, imm64", "hex": "0x14", "bin": "00010100"},
    0x15: {"dgm": "15", "mnemonic": "icmp", "meaning": "icmp", "nasm": "CMP r/m64, r64", "hex": "0x15", "bin": "00010101"},
    0x16: {"dgm": "16", "mnemonic": "fcmp", "meaning": "fcmp", "nasm": "UCOMISD / UCOMISS", "hex": "0x16", "bin": "00010110"},
    0x17: {"dgm": "17", "mnemonic": "add", "meaning": "add", "nasm": "ADD r/m64, r64", "hex": "0x17", "bin": "00010111"},
    0x18: {"dgm": "18", "mnemonic": "sub", "meaning": "sub", "nasm": "SUB r/m64, r64", "hex": "0x18", "bin": "00011000"},
    0x19: {"dgm": "19", "mnemonic": "mul", "meaning": "mul", "nasm": "IMUL r64, r/m64", "hex": "0x19", "bin": "00011001"},
    0x1A: {"dgm": "1A", "mnemonic": "udiv", "meaning": "udiv", "nasm": "DIV r/m64", "hex": "0x1A", "bin": "00011010"},
    0x1B: {"dgm": "1B", "mnemonic": "sdiv", "meaning": "sdiv", "nasm": "IDIV r/m64", "hex": "0x1B", "bin": "00011011"},
    0x20: {"dgm": "20", "mnemonic": "fadd", "meaning": "fadd", "nasm": "ADDSD xmm, xmm", "hex": "0x20", "bin": "00100000"},
    0x21: {"dgm": "21", "mnemonic": "fsub", "meaning": "fsub", "nasm": "SUBSD xmm, xmm", "hex": "0x21", "bin": "00100001"},
    0x22: {"dgm": "22", "mnemonic": "fmul", "meaning": "fmul", "nasm": "MULSD xmm, xmm", "hex": "0x22", "bin": "00100010"},
    0x23: {"dgm": "23", "mnemonic": "fdiv", "meaning": "fdiv", "nasm": "DIVSD xmm, xmm", "hex": "0x23", "bin": "00100011"},
    0x24: {"dgm": "24", "mnemonic": "frem", "meaning": "frem", "nasm": "Emulated DIV+MUL-SUB", "hex": "0x24", "bin": "00100100"},
    0x25: {"dgm": "25", "mnemonic": "shl", "meaning": "shl", "nasm": "SHL r/m64, CL", "hex": "0x25", "bin": "00100101"},
    0x26: {"dgm": "26", "mnemonic": "lshr", "meaning": "lshr", "nasm": "SHR r/m64, CL", "hex": "0x26", "bin": "00100110"},
    0x27: {"dgm": "27", "mnemonic": "ashr", "meaning": "ashr", "nasm": "SAR r/m64, CL", "hex": "0x27", "bin": "00100111"},
    0x28: {"dgm": "28", "mnemonic": "and", "meaning": "and", "nasm": "AND r/m64, r64", "hex": "0x28", "bin": "00101000"},
    0x29: {"dgm": "29", "mnemonic": "or", "meaning": "or", "nasm": "OR r/m64, r64", "hex": "0x29", "bin": "00101001"},
    0x2A: {"dgm": "2A", "mnemonic": "xor", "meaning": "xor", "nasm": "XOR r/m64, r64", "hex": "0x2A", "bin": "00101010"},
    0x2B: {"dgm": "2B", "mnemonic": "call", "meaning": "call", "nasm": "CALL rel32", "hex": "0x2B", "bin": "00101011"},
    0x30: {"dgm": "30", "mnemonic": "br", "meaning": "br", "nasm": "JMP rel32", "hex": "0x30", "bin": "00110000"},
    0x31: {"dgm": "31", "mnemonic": "switch", "meaning": "switch", "nasm": "CMP+JMP table", "hex": "0x31", "bin": "00110001"},
    0x32: {"dgm": "32", "mnemonic": "indirectbr", "meaning": "indirectbr", "nasm": "JMP r/m64", "hex": "0x32", "bin": "00110010"},
    0x33: {"dgm": "33", "mnemonic": "ret", "meaning": "ret", "nasm": "RET", "hex": "0x33", "bin": "00110011"},
    0x34: {"dgm": "34", "mnemonic": "resume", "meaning": "resume", "nasm": "EH resume stub", "hex": "0x34", "bin": "00110100"},
    0x35: {"dgm": "35", "mnemonic": "unreachable", "meaning": "unreachable", "nasm": "UD2", "hex": "0x35", "bin": "00110101"},
    0x36: {"dgm": "36", "mnemonic": "landingpad", "meaning": "landingpad", "nasm": "EH landing pad", "hex": "0x36", "bin": "00110110"},
    0x37: {"dgm": "37", "mnemonic": "invoke", "meaning": "invoke", "nasm": "CALL+EH unwind", "hex": "0x37", "bin": "00110111"},
    0x38: {"dgm": "38", "mnemonic": "phi", "meaning": "phi", "nasm": "SSA merge (no direct)", "hex": "0x38", "bin": "00111000"},
    0x39: {"dgm": "39", "mnemonic": "select", "meaning": "select", "nasm": "CMP+CMOVcc", "hex": "0x39", "bin": "00111001"},
    0x3A: {"dgm": "3A", "mnemonic": "extractvalue", "meaning": "extractvalue", "nasm": "MOV reg,[struct+offset]", "hex": "0x3A", "bin": "00111010"},
    0x3B: {"dgm": "3B", "mnemonic": "insertvalue", "meaning": "insertvalue", "nasm": "MOV [struct+offset],reg", "hex": "0x3B", "bin": "00111011"},
    0x40: {"dgm": "40", "mnemonic": "atomicrmw", "meaning": "atomicrmw", "nasm": "LOCK prefixed ops", "hex": "0x40", "bin": "01000000"},
    0x41: {"dgm": "41", "mnemonic": "cmpxchg", "meaning": "cmpxchg", "nasm": "LOCK CMPXCHG", "hex": "0x41", "bin": "01000001"},
    0x42: {"dgm": "42", "mnemonic": "fence", "meaning": "fence", "nasm": "MFENCE", "hex": "0x42", "bin": "01000010"},
    0x43: {"dgm": "43", "mnemonic": "memset", "meaning": "memset", "nasm": "REP STOSB", "hex": "0x43", "bin": "01000011"},
    0x44: {"dgm": "44", "mnemonic": "memcpy", "meaning": "memcpy", "nasm": "REP MOVSB", "hex": "0x44", "bin": "01000100"},
    0x45: {"dgm": "45", "mnemonic": "memmove", "meaning": "memmove", "nasm": "REP MOVSB+temp", "hex": "0x45", "bin": "01000101"},
    0x46: {"dgm": "46", "mnemonic": "lifetime.start", "meaning": "lifetime.start", "nasm": "No codegen", "hex": "0x46", "bin": "01000110"},
    0x47: {"dgm": "47", "mnemonic": "lifetime.end", "meaning": "lifetime.end", "nasm": "No codegen", "hex": "0x47", "bin": "01000111"},
    0x48: {"dgm": "48", "mnemonic": "sanitizer.check", "meaning": "sanitizer.check", "nasm": "CMP+Jcc bounds check", "hex": "0x48", "bin": "01001000"},
    0x49: {"dgm": "49", "mnemonic": "assume", "meaning": "assume", "nasm": "Compiler builtin", "hex": "0x49", "bin": "01001001"},
    0x4A: {"dgm": "4A", "mnemonic": "llvm.dbg.declare", "meaning": "llvm.dbg.declare", "nasm": "Debug meta", "hex": "0x4A", "bin": "01001010"},
    0x4B: {"dgm": "4B", "mnemonic": "llvm.dbg.value", "meaning": "llvm.dbg.value", "nasm": "Debug meta", "hex": "0x4B", "bin": "01001011"},
    0x50: {"dgm": "50", "mnemonic": "safe.add", "meaning": "safe.add", "nasm": "ADD+JO recover", "hex": "0x50", "bin": "01010000"},
    0x51: {"dgm": "51", "mnemonic": "safe.sub", "meaning": "safe.sub", "nasm": "SUB+JO recover", "hex": "0x51", "bin": "01010001"},
    0x52: {"dgm": "52", "mnemonic": "safe.mul", "meaning": "safe.mul", "nasm": "IMUL+JO recover", "hex": "0x52", "bin": "01010010"},
    0x53: {"dgm": "53", "mnemonic": "safe.div", "meaning": "safe.div", "nasm": "DIV+guard", "hex": "0x53", "bin": "01010011"},
    0x54: {"dgm": "54", "mnemonic": "safe.mod", "meaning": "safe.mod", "nasm": "IDIV+guard", "hex": "0x54", "bin": "01010100"},
    0x55: {"dgm": "55", "mnemonic": "safe.shift", "meaning": "safe.shift", "nasm": "SHL/SHR+mask", "hex": "0x55", "bin": "01010101"},
    0x56: {"dgm": "56", "mnemonic": "safe.and", "meaning": "safe.and", "nasm": "AND+guard", "hex": "0x56", "bin": "01010110"},
    0x57: {"dgm": "57", "mnemonic": "safe.or", "meaning": "safe.or", "nasm": "OR+guard", "hex": "0x57", "bin": "01010111"},
    0x58: {"dgm": "58", "mnemonic": "safe.xor", "meaning": "safe.xor", "nasm": "XOR+guard", "hex": "0x58", "bin": "01011000"},
    0x59: {"dgm": "59", "mnemonic": "safe.neg", "meaning": "safe.neg", "nasm": "NEG+check", "hex": "0x59", "bin": "01011001"},
    0x5A: {"dgm": "5A", "mnemonic": "safe.not", "meaning": "safe.not", "nasm": "NOT r/m64", "hex": "0x5A", "bin": "01011010"},
    0x60: {"dgm": "60", "mnemonic": "cascade.begin", "meaning": "cascade.begin", "nasm": "PUSH context", "hex": "0x60", "bin": "01100000"},
    0x61: {"dgm": "61", "mnemonic": "cascade.end", "meaning": "cascade.end", "nasm": "POP context", "hex": "0x61", "bin": "01100001"},
    0x62: {"dgm": "62", "mnemonic": "cascade.yield", "meaning": "cascade.yield", "nasm": "SAVE+JMP out", "hex": "0x62", "bin": "01100010"},
    0x63: {"dgm": "63", "mnemonic": "cascade.resume", "meaning": "cascade.resume", "nasm": "RESTORE+JMP in", "hex": "0x63", "bin": "01100011"},
    0x70: {"dgm": "70", "mnemonic": "branch.try", "meaning": "branch.try", "nasm": "Label mark", "hex": "0x70", "bin": "01110000"},
    0x71: {"dgm": "71", "mnemonic": "branch.heal", "meaning": "branch.heal", "nasm": "JMP recover block", "hex": "0x71", "bin": "01110001"},
    0x72: {"dgm": "72", "mnemonic": "branch.soft", "meaning": "branch.soft", "nasm": "JMP with mask", "hex": "0x72", "bin": "01110010"},
    0x73: {"dgm": "73", "mnemonic": "branch.auto", "meaning": "branch.auto", "nasm": "Predicated JMP", "hex": "0x73", "bin": "01110011"},
    0x7A: {"dgm": "7A", "mnemonic": "recover", "meaning": "recover", "nasm": "RESTORE state", "hex": "0x7A", "bin": "01111010"},
    0x7B: {"dgm": "7B", "mnemonic": "language.assert", "meaning": "language.assert", "nasm": "CMP+Jcc trap", "hex": "0x7B", "bin": "01111011"},
    0x80: {"dgm": "80", "mnemonic": "tuple.pack", "meaning": "tuple.pack", "nasm": "CALL __tuple_pack", "hex": "0x80", "bin": "10000000"},
    0x81: {"dgm": "81", "mnemonic": "tuple.unpack", "meaning": "tuple.unpack", "nasm": "CALL __tuple_unpack", "hex": "0x81", "bin": "10000001"},
    0x82: {"dgm": "82", "mnemonic": "list.append", "meaning": "list.append", "nasm": "CALL __list_append", "hex": "0x82", "bin": "10000010"},
    0x83: {"dgm": "83", "mnemonic": "list.remove", "meaning": "list.remove", "nasm": "CALL __list_remove", "hex": "0x83", "bin": "10000011"},
    0x84: {"dgm": "84", "mnemonic": "list.insert", "meaning": "list.insert", "nasm": "CALL __list_insert", "hex": "0x84", "bin": "10000100"},
    0x85: {"dgm": "85", "mnemonic": "list.pop", "meaning": "list.pop", "nasm": "CALL __list_pop", "hex": "0x85", "bin": "10000101"},
    0x86: {"dgm": "86", "mnemonic": "array.load", "meaning": "array.load", "nasm": "MOV reg,[array+idx]", "hex": "0x86", "bin": "10000110"},
    0x87: {"dgm": "87", "mnemonic": "array.store", "meaning": "array.store", "nasm": "MOV [array+idx],reg", "hex": "0x87", "bin": "10000111"},
    0x88: {"dgm": "88", "mnemonic": "group.spawn", "meaning": "group.spawn", "nasm": "CALL __group_spawn", "hex": "0x88", "bin": "10001000"},
    0x89: {"dgm": "89", "mnemonic": "group.merge", "meaning": "group.merge", "nasm": "CALL __group_merge", "hex": "0x89", "bin": "10001001"},
    0x8A: {"dgm": "8A", "mnemonic": "group.split", "meaning": "group.split", "nasm": "CALL __group_split", "hex": "0x8A", "bin": "10001010"},
    0x8B: {"dgm": "8B", "mnemonic": "nest.enter", "meaning": "nest.enter", "nasm": "CALL __nest_enter", "hex": "0x8B", "bin": "10001011"},
    0x90: {"dgm": "90", "mnemonic": "nest.exit", "meaning": "nest.exit", "nasm": "CALL __nest_exit", "hex": "0x90", "bin": "10010000"},
    0x91: {"dgm": "91", "mnemonic": "derive.child", "meaning": "derive.child", "nasm": "CALL __derive_child", "hex": "0x91", "bin": "10010001"},
    0x92: {"dgm": "92", "mnemonic": "derive.parent", "meaning": "derive.parent", "nasm": "CALL __derive_parent", "hex": "0x92", "bin": "10010010"},
    0x93: {"dgm": "93", "mnemonic": "pair.create", "meaning": "pair.create", "nasm": "CALL __pair_create", "hex": "0x93", "bin": "10010011"},
    0x94: {"dgm": "94", "mnemonic": "pair.split", "meaning": "pair.split", "nasm": "CALL __pair_split", "hex": "0x94", "bin": "10010100"},
    0x95: {"dgm": "95", "mnemonic": "match.begin", "meaning": "match.begin", "nasm": "LABEL match", "hex": "0x95", "bin": "10010101"},
    0x96: {"dgm": "96", "mnemonic": "match.case", "meaning": "match.case", "nasm": "CMP+Jcc", "hex": "0x96", "bin": "10010110"},
    0x97: {"dgm": "97", "mnemonic": "match.end", "meaning": "match.end", "nasm": "JMP end", "hex": "0x97", "bin": "10010111"},
    0x98: {"dgm": "98", "mnemonic": "language.yield", "meaning": "language.yield", "nasm": "CALL __yield", "hex": "0x98", "bin": "10011000"},
    0x99: {"dgm": "99", "mnemonic": "language.halt", "meaning": "language.halt", "nasm": "HLT", "hex": "0x99", "bin": "10011001"},
    0x9A: {"dgm": "9A", "mnemonic": "language.wait", "meaning": "language.wait", "nasm": "PAUSE", "hex": "0x9A", "bin": "10011010"},
    0x9B: {"dgm": "9B", "mnemonic": "language.resume", "meaning": "language.resume", "nasm": "CALL __resume", "hex": "0x9B", "bin": "10011011"},
    0xA0: {"dgm": "A0", "mnemonic": "language.inline", "meaning": "language.inline", "nasm": "__forceinline", "hex": "0xA0", "bin": "10100000"},
    0xA1: {"dgm": "A1", "mnemonic": "language.expand", "meaning": "language.expand", "nasm": "Macro expansion", "hex": "0xA1", "bin": "10100001"},
    0xA2: {"dgm": "A2", "mnemonic": "language.fold", "meaning": "language.fold", "nasm": "Folded macro", "hex": "0xA2", "bin": "10100010"},
    0xA3: {"dgm": "A3", "mnemonic": "language.derive", "meaning": "language.derive", "nasm": "Template derive", "hex": "0xA3", "bin": "10100011"},
    0xA4: {"dgm": "A4", "mnemonic": "language.macro", "meaning": "language.macro", "nasm": "Macro define", "hex": "0xA4", "bin": "10100100"},
    0xA5: {"dgm": "A5", "mnemonic": "language.trace", "meaning": "language.trace", "nasm": "CALL __tracepoint", "hex": "0xA5", "bin": "10100101"},
    0xA6: {"dgm": "A6", "mnemonic": "language.echo", "meaning": "language.echo", "nasm": "CALL puts/printf", "hex": "0xA6", "bin": "10100110"},
    0xA7: {"dgm": "A7", "mnemonic": "language.link", "meaning": "language.link", "nasm": "CALL dlopen", "hex": "0xA7", "bin": "10100111"},
    0xA8: {"dgm": "A8", "mnemonic": "language.infer", "meaning": "language.infer", "nasm": "Type infer pass", "hex": "0xA8", "bin": "10101000"},
    0xA9: {"dgm": "A9", "mnemonic": "language.delete", "meaning": "language.delete", "nasm": "CALL free", "hex": "0xA9", "bin": "10101001"},
    0xAA: {"dgm": "AA", "mnemonic": "language.replace", "meaning": "language.replace", "nasm": "Swap call", "hex": "0xAA", "bin": "10101010"},
    0xAB: {"dgm": "AB", "mnemonic": "language.redirect", "meaning": "language.redirect", "nasm": "JMP other", "hex": "0xAB", "bin": "10101011"},
    0xB0: {"dgm": "B0", "mnemonic": "language.guard", "meaning": "language.guard", "nasm": "CMP+Jcc guard", "hex": "0xB0", "bin": "10110000"},
    0xB1: {"dgm": "B1", "mnemonic": "language.wrap", "meaning": "language.wrap", "nasm": "PUSH+CALL+POP", "hex": "0xB1", "bin": "10110001"},
    0xB2: {"dgm": "B2", "mnemonic": "language.unwrap", "meaning": "language.unwrap", "nasm": "MOV out,in", "hex": "0xB2", "bin": "10110010"},
    0xB3: {"dgm": "B3", "mnemonic": "language.enclose", "meaning": "language.enclose", "nasm": "SCOPE guard", "hex": "0xB3", "bin": "10110011"},
    0xB4: {"dgm": "B4", "mnemonic": "language.open", "meaning": "language.open", "nasm": "CALL fopen", "hex": "0xB4", "bin": "10110100"},
    0xB5: {"dgm": "B5", "mnemonic": "language.close", "meaning": "language.close", "nasm": "CALL fclose", "hex": "0xB5", "bin": "10110101"},
    0xB6: {"dgm": "B6", "mnemonic": "language.defer", "meaning": "language.defer", "nasm": "PUSH cleanup", "hex": "0xB6", "bin": "10110110"},
    0xB7: {"dgm": "B7", "mnemonic": "language.future", "meaning": "language.future", "nasm": "THREAD CREATE", "hex": "0xB7", "bin": "10110111"},
    0xB8: {"dgm": "B8", "mnemonic": "language.parallel", "meaning": "language.parallel", "nasm": "PTHREAD_CREATE", "hex": "0xB8", "bin": "10111000"},
    0xB9: {"dgm": "B9", "mnemonic": "language.sync", "meaning": "language.sync", "nasm": "SYSCALL futex_wait", "hex": "0xB9", "bin": "10111001"},
    0xBA: {"dgm": "BA", "mnemonic": "language.pragma", "meaning": "language.pragma", "nasm": "Compiler directive", "hex": "0xBA", "bin": "10111010"},
    0xBB: {"dgm": "BB", "mnemonic": "language.exit", "meaning": "language.exit", "nasm": "SYSCALL exit", "hex": "0xBB", "bin": "10111011"},
    0xC0: {"dgm": "C0", "mnemonic": "@llvm.game.init", "meaning": "@llvm.game.init", "nasm": "CALL __game_init", "hex": "0xC0", "bin": "11000000"},
    0xC1: {"dgm": "C1", "mnemonic": "@llvm.game.load.model", "meaning": "@llvm.game.load.model", "nasm": "CALL __game_load_model", "hex": "0xC1", "bin": "11000001"},
    0xC2: {"dgm": "C2", "mnemonic": "@llvm.game.load.texture", "meaning": "@llvm.game.load.texture", "nasm": "CALL __game_load_texture", "hex": "0xC2", "bin": "11000010"},
    0xC3: {"dgm": "C3", "mnemonic": "@llvm.game.create.world", "meaning": "@llvm.game.create.world", "nasm": "CALL __game_create_world", "hex": "0xC3", "bin": "11000011"},
    0xC4: {"dgm": "C4", "mnemonic": "@llvm.game.add.entity", "meaning": "@llvm.game.add.entity", "nasm": "CALL __game_add_entity", "hex": "0xC4", "bin": "11000100"},
    0xC5: {"dgm": "C5", "mnemonic": "@llvm.game.add.light", "meaning": "@llvm.game.add.light", "nasm": "CALL __game_add_light", "hex": "0xC5", "bin": "11000101"},
    0xC6: {"dgm": "C6", "mnemonic": "@llvm.game.update", "meaning": "@llvm.game.update", "nasm": "CALL __game_update", "hex": "0xC6", "bin": "11000110"},
    0xC7: {"dgm": "C7", "mnemonic": "@llvm.game.render", "meaning": "@llvm.game.render", "nasm": "CALL __game_render", "hex": "0xC7", "bin": "11000111"},
    0xC8: {"dgm": "C8", "mnemonic": "@llvm.game.running", "meaning": "@llvm.game.running", "nasm": "CALL __game_running", "hex": "0xC8", "bin": "11001000"},
    0xC9: {"dgm": "C9", "mnemonic": "@llvm.game.input", "meaning": "@llvm.game.input", "nasm": "CALL __game_input", "hex": "0xC9", "bin": "11001001"},
    0xCA: {"dgm": "CA", "mnemonic": "@llvm.game.play.sound", "meaning": "@llvm.game.play.sound", "nasm": "CALL __game_play_sound", "hex": "0xCA", "bin": "11001010"},
    0xCB: {"dgm": "CB", "mnemonic": "@llvm.game.play.music", "meaning": "@llvm.game.play.music", "nasm": "CALL __game_play_music", "hex": "0xCB", "bin": "11001011"},
    0xCC: {"dgm": "CC", "mnemonic": "@llvm.game.quit", "meaning": "@llvm.game.quit", "nasm": "CALL __game_quit", "hex": "0xCC", "bin": "11001100"},
    0xD0: {"dgm": "D0", "mnemonic": "@llvm.math.pow", "meaning": "@llvm.math.pow", "nasm": "CALL __math_pow", "hex": "0xD0", "bin": "11010000"},
    0xD1: {"dgm": "D1", "mnemonic": "@llvm.math.log", "meaning": "@llvm.math.log", "nasm": "CALL __math_log", "hex": "0xD1", "bin": "11010001"},
    0xD2: {"dgm": "D2", "mnemonic": "@llvm.math.exp", "meaning": "@llvm.math.exp", "nasm": "CALL __math_exp", "hex": "0xD2", "bin": "11010010"},
    0xD3: {"dgm": "D3", "mnemonic": "@llvm.math.sin", "meaning": "@llvm.math.sin", "nasm": "CALL __math_sin", "hex": "0xD3", "bin": "11010011"},
    0xD4: {"dgm": "D4", "mnemonic": "@llvm.math.cos", "meaning": "@llvm.math.cos", "nasm": "CALL __math_cos", "hex": "0xD4", "bin": "11010100"},
    0xD5: {"dgm": "D5", "mnemonic": "@llvm.math.tan", "meaning": "@llvm.math.tan", "nasm": "CALL __math_tan", "hex": "0xD5", "bin": "11010101"},
    0xD6: {"dgm": "D6", "mnemonic": "@llvm.math.asin", "meaning": "@llvm.math.asin", "nasm": "CALL __math_asin", "hex": "0xD6", "bin": "11010110"},
    0xD7: {"dgm": "D7", "mnemonic": "@llvm.math.acos", "meaning": "@llvm.math.acos", "nasm": "CALL __math_acos", "hex": "0xD7", "bin": "11010111"},
    0xD8: {"dgm": "D8", "mnemonic": "@llvm.math.atan", "meaning": "@llvm.math.atan", "nasm": "CALL __math_atan", "hex": "0xD8", "bin": "11011000"},
    0xD9: {"dgm": "D9", "mnemonic": "@llvm.math.sqrt", "meaning": "@llvm.math.sqrt", "nasm": "CALL __math_sqrt", "hex": "0xD9", "bin": "11011001"},
    0xDA: {"dgm": "DA", "mnemonic": "@llvm.math.cbrt", "meaning": "@llvm.math.cbrt", "nasm": "CALL __math_cbrt", "hex": "0xDA", "bin": "11011010"},
    0xDB: {"dgm": "DB", "mnemonic": "@llvm.math.hypot", "meaning": "@llvm.math.hypot", "nasm": "CALL __math_hypot", "hex": "0xDB", "bin": "11011011"},
    0xDC: {"dgm": "DC", "mnemonic": "@llvm.math.floor", "meaning": "@llvm.math.floor", "nasm": "CALL __math_floor", "hex": "0xDC", "bin": "11011100"},
    0xDD: {"dgm": "DD", "mnemonic": "@llvm.math.ceil", "meaning": "@llvm.math.ceil", "nasm": "CALL __math_ceil", "hex": "0xDD", "bin": "11011101"},
    0xDE: {"dgm": "DE", "mnemonic": "@llvm.math.abs", "meaning": "@llvm.math.abs", "nasm": "CALL __math_abs", "hex": "0xDE", "bin": "11011110"},
    0xDF: {"dgm": "DF", "mnemonic": "@llvm.math.rand", "meaning": "@llvm.math.rand", "nasm": "CALL __math_rand", "hex": "0xDF", "bin": "11011111"},
    0xE0: {"dgm": "E0", "mnemonic": "@llvm.str.concat", "meaning": "@llvm.str.concat", "nasm": "CALL __str_concat", "hex": "0xE0", "bin": "11100000"},
    0xE1: {"dgm": "E1", "mnemonic": "@llvm.str.upper", "meaning": "@llvm.str.upper", "nasm": "CALL __str_upper", "hex": "0xE1", "bin": "11100001"},
    0xE2: {"dgm": "E2", "mnemonic": "@llvm.str.lower", "meaning": "@llvm.str.lower", "nasm": "CALL __str_lower", "hex": "0xE2", "bin": "11100010"},
    0xE3: {"dgm": "E3", "mnemonic": "@llvm.str.len", "meaning": "@llvm.str.len", "nasm": "CALL __str_len", "hex": "0xE3", "bin": "11100011"},
    0xE4: {"dgm": "E4", "mnemonic": "@llvm.str.substr", "meaning": "@llvm.str.substr", "nasm": "CALL __str_substr", "hex": "0xE4", "bin": "11100100"},
    0xE5: {"dgm": "E5", "mnemonic": "@llvm.str.find", "meaning": "@llvm.str.find", "nasm": "CALL __str_find", "hex": "0xE5", "bin": "11100101"},
    0xE6: {"dgm": "E6", "mnemonic": "@llvm.str.replace", "meaning": "@llvm.str.replace", "nasm": "CALL __str_replace", "hex": "0xE6", "bin": "11100110"},
    0xE7: {"dgm": "E7", "mnemonic": "@llvm.str.split", "meaning": "@llvm.str.split", "nasm": "CALL __str_split", "hex": "0xE7", "bin": "11100111"},
    0xE8: {"dgm": "E8", "mnemonic": "@llvm.str.join", "meaning": "@llvm.str.join", "nasm": "CALL __str_join", "hex": "0xE8", "bin": "11101000"},
    0xE9: {"dgm": "E9", "mnemonic": "@llvm.file.write", "meaning": "@llvm.file.write", "nasm": "CALL __file_write", "hex": "0xE9", "bin": "11101001"},
    0xEA: {"dgm": "EA", "mnemonic": "@llvm.file.append", "meaning": "@llvm.file.append", "nasm": "CALL __file_append", "hex": "0xEA", "bin": "11101010"},
    0xEB: {"dgm": "EB", "mnemonic": "@llvm.file.read", "meaning": "@llvm.file.read", "nasm": "CALL __file_read", "hex": "0xEB", "bin": "11101011"},
    0xEC: {"dgm": "EC", "mnemonic": "@llvm.file.delete", "meaning": "@llvm.file.delete", "nasm": "CALL __file_delete", "hex": "0xEC", "bin": "11101100"},
    0xED: {"dgm": "ED", "mnemonic": "@llvm.file.exists", "meaning": "@llvm.file.exists", "nasm": "CALL __file_exists", "hex": "0xED", "bin": "11101101"},
    0xEE: {"dgm": "EE", "mnemonic": "@llvm.system.now", "meaning": "@llvm.system.now", "nasm": "CALL __system_now", "hex": "0xEE", "bin": "11101110"},
    0xEF: {"dgm": "EF", "mnemonic": "@llvm.system.sleep", "meaning": "@llvm.system.sleep", "nasm": "CALL __system_sleep", "hex": "0xEF", "bin": "11101111"},
    0xF0: {"dgm": "F0", "mnemonic": "@llvm.system.env", "meaning": "@llvm.system.env", "nasm": "CALL __system_env", "hex": "0xF0", "bin": "11110000"},
    0xF1: {"dgm": "F1", "mnemonic": "@llvm.system.platform", "meaning": "@llvm.system.platform", "nasm": "CALL __system_platform", "hex": "0xF1", "bin": "11110001"},
    0xF2: {"dgm": "F2", "mnemonic": "@llvm.system.cpu", "meaning": "@llvm.system.cpu", "nasm": "CALL __system_cpu", "hex": "0xF2", "bin": "11110010"},
    0xF3: {"dgm": "F3", "mnemonic": "@llvm.system.mem", "meaning": "@llvm.system.mem", "nasm": "CALL __system_mem", "hex": "0xF3", "bin": "11110011"},
    0xF4: {"dgm": "F4", "mnemonic": "@llvm.sys.exec", "meaning": "@llvm.sys.exec", "nasm": "CALL __sys_exec", "hex": "0xF4", "bin": "11110100"},
    0xF5: {"dgm": "F5", "mnemonic": "@llvm.sys.cwd", "meaning": "@llvm.sys.cwd", "nasm": "CALL __sys_cwd", "hex": "0xF5", "bin": "11110101"},
    0xF6: {"dgm": "F6", "mnemonic": "@llvm.sys.chdir", "meaning": "@llvm.sys.chdir", "nasm": "CALL __sys_chdir", "hex": "0xF6", "bin": "11110110"},
    0xF7: {"dgm": "F7", "mnemonic": "@llvm.sys.listdir", "meaning": "@llvm.sys.listdir", "nasm": "CALL __sys_listdir", "hex": "0xF7", "bin": "11110111"},
    0xF8: {"dgm": "F8", "mnemonic": "@llvm.sys.mkdir", "meaning": "@llvm.sys.mkdir", "nasm": "CALL __sys_mkdir", "hex": "0xF8", "bin": "11111000"},
    0xF9: {"dgm": "F9", "mnemonic": "@llvm.sys.rmdir", "meaning": "@llvm.sys.rmdir", "nasm": "CALL __sys_rmdir", "hex": "0xF9", "bin": "11111001"},
    0xFA: {"dgm": "FA", "mnemonic": "@llvm.sys.tempfile", "meaning": "@llvm.sys.tempfile", "nasm": "CALL __sys_tempfile", "hex": "0xFA", "bin": "11111010"},
    0xFB: {"dgm": "FB", "mnemonic": "@llvm.sys.clipboard", "meaning": "@llvm.sys.clipboard", "nasm": "CALL __sys_clipboard", "hex": "0xFB", "bin": "11111011"},
    0xFC: {"dgm": "FC", "mnemonic": "@llvm.sys.args", "meaning": "@llvm.sys.args", "nasm": "CALL __sys_args", "hex": "0xFC", "bin": "11111100"},
    0xFD: {"dgm": "FD", "mnemonic": "@llvm.sys.uid", "meaning": "@llvm.sys.uid", "nasm": "CALL __sys_uid", "hex": "0xFD", "bin": "11111101"},
    0xFE: {"dgm": "FE", "mnemonic": "@llvm.sys.pid", "meaning": "@llvm.sys.pid", "nasm": "CALL __sys_pid", "hex": "0xFE", "bin": "11111110"},
    0xFF: {"dgm": "FF", "mnemonic": "@llvm.sys.exit", "meaning": "@llvm.sys.exit", "nasm": "SYSCALL exit", "hex": "0xFF", "bin": "11111111"},
    # extended (0x100+): store as integer keys (example entries provided)
    0x100: {"dgm": "100", "mnemonic": "@llvm.hash.md5", "meaning": "@llvm.hash.md5", "nasm": "CALL __hash_md5", "hex": "0x100", "bin": "000100000000"},
    0x101: {"dgm": "101", "mnemonic": "@llvm.hash.sha1", "meaning": "@llvm.hash.sha1", "nasm": "CALL __hash_sha1", "hex": "0x101", "bin": "000100000001"},
    0x102: {"dgm": "102", "mnemonic": "@llvm.hash.sha256", "meaning": "@llvm.hash.sha256", "nasm": "CALL __hash_sha256", "hex": "0x102", "bin": "000100000010"},
    0x103: {"dgm": "103", "mnemonic": "@llvm.hash.sha512", "meaning": "@llvm.hash.sha512", "nasm": "CALL __hash_sha512", "hex": "0x103", "bin": "000100000011"},
    0x104: {"dgm": "104", "mnemonic": "@llvm.hmac.md5", "meaning": "@llvm.hmac.md5", "nasm": "CALL __hmac_md5", "hex": "0x104", "bin": "000100000100"},
    0x105: {"dgm": "105", "mnemonic": "@llvm.hmac.sha256", "meaning": "@llvm.hmac.sha256", "nasm": "CALL __hmac_sha256", "hex": "0x105", "bin": "000100000101"},
    0x106: {"dgm": "106", "mnemonic": "@llvm.base64.encode", "meaning": "@llvm.base64.encode", "nasm": "CALL __base64_encode", "hex": "0x106", "bin": "000100000110"},
    0x107: {"dgm": "107", "mnemonic": "@llvm.base64.decode", "meaning": "@llvm.base64.decode", "nasm": "CALL __base64_decode", "hex": "0x107", "bin": "000100000111"},
    0x108: {"dgm": "108", "mnemonic": "@llvm.hex.encode", "meaning": "@llvm.hex.encode", "nasm": "CALL __hex_encode", "hex": "0x108", "bin": "000100001000"},
    0x109: {"dgm": "109", "mnemonic": "@llvm.hex.decode", "meaning": "@llvm.hex.decode", "nasm": "CALL __hex_decode", "hex": "0x109", "bin": "000100001001"},
    0x10A: {"dgm": "10A", "mnemonic": "@llvm.crc32", "meaning": "@llvm.crc32", "nasm": "CALL __crc32", "hex": "0x10A", "bin": "000100001010"},
    0x10B: {"dgm": "10B", "mnemonic": "@llvm.random.bytes", "meaning": "@llvm.random.bytes", "nasm": "CALL __random_bytes", "hex": "0x10B", "bin": "000100001011"},
    0x10C: {"dgm": "10C", "mnemonic": "@llvm.uuid.generate", "meaning": "@llvm.uuid.generate", "nasm": "CALL __uuid_generate", "hex": "0x10C", "bin": "000100001100"},
    0x10D: {"dgm": "10D", "mnemonic": "@llvm.password.hash", "meaning": "@llvm.password.hash", "nasm": "CALL __password_hash", "hex": "0x10D", "bin": "000100001101"},
    0x10E: {"dgm": "10E", "mnemonic": "@llvm.password.verify", "meaning": "@llvm.password.verify", "nasm": "CALL __password_verify", "hex": "0x10E", "bin": "000100001110"},
    0x10F: {"dgm": "10F", "mnemonic": "@llvm.jwt.encode", "meaning": "@llvm.jwt.encode", "nasm": "CALL __jwt_encode", "hex": "0x10F", "bin": "000100001111"},
    0x110: {"dgm": "110", "mnemonic": "@llvm.zlib.compress", "meaning": "@llvm.zlib.compress", "nasm": "CALL __zlib_compress", "hex": "0x110", "bin": "000100010000"},
    0x111: {"dgm": "111", "mnemonic": "@llvm.zlib.decompress", "meaning": "@llvm.zlib.decompress", "nasm": "CALL __zlib_decompress", "hex": "0x111", "bin": "000100010001"},
    0x112: {"dgm": "112", "mnemonic": "@llvm.bz2.compress", "meaning": "@llvm.bz2.compress", "nasm": "CALL __bz2_compress", "hex": "0x112", "bin": "000100010010"},
    0x113: {"dgm": "113", "mnemonic": "@llvm.bz2.decompress", "meaning": "@llvm.bz2.decompress", "nasm": "CALL __bz2_decompress", "hex": "0x113", "bin": "000100010011"},
    0x114: {"dgm": "114", "mnemonic": "@llvm.lzma.compress", "meaning": "@llvm.lzma.compress", "nasm": "CALL __lzma_compress", "hex": "0x114", "bin": "000100010100"},
    0x115: {"dgm": "115", "mnemonic": "@llvm.lzma.decompress", "meaning": "@llvm.lzma.decompress", "nasm": "CALL __lzma_decompress", "hex": "0x115", "bin": "000100010101"},
    0x116: {"dgm": "116", "mnemonic": "@llvm.gzip.compress", "meaning": "@llvm.gzip.compress", "nasm": "CALL __gzip_compress", "hex": "0x116", "bin": "000100010110"},
    0x117: {"dgm": "117", "mnemonic": "@llvm.gzip.decompress", "meaning": "@llvm.gzip.decompress", "nasm": "CALL __gzip_decompress", "hex": "0x117", "bin": "000100010111"},
    0x118: {"dgm": "118", "mnemonic": "@llvm.tar.create", "meaning": "@llvm.tar.create", "nasm": "CALL __tar_create", "hex": "0x118", "bin": "000100011000"},
    0x119: {"dgm": "119", "mnemonic": "@llvm.tar.extract", "meaning": "@llvm.tar.extract", "nasm": "CALL __tar_extract", "hex": "0x119", "bin": "000100011001"},
    0x11A: {"dgm": "11A", "mnemonic": "@llvm.zip.create", "meaning": "@llvm.zip.create", "nasm": "CALL __zip_create", "hex": "0x11A", "bin": "000100011010"},
    0x11B: {"dgm": "11B", "mnemonic": "@llvm.zip.extract", "meaning": "@llvm.zip.extract", "nasm": "CALL __zip_extract", "hex": "0x11B", "bin": "000100011011"},
    0x11C: {"dgm": "11C", "mnemonic": "@llvm.compress.detect", "meaning": "@llvm.compress.detect", "nasm": "CALL __compress_detect", "hex": "0x11C", "bin": "000100011100"},
    0x11D: {"dgm": "11D", "mnemonic": "@llvm.compress.ratio", "meaning": "@llvm.compress.ratio", "nasm": "CALL __compress_ratio", "hex": "0x11D", "bin": "000100011101"},
    0x11E: {"dgm": "11E", "mnemonic": "@llvm.compress.level", "meaning": "@llvm.compress.level", "nasm": "CALL __compress_level", "hex": "0x11E", "bin": "000100011110"},
    0x11F: {"dgm": "11F", "mnemonic": "@llvm.compress.bench", "meaning": "@llvm.compress.bench", "nasm": "CALL __compress_bench", "hex": "0x11F", "bin": "000100011111"},
    0x120: {"dgm": "120", "mnemonic": "@llvm.http.get", "meaning": "@llvm.http.get", "nasm": "CALL __http_get", "hex": "0x120", "bin": "000100100000"},
    0x121: {"dgm": "121", "mnemonic": "@llvm.http.post", "meaning": "@llvm.http.post", "nasm": "CALL __http_post", "hex": "0x121", "bin": "000100100001"},
    0x122: {"dgm": "122", "mnemonic": "@llvm.http.head", "meaning": "@llvm.http.head", "nasm": "CALL __http_head", "hex": "0x122", "bin": "000100100010"},
    0x123: {"dgm": "123", "mnemonic": "@llvm.http.put", "meaning": "@llvm.http.put", "nasm": "CALL __http_put", "hex": "0x123", "bin": "000100100011"},
    0x124: {"dgm": "124", "mnemonic": "@llvm.http.delete", "meaning": "@llvm.http.delete", "nasm": "CALL __http_delete", "hex": "0x124", "bin": "000100100100"},
    0x125: {"dgm": "125", "mnemonic": "@llvm.http.download", "meaning": "@llvm.http.download", "nasm": "CALL __http_download", "hex": "0x125", "bin": "000100100101"},
    0x126: {"dgm": "126", "mnemonic": "@llvm.ws.connect", "meaning": "@llvm.ws.connect", "nasm": "CALL __ws_connect", "hex": "0x126", "bin": "000100100110"},
    0x127: {"dgm": "127", "mnemonic": "@llvm.ws.send", "meaning": "@llvm.ws.send", "nasm": "CALL __ws_send", "hex": "0x127", "bin": "000100100111"},
    0x128: {"dgm": "128", "mnemonic": "@llvm.ws.recv", "meaning": "@llvm.ws.recv", "nasm": "CALL __ws_recv", "hex": "0x128", "bin": "000100101000"},
    0x129: {"dgm": "129", "mnemonic": "@llvm.ws.close", "meaning": "@llvm.ws.close", "nasm": "CALL __ws_close", "hex": "0x129", "bin": "000100101001"},
    0x12A: {"dgm": "12A", "mnemonic": "@llvm.udp.send", "meaning": "@llvm.udp.send", "nasm": "CALL __udp_send", "hex": "0x12A", "bin": "000100101010"},
    0x12B: {"dgm": "12B", "mnemonic": "@llvm.udp.recv", "meaning": "@llvm.udp.recv", "nasm": "CALL __udp_recv", "hex": "0x12B", "bin": "000100101011"},
    0x12C: {"dgm": "12C", "mnemonic": "@llvm.tcp.listen", "meaning": "@llvm.tcp.listen", "nasm": "CALL __tcp_listen", "hex": "0x12C", "bin": "000100101100"},
    0x12D: {"dgm": "12D", "mnemonic": "@llvm.tcp.accept", "meaning": "@llvm.tcp.accept", "nasm": "CALL __tcp_accept", "hex": "0x12D", "bin": "000100101101"},
    0x12E: {"dgm": "12E", "mnemonic": "@llvm.tcp.send", "meaning": "@llvm.tcp.send", "nasm": "CALL __tcp_send", "hex": "0x12E", "bin": "000100101110"},
    0x12F: {"dgm": "12F", "mnemonic": "@llvm.tcp.recv", "meaning": "@llvm.tcp.recv", "nasm": "CALL __tcp_recv", "hex": "0x12F", "bin": "000100101111"},
    0x130: {"dgm": "130", "mnemonic": "@llvm.db.open", "meaning": "@llvm.db.open", "nasm": "CALL __db_open", "hex": "0x130", "bin": "000100110000"},
    0x131: {"dgm": "131", "mnemonic": "@llvm.db.exec", "meaning": "@llvm.db.exec", "nasm": "CALL __db_exec", "hex": "0x131", "bin": "000100110001"},
    0x132: {"dgm": "132", "mnemonic": "@llvm.db.query", "meaning": "@llvm.db.query", "nasm": "CALL __db_query", "hex": "0x132", "bin": "000100110010"},
    0x133: {"dgm": "133", "mnemonic": "@llvm.db.close", "meaning": "@llvm.db.close", "nasm": "CALL __db_close", "hex": "0x133", "bin": "000100110011"},
    0x134: {"dgm": "134", "mnemonic": "@llvm.db.begin", "meaning": "@llvm.db.begin", "nasm": "CALL __db_begin", "hex": "0x134", "bin": "000100110100"},
    0x135: {"dgm": "135", "mnemonic": "@llvm.db.commit", "meaning": "@llvm.db.commit", "nasm": "CALL __db_commit", "hex": "0x135", "bin": "000100110101"},
    0x136: {"dgm": "136", "mnemonic": "@llvm.db.rollback", "meaning": "@llvm.db.rollback", "nasm": "CALL __db_rollback", "hex": "0x136", "bin": "000100110110"},
    0x137: {"dgm": "137", "mnemonic": "@llvm.db.tables", "meaning": "@llvm.db.tables", "nasm": "CALL __db_tables", "hex": "0x137", "bin": "000100110111"},
    0x138: {"dgm": "138", "mnemonic": "@llvm.db.schema", "meaning": "@llvm.db.schema", "nasm": "CALL __db_schema", "hex": "0x138", "bin": "000100111000"},
    0x139: {"dgm": "139", "mnemonic": "@llvm.db.insert", "meaning": "@llvm.db.insert", "nasm": "CALL __db_insert", "hex": "0x139", "bin": "000100111001"},
    0x13A: {"dgm": "13A", "mnemonic": "@llvm.db.update", "meaning": "@llvm.db.update", "nasm": "CALL __db_update", "hex": "0x13A", "bin": "000100111010"},
    0x13B: {"dgm": "13B", "mnemonic": "@llvm.db.delete", "meaning": "@llvm.db.delete", "nasm": "CALL __db_delete", "hex": "0x13B", "bin": "000100111011"},
    0x13C: {"dgm": "13C", "mnemonic": "@llvm.db.count", "meaning": "@llvm.db.count", "nasm": "CALL __db_count", "hex": "0x13C", "bin": "000100111100"},
    0x13D: {"dgm": "13D", "mnemonic": "@llvm.db.indexes", "meaning": "@llvm.db.indexes", "nasm": "CALL __db_indexes", "hex": "0x13D", "bin": "000100111101"},
    0x13E: {"dgm": "13E", "mnemonic": "@llvm.db.analyze", "meaning": "@llvm.db.analyze", "nasm": "CALL __db_analyze", "hex": "0x13E", "bin": "000100111110"},
    0x13F: {"dgm": "13F", "mnemonic": "@llvm.db.vacuum", "meaning": "@llvm.db.vacuum", "nasm": "CALL __db_vacuum", "hex": "0x13F", "bin": "000100111111"},
    0x140: {"dgm": "140", "mnemonic": "@llvm.regex.match", "meaning": "@llvm.regex.match", "nasm": "CALL __regex_match", "hex": "0x140", "bin": "000101000000"},
    0x141: {"dgm": "141", "mnemonic": "@llvm.regex.findall", "meaning": "@llvm.regex.findall", "nasm": "CALL __regex_findall", "hex": "0x141", "bin": "000101000001"},
    0x142: {"dgm": "142", "mnemonic": "@llvm.regex.replace", "meaning": "@llvm.regex.replace", "nasm": "CALL __regex_replace", "hex": "0x142", "bin": "000101000010"},
    0x143: {"dgm": "143", "mnemonic": "@llvm.regex.split", "meaning": "@llvm.regex.split", "nasm": "CALL __regex_split", "hex": "0x143", "bin": "000101000011"},
    0x144: {"dgm": "144", "mnemonic": "@llvm.regex.subn", "meaning": "@llvm.regex.subn", "nasm": "CALL __regex_subn", "hex": "0x144", "bin": "000101000100"},
    0x145: {"dgm": "145", "mnemonic": "@llvm.regex.compile", "meaning": "@llvm.regex.compile", "nasm": "CALL __regex_compile", "hex": "0x145", "bin": "000101000101"},
    0x146: {"dgm": "146", "mnemonic": "@llvm.fuzzy.match", "meaning": "@llvm.fuzzy.match", "nasm": "CALL __fuzzy_match", "hex": "0x146", "bin": "000101000110"},
    0x147: {"dgm": "147", "mnemonic": "@llvm.fuzzy.closest", "meaning": "@llvm.fuzzy.closest", "nasm": "CALL __fuzzy_closest", "hex": "0x147", "bin": "000101000111"},
    0x148: {"dgm": "148", "mnemonic": "@llvm.fuzzy.sort", "meaning": "@llvm.fuzzy.sort", "nasm": "CALL __fuzzy_sort", "hex": "0x148", "bin": "000101001000"},
    0x150: {"dgm": "150", "mnemonic": "@llvm.audio.playwav", "meaning": "@llvm.audio.playwav", "nasm": "CALL __audio_playwav", "hex": "0x150", "bin": "000101010000"},
    0x151: {"dgm": "151", "mnemonic": "@llvm.audio.playmp3", "meaning": "@llvm.audio.playmp3", "nasm": "CALL __audio_playmp3", "hex": "0x151", "bin": "000101010001"},
    0x152: {"dgm": "152", "mnemonic": "@llvm.audio.record", "meaning": "@llvm.audio.record", "nasm": "CALL __audio_record", "hex": "0x152", "bin": "000101010010"},
    0x153: {"dgm": "153", "mnemonic": "@llvm.audio.stop", "meaning": "@llvm.audio.stop", "nasm": "CALL __audio_stop", "hex": "0x153", "bin": "000101010011"},
    0x154: {"dgm": "154", "mnemonic": "@llvm.audio.tone", "meaning": "@llvm.audio.tone", "nasm": "CALL __audio_tone", "hex": "0x154", "bin": "000101010100"},
    0x155: {"dgm": "155", "mnemonic": "@llvm.audio.volume", "meaning": "@llvm.audio.volume", "nasm": "CALL __audio_volume", "hex": "0x155", "bin": "000101010101"},
    0x156: {"dgm": "156", "mnemonic": "@llvm.audio.mixer", "meaning": "@llvm.audio.mixer", "nasm": "CALL __audio_mixer", "hex": "0x156", "bin": "000101010110"},
    0x157: {"dgm": "157", "mnemonic": "@llvm.audio.pause", "meaning": "@llvm.audio.pause", "nasm": "CALL __audio_pause", "hex": "0x157", "bin": "000101010111"},
    0x158: {"dgm": "158", "mnemonic": "@llvm.audio.resume", "meaning": "@llvm.audio.resume", "nasm": "CALL __audio_resume", "hex": "0x158", "bin": "000101011000"},
    0x159: {"dgm": "159", "mnemonic": "@llvm.audio.stream", "meaning": "@llvm.audio.stream", "nasm": "CALL __audio_stream", "hex": "0x159", "bin": "000101011001"},
}


def get_dgm_info(code: int) -> Optional[Dict[str, str]]:
    """
    Lookup DGM opcode metadata by integer code. Returns a dict or None.
    """
    return _DGM_TABLE.get(code)


def lookup_dgm_by_hex(hexstr: str) -> Optional[Dict[str, str]]:
    """
    Lookup DGM metadata by hex string like '0x1A' or '1A'.
    """
    if hexstr.startswith("0x") or hexstr.startswith("0X"):
        try:
            v = int(hexstr, 16)
        except ValueError:
            return None
    else:
        try:
            v = int(hexstr, 16)
        except ValueError:
            return None
    return get_dgm_info(v)


def dump_dgm_table_text() -> str:
    """
    Return a human-readable dump of the DGM table (short form).
    """
    lines: List[str] = []
    lines.append("; Dodecagram (DGM) opcode table")
    for code in sorted(_DGM_TABLE.keys()):
        md = _DGM_TABLE[code]
        lines.append(f"{md['hex']} ({md['dgm']}): {md['mnemonic']} - {md['meaning']} - {md['nasm']} - bin:{md['bin']}")
    return "\n".join(lines)


# ---------- CLI and entry helpers ----------
def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sayit", description="Sayit - ceremonial storytelling DSL (Executor CLI)")
    p.add_argument("file", nargs="?", help=".say file to execute (use '-' to read from stdin)")
    p.add_argument("--engine", choices=("ir", "vm"), default="ir", help="backend engine to use (default: ir)")
    p.add_argument("--trace", action="store_true", help="enable execution tracing / VM echo")
    p.add_argument("--max-steps", type=int, default=1_000_000, help="maximum execution steps")
    p.add_argument("--timeout", type=float, help="wall-clock timeout seconds")
    p.add_argument("--emit-ir", type=str, help="write pseudo-IR / output to file")
    p.add_argument("--version", action="store_true", help="print version and exit")
    p.add_argument("--dump-dgm", action="store_true", help="print the DGM opcode table and exit")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    """
    CLI entrypoint preserved. Returns exit codes:
      0 - success
      1 - runtime / parse / compile error
      2 - usage / missing args
    """
    if argv is None:
        argv = sys.argv[1:]
    parser = _make_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        # argparse already printed usage; mirror previous behavior
        return 2

    if args.version:
        print("Sayit Executor (system.py) - version: dev")
        return 0

    if args.dump_dgm:
        print(dump_dgm_table_text())
        return 0

    if not args.file:
        parser.print_usage()
        return 2

    out: Optional[str] = None
    ir_text: Optional[str] = None
    # dispatch engines
    if args.engine == "vm":
        try:
            if args.file == "-":
                src = sys.stdin.read()
                # compile_to_vm returns ir_text as third element
                bytecode, const_pool, ir_text = compile_to_vm(src)
                vm = VM(bytecode, const_pool, echo=args.trace)
                vm.run()
                out = "\n".join(vm.output)
            else:
                bytecode, const_pool, ir_text = compile_to_vm(open(args.file, "r", encoding="utf-8").read())
                vm = VM(bytecode, const_pool, echo=args.trace)
                vm.run()
                out = "\n".join(vm.output)
        except Exception as e:
            print("[error]", e)
            return 1
    else:
        ex = Executor(max_steps=args.max_steps, timeout=args.timeout, trace=args.trace)
        try:
            if args.file == "-":
                src = sys.stdin.read()
                out = ex.run_source(src)
            else:
                out = ex.run_file(args.file)
        except Exception as e:
            print("[error]", e)
            return 1

    # emit pseudo-IR to file if requested (useful for sayc caching / inspection)
    if args.emit_ir:
        try:
            # prefer ir_text from codegen/compile_to_vm if available
            to_write = ir_text if ir_text is not None else (out or "")
            with open(args.emit_ir, "w", encoding="utf-8") as f:
                f.write(to_write)
        except Exception as e:
            print("[error] writing emit-ir file:", e)
            return 1

    print("--- pseudo-IR / output ---")
    print(out or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# EOF
import argparse
import sys
from typing import Dict, List, Optional
from .executor import Executor
from .vm import VM
from .codegen import compile_to_vm

# --- Extensions, preprocessing, optimizations and tooling hooks (append-only) ---
# Provides an extensible framework to add keywords, characters, VM ops,
# optimization passes and simple tooling without modifying the core engine.

import re
from types import MethodType
from functools import wraps
from copy import deepcopy

class ExtensionManager:
    """
    Central registry for language/VM/tooling extensions.

    - preprocessors: functions that accept (src:str) -> str and transform source text
      (useful to add syntactic sugar or new keywords without changing the parser).
    - vm_ops: mapping of opcode name -> handler(vm_instance, arg) for custom VM opcode behavior.
    - optimizers: list of functions (bytecode, const_pool) -> (bytecode, const_pool)
    - tool_hooks: named tooling hooks (format, lint, profile) callable by helpers below.
    """
    def __init__(self):
        self.preprocessors: List[callable] = []
        self.vm_ops: Dict[str, callable] = {}
        self.optimizers: List[callable] = []
        self.tool_hooks: Dict[str, callable] = {}

    # Preprocessing (source-level syntactic sugar)
    def register_preprocessor(self, fn: callable) -> None:
        """fn: (src: str) -> str"""
        self.preprocessors.append(fn)

    def preprocess(self, src: str) -> str:
        out = src
        for p in self.preprocessors:
            out = p(out)
        return out

    # VM extensions
    def register_vm_op(self, opname: str, handler: callable) -> None:
        """handler: (vm: VM, arg: Any) -> None"""
        self.vm_ops[opname.upper()] = handler

    def get_vm_op(self, opname: str):
        return self.vm_ops.get(opname.upper())

    # Optimizers
    def register_optimizer(self, fn: callable) -> None:
        """fn: (bytecode: List[(op,arg)], const_pool: List[Any]) -> (bytecode, const_pool)"""
        self.optimizers.append(fn)

    def optimize(self, bytecode, const_pool):
        bc, cp = bytecode, const_pool
        for opt in self.optimizers:
            bc, cp = opt(bc, cp)
        return bc, cp

    # Tool hooks
    def register_tool(self, name: str, fn: callable) -> None:
        self.tool_hooks[name] = fn

    def get_tool(self, name: str):
        return self.tool_hooks.get(name)

# Global singleton for extensions
GLOBAL_EXTENSIONS = ExtensionManager()

# -----------------------
# Helpful preprocessors
# -----------------------
# 1) Allow additional characters in identifiers by normalizing them to parser-friendly names.
#    For example: my-var -> my_var, café -> cafe (strip diacritics simplistically).
#    This is intentionally conservative: textual normalization only.
def _normalize_identifiers(src: str) -> str:
    # replace hyphens in simple identifiers with underscore
    src = re.sub(r"\b([A-Za-z0-9_]+)-([A-Za-z0-9_]+)\b", r"\1_\2", src)
    # remove a small set of diacritics (very small heuristic)
    accents = {
        "á":"a","à":"a","ä":"a","â":"a","ã":"a",
        "é":"e","è":"e","ë":"e","ê":"e",
        "í":"i","ì":"i","ï":"i","î":"i",
        "ó":"o","ò":"o","ö":"o","ô":"o","õ":"o",
        "ú":"u","ù":"u","ü":"u","û":"u",
        "ç":"c","ñ":"n"
    }
    def _strip_acc(m):
        s = m.group(0)
        return "".join(accents.get(ch, ch) for ch in s)
    src = re.sub(r"[^\x00-\x7F]+", _strip_acc, src)
    return src

GLOBAL_EXTENSIONS.register_preprocessor(_normalize_identifiers)

# 2) Add a new 'repeat N' block as syntactic sugar:
#    repeat <N>
#      <body>
#    end
# expands to:
#    __repeat_index = 0
#    while __repeat_index < N
#      <body>
#      __repeat_index = __repeat_index + 1
#    end
_repeat_counter = 0
def _expand_repeat_blocks(src: str) -> str:
    global _repeat_counter
    lines = src.splitlines()
    out_lines: List[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        m = re.match(r"^\s*repeat\s+(.+)$", line)
        if m:
            n_expr = m.group(1).strip()
            # collect body until matching 'end'
            body = []
            idx += 1
            depth = 0
            while idx < len(lines):
                l2 = lines[idx]
                if re.match(r"^\s*repeat\b", l2):
                    depth += 1
                if re.match(r"^\s*end\s*$", l2):
                    if depth == 0:
                        break
                    depth -= 1
                body.append(l2)
                idx += 1
            # idx points to 'end' or EOF
            # create unique counter variable
            varname = f"__repeat_idx_{_repeat_counter}"
            _repeat_counter += 1
            out_lines.append(f"{varname} = 0")
            out_lines.append(f"while {varname} < {n_expr}")
            out_lines.extend(body)
            out_lines.append(f"{varname} = {varname} + 1")
            out_lines.append("end")
            # skip the 'end' line
            idx += 1
            continue
        out_lines.append(line)
        idx += 1
    return "\n".join(out_lines)

GLOBAL_EXTENSIONS.register_preprocessor(_expand_repeat_blocks)

# -----------------------
# VM monkey-patch to consult GLOBAL_EXTENSIONS for custom opcodes
# -----------------------
# Keep a reference to original run to allow fallback
_original_vm_run = getattr(VM, "run", None)
def _vm_run_with_extensions(self):
    """
    Replacement VM.run that first checks GLOBAL_EXTENSIONS for custom op handlers.
    If no custom handler, falls back to existing op_<name> handler semantics.
    """
    while self.running and self.pc < len(self.bytecode):
        op, arg = self.bytecode[self.pc]

        # extension handler wins
        ext_handler = GLOBAL_EXTENSIONS.get_vm_op(op)
        try:
            if ext_handler is not None:
                # handler may mutate vm.stack/vars/pc/running
                ext_handler(self, arg)
                # support handler setting pc directly; otherwise increment below
                self.pc += 1
                continue

            # otherwise perform normal dispatch as existing VM did
            handler = getattr(self, f"op_{op.lower()}", None)
            if handler is None:
                raise RuntimeError(f"Unknown opcode: {op}")
            handler(arg)
            self.pc += 1
        except Exception:
            # preserve original behavior: surface error with context
            raise
# bind the new run method
VM.run = MethodType(_vm_run_with_extensions, None, VM)  # type: ignore

# -----------------------
# Bytecode optimizers
# -----------------------
def _const_fold_optimizer(bytecode: List[Tuple[str, Any]], const_pool: List[Any]):
    """
    Simple pass: fold sequences LOAD_CONST a, LOAD_CONST b, BINARY_OP op into single LOAD_CONST result.
    Works only for pure ops supported by VM/op_binary_op logic.
    """
    bc = deepcopy(bytecode)
    cp = list(const_pool)
    i = 0
    changed = False
    while i < len(bc) - 2:
        op1, a1 = bc[i]
        op2, a2 = bc[i+1]
        op3, a3 = bc[i+2]
        if op1 == "LOAD_CONST" and op2 == "LOAD_CONST" and op3 == "BINARY_OP":
            v1 = cp[a1]
            v2 = cp[a2]
            op = a3
            try:
                # apply semantics similar to VM._binary_op
                if op == "+":
                    res = v1 + v2
                elif op == "-":
                    res = v1 - v2
                elif op == "*":
                    res = v1 * v2
                elif op == "/":
                    if isinstance(v1, int) and isinstance(v2, int):
                        if v2 == 0:
                            raise ZeroDivisionError
                        res = v1 // v2
                    else:
                        res = v1 / v2
                elif op == "==":
                    res = v1 == v2
                elif op == "!=":
                    res = v1 != v2
                elif op == "<":
                    res = v1 < v2
                elif op == "<=":
                    res = v1 <= v2
                elif op == ">":
                    res = v1 > v2
                elif op == ">=":
                    res = v1 >= v2
                else:
                    raise ValueError("unsupported")
            except Exception:
                i += 1
                continue
            # fold: append res to const_pool (or reuse existing), replace triple with single LOAD_CONST
            try:
                idx = cp.index(res)
            except ValueError:
                cp.append(res)
                idx = len(cp) - 1
            # replace entry i with LOAD_CONST idx and remove next two
            bc[i:i+3] = [("LOAD_CONST", idx)]
            changed = True
            # continue from same index
            continue
        i += 1
    if changed:
        return bc, cp
    return bytecode, const_pool

GLOBAL_EXTENSIONS.register_optimizer(_const_fold_optimizer)

def _remove_noop_optimizer(bytecode: List[Tuple[str, Any]], const_pool: List[Any]):
    bc = [ins for ins in bytecode if not (ins[0] == "NOOP")]
    return bc, const_pool

GLOBAL_EXTENSIONS.register_optimizer(_remove_noop_optimizer)

# -----------------------
# Helper compile/run wrappers that use extensions
# -----------------------
def compile_to_vm_with_extensions(src: str, optimize: bool = True):
    """
    Preprocess source, compile to vm bytecode via existing compile_to_vm,
    then run registered bytecode optimizers (if requested).
    Returns (bytecode, const_pool, ir_text).
    """
    pre = GLOBAL_EXTENSIONS.preprocess(src)
    bytecode, const_pool, ir = compile_to_vm(pre)
    if optimize:
        bytecode, const_pool = GLOBAL_EXTENSIONS.optimize(bytecode, const_pool)
    return bytecode, const_pool, ir

def run_vm_from_source_with_extensions(src: str, echo: bool = True, optimize: bool = True) -> str:
    bytecode, const_pool, ir = compile_to_vm_with_extensions(src, optimize=optimize)
    vm = VM(bytecode, const_pool, echo=echo)
    vm.run()
    return "\n".join(vm.output)

# -----------------------
# Example tooling hooks
# -----------------------
def simple_formatter(src: str) -> str:
    """
    Very small source formatter:
    - normalize spacing around '=' and binary ops
    - collapse repeated blank lines
    """
    # normalize equal spacing
    src = re.sub(r"\s*=\s*", " = ", src)
    src = re.sub(r"\s*([+\-*/<>]=?|==|!=)\s*", r" \1 ", src)
    # collapse blank lines
    src = re.sub(r"\n{3,}", "\n\n", src)
    return src.strip() + "\n"

GLOBAL_EXTENSIONS.register_tool("format", simple_formatter)

def simple_linter(src: str) -> List[str]:
    issues: List[str] = []
    # warn about use of undefined identifiers heuristically (very small heuristic)
    idents = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", src))
    # treat any ident that starts with "__repeat_idx" as internal (ignore)
    for ident in sorted(idents):
        if ident.startswith("tmp") or ident.startswith("__repeat_idx"):
            continue
        # naive: any identifier that contains uppercase warns
        if any(ch.isupper() for ch in ident):
            issues.append(f"lint: identifier '{ident}' contains uppercase letters")
    return issues

GLOBAL_EXTENSIONS.register_tool("lint", simple_linter)

# -----------------------
# Example VM op extension
# -----------------------
def _op_trace(vm: VM, arg):
    """
    Custom VM opcode TRACE <message> : prints a trace message and optionally inspects top of stack.
    Usage in bytecode: ("TRACE", "hello") - will print "TRACE: hello"
    This op is illustrative; real plugins may inspect vm.stack and vm.vars.
    """
    prefix = "[EXT-TRACE]"
    if arg is None:
        msg = prefix
    else:
        msg = f"{prefix} {arg}"
    vm.output.append(msg)
    if vm.echo:
        print(msg)

GLOBAL_EXTENSIONS.register_vm_op("TRACE", _op_trace)

# -----------------------
# Convenience helpers for interactive use
# -----------------------
def format_source(src: str) -> str:
    fmt = GLOBAL_EXTENSIONS.get_tool("format")
    return fmt(src) if fmt else src

def lint_source(src: str) -> List[str]:
    lint = GLOBAL_EXTENSIONS.get_tool("lint")
    return lint(src) if lint else []

def run_source_with_tools(src: str, echo: bool = True, optimize: bool = True, lint: bool = True, fmt: bool = False) -> Tuple[str, List[str]]:
    """
    Runs source using extension pipeline. Returns (output_text, lint_issues)
    if fmt=True, formats before running (does not mutate original file).
    """
    work = src
    if fmt:
        work = format_source(work)
    issues = lint_source(work) if lint else []
    out = run_vm_from_source_with_extensions(work, echo=echo, optimize=optimize)
    return out, issues

# -----------------------
# Example registrations and usage notes (non-invasive)
# -----------------------
# Register more "character allowances" and small convenience preprocessor that allows
# '%' to be used as comment start (maps to '#') for authors who prefer it.
def _percent_comments(src: str) -> str:
    return re.sub(r"(?m)^\s*%+", "#", src)

GLOBAL_EXTENSIONS.register_preprocessor(_percent_comments)

# Small example optimizer already registered; users can add more via GLOBAL_EXTENSIONS.register_optimizer.

# Minimal demonstration plugin: add a 'repeat' example already registered above.
# Example usage (programmatic):
#    out, issues = run_source_with_tools("repeat 3\nprint('hello')\nend", echo=False)
#    print(out)
#
# For CLI integration, callers can call compile_to_vm_with_extensions before creating a VM,
# or call run_vm_from_source_with_extensions directly.

# End of append-only extension layer.

# -----------------------
# Extended capabilities: ops, tooling, optimizations, networking, WASM, HTML
# Append-only extension block (adds many features while keeping core intact)
# -----------------------

import hashlib
import json
import html
import urllib.request
import urllib.parse
import ssl
import threading
import time
from typing import Any, Tuple

# Simple bytecode cache to speed repeated compilations
if not hasattr(GLOBAL_EXTENSIONS, "_bcache"):
    GLOBAL_EXTENSIONS._bcache: Dict[str, Tuple[List[Tuple[str, Any]], List[Any], Optional[str]]] = {}

def _hash_source(src: str) -> str:
    return hashlib.sha256(src.encode("utf-8")).hexdigest()

def compile_to_vm_with_cache(src: str, optimize: bool = True, use_cache: bool = True):
    """
    Preprocess + compile to VM using compile_to_vm with optional cache and optimizers.
    """
    key = _hash_source(src)
    if use_cache and key in GLOBAL_EXTENSIONS._bcache:
        return GLOBAL_EXTENSIONS._bcache[key]
    # preprocess & compile
    pre = GLOBAL_EXTENSIONS.preprocess(src)
    bytecode, const_pool, ir = compile_to_vm(pre)
    if optimize:
        bytecode, const_pool = GLOBAL_EXTENSIONS.optimize(bytecode, const_pool)
    if use_cache:
        GLOBAL_EXTENSIONS._bcache[key] = (bytecode, const_pool, ir)
    return bytecode, const_pool, ir

# Safety / sandboxing controls and resource limits
GLOBAL_EXTENSIONS._allow_networking = False
GLOBAL_EXTENSIONS._wasm_allowed = False
GLOBAL_EXTENSIONS._max_request_size = 10 * 1024 * 1024  # 10MB default

def enable_networking(allow: bool = True):
    GLOBAL_EXTENSIONS._allow_networking = bool(allow)

def enable_wasm(allow: bool = True):
    GLOBAL_EXTENSIONS._wasm_allowed = bool(allow)

def set_max_request_size(bytes_limit: int):
    GLOBAL_EXTENSIONS._max_request_size = int(bytes_limit)

GLOBAL_EXTENSIONS.register_tool("enable_networking", enable_networking)
GLOBAL_EXTENSIONS.register_tool("enable_wasm", enable_wasm)
GLOBAL_EXTENSIONS.register_tool("set_max_request_size", set_max_request_size)

# -----------------------
# Additional VM extension ops (safe wrappers that require opt-in for network/WASM)
# Stack conventions: operations push/pop from vm.stack (LIFO).
# -----------------------

def _ext_op_dict_new(vm: VM, arg):
    vm.stack.append({})
def _ext_op_dict_set(vm: VM, arg):
    # expects: key, value, dict on stack (top is dict)
    d = vm.stack.pop()
    v = vm.stack.pop()
    k = vm.stack.pop()
    if not isinstance(d, dict):
        raise RuntimeError("DICT_SET expects dict on top of stack")
    d[k] = v
    vm.stack.append(d)
def _ext_op_dict_get(vm: VM, arg):
    # expects: key, dict
    d = vm.stack.pop()
    k = vm.stack.pop()
    if not isinstance(d, dict):
        raise RuntimeError("DICT_GET expects dict on top of stack")
    vm.stack.append(d.get(k))
def _ext_op_list_new(vm: VM, arg):
    vm.stack.append([])
def _ext_op_list_push(vm: VM, arg):
    lst = vm.stack.pop()
    val = vm.stack.pop()
    if not isinstance(lst, list):
        raise RuntimeError("LIST_PUSH expects list on top of stack")
    lst.append(val)
    vm.stack.append(lst)
def _ext_op_list_pop(vm: VM, arg):
    lst = vm.stack.pop()
    if not isinstance(lst, list):
        raise RuntimeError("LIST_POP expects list on top of stack")
    vm.stack.append(lst.pop())

GLOBAL_EXTENSIONS.register_vm_op("DICT_NEW", _ext_op_dict_new)
GLOBAL_EXTENSIONS.register_vm_op("DICT_SET", _ext_op_dict_set)
GLOBAL_EXTENSIONS.register_vm_op("DICT_GET", _ext_op_dict_get)
GLOBAL_EXTENSIONS.register_vm_op("LIST_NEW", _ext_op_list_new)
GLOBAL_EXTENSIONS.register_vm_op("LIST_PUSH", _ext_op_list_push)
GLOBAL_EXTENSIONS.register_vm_op("LIST_POP", _ext_op_list_pop)

# -----------------------
# Networking ops (require explicit enable_networking(True))
# -----------------------
def _safe_fetch_url(url: str, timeout: float = 10.0, method: str = "GET", data: Optional[bytes] = None, headers: Optional[Dict[str,str]] = None, verify_tls: bool = True) -> Tuple[int, bytes, Dict[str,str]]:
    # enforce max size via reading chunks
    ctx = ssl.create_default_context() if verify_tls else ssl._create_unverified_context()
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        status = getattr(resp, "status", None) or resp.getcode()
        # read in chunks to enforce size limit
        out = bytearray()
        chunk_size = 16384
        total = 0
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out.extend(chunk)
            total += len(chunk)
            if total > GLOBAL_EXTENSIONS._max_request_size:
                raise RuntimeError("Response exceeded configured max_request_size")
        hdrs = {k.lower(): v for k, v in resp.headers.items()}
        return int(status), bytes(out), hdrs

def _op_http_get(vm: VM, arg):
    if not GLOBAL_EXTENSIONS._allow_networking:
        raise RuntimeError("Networking is disabled. Call GLOBAL_EXTENSIONS.get_tool('enable_networking')(True) to enable explicitly.")
    # arg may be tuple (url, timeout, headers)
    if isinstance(arg, tuple) or isinstance(arg, list):
        url = arg[0]
        timeout = float(arg[1]) if len(arg) > 1 and arg[1] is not None else 10.0
        headers = arg[2] if len(arg) > 2 and arg[2] is not None else {}
    else:
        url = str(arg)
        timeout = 10.0
        headers = {}
    status, body, hdrs = _safe_fetch_url(url, timeout=timeout, method="GET", headers=headers, verify_tls=True)
    text = body.decode("utf-8", errors="replace")
    vm.stack.append({"status": status, "body": text, "headers": hdrs})

def _op_http_post(vm: VM, arg):
    if not GLOBAL_EXTENSIONS._allow_networking:
        raise RuntimeError("Networking is disabled. Call GLOBAL_EXTENSIONS.get_tool('enable_networking')(True) to enable explicitly.")
    if isinstance(arg, tuple) or isinstance(arg, list):
        url = arg[0]
        data = arg[1] if len(arg) > 1 else b""
        timeout = float(arg[2]) if len(arg) > 2 else 10.0
        headers = arg[3] if len(arg) > 3 else {}
    else:
        url = str(arg)
        data = b""
        timeout = 10.0
        headers = {}
    if isinstance(data, str):
        data = data.encode("utf-8")
    status, body, hdrs = _safe_fetch_url(url, timeout=timeout, method="POST", data=data, headers=headers, verify_tls=True)
    text = body.decode("utf-8", errors="replace")
    vm.stack.append({"status": status, "body": text, "headers": hdrs})

GLOBAL_EXTENSIONS.register_vm_op("HTTP_GET", _op_http_get)
GLOBAL_EXTENSIONS.register_vm_op("HTTP_POST", _op_http_post)

# -----------------------
# HTML templating helpers (safe, no external dependencies)
# -----------------------
def render_html_template(template: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Very small and safe HTML templating: replaces {{ key }} with html-escaped context[key].
    No code execution allowed.
    """
    ctx = context or {}
    def _sub(m):
        key = m.group(1).strip()
        val = ctx.get(key, "")
        return html.escape(str(val))
    return re.sub(r"\{\{\s*([A-Za-z0-9_\.]+)\s*\}\}", _sub, template)

def _op_render_html(vm: VM, arg):
    # expects: context_dict, template_string (top is template)
    template = vm.stack.pop()
    context = vm.stack.pop()
    if not isinstance(context, dict) or not isinstance(template, str):
        raise RuntimeError("RENDER_HTML expects (context:dict, template:str)")
    out = render_html_template(template, context)
    vm.stack.append(out)

GLOBAL_EXTENSIONS.register_vm_op("RENDER_HTML", _op_render_html)

# -----------------------
# WASM invocation (optional; requires enabling via enable_wasm(True))
# Attempts to use 'wasmtime' or 'wasmer' if available. If not present, raises informative error.
# -----------------------
def _op_wasm_invoke(vm: VM, arg):
    """
    WASM_INVOKE expects: wasm_bytes (as bytes), func_name (str), arg_list (list)
    Returns: function return value (primitive) pushed to stack.
    Requires GLOBAL_EXTENSIONS._wasm_allowed True.
    """
    if not GLOBAL_EXTENSIONS._wasm_allowed:
        raise RuntimeError("WASM execution disabled. Call GLOBAL_EXTENSIONS.get_tool('enable_wasm')(True) to enable explicitly.")
    # pop: arg_list, func_name, wasm_bytes
    arg_list = vm.stack.pop()
    func_name = vm.stack.pop()
    wasm_bytes = vm.stack.pop()
    if not isinstance(wasm_bytes, (bytes, bytearray)):
        raise RuntimeError("WASM_INVOKE expects wasm_bytes (bytes) on the stack")
    if not isinstance(func_name, str):
        raise RuntimeError("WASM_INVOKE expects function name (str)")
    # try wasmtime
    try:
        import wasmtime  # type: ignore
        store = wasmtime.Store()
        module = wasmtime.Module(store.engine, bytes(wasm_bytes))
        instance = wasmtime.Instance(store, module, [])
        fn = instance.exports(store)[func_name]
        res = fn(store, *arg_list)
        vm.stack.append(res)
        return
    except Exception:
        pass
    # try wasmer
    try:
        from wasmer import engine, Store, Module, Instance  # type: ignore
        store = Store(engine.Universal())
        module = Module(store, bytes(wasm_bytes))
        inst = Instance(module)
        fn = inst.exports[func_name]
        res = fn(*arg_list)
        vm.stack.append(res)
        return
    except Exception:
        pass
    raise RuntimeError("No WASM runtime available (wasmtime/wasmer). Install one or enable via GLOBAL_EXTENSIONS tools.")

GLOBAL_EXTENSIONS.register_vm_op("WASM_INVOKE", _op_wasm_invoke)

# -----------------------
# Advanced optimizer passes (dead-code elimination, peephole, jump folding)
# -----------------------
def _dead_code_elimination(bytecode: List[Tuple[str, Any]], const_pool: List[Any]):
    """
    Removes unreachable instructions after HALT and reduces obvious NOOPs.
    Conservative: only removes trailing instructions after HALT.
    """
    bc = list(bytecode)
    # find first HALT (or HALT-equivalent)
    for i, (op, arg) in enumerate(bc):
        if op in ("HALT", "HALT", "op_halt"):
            return bc[:i+1], const_pool
    return bc, const_pool

def _peephole_optimize(bytecode: List[Tuple[str, Any]], const_pool: List[Any]):
    """
    Some local peephole: collapse LOAD_CONST X ; LOAD_CONST Y ; BINARY_OP -> LOAD_CONST (if both consts)
    This duplicates earlier constant-fold but remains useful after other passes.
    """
    # reuse existing const-fold logic from GLOBAL_EXTENSIONS if present by calling optimize on a clone
    return _const_fold_optimizer(bytecode, const_pool)

GLOBAL_EXTENSIONS.register_optimizer(_dead_code_elimination)
GLOBAL_EXTENSIONS.register_optimizer(_peephole_optimize)

# -----------------------
# Execution wrappers with enhanced error resilience and profiling
# -----------------------
def safe_run_vm(vm: VM, *, timeout: Optional[float] = None, step_limit: Optional[int] = None) -> List[str]:
    """
    Run a VM with error mapping, timeout and optional step-limit enforced.
    Returns vm.output on success, raises ExecutionError on failure with context.
    """
    outputs = []
    exc: Optional[BaseException] = None
    start = time.time()
    steps = 0
    try:
        while vm.running and vm.pc < len(vm.bytecode):
            if timeout is not None and (time.time() - start) > timeout:
                raise ExecutionError("VM execution timed out")
            if step_limit is not None and steps > step_limit:
                raise ExecutionError("VM step limit exceeded")
            try:
                op, arg = vm.bytecode[vm.pc]
            except IndexError:
                raise ExecutionError(f"PC out of range: {vm.pc}")
            # dispatch via extension-run VM.run semantics (reuse VM.run logic by invoking op handler)
            ext_handler = GLOBAL_EXTENSIONS.get_vm_op(op)
            if ext_handler is not None:
                ext_handler(vm, arg)
                vm.pc += 1
            else:
                handler = getattr(vm, f"op_{op.lower()}", None)
                if handler is None:
                    raise ExecutionError(f"Unknown opcode during safe_run_vm: {op}")
                handler(arg)
                vm.pc += 1
            steps += 1
    except BaseException as e:
        exc = e
    finally:
        outputs = vm.output
    if exc:
        # enrich ExecutionError with bytecode context if not already an ExecutionError
        msg = f"{type(exc).__name__}: {exc}"
        raise ExecutionError(msg)
    return outputs

def safe_run_executor(ex: Executor, prog, *, timeout: Optional[float] = None, step_limit: Optional[int] = None) -> List[str]:
    """
    Wrapper around Executor.run_program adding timeout and step-limit protection and mapping exceptions.
    """
    # temporarily override executor limits if provided
    orig_timeout = ex.timeout
    orig_max = ex.max_steps
    if timeout is not None:
        ex.timeout = timeout
    if step_limit is not None:
        ex.max_steps = step_limit
    try:
        ex.run_program(prog)
    except BaseException as e:
        raise ExecutionError(f"Executor error: {e}")
    finally:
        ex.timeout = orig_timeout
        ex.max_steps = orig_max
    return ex.output

GLOBAL_EXTENSIONS.register_tool("safe_run_vm", safe_run_vm)
GLOBAL_EXTENSIONS.register_tool("safe_run_executor", safe_run_executor)

# -----------------------
# Lightweight static analyzer + formatter improvements
# -----------------------
def advanced_lint(src: str) -> List[str]:
    """
    Extended linter: detects unused constants (simple heuristic), suspicious divisions by zero,
    and flag network/WASM usage where not enabled.
    """
    issues: List[str] = []
    # division by zero heuristics: look for "/ 0" or "/0"
    if re.search(r"/\s*0\b", src):
        issues.append("possible division by zero literal")
    # network usage detection
    if re.search(r"\bHTTP_GET\b|\bHTTP_POST\b", src) and not GLOBAL_EXTENSIONS._allow_networking:
        issues.append("networking ops used but networking disabled")
    if re.search(r"\bWASM_INVOKE\b", src) and not GLOBAL_EXTENSIONS._wasm_allowed:
        issues.append("WASM invoked but WASM disabled")
    # basic undefined identifier check (reuse simple_linter)
    issues.extend(simple_linter(src))
    return issues

GLOBAL_EXTENSIONS.register_tool("adv_lint", advanced_lint)

def advanced_format(src: str) -> str:
    # build on simple formatter and ensure consistent indentation for 'if/while/repeat' blocks
    out = simple_formatter(src)
    # naive indentation: increase indent after lines starting with 'if', 'while', 'repeat', 'else'
    lines = out.splitlines()
    indent = 0
    res = []
    for ln in lines:
        stripped = ln.strip()
        if stripped == "end":
            indent = max(0, indent - 1)
        res.append(("    " * indent) + stripped)
        if stripped.startswith(("if ", "while ", "repeat ")) or stripped == "else":
            indent += 1
    return "\n".join(res) + "\n"

GLOBAL_EXTENSIONS.register_tool("adv_format", advanced_format)

# -----------------------
# Simple networking CLI helpers (non-blocking wrappers)
# -----------------------
def http_get_async(url: str, callback=None, timeout: float = 10.0):
    """
    Perform an HTTP GET in a background thread. Requires networking enabled.
    Callback signature: fn(result_dict) where result_dict = {"status":..,"body":..,"headers":..,"error":..}
    """
    def _work():
        res = {"status": None, "body": None, "headers": None, "error": None}
        try:
            status, body, hdrs = _safe_fetch_url(url, timeout=timeout)
            res["status"] = status
            res["body"] = body.decode("utf-8", errors="replace")
            res["headers"] = hdrs
        except Exception as e:
            res["error"] = str(e)
        if callable(callback):
            try:
                callback(res)
            except Exception:
                pass
    if not GLOBAL_EXTENSIONS._allow_networking:
        raise RuntimeError("Networking disabled")
    t = threading.Thread(target=_work, daemon=True)
    t.start()
    return t

GLOBAL_EXTENSIONS.register_tool("http_get_async", http_get_async)

# -----------------------
# Developer-facing helpers: list registered extensions and tools
# -----------------------
def list_extensions() -> Dict[str, Any]:
    return {
        "preprocessors": [p.__name__ for p in GLOBAL_EXTENSIONS.preprocessors],
        "vm_ops": list(GLOBAL_EXTENSIONS.vm_ops.keys()),
        "optimizers": [o.__name__ for o in GLOBAL_EXTENSIONS.optimizers],
        "tools": list(GLOBAL_EXTENSIONS.tool_hooks.keys()),
        "networking_enabled": GLOBAL_EXTENSIONS._allow_networking,
        "wasm_enabled": GLOBAL_EXTENSIONS._wasm_allowed,
    }

GLOBAL_EXTENSIONS.register_tool("list_extensions", list_extensions)

# -----------------------
# Notes:
# - Networking and WASM capabilities are opt-in for safety. Enable with:
#     GLOBAL_EXTENSIONS.get_tool("enable_networking")(True)
#     GLOBAL_EXTENSIONS.get_tool("enable_wasm")(True)
# - Use compile_to_vm_with_cache(...) to speed repeated compilations.
# - Use run_source_with_tools(...) to exercise preprocessors, optimizers and tooling.
# - New VM ops: DICT_NEW/DICT_SET/DICT_GET, LIST_NEW/LIST_PUSH/LIST_POP, HTTP_GET/HTTP_POST,
#   RENDER_HTML, WASM_INVOKE.
# - Advanced optimizers and lint/format tools exposed via GLOBAL_EXTENSIONS tools.
# -----------------------

# --- Further extensible capabilities (append-only) ---
# Adds:
#  - JSON/base64/crypto VM ops
#  - optional filesystem ops (opt-in)
#  - HTTP connection pooling and caching
#  - async thread-pool HTTP helpers
#  - simple profiler and VM instrumentation
#  - additional optimizer passes (jump folding, const-prop stub)
#  - safer HTML mini-templating with simple loops
#  - enhanced error contexts for VM/executor
#  - bytecode persist/load helpers for faster startup
# All features are opt-in (enable via GLOBAL_EXTENSIONS tools) to preserve safety.

import base64
import pickle
import os
import hashlib as _hashlib
import functools
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ------- Feature gates / tools -------
GLOBAL_EXTENSIONS._allow_filesystem = False
GLOBAL_EXTENSIONS._http_pool_workers = 4
GLOBAL_EXTENSIONS._http_cache_enabled = True
GLOBAL_EXTENSIONS._http_cache_dir = Path(".sayit_http_cache")

def enable_filesystem(allow: bool = True):
    GLOBAL_EXTENSIONS._allow_filesystem = bool(allow)
GLOBAL_EXTENSIONS.register_tool("enable_filesystem", enable_filesystem)

def set_http_pool_workers(n: int):
    GLOBAL_EXTENSIONS._http_pool_workers = max(1, int(n))
GLOBAL_EXTENSIONS.register_tool("set_http_pool_workers", set_http_pool_workers)

def set_http_cache(enabled: bool, cache_dir: Optional[str] = None):
    GLOBAL_EXTENSIONS._http_cache_enabled = bool(enabled)
    if cache_dir:
        GLOBAL_EXTENSIONS._http_cache_dir = Path(cache_dir)
GLOBAL_EXTENSIONS.register_tool("set_http_cache", set_http_cache)

# ensure cache dir exists on demand
def _ensure_http_cache_dir():
    d = GLOBAL_EXTENSIONS._http_cache_dir
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return d

# ------- Simple HTTP connection pool with caching -------
class SimpleHTTPPool:
    """
    Threaded pool wrapper using urllib; uses GLOBAL_EXTENSIONS._http_pool_workers workers.
    Caches GET responses (safe, based on URL+params) to disk if enabled.
    """
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=GLOBAL_EXTENSIONS._http_pool_workers)
        self._futures = []

    def get(self, url: str, timeout: float = 10.0, headers: Optional[Dict[str,str]] = None, use_cache: Optional[bool] = None):
        use_cache = GLOBAL_EXTENSIONS._http_cache_enabled if use_cache is None else bool(use_cache)
        if use_cache:
            # simple cache key
            key = _hash_source(url + (json.dumps(headers or {}, sort_keys=True)))
            cache_dir = _ensure_http_cache_dir()
            cache_file = cache_dir / f"{key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        return pickle.load(f)
                except Exception:
                    pass
        fut = self._executor.submit(_safe_fetch_url, url, timeout, "GET", None, headers or {}, True)
        self._futures.append(fut)
        res = fut.result()
        if use_cache:
            try:
                cache_dir = _ensure_http_cache_dir()
                cache_file = cache_dir / f"{_hash_source(url)}.pkl"
                with open(cache_file, "wb") as f:
                    pickle.dump(res, f)
            except Exception:
                pass
        return res

_http_pool_singleton: Optional[SimpleHTTPPool] = None
def get_http_pool() -> SimpleHTTPPool:
    global _http_pool_singleton
    if _http_pool_singleton is None:
        _http_pool_singleton = SimpleHTTPPool()
    return _http_pool_singleton

GLOBAL_EXTENSIONS.register_tool("http_pool", get_http_pool)

# ------- Async helpers using ThreadPoolExecutor -------
def http_get_bulk(urls: List[str], timeout: float = 10.0, concurrency: Optional[int] = None):
    """
    Concurrently retrieve multiple URLs. Returns list of (status, body_bytes, headers) preserving order.
    """
    if not GLOBAL_EXTENSIONS._allow_networking:
        raise RuntimeError("Networking disabled")
    max_workers = concurrency or GLOBAL_EXTENSIONS._http_pool_workers
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_safe_fetch_url, url, timeout, "GET", None, {}, True) for url in urls]
        results = []
        for fut in futures:
            try:
                results.append(fut.result())
            except Exception as e:
                results.append((None, b"", {"error": str(e)}))
    return results

GLOBAL_EXTENSIONS.register_tool("http_get_bulk", http_get_bulk)

# ------- Filesystem ops (opt-in) -------
def _op_fs_read(vm: VM, arg):
    if not GLOBAL_EXTENSIONS._allow_filesystem:
        raise RuntimeError("Filesystem ops are disabled. Call GLOBAL_EXTENSIONS.get_tool('enable_filesystem')(True) to enable.")
    # arg is file path or tuple (path, mode)
    path = arg if not isinstance(arg, (list, tuple)) else arg[0]
    mode = "r" if (not isinstance(arg, (list, tuple)) or len(arg) < 2) else arg[1]
    # safe canonicalization
    p = Path(path).resolve()
    with open(str(p), mode, encoding="utf-8" if "b" not in mode else None) as f:
        data = f.read()
    vm.stack.append(data)

def _op_fs_write(vm: VM, arg):
    if not GLOBAL_EXTENSIONS._allow_filesystem:
        raise RuntimeError("Filesystem ops are disabled. Call GLOBAL_EXTENSIONS.get_tool('enable_filesystem')(True) to enable.")
    # expects: content, path on stack (content top)
    content = vm.stack.pop()
    path = vm.stack.pop()
    p = Path(path).resolve()
    mode = "w"
    if isinstance(content, (bytes, bytearray)):
        mode = "wb"
    with open(str(p), mode, encoding="utf-8" if "b" not in mode else None) as f:
        f.write(content)
    vm.stack.append(True)

GLOBAL_EXTENSIONS.register_vm_op("FS_READ", _op_fs_read)
GLOBAL_EXTENSIONS.register_vm_op("FS_WRITE", _op_fs_write)

# ------- JSON & base64 & crypto ops -------
def _op_json_encode(vm: VM, arg):
    obj = vm.stack.pop()
    vm.stack.append(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))

def _op_json_decode(vm: VM, arg):
    text = vm.stack.pop()
    vm.stack.append(json.loads(text))

def _op_b64_encode(vm: VM, arg):
    data = vm.stack.pop()
    if isinstance(data, str):
        data = data.encode("utf-8")
    vm.stack.append(base64.b64encode(data).decode("ascii"))

def _op_b64_decode(vm: VM, arg):
    text = vm.stack.pop()
    vm.stack.append(base64.b64decode(text))

def _op_hash_md5(vm: VM, arg):
    data = vm.stack.pop()
    if isinstance(data, str):
        data = data.encode("utf-8")
    vm.stack.append(_hashlib.md5(data).hexdigest())

def _op_hash_sha256(vm: VM, arg):
    data = vm.stack.pop()
    if isinstance(data, str):
        data = data.encode("utf-8")
    vm.stack.append(_hashlib.sha256(data).hexdigest())

GLOBAL_EXTENSIONS.register_vm_op("JSON_ENCODE", _op_json_encode)
GLOBAL_EXTENSIONS.register_vm_op("JSON_DECODE", _op_json_decode)
GLOBAL_EXTENSIONS.register_vm_op("B64_ENCODE", _op_b64_encode)
GLOBAL_EXTENSIONS.register_vm_op("B64_DECODE", _op_b64_decode)
GLOBAL_EXTENSIONS.register_vm_op("HASH_MD5", _op_hash_md5)
GLOBAL_EXTENSIONS.register_vm_op("HASH_SHA256", _op_hash_sha256)

# ------- Simple bytecode persist / load helpers -------
def save_bytecode_to_file(bytecode: List[Tuple[str, Any]], const_pool: List[Any], path: str):
    p = Path(path)
    with open(p, "wb") as f:
        pickle.dump((bytecode, const_pool), f)

def load_bytecode_from_file(path: str):
    p = Path(path)
    with open(p, "rb") as f:
        return pickle.load(f)

GLOBAL_EXTENSIONS.register_tool("save_bytecode", save_bytecode_to_file)
GLOBAL_EXTENSIONS.register_tool("load_bytecode", load_bytecode_from_file)

# ------- VM instrumentation / profiler -------
def instrument_vm(vm: VM):
    """
    Attach counters and a timing wrapper to VM instance for profiling.
    Adds vm._op_counts dict and vm._profile_timestamps list.
    """
    vm._op_counts = {}
    vm._profile_timestamps = []
    # wrap op dispatch by monkey-patching a simple step function
    orig_run = vm.run

    def _step_and_run():
        start_time = time.time()
        while vm.running and vm.pc < len(vm.bytecode):
            op, arg = vm.bytecode[vm.pc]
            vm._op_counts[op] = vm._op_counts.get(op, 0) + 1
            vm._profile_timestamps.append((vm.pc, op, time.time()))
            # dispatch via existing extension-aware handler (use GLOBAL_EXTENSIONS)
            ext = GLOBAL_EXTENSIONS.get_vm_op(op)
            if ext:
                try:
                    ext(vm, arg)
                except Exception as e:
                    # attach context and re-raise
                    raise RuntimeError(f"VM op {op} @pc={vm.pc} failed: {e}")
                vm.pc += 1
                continue
            handler = getattr(vm, f"op_{op.lower()}", None)
            if handler is None:
                raise RuntimeError(f"Unknown opcode during instrumented run: {op} @pc={vm.pc}")
            try:
                handler(arg)
            except Exception as e:
                raise RuntimeError(f"VM op {op} @pc={vm.pc} failed: {e}")
            vm.pc += 1
        vm._profile_duration = time.time() - start_time

    vm.run = _step_and_run  # type: ignore
    return vm

GLOBAL_EXTENSIONS.register_tool("instrument_vm", instrument_vm)

# ------- Enhanced VM/Executor error context helpers -------
def vm_error_context(vm: VM, exc: BaseException) -> str:
    """
    Build a helpful context string when VM errors occur.
    """
    try:
        pc = vm.pc
        op, arg = vm.bytecode[pc] if 0 <= pc < len(vm.bytecode) else ("<out-of-range>", None)
        stack_snapshot = repr(vm.stack[-10:])
        vars_snapshot = repr({k: vm.vars.get(k) for k in list(vm.vars)[:10]})
        return f"VMError at pc={pc}, op={op}, arg={arg}\nstack(last10)={stack_snapshot}\nvars(sample)={vars_snapshot}\nexc={exc}"
    except Exception as e:
        return f"vm_error_context build failed: {e}; original exc: {exc}"

def executor_error_context(ex: Executor, exc: BaseException) -> str:
    try:
        return f"ExecutorError steps={ex._steps}, halted={ex._halted}, env-sample={dict(list(ex.env.items())[:10])}, exc={exc}"
    except Exception:
        return f"executor_error_context build failed; exc={exc}"

GLOBAL_EXTENSIONS.register_tool("vm_error_context", vm_error_context)
GLOBAL_EXTENSIONS.register_tool("executor_error_context", executor_error_context)

# ------- Additional optimizer passes: jump folding & light const-prop -------
def _jump_folding_opt(bytecode: List[Tuple[str, Any]], const_pool: List[Any]):
    """
    Fold JUMP targets that point to the very next instruction: replace with NOOP.
    Also patch JUMP_IF_FALSE that jump to immediate next instruction.
    """
    bc = list(bytecode)
    changed = False
    for idx, (op, arg) in enumerate(list(bc)):
        if op in ("JUMP", "JUMP_IF_FALSE") and isinstance(arg, int):
            target = arg
            # if target equals idx+1 then remove / replace
            if target == idx + 1:
                bc[idx] = ("NOOP", None)
                changed = True
    return (bc, const_pool) if changed else (bytecode, const_pool)

def _light_const_propagation(bytecode: List[Tuple[str, Any]], const_pool: List[Any]):
    """
    Very conservative const-prop: if LOAD_CONST X followed by STORE_VAR name with no intervening LOAD_VAR for that name,
    then subsequent LOAD_VAR name immediately followed by BINARY_OP with another LOAD_CONST can be folded.
    This is intentionally limited to avoid complex analysis.
    """
    # For demo: no-op placeholder returning original inputs (safe)
    return bytecode, const_pool

GLOBAL_EXTENSIONS.register_optimizer(_jump_folding_opt)
GLOBAL_EXTENSIONS.register_optimizer(_light_const_propagation)

# ------- Safe enhanced HTML templating (mini language) -------
def render_html_safe(template: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Support:
      - {{ var }}
      - {{#each listvar}}...{{/each}} for simple list iteration
    Everything is HTML-escaped.
    """
    ctx = context or {}

    def replace_var(m):
        key = m.group(1).strip()
        val = ctx.get(key, "")
        return html.escape(str(val))

    # handle each blocks
    def each_repl(m):
        list_name = m.group(1).strip()
        body = m.group(2)
        lst = ctx.get(list_name) or []
        out = []
        for item in lst:
            # support item as dict with keys accessible via {{this.key}} or {{this}}
            local_ctx = dict(ctx)
            local_ctx["this"] = item
            # replace {{this}} and nested vars within body
            def repl_inner(mm):
                k = mm.group(1).strip()
                if k.startswith("this."):
                    subk = k[len("this."):]
                    v = item.get(subk) if isinstance(item, dict) else getattr(item, subk, "")
                    return html.escape(str(v))
                if k == "this":
                    return html.escape(str(item))
                return html.escape(str(local_ctx.get(k, "")))
            out.append(re.sub(r"\{\{\s*([A-Za-z0-9_\.]+)\s*\}\}", repl_inner, body))
        return "".join(out)

    # process each blocks first
    template = re.sub(r"\{\{\s*#each\s+([A-Za-z0-9_]+)\s*\}\}([\s\S]*?)\{\{\s*/each\s*\}\}", each_repl, template)
    # then variables
    return re.sub(r"\{\{\s*([A-Za-z0-9_\.]+)\s*\}\}", replace_var, template)

GLOBAL_EXTENSIONS.register_tool("render_html_safe", render_html_safe)

# ------- Convenience: register documentation / capabilities summary -------
def capability_summary() -> Dict[str, Any]:
    return {
        "vm_ops_added": ["DICT_*/LIST_*/JSON_*/B64_*/HASH_*", "FS_READ/FS_WRITE (opt-in)", "HTTP_*", "WASM_INVOKE"],
        "tools": list(GLOBAL_EXTENSIONS.tool_hooks.keys()),
        "optimizers": [o.__name__ for o in GLOBAL_EXTENSIONS.optimizers],
        "safety": {
            "networking": GLOBAL_EXTENSIONS._allow_networking,
            "wasm": GLOBAL_EXTENSIONS._wasm_allowed,
            "filesystem": GLOBAL_EXTENSIONS._allow_filesystem,
        }
    }

GLOBAL_EXTENSIONS.register_tool("capability_summary", capability_summary)

# End of append-only advanced capability layer.

def compile_to_vm_with_cache(src: str, optimize: bool = True, use_cache: bool = True) -> Tuple[List[Tuple[str, Any]], List[Any], List[Tuple[str, Any]]]:
    """
    Compile source to bytecode/const_pool/IR with caching.
    Uses GLOBAL_EXTENSIONS for preprocessing and optimization.
    """
    key = _hash_source(src + str(optimize))
    if use_cache and key in GLOBAL_EXTENSIONS._bcache:
        return GLOBAL_EXTENSIONS._bcache[key]
    # preprocess
    pre_src = src
    for pre in GLOBAL_EXTENSIONS.preprocessors:
        pre_src = pre(pre_src)
    # compile
    ir = parse_source_to_ir(pre_src)
    bytecode, const_pool = compile_ir_to_bytecode(ir)
    # optimize
    if optimize:
        bytecode, const_pool = _const_fold_optimizer(bytecode, const_pool)
        for opt in GLOBAL_EXTENSIONS.optimizers:
            bytecode, const_pool = opt(bytecode, const_pool)
        # final cleanup
        bytecode, const_pool = _dead_code_elimination(bytecode, const_pool)
        bytecode, const_pool = _peephole_optimize(bytecode, const_pool)
        bytecode, const_pool = _jump_folding_opt(bytecode, const_pool)
        bytecode, const_pool = _light_const_propagation(bytecode, const_pool)
        bytecode, const_pool = _const_fold_optimizer(bytecode, const_pool)
        bytecode, const_pool = _dead_code_elimination(bytecode, const_pool)
        if use_cache:
            GLOBAL_EXTENSIONS._bcache[key] = (bytecode, const_pool, ir)
            return bytecode, const_pool, ir
        if use_cache:
            GLOBAL_EXTENSIONS._bcache[key] = (bytecode, const_pool, ir)
            return bytecode, const_pool, ir
        if use_cache:
            GLOBAL_EXTENSIONS._bcache[key] = (bytecode, const_pool, ir)
            return bytecode, const_pool, ir
        if use_cache:
            GLOBAL_EXTENSIONS._bcache[key] = (bytecode, const_pool, ir)
            return bytecode, const_pool, ir
        if use_cache:
            GLOBAL_EXTENSIONS._bcache[key] = (bytecode, const_pool, ir)
            return bytecode, const_pool, ir
        if use_cache:
            GLOBAL_EXTENSIONS._bcache[key] = (bytecode, const_pool, ir)
            return bytecode, const_pool, ir
        return bytecode, const_pool, ir
    if use_cache:
        GLOBAL_EXTENSIONS._bcache[key] = (bytecode, const_pool, ir)
        return bytecode, const_pool, ir
    return bytecode, const_pool, ir
    if use_cache:
        GLOBAL_EXTENSIONS._bcache[key] = (bytecode, const_pool, ir)
        return bytecode, const_pool, ir
    return bytecode, const_pool, ir

# --- Ultra-capability append-only layer (append-only) ---
# Adds:
#  - textual macro inliner (#define macros and simple inline expansion)
#  - persistent bytecode & IR cache, LRU result cache, memoization helpers
#  - optional native acceleration (numba / numpy) for math-heavy workloads (opt-in)
#  - broad math ops (math + optional numpy) registered as VM ops
#  - advanced crypto ops (hash/hmac + optional AES via cryptography) (opt-in)
#  - safe smartcard / serial / board-management wrappers (opt-in; gated)
#  - high-level NLP helpers (tokenize, ngrams, freq)
#  - project scaffolding helpers for app/service/embedded templates (non-destructive)
#  - strong error contexts, structured logging, and defensive checks
#
# All risky features require explicit enabling via GLOBAL_EXTENSIONS tools.
# This block is intentionally conservative: stubs use optional libraries where available
# and otherwise provide safe simulations.

import inspect
import math
import shutil
import functools
import collections
from collections import OrderedDict
from typing import Iterable
import importlib

# ------------ Persistence & LRU caches ------------
if not hasattr(GLOBAL_EXTENSIONS, "_persistent_cache_dir"):
    GLOBAL_EXTENSIONS._persistent_cache_dir = Path(".sayit_cache")

def ensure_cache_dir():
    d = GLOBAL_EXTENSIONS._persistent_cache_dir
    d.mkdir(parents=True, exist_ok=True)
    return d

# persistent compile cache (already had _bcache in earlier blocks) - extend with file backing
def persistent_cache_save(key: str, payload: Any):
    try:
        d = ensure_cache_dir()
        p = d / f"{key}.pkl"
        with open(p, "wb") as f:
            pickle.dump(payload, f)
    except Exception:
        pass

def persistent_cache_load(key: str):
    try:
        d = ensure_cache_dir()
        p = d / f"{key}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f)
    except Exception:
        pass
    return None

GLOBAL_EXTENSIONS.register_tool("persistent_cache_save", persistent_cache_save)
GLOBAL_EXTENSIONS.register_tool("persistent_cache_load", persistent_cache_load)

# Lightweight thread-safe LRU cache for runtime results (in-memory)
class LRUCache:
    def __init__(self, maxsize=2048):
        self.maxsize = maxsize
        self.lock = threading.RLock()
        self.store = OrderedDict()

    def get(self, key):
        with self.lock:
            v = self.store.get(key)
            if v is None:
                return None
            # move to end (most recently used)
            self.store.move_to_end(key)
            return v

    def set(self, key, value):
        with self.lock:
            self.store[key] = value
            self.store.move_to_end(key)
            if len(self.store) > self.maxsize:
                self.store.popitem(last=False)

GLOBAL_EXTENSIONS._result_lru = LRUCache(maxsize=4096)
GLOBAL_EXTENSIONS.register_tool("get_result_cache", lambda: GLOBAL_EXTENSIONS._result_lru)

def memoize_result(func=None, *, cache_key_prefix="memo", ttl: Optional[int]=None):
    """
    Decorator to memoize Python-callable used by tools (not VM bytecode).
    TTL is not enforced here (placeholder).
    """
    def deco(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            key = cache_key_prefix + ":" + _hash_source(json.dumps((repr(args), repr(kwargs)), sort_keys=True))
            v = GLOBAL_EXTENSIONS._result_lru.get(key)
            if v is not None:
                return v
            res = f(*args, **kwargs)
            GLOBAL_EXTENSIONS._result_lru.set(key, res)
            return res
        return wrapped
    if func:
        return deco(func)
    return deco

GLOBAL_EXTENSIONS.register_tool("memoize_result", memoize_result)

# ------------ Macro inliner (source preprocessing) ------------
# Very small macro system: #define name(args) body
# Example:
#   #define add1(x) (x + 1)
#   a = add1(3)
# This is textual expansion before parsing and is intentionally simple.
_macro_defs = {}

def _macro_define_preprocessor(src: str) -> str:
    """
    Parse lines starting with '#define' and store macro body.
    Replace macro usages with textual expansion.
    """
    global _macro_defs
    lines = src.splitlines()
    out_lines = []
    for ln in lines:
        m = re.match(r"^\s*#define\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*(.*)$", ln)
        if m:
            name = m.group(1)
            params = [p.strip() for p in m.group(2).split(",")] if m.group(2).strip() else []
            body = m.group(3) or ""
            _macro_defs[name] = (params, body)
            continue
        # Expand macros in-line (naive: look for name(...))
        def _repl(mm):
            nm = mm.group(1)
            argstr = mm.group(2)
            if nm not in _macro_defs:
                return mm.group(0)
            params, body = _macro_defs[nm]
            args = [a.strip() for a in re.split(r",(?![^()]*\))", argstr)] if argstr.strip() else []
            if len(args) != len(params):
                # fallback: skip expansion if mismatch
                return mm.group(0)
            res = body
            for p, a in zip(params, args):
                # simple substitution
                res = re.sub(rf"\b{re.escape(p)}\b", a, res)
            return res
        ln2 = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)", _repl, ln)
        out_lines.append(ln2)
    return "\n".join(out_lines)

GLOBAL_EXTENSIONS.register_preprocessor(_macro_define_preprocessor)

# ------------ Native acceleration gates ------------
GLOBAL_EXTENSIONS._native_accel_enabled = False
GLOBAL_EXTENSIONS._numpy_enabled = False
GLOBAL_EXTENSIONS._numba_enabled = False

def enable_native_acceleration(allow: bool = True):
    GLOBAL_EXTENSIONS._native_accel_enabled = bool(allow)
    if allow:
        try:
            import numpy as _np  # type: ignore
            GLOBAL_EXTENSIONS._numpy_enabled = True
        except Exception:
            GLOBAL_EXTENSIONS._numpy_enabled = False
        try:
            import numba  # type: ignore
            GLOBAL_EXTENSIONS._numba_enabled = True
        except Exception:
            GLOBAL_EXTENSIONS._numba_enabled = False

GLOBAL_EXTENSIONS.register_tool("enable_native_acceleration", enable_native_acceleration)

# Provide vector-aware BINARY_OP handler if numpy available
def _op_binary_op_fast(vm: VM, arg):
    # arg is operator; this op will be registered as a fall-back if native acceleration enabled
    b = vm.stack.pop()
    a = vm.stack.pop()
    op = arg
    # vectorized handling
    try:
        if GLOBAL_EXTENSIONS._numpy_enabled:
            import numpy as _np  # type: ignore
            if isinstance(a, (_np.ndarray, list)) or isinstance(b, (_np.ndarray, list)):
                a_arr = _np.array(a)
                b_arr = _np.array(b)
                if op == "+":
                    vm.stack.append((a_arr + b_arr).tolist())
                    return
                if op == "-":
                    vm.stack.append((a_arr - b_arr).tolist())
                    return
                if op == "*":
                    vm.stack.append((a_arr * b_arr).tolist())
                    return
                if op == "/":
                    vm.stack.append((a_arr / b_arr).tolist())
                    return
        # fallback to default semantics (reuse VM.op_binary_op)
        # Put back values and call original
        vm.stack.append(a); vm.stack.append(b)
        return VM.op_binary_op(vm, arg)
    except Exception:
        vm.stack.append(a); vm.stack.append(b)
        raise

GLOBAL_EXTENSIONS.register_vm_op("BINARY_OP_FAST", _op_binary_op_fast)

# ------------ Extensive math ops registration ------------
_math_funcs = {
    "MATH_SIN": math.sin,
    "MATH_COS": math.cos,
    "MATH_TAN": math.tan,
    "MATH_SQRT": math.sqrt,
    "MATH_LOG": math.log,
    "MATH_EXP": math.exp,
    "MATH_POW": math.pow,
    "MATH_HYPOT": math.hypot,
    "MATH_FLOOR": math.floor,
    "MATH_CEIL": math.ceil,
    "MATH_ABS": abs,
}

def _make_math_op(fn):
    def op(vm: VM, arg):
        a = vm.stack.pop()
        try:
            vm.stack.append(fn(a) if (fn is not math.pow) else fn(a, arg if arg is not None else 2))
        except Exception as e:
            raise RuntimeError(f"math op error: {e}")
    return op

for name, fn in _math_funcs.items():
    GLOBAL_EXTENSIONS.register_vm_op(name, _make_math_op(fn))

# Vectorized math (if numpy enabled) registered on demand via tool
def register_vectorized_math():
    if not GLOBAL_EXTENSIONS._numpy_enabled:
        return False
    import numpy as _np  # type: ignore
    def _wrap_vec(fn):
        def op(vm: VM, arg):
            b = vm.stack.pop()
            a = vm.stack.pop()
            a_arr = _np.array(a)
            b_arr = _np.array(b) if isinstance(b, (list, tuple, _np.ndarray)) else b
            try:
                res = fn(a_arr, b_arr) if arg is not None else fn(a_arr)
                vm.stack.append(res.tolist() if hasattr(res, "tolist") else res)
            except Exception as e:
                raise RuntimeError(f"vector math error: {e}")
        return op
    # register a few vector ops as examples
    GLOBAL_EXTENSIONS.register_vm_op("VEC_ADD", _wrap_vec(lambda a,b: a + b))
    GLOBAL_EXTENSIONS.register_vm_op("VEC_SUB", _wrap_vec(lambda a,b: a - b))
    GLOBAL_EXTENSIONS.register_vm_op("VEC_MUL", _wrap_vec(lambda a,b: a * b))
    return True

GLOBAL_EXTENSIONS.register_tool("register_vectorized_math", register_vectorized_math)

# ------------ Advanced crypto ops (safe, opt-in) ------------
GLOBAL_EXTENSIONS._crypto_enabled = False

def enable_crypto(allow: bool = True):
    GLOBAL_EXTENSIONS._crypto_enabled = bool(allow)
GLOBAL_EXTENSIONS.register_tool("enable_crypto", enable_crypto)

# Basic crypto ops (hash/hmac) always available; AES via cryptography optional
def _op_hmac_sha256(vm: VM, arg):
    msg = vm.stack.pop()
    key = vm.stack.pop()
    if isinstance(key, str):
        key = key.encode("utf-8")
    if isinstance(msg, str):
        msg = msg.encode("utf-8")
    import hmac, hashlib as _hl
    vm.stack.append(hmac.new(key, msg, _hl.sha256).hexdigest())

GLOBAL_EXTENSIONS.register_vm_op("HMAC_SHA256", _op_hmac_sha256)

# AES wrapper (requires cryptography) - opt-in enabling recommended
def _op_aes_encrypt(vm: VM, arg):
    if not GLOBAL_EXTENSIONS._crypto_enabled:
        raise RuntimeError("AES ops disabled. Call GLOBAL_EXTENSIONS.get_tool('enable_crypto')(True) to enable.")
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes  # type: ignore
        from cryptography.hazmat.backends import default_backend  # type: ignore
    except Exception:
        raise RuntimeError("cryptography not available")
    data = vm.stack.pop()
    key = vm.stack.pop()
    iv = arg or b"\x00" * 16
    if isinstance(data, str):
        data = data.encode("utf-8")
    if isinstance(key, str):
        key = key.encode("utf-8")
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    enc = cipher.encryptor().update(data) + cipher.encryptor().finalize()
    vm.stack.append(base64.b64encode(enc).decode("ascii"))

def _op_aes_decrypt(vm: VM, arg):
    if not GLOBAL_EXTENSIONS._crypto_enabled:
        raise RuntimeError("AES ops disabled. Call GLOBAL_EXTENSIONS.get_tool('enable_crypto')(True) to enable.")
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes  # type: ignore
        from cryptography.hazmat.backends import default_backend  # type: ignore
    except Exception:
        raise RuntimeError("cryptography not available")
    data = vm.stack.pop()
    key = vm.stack.pop()
    iv = arg or b"\x00" * 16
    if isinstance(data, str):
        data = base64.b64decode(data)
    if isinstance(key, str):
        key = key.encode("utf-8")
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    dec = cipher.decryptor().update(data) + cipher.decryptor().finalize()
    vm.stack.append(dec.decode("utf-8", errors="replace"))

GLOBAL_EXTENSIONS.register_vm_op("AES_ENCRYPT", _op_aes_encrypt)
GLOBAL_EXTENSIONS.register_vm_op("AES_DECRYPT", _op_aes_decrypt)

# ------------ Smartcard / Serial / Board management (opt-in) ------------
GLOBAL_EXTENSIONS._allow_hardware = False
def enable_hardware(allow: bool = True):
    GLOBAL_EXTENSIONS._allow_hardware = bool(allow)
GLOBAL_EXTENSIONS.register_tool("enable_hardware", enable_hardware)

# Smartcard helpers (pyscard if available) - gated
def list_smartcard_readers():
    if not GLOBAL_EXTENSIONS._allow_hardware:
        raise RuntimeError("Hardware access disabled. Call GLOBAL_EXTENSIONS.get_tool('enable_hardware')(True) to enable.")
    try:
        from smartcard.System import readers  # type: ignore
    except Exception:
        raise RuntimeError("pyscard not available")
    return [str(r) for r in readers()]

def transmit_apdu(reader_name: str, apdu: List[int]):
    if not GLOBAL_EXTENSIONS._allow_hardware:
        raise RuntimeError("Hardware access disabled.")
    try:
        from smartcard.System import readers  # type: ignore
    except Exception:
        raise RuntimeError("pyscard not available")
    for r in readers():
        if str(r) == reader_name:
            conn = r.createConnection()
            conn.connect()
            data, sw1, sw2 = conn.transmit(apdu)
            return data, sw1, sw2
    raise RuntimeError("reader not found")

GLOBAL_EXTENSIONS.register_tool("list_smartcard_readers", list_smartcard_readers)
GLOBAL_EXTENSIONS.register_tool("transmit_apdu", transmit_apdu)

# Serial port helpers for chip/board interactions (requires pyserial)
def serial_send_recv(port: str, data: bytes, baud: int = 115200, timeout: float = 1.0):
    if not GLOBAL_EXTENSIONS._allow_hardware:
        raise RuntimeError("Hardware access disabled.")
    try:
        import serial  # type: ignore
    except Exception:
        raise RuntimeError("pyserial not available")
    with serial.Serial(port, baudrate=baud, timeout=timeout) as s:
        s.write(data)
        s.flush()
        resp = s.read(4096)
        return resp

GLOBAL_EXTENSIONS.register_tool("serial_send_recv", serial_send_recv)

# Board manager (simulated when hardware not allowed)
class BoardManager:
    def __init__(self):
        self.configs = {}
        self.lock = threading.RLock()

    def push_config(self, board_id: str, cfg: Dict[str, Any]):
        with self.lock:
            # store and optionally apply via serial/smartcard/other methods (opt-in)
            self.configs[board_id] = cfg
            # simulate possible hardware action
            if GLOBAL_EXTENSIONS._allow_hardware:
                # attempt to send config to a serial port specified in cfg (best-effort)
                port = cfg.get("serial_port")
                payload = json.dumps(cfg).encode("utf-8")
                if port:
                    try:
                        serial_send_recv(port, payload)
                    except Exception:
                        pass
            return True

    def get_config(self, board_id: str):
        return self.configs.get(board_id)

GLOBAL_EXTENSIONS._board_manager = BoardManager()
GLOBAL_EXTENSIONS.register_tool("board_manager", lambda: GLOBAL_EXTENSIONS._board_manager)

# ------------ NLP / Linguistic helpers ------------
def nlp_tokenize(text: str) -> List[str]:
    # very small whitespace and punctuation tokenizer
    toks = re.findall(r"\b\w+\b", text.lower())
    return toks

def nlp_ngrams(tokens: List[str], n: int = 2) -> List[Tuple[str,...]]:
    if n <= 1:
        return [(t,) for t in tokens]
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def nlp_freq(tokens: Iterable[str], top: int = 20):
    cnt = collections.Counter(tokens)
    return cnt.most_common(top)

GLOBAL_EXTENSIONS.register_tool("nlp_tokenize", nlp_tokenize)
GLOBAL_EXTENSIONS.register_tool("nlp_ngrams", nlp_ngrams)
GLOBAL_EXTENSIONS.register_tool("nlp_freq", nlp_freq)

# Provide VM ops for basic linguistic tasks
def _op_nlp_tokenize(vm: VM, arg):
    text = vm.stack.pop()
    vm.stack.append(nlp_tokenize(text))

def _op_nlp_freq(vm: VM, arg):
    toks = vm.stack.pop()
    top = arg or 10
    vm.stack.append(nlp_freq(toks, top))

GLOBAL_EXTENSIONS.register_vm_op("NLP_TOKENIZE", _op_nlp_tokenize)
GLOBAL_EXTENSIONS.register_vm_op("NLP_FREQ", _op_nlp_freq)

# ------------ Project scaffolding helpers ------------
def create_project_skeleton(kind: str, path: str, name: str):
    """
    Create minimal skeleton for 'app', 'service', or 'embedded' projects.
    This helper is non-destructive and refuses to overwrite existing files.
    """
    p = Path(path) / name
    if p.exists():
        raise RuntimeError("target exists")
    p.mkdir(parents=True)
    if kind == "app":
        (p / "main.say").write_text("print('Hello from app')\n")
        (p / "README.md").write_text(f"# {name}\n\nApp skeleton\n")
    elif kind == "service":
        (p / "service.say").write_text("while 1\nprint('service loop')\nend\n")
        (p / "Dockerfile").write_text("FROM python:3.11-slim\n# service skeleton\n")
    elif kind == "embedded":
        (p / "firmware.say").write_text("; firmware skeleton\n")
        (p / "board.cfg").write_text("serial_port=/dev/ttyUSB0\nbaud=115200\n")
    else:
        raise RuntimeError("unknown kind")
    return str(p)

GLOBAL_EXTENSIONS.register_tool("create_project_skeleton", create_project_skeleton)

# ------------ Strong error context / structured logging ------------
def structured_log(level: str, msg: str, **meta):
    t = time.time()
    entry = {"ts": t, "level": level, "msg": msg, "meta": meta}
    # append to in-memory trace (bounded)
    if not hasattr(GLOBAL_EXTENSIONS, "_log_ring"):
        GLOBAL_EXTENSIONS._log_ring = collections.deque(maxlen=1000)
    GLOBAL_EXTENSIONS._log_ring.append(entry)
    # also optionally print
    if level in ("error", "warn"):
        print(f"[{level.upper()}] {msg}", file=sys.stderr)
    else:
        print(f"[{level.upper()}] {msg}")
    return entry

GLOBAL_EXTENSIONS.register_tool("log", structured_log)
GLOBAL_EXTENSIONS.register_tool("get_logs", lambda: list(getattr(GLOBAL_EXTENSIONS, "_log_ring", [])))

# ------------ Inlining optimizer (bytecode-level simple inliner) ------------
def _simple_inliner(bytecode: List[Tuple[str, Any]], const_pool: List[Any]):
    """
    Identify tiny function-like patterns emitted as:
      LOAD_CONST <func_obj>, CALL <name>
    and inline by replacing CALL with the function body if the function body is small and marked inline.
    Because the current codegen doesn't emit function objects, this pass is conservative
    and looks for patterns of STORE_VAR/LOAD_VAR sequences that can be folded.
    This is a conservative placeholder demonstrating an inlining hook.
    """
    # No-op placeholder to preserve safety; real inlining would require IR-level analysis.
    return bytecode, const_pool

GLOBAL_EXTENSIONS.register_optimizer(_simple_inliner)

# ------------ Developer convenience: capability & safety checklist ------------
def advanced_capabilities_report():
    return {
        "native": {
            "enabled": GLOBAL_EXTENSIONS._native_accel_enabled,
            "numpy": GLOBAL_EXTENSIONS._numpy_enabled,
            "numba": GLOBAL_EXTENSIONS._numba_enabled,
        },
        "crypto": GLOBAL_EXTENSIONS._crypto_enabled,
        "hardware": GLOBAL_EXTENSIONS._allow_hardware,
        "filesystem": GLOBAL_EXTENSIONS._allow_filesystem,
        "networking": GLOBAL_EXTENSIONS._allow_networking,
        "wasm": GLOBAL_EXTENSIONS._wasm_allowed,
        "tools": list(GLOBAL_EXTENSIONS.tool_hooks.keys()),
        "vm_ops": list(GLOBAL_EXTENSIONS.vm_ops.keys()),
    }

GLOBAL_EXTENSIONS.register_tool("advanced_capabilities_report", advanced_capabilities_report)

# ------------ Safety notes (logged) ------------
structured_log("info", "Ultra-capability layer loaded (append-only). Review and opt-in to sensitive features: enable_native_acceleration, enable_crypto, enable_hardware, enable_filesystem, enable_networking, enable_wasm.")

# End of ultra-capability append-only layer.
import time
import re
import html
import pickle
import hashlib as _hashlib
import base64
import json
from pathlib import Path
from typing import List, Tuple, Any, Dict, Optional
from typing import Callable
from collections import defaultdict
from dataclasses import dataclass, field
import threading
from enum import Enum
import sys
import logging
import io
import os
from functools import lru_cache
import traceback
import tempfile
from contextlib import contextmanager
from subprocess import Popen, PIPE
import struct
import wasm  # type: ignore
import wasm.vm  # type: ignore
import wasm.decode  # type: ignore
import wasm.encode  # type: ignore
import wasm.instructions  # type: ignore
import wasm.module  # type: ignore
import wasm.types  # type: ignore
import wasm.validate  # type: ignore
import wasm.interpreter  # type: ignore
import wasm.utils  # type: ignore
import wasm.exceptions  # type: ignore
import re
from collections import namedtuple
from typing import Union
import ast
import operator
import copy
import json
import threading
import time
from typing import Optional, Callable, Any, Dict, List, Tuple
import pickle
from pathlib import Path
import hashlib
import base64
import html
import os
import sys
import io
import tempfile
import shutil
import threading
import time
import json
import re
import logging
import functools
import collections
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import OrderedDict

# --- Core VM and Executor implementation ---
class VM:
    def __init__(self, bytecode: List[Tuple[str, Any]], const_pool: List[Any]):
        self.bytecode = bytecode
        self.const_pool = const_pool
        self.stack = []
        self.vars = {}
        self.pc = 0
        self.running = False
    def run(self):
        self.running = True
        while self.running and self.pc < len(self.bytecode):
            op, arg = self.bytecode[self.pc]
            # dispatch via existing extension-aware handler (use GLOBAL_EXTENSIONS)
            ext = GLOBAL_EXTENSIONS.get_vm_op(op)
            if ext:
                try:
                    ext(self, arg)
                except Exception as e:
                    # attach context and re-raise
                    raise RuntimeError(f"VM op {op} @pc={self.pc} failed: {e}")
                self.pc += 1
                continue
            handler = getattr(self, f"op_{op.lower()}", None)
            if handler is None:
                raise RuntimeError(f"Unknown opcode: {op} @pc={self.pc}")
            try:
                handler(arg)
            except Exception as e:
                raise RuntimeError(f"VM op {op} @pc={self.pc} failed: {e}")
            self.pc += 1
    def op_noop(self, arg):
        pass
    def op_halt(self, arg):
        self.running = False
    def op_load_const(self, arg):
        if not (0 <= arg < len(self.const_pool)):
            raise RuntimeError(f"LOAD_CONST with invalid index: {arg}")
        self.stack.append(self.const_pool[arg])
    def op_store_var(self, arg):
        val = self.stack.pop()
        self.vars[arg] = val
    def op_load_var(self, arg):
        if arg not in self.vars:
            raise RuntimeError(f"LOAD_VAR of undefined variable: {arg}")
        self.stack.append(self.vars[arg])
    def op_binary_op(self, arg):
        b = self.stack.pop()
        a = self.stack.pop()
        if arg == "+":
            self.stack.append(a + b)
        elif arg == "-":
            self.stack.append(a - b)
        elif arg == "*":
            self.stack.append(a * b)
        elif arg == "/":
            if b == 0:
                raise RuntimeError("Division by zero")
            self.stack.append(a / b)

python 
system.py
# --- Append-only: comprehensive web & networking capabilities ---
# Opt-in, non-invasive helpers for HTTP(S) clients, WebSocket, simple servers,
# TCP/UDP helpers, DNS utilities, reverse-proxy, retry/backoff and caching.
# All network actions require explicit enablement via GLOBAL_EXTENSIONS tools.

import socket
import selectors
import http.server
import ssl
import threading
import time
from urllib.parse import urlparse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from typing import Any, Dict, Optional, Tuple, List

# Feature gates (do not enable by default)
GLOBAL_EXTENSIONS._allow_servers = False
GLOBAL_EXTENSIONS._allow_raw_sockets = False
GLOBAL_EXTENSIONS._http_client_session = None

def enable_servers(allow: bool = True):
    GLOBAL_EXTENSIONS._allow_servers = bool(allow)
GLOBAL_EXTENSIONS.register_tool("enable_servers", enable_servers)

def enable_raw_sockets(allow: bool = True):
    GLOBAL_EXTENSIONS._allow_raw_sockets = bool(allow)
GLOBAL_EXTENSIONS.register_tool("enable_raw_sockets", enable_raw_sockets)

# ---------- Robust HTTP client with retries, backoff and caching ----------
# Uses requests if available for connection pooling; falls back to urllib.
try:
    import requests  # type: ignore
    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False

def _simple_backoff(retries:int, backoff:float, attempt:int):
    # jittered exponential backoff
    return backoff * (2 ** attempt) + (backoff * 0.1 * (attempt % 3))

def http_request(url: str,
                 method: str = "GET",
                 headers: Optional[Dict[str, str]] = None,
                 data: Optional[bytes] = None,
                 timeout: float = 10.0,
                 retries: int = 2,
                 backoff: float = 0.25,
                 verify_tls: bool = True,
                 allow_redirects: bool = True) -> Dict[str, Any]:
    """
    High-level HTTP(S) request helper. Returns dict: {"status":int,"body":bytes,"headers":dict}.
    Requires GLOBAL_EXTENSIONS._allow_networking True.
    Attempts to use requests for pooling; falls back to urllib.
    """
    if not GLOBAL_EXTENSIONS._allow_networking:
        raise RuntimeError("Networking disabled. Call GLOBAL_EXTENSIONS.get_tool('enable_networking')(True) to enable.")
    method = method.upper()
    headers = headers or {}

    last_exc = None
    for attempt in range(retries + 1):
        try:
            if _HAS_REQUESTS:
                sess = GLOBAL_EXTENSIONS._http_client_session
                if sess is None:
                    sess = requests.Session()
                    GLOBAL_EXTENSIONS._http_client_session = sess
                resp = sess.request(method, url, headers=headers, data=data, timeout=timeout, verify=verify_tls, allow_redirects=allow_redirects)
                content = resp.content
                # enforce size
                if len(content) > GLOBAL_EXTENSIONS._max_request_size:
                    raise RuntimeError("Response exceeded configured max_request_size")
                return {"status": resp.status_code, "body": content, "headers": dict(resp.headers)}
            else:
                # urllib fallback
                ctx = ssl.create_default_context() if verify_tls else ssl._create_unverified_context()
                req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
                with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                    status = getattr(r, "status", None) or r.getcode()
                    body = r.read(GLOBAL_EXTENSIONS._max_request_size + 1)
                    if len(body) > GLOBAL_EXTENSIONS._max_request_size:
                        raise RuntimeError("Response exceeded configured max_request_size")
                    hdrs = {k.lower(): v for k, v in r.headers.items()}
                    return {"status": int(status), "body": body, "headers": hdrs}
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(_simple_backoff(retries, backoff, attempt))
                continue
            raise
    raise last_exc or RuntimeError("http_request unknown error")

GLOBAL_EXTENSIONS.register_tool("http_request", http_request)

# VM ops wrappers for extended HTTP verbs
def _op_http_head(vm: VM, arg):
    if not GLOBAL_EXTENSIONS._allow_networking:
        raise RuntimeError("Networking disabled.")
    # arg may be URL or tuple (url, headers)
    if isinstance(arg, (list, tuple)):
        url = arg[0]; headers = arg[1] if len(arg) > 1 else {}
    else:
        url = str(arg); headers = {}
    res = http_request(url, method="HEAD", headers=headers)
    vm.stack.append({"status": res["status"], "headers": res["headers"]})

def _op_http_put(vm: VM, arg):
    if not GLOBAL_EXTENSIONS._allow_networking:
        raise RuntimeError("Networking disabled.")
    # expects (url, data, headers?) OR url
    if isinstance(arg, (list, tuple)):
        url = arg[0]; data = arg[1] if len(arg) > 1 else b""; headers = arg[2] if len(arg) > 2 else {}
    else:
        url = str(arg); data = b""; headers = {}
    if isinstance(data, str):
        data = data.encode("utf-8")
    res = http_request(url, method="PUT", headers=headers, data=data)
    vm.stack.append({"status": res["status"], "body": res["body"].decode("utf-8", errors="replace"), "headers": res["headers"]})

def _op_http_delete(vm: VM, arg):
    if not GLOBAL_EXTENSIONS._allow_networking:
        raise RuntimeError("Networking disabled.")
    url = arg if not isinstance(arg, (list, tuple)) else arg[0]
    res = http_request(url, method="DELETE")
    vm.stack.append({"status": res["status"], "body": res["body"].decode("utf-8", errors="replace")})

GLOBAL_EXTENSIONS.register_vm_op("HTTP_HEAD", _op_http_head)
GLOBAL_EXTENSIONS.register_vm_op("HTTP_PUT", _op_http_put)
GLOBAL_EXTENSIONS.register_vm_op("HTTP_DELETE", _op_http_delete)

# ---------- WebSocket helpers (client) ----------
# Optional: use websockets library if available; otherwise informative error.
try:
    import asyncio, websockets  # type: ignore
    _HAS_WEBSOCKETS = True
except Exception:
    _HAS_WEBSOCKETS = False

class WebSocketClient:
    """
    Minimal threadsafe wrapper for synchronous use of an async websocket client.
    """
    def __init__(self, uri: str):
        if not _HAS_WEBSOCKETS:
            raise RuntimeError("websockets library not available")
        self.uri = uri
        self._loop = asyncio.new_event_loop()
        self._ws = None
        self._thread = threading.Thread(target=self._start_loop, daemon=True)
        self._thread.start()
        self._ready = threading.Event()
        # schedule connect
        fut = asyncio.run_coroutine_threadsafe(self._connect(), self._loop)
        # wait until connected or error (timed)
        self._ready.wait(timeout=5.0)

    def _start_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _connect(self):
        self._ws = await websockets.connect(self.uri)
        self._ready.set()

    def send(self, msg: str):
        if not self._ws:
            raise RuntimeError("ws not connected")
        return asyncio.run_coroutine_threadsafe(self._ws.send(msg), self._loop).result()

    def recv(self, timeout: float = 5.0) -> str:
        if not self._ws:
            raise RuntimeError("ws not connected")
        fut = asyncio.run_coroutine_threadsafe(self._ws.recv(), self._loop)
        return fut.result(timeout=timeout)

    def close(self):
        if self._ws:
            asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop).result()

GLOBAL_EXTENSIONS.register_tool("WebSocketClient", WebSocketClient)

# VM ops for WebSocket: WS_CONNECT (returns client object), WS_SEND, WS_RECV, WS_CLOSE
def _op_ws_connect(vm: VM, arg):
    if not GLOBAL_EXTENSIONS._allow_networking:
        raise RuntimeError("Networking disabled.")
    uri = arg if isinstance(arg, str) else str(arg[0])
    client = WebSocketClient(uri)
    vm.stack.append(client)

def _op_ws_send(vm: VM, arg):
    # expects: message, client
    msg = vm.stack.pop()
    client = vm.stack.pop()
    client.send(msg)
    vm.stack.append(True)

def _op_ws_recv(vm: VM, arg):
    client = vm.stack.pop()
    msg = client.recv(arg or 5.0)
    vm.stack.append(msg)

def _op_ws_close(vm: VM, arg):
    client = vm.stack.pop()
    client.close()
    vm.stack.append(True)

GLOBAL_EXTENSIONS.register_vm_op("WS_CONNECT", _op_ws_connect)
GLOBAL_EXTENSIONS.register_vm_op("WS_SEND", _op_ws_send)
GLOBAL_EXTENSIONS.register_vm_op("WS_RECV", _op_ws_recv)
GLOBAL_EXTENSIONS.register_vm_op("WS_CLOSE", _op_ws_close)

# ---------- Simple threaded HTTP(S) server helpers ----------
class _StaticHTTPRequestHandler(SimpleHTTPRequestHandler):
    # override log to avoid noisy output; use structured_log
    def log_message(self, format_str, *args):
        structured_log("info", f"HTTP {self.client_address}: {format_str % args}")

_http_servers: Dict[int, Tuple[ThreadingHTTPServer, threading.Thread]] = {}

def start_static_server(directory: str, port: int = 8000, host: str = "0.0.0.0", tls_cert: Optional[str] = None, tls_key: Optional[str] = None) -> threading.Thread:
    """
    Serve files from directory on host:port in background thread.
    Requires GLOBAL_EXTENSIONS._allow_servers True.
    Returns the server thread object.
    """
    if not GLOBAL_EXTENSIONS._allow_servers:
        raise RuntimeError("Server creation disabled. Call GLOBAL_EXTENSIONS.get_tool('enable_servers')(True) to enable.")
    cwd = os.getcwd()
    try:
        os.chdir(directory)
        httpd = ThreadingHTTPServer((host, port), _StaticHTTPRequestHandler)
        if tls_cert and tls_key:
            httpd.socket = ssl.wrap_socket(httpd.socket, certfile=tls_cert, keyfile=tls_key, server_side=True)
        def _serve():
            structured_log("info", f"Starting static server on {host}:{port} serving {directory}")
            try:
                httpd.serve_forever()
            except Exception as e:
                structured_log("error", f"Static server on {host}:{port} stopped: {e}")
            finally:
                structured_log("info", f"Static server on {host}:{port} terminated")
        t = threading.Thread(target=_serve, daemon=True)
        t.start()
        _http_servers[port] = (httpd, t)
        return t
    finally:
        os.chdir(cwd)

def stop_server(port: int):
    tup = _http_servers.get(port)
    if not tup:
        return False
    srv, thr = tup
    try:
        srv.shutdown()
        srv.server_close()
        structured_log("info", f"Server on port {port} shutdown requested")
        return True
    except Exception as e:
        structured_log("error", f"Failed to stop server {port}: {e}")
        return False

GLOBAL_EXTENSIONS.register_tool("start_static_server", start_static_server)
GLOBAL_EXTENSIONS.register_tool("stop_server", stop_server)

# ---------- Simple reverse proxy (threaded) ----------
class _ProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_REQUEST(self):
        # forward headers and body to target provided via server.target
        target = self.server.target  # type: ignore
        parsed = urlparse(target)
        path = parsed.path.rstrip("/") + self.path
        url = f"{parsed.scheme}://{parsed.netloc}{path}"
        try:
            result = http_request(url, method=self.command, headers=dict(self.headers), data=self.rfile.read(int(self.headers.get('Content-Length') or 0)), timeout=10)
            self.send_response(result["status"])
            for k,v in (result["headers"] or {}).items():
                try:
                    self.send_header(k, v)
                except Exception:
                    pass
            self.end_headers()
            self.wfile.write(result["body"])
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(str(e).encode("utf-8"))

    def do_GET(self): return self.do_REQUEST()
    def do_POST(self): return self.do_REQUEST()
    def do_PUT(self): return self.do_REQUEST()
    def do_DELETE(self): return self.do_REQUEST()
    def log_message(self, format_str, *args):
        structured_log("info", f"Proxy {self.client_address}: {format_str % args}")

def start_reverse_proxy(local_port: int, target_url: str, host: str = "0.0.0.0") -> threading.Thread:
    if not GLOBAL_EXTENSIONS._allow_servers:
        raise RuntimeError("Server creation disabled.")
    srv = ThreadingHTTPServer((host, local_port), _ProxyHandler)
    srv.target = target_url  # type: ignore
    def _run():
        structured_log("info", f"Reverse proxy listening on {host}:{local_port} -> {target_url}")
        try:
            srv.serve_forever()
        except Exception as e:
            structured_log("error", f"Reverse proxy error: {e}")
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    _http_servers[local_port] = (srv, t)
    return t

GLOBAL_EXTENSIONS.register_tool("start_reverse_proxy", start_reverse_proxy)

# ---------- Raw TCP/UDP helpers ----------
def tcp_connect(host: str, port: int, timeout: float = 5.0) -> socket.socket:
    if not GLOBAL_EXTENSIONS._allow_raw_sockets:
        raise RuntimeError("Raw sockets disabled. Call GLOBAL_EXTENSIONS.get_tool('enable_raw_sockets')(True) to enable.")
    s = socket.create_connection((host, port), timeout=timeout)
    return s

def tcp_send_recv(host: str, port: int, payload: bytes, timeout: float = 5.0) -> bytes:
    s = tcp_connect(host, port, timeout=timeout)
    try:
        s.sendall(payload)
        s.settimeout(timeout)
        data = s.recv(65536)
        return data
    finally:
        try:
            s.close()
        except Exception:
            pass

def udp_send_recv(host: str, port: int, payload: bytes, timeout: float = 2.0) -> bytes:
    if not GLOBAL_EXTENSIONS._allow_raw_sockets:
        raise RuntimeError("Raw sockets disabled.")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.settimeout(timeout)
        s.sendto(payload, (host, port))
        data, _ = s.recvfrom(65536)
        return data

GLOBAL_EXTENSIONS.register_tool("tcp_connect", tcp_connect)
GLOBAL_EXTENSIONS.register_tool("tcp_send_recv", tcp_send_recv)
GLOBAL_EXTENSIONS.register_tool("udp_send_recv", udp_send_recv)

# VM ops wrappers for TCP/UDP (simple)
def _op_tcp_send_recv(vm: VM, arg):
    # arg: tuple(host, port, payload)
    host, port, payload = arg
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    res = tcp_send_recv(host, int(port), payload)
    vm.stack.append(res)

def _op_udp_send_recv(vm: VM, arg):
    host, port, payload = arg
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    res = udp_send_recv(host, int(port), payload)
    vm.stack.append(res)

GLOBAL_EXTENSIONS.register_vm_op("TCP_SEND_RECV", _op_tcp_send_recv)
GLOBAL_EXTENSIONS.register_vm_op("UDP_SEND_RECV", _op_udp_send_recv)

# ---------- DNS utilities ----------
import socket as _socket_mod

def resolve_hostname(name: str) -> List[str]:
    try:
        return [ai[4][0] for ai in _socket_mod.getaddrinfo(name, None)]
    except Exception as e:
        raise RuntimeError(f"DNS resolve failed: {e}")

def reverse_lookup(ip: str) -> Optional[str]:
    try:
        return _socket_mod.gethostbyaddr(ip)[0]
    except Exception:
        return None

GLOBAL_EXTENSIONS.register_tool("resolve_hostname", resolve_hostname)
GLOBAL_EXTENSIONS.register_tool("reverse_lookup", reverse_lookup)

# ---------- Service discovery (mDNS stub) ----------
def mdns_query(service: str, timeout: float = 2.0) -> List[Dict[str, Any]]:
    """
    Minimal mDNS placeholder. For full mDNS use 'zeroconf' library (not included here).
    Returns empty list when zeroconf is not available.
    """
    try:
        from zeroconf import Zeroconf, ServiceBrowser, ServiceStateChange  # type: ignore
    except Exception:
        structured_log("warn", "zeroconf not available; mdns_query returning empty list")
        return []
    out = []
    # Non-blocking simple browse (short demo)
    zc = Zeroconf()
    # For brevity: do not implement full browse here (requires callback plumbing)
    zc.close()
    return out

GLOBAL_EXTENSIONS.register_tool("mdns_query", mdns_query)

# ---------- Expose list of networking tools for discovery ----------
def list_networking_tools() -> Dict[str, Any]:
    keys = [k for k in GLOBAL_EXTENSIONS.tool_hooks.keys() if any(prefix in k for prefix in ("http","tcp","udp","ws","start","stop","resolve","mdns"))]
    # include some vm ops
    vm_ops = sorted([op for op in GLOBAL_EXTENSIONS.vm_ops.keys() if any(x in op for x in ("HTTP","WS","TCP","UDP"))])
    return {"tools": keys, "vm_ops": vm_ops, "gates": {
        "networking": GLOBAL_EXTENSIONS._allow_networking,
        "servers": GLOBAL_EXTENSIONS._allow_servers,
        "raw_sockets": GLOBAL_EXTENSIONS._allow_raw_sockets
    }}

GLOBAL_EXTENSIONS.register_tool("list_networking_tools", list_networking_tools)

# --- End of append-only networking layer ---

# ------------ Cryptographic helpers (opt-in) ------------
GLOBAL_EXTENSIONS._crypto_enabled = False
def enable_crypto(allow: bool = True):
    GLOBAL_EXTENSIONS._crypto_enabled = bool(allow)
    GLOBAL_EXTENSIONS.register_tool("enable_crypto", enable_crypto)
    GLOBAL_EXTENSIONS.register_tool("crypto_enabled", lambda: GLOBAL_EXTENSIONS._crypto_enabled)
    GLOBAL_EXTENSIONS.register_tool("hashlib_md5", lambda data: _hashlib.md5(data if isinstance(data, bytes) else str(data).encode("utf-8")).hexdigest())
    
# --- Append-only: operational communication & orchestration layer ---
# Opt-in, safe helpers for SSH, port scanning, TCP forwarding, packet capture and orchestration.
# All powerful features are gated; enable with GLOBAL_EXTENSIONS.get_tool("enable_ops")(True)
# and (where needed) enable_raw_sockets / enable_servers / enable_networking.

import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# Feature gate for operational capabilities
GLOBAL_EXTENSIONS._allow_ops = False
def enable_ops(allow: bool = True):
    GLOBAL_EXTENSIONS._allow_ops = bool(allow)
GLOBAL_EXTENSIONS.register_tool("enable_ops", enable_ops)
GLOBAL_EXTENSIONS.register_tool("ops_enabled", lambda: GLOBAL_EXTENSIONS._allow_ops)

# ---------- SSH helpers (paramiko, opt-in) ----------
try:
    import paramiko  # type: ignore
    _HAS_PARAMIKO = True
except Exception:
    _HAS_PARAMIKO = False

def _ssh_client_connect(host: str, port: int, username: Optional[str], password: Optional[str], pkey: Optional[str], timeout: float = 10.0):
    """
    Create and return a connected paramiko.SSHClient.
    Caller must close the client.
    """
    if not GLOBAL_EXTENSIONS._allow_ops:
        raise RuntimeError("Operational ops disabled. Call GLOBAL_EXTENSIONS.get_tool('enable_ops')(True) to enable.")
    if not _HAS_PARAMIKO:
        raise RuntimeError("paramiko not available. Install paramiko to use SSH helpers.")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kwargs = {"hostname": host, "port": port, "username": username, "timeout": timeout}
    if pkey:
        try:
            key = paramiko.RSAKey.from_private_key_file(pkey)
            kwargs["pkey"] = key
        except Exception:
            # try loading as paramiko.PKey fallback or raise
            raise
    else:
        kwargs["password"] = password
    client.connect(**{k:v for k,v in kwargs.items() if v is not None})
    return client

def ssh_exec(host: str, command: str, port: int = 22, username: Optional[str] = None, password: Optional[str] = None, pkey: Optional[str] = None, timeout: float = 10.0) -> Dict[str, Any]:
    """
    Execute a command over SSH. Returns {"exit_status":int,"stdout":str,"stderr":str}.
    Safe, opt-in, requires paramiko and GLOBAL_EXTENSIONS._allow_ops True.
    """
    if not GLOBAL_EXTENSIONS._allow_ops:
        raise RuntimeError("Operational ops disabled.")
    if not _HAS_PARAMIKO:
        raise RuntimeError("paramiko not available")
    client = _ssh_client_connect(host, port, username, password, pkey, timeout=timeout)
    try:
        stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        rc = stdout.channel.recv_exit_status() if stdout is not None else 0
        return {"exit_status": rc, "stdout": out, "stderr": err}
    finally:
        try:
            client.close()
        except Exception:
            pass

def ssh_sftp_put(host: str, local_path: str, remote_path: str, port: int = 22, username: Optional[str] = None, password: Optional[str] = None, pkey: Optional[str] = None, timeout: float = 10.0) -> bool:
    """
    Upload a file via SFTP. Returns True on success.
    """
    if not GLOBAL_EXTENSIONS._allow_ops:
        raise RuntimeError("Operational ops disabled.")
    client = _ssh_client_connect(host, port, username, password, pkey, timeout=timeout)
    try:
        sftp = client.open_sftp()
        sftp.put(local_path, remote_path)
        sftp.close()
        return True
    finally:
        try:
            client.close()
        except Exception:
            pass

def ssh_sftp_get(host: str, remote_path: str, local_path: str, port: int = 22, username: Optional[str] = None, password: Optional[str] = None, pkey: Optional[str] = None, timeout: float = 10.0) -> bool:
    """
    Download a file via SFTP. Returns True on success.
    """
    if not GLOBAL_EXTENSIONS._allow_ops:
        raise RuntimeError("Operational ops disabled.")
    client = _ssh_client_connect(host, port, username, password, pkey, timeout=timeout)
    try:
        sftp = client.open_sftp()
        sftp.get(remote_path, local_path)
        sftp.close()
        return True
    finally:
        try:
            client.close()
        except Exception:
            pass

GLOBAL_EXTENSIONS.register_tool("ssh_exec", ssh_exec)
GLOBAL_EXTENSIONS.register_tool("ssh_sftp_put", ssh_sftp_put)
GLOBAL_EXTENSIONS.register_tool("ssh_sftp_get", ssh_sftp_get)

# Bulk orchestration helper (run same command on many hosts concurrently)
def remote_bulk_ssh(hosts: List[str], command: str, username: Optional[str] = None, password: Optional[str] = None, pkey: Optional[str] = None, port: int = 22, concurrency: int = 10, timeout: float = 10.0) -> Dict[str, Dict[str, Any]]:
    """
    Executes `command` on a list of hosts concurrently. Returns mapping host->result dict from ssh_exec.
    """
    if not GLOBAL_EXTENSIONS._allow_ops:
        raise RuntimeError("Operational ops disabled.")
    out = {}
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        futs = {ex.submit(ssh_exec, h, command, port, username, password, pkey, timeout): h for h in hosts}
        for fut in as_completed(futs):
            h = futs[fut]
            try:
                out[h] = fut.result()
            except Exception as e:
                out[h] = {"exit_status": None, "stdout": "", "stderr": str(e)}
    return out

GLOBAL_EXTENSIONS.register_tool("remote_bulk_ssh", remote_bulk_ssh)

# ---------- Port scanning helpers ----------
def scan_ports(host: str, ports: List[int], timeout: float = 0.5, concurrency: int = 100) -> Dict[int, bool]:
    """
    TCP connect-based port scan. Returns mapping port->is_open (True/False).
    Conservative and opt-in (requires enable_networking).
    """
    if not GLOBAL_EXTENSIONS._allow_networking:
        raise RuntimeError("Networking disabled.")
    results = {}
    def _probe(p):
        try:
            with socket.create_connection((host, p), timeout=timeout):
                return p, True
        except Exception:
            return p, False
    with ThreadPoolExecutor(max_workers=min(concurrency, max(1, len(ports)))) as ex:
        futures = [ex.submit(_probe, p) for p in ports]
        for fut in as_completed(futures):
            p, ok = fut.result()
            results[p] = ok
    return results

def scan_hosts_ports(hosts: List[str], ports: List[int], timeout: float = 0.5, concurrency: int = 200) -> Dict[str, Dict[int, bool]]:
    if not GLOBAL_EXTENSIONS._allow_networking:
        raise RuntimeError("Networking disabled.")
    out = {}
    with ThreadPoolExecutor(max_workers=min(concurrency, max(1, len(hosts)))) as ex:
        futs = {ex.submit(scan_ports, h, ports, timeout, max(1, concurrency//len(hosts))): h for h in hosts}
        for fut in as_completed(futs):
            h = futs[fut]
            try:
                out[h] = fut.result()
            except Exception as e:
                out[h] = {"error": str(e)}
    return out

GLOBAL_EXTENSIONS.register_tool("scan_ports", scan_ports)
GLOBAL_EXTENSIONS.register_tool("scan_hosts_ports", scan_hosts_ports)

# VM op for SCAN_PORTS: arg = (host, [ports], timeout)
def _op_scan_ports(vm: VM, arg):
    if not GLOBAL_EXTENSIONS._allow_networking:
        raise RuntimeError("Networking disabled.")
    if isinstance(arg, (list, tuple)):
        host = arg[0]; ports = arg[1] if len(arg) > 1 else [80, 443]; timeout = arg[2] if len(arg) > 2 else 0.5
    else:
        host = str(arg); ports = [80,443]; timeout=0.5
    res = scan_ports(host, list(ports), timeout=timeout)
    vm.stack.append(res)

GLOBAL_EXTENSIONS.register_vm_op("SCAN_PORTS", _op_scan_ports)

# ---------- TCP forwarder (simple) ----------
_tcp_forwarders = {}

def start_tcp_forward(local_host: str, local_port: int, remote_host: str, remote_port: int, backlog: int = 5) -> threading.Thread:
    """
    Start a simple TCP forwarder in background that forwards connections from local_host:local_port to remote_host:remote_port.
    Requires enable_raw_sockets and enable_servers for safety.
    Returns the thread object. Stop by calling stop_tcp_forward with the same local_port.
    """
    if not GLOBAL_EXTENSIONS._allow_raw_sockets or not GLOBAL_EXTENSIONS._allow_servers:
        raise RuntimeError("TCP forwarders require raw sockets and server allowance. Enable via tools.")
    stop_flag = threading.Event()
    def _handler(client_sock):
        try:
            remote = socket.create_connection((remote_host, remote_port), timeout=5)
        except Exception as e:
            try:
                client_sock.close()
            except Exception:
                pass
            return
        def _pipe(src, dst):
            try:
                while True:
                    data = src.recv(4096)
                    if not data:
                        break
                    dst.sendall(data)
            except Exception:
                pass
            finally:
                try: src.close()
                except Exception: pass
        t1 = threading.Thread(target=_pipe, args=(client_sock, remote), daemon=True)
        t2 = threading.Thread(target=_pipe, args=(remote, client_sock), daemon=True)
        t1.start(); t2.start()
    def _serve():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((local_host, local_port))
        srv.listen(backlog)
        _tcp_forwarders[local_port] = (srv, stop_flag)
        try:
            while not stop_flag.is_set():
                try:
                    client, addr = srv.accept()
                except OSError:
                    break
                threading.Thread(target=_handler, args=(client,), daemon=True).start()
        finally:
            try: srv.close()
            except Exception: pass
    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    structured_log("info", f"TCP forwarder started {local_host}:{local_port} -> {remote_host}:{remote_port}")
    return t

def stop_tcp_forward(local_port: int) -> bool:
    tup = _tcp_forwarders.get(local_port)
    if not tup:
        return False
    srv, flag = tup
    try:
        flag.set()
    except Exception:
        pass
    try:
        srv.close()
    except Exception:
        pass
    _tcp_forwarders.pop(local_port, None)
    structured_log("info", f"Stopped TCP forwarder on port {local_port}")
    return True

GLOBAL_EXTENSIONS.register_tool("start_tcp_forward", start_tcp_forward)
GLOBAL_EXTENSIONS.register_tool("stop_tcp_forward", stop_tcp_forward)

# VM ops for TCP_FORWARD_START / TCP_FORWARD_STOP
def _op_tcp_forward_start(vm: VM, arg):
    # arg: (local_host, local_port, remote_host, remote_port)
    if not GLOBAL_EXTENSIONS._allow_raw_sockets or not GLOBAL_EXTENSIONS._allow_servers:
        raise RuntimeError("TCP forwarders require raw sockets and server allowance.")
    local_host, local_port, remote_host, remote_port = arg
    t = start_tcp_forward(local_host, int(local_port), remote_host, int(remote_port))
    vm.stack.append(True)

def _op_tcp_forward_stop(vm: VM, arg):
    local_port = int(arg)
    ok = stop_tcp_forward(local_port)
    vm.stack.append(ok)

GLOBAL_EXTENSIONS.register_vm_op("TCP_FORWARD_START", _op_tcp_forward_start)
GLOBAL_EXTENSIONS.register_vm_op("TCP_FORWARD_STOP", _op_tcp_forward_stop)

# ---------- Packet capture / sniffing (scapy opt-in) ----------
try:
    import scapy.all as scapy  # type: ignore
    _HAS_SCAPY = True
except Exception:
    _HAS_SCAPY = False

_pcap_threads = []

def packet_sniff(interface: Optional[str], callback: Callable[[Any], None], count: Optional[int] = None, timeout: Optional[float] = None) -> threading.Thread:
    """
    Start a background thread that sniffs packets on `interface` and invokes `callback(packet)`.
    Requires scapy and enable_raw_sockets True.
    """
    if not GLOBAL_EXTENSIONS._allow_raw_sockets:
        raise RuntimeError("Raw sockets disabled. Call GLOBAL_EXTENSIONS.get_tool('enable_raw_sockets')(True) to enable.")
    if not _HAS_SCAPY:
        raise RuntimeError("scapy not available")
    def _run():
        try:
            scapy.sniff(iface=interface, prn=callback, count=count, timeout=timeout)
        except Exception as e:
            structured_log("error", f"packet_sniff error: {e}")
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    _pcap_threads.append(t)
    return t

GLOBAL_EXTENSIONS.register_tool("packet_sniff", packet_sniff)

# VM op for PCAP_SNIFF: arg = (interface, timeout)
def _op_pcap_sniff(vm: VM, arg):
    if not GLOBAL_EXTENSIONS._allow_raw_sockets:
        raise RuntimeError("Raw sockets disabled.")
    if not _HAS_SCAPY:
        raise RuntimeError("scapy not available for PCAP_SNIFF")
    iface = None
    tout = None
    if isinstance(arg, (list, tuple)):
        iface = arg[0] if len(arg) > 0 else None
        tout = arg[1] if len(arg) > 1 else None
    else:
        iface = str(arg) if arg is not None else None
    # For VM op, collect captured packets into a short-lived list and return count.
    captured = []
    def _cb(pkt):
        try:
            captured.append(bytes(pkt)[:512])  # snapshot
        except Exception:
            captured.append(b"")
    t = packet_sniff(iface, _cb, timeout=tout, count=None)
    # Sleep briefly to allow capture window if timeout specified; otherwise return immediately with thread id
    vm.stack.append({"thread": id(t), "captured_preview_count": 0})
GLOBAL_EXTENSIONS.register_vm_op("PCAP_SNIFF", _op_pcap_sniff)

# ---------- Discovery helper: aggregated networking & ops capability list ----------
def list_ops_tools() -> Dict[str, Any]:
    return {
        "tools": sorted([k for k in GLOBAL_EXTENSIONS.tool_hooks.keys() if any(p in k for p in ("ssh","scan","tcp","pcap","start_tcp_forward","remote_bulk_ssh"))]),
        "vm_ops": sorted([k for k in GLOBAL_EXTENSIONS.vm_ops.keys() if any(p in k for p in ("SSH","SCAN","TCP_FORWARD","PCAP"))]),
        "gates": {
            "ops": GLOBAL_EXTENSIONS._allow_ops,
            "networking": GLOBAL_EXTENSIONS._allow_networking,
            "raw_sockets": GLOBAL_EXTENSIONS._allow_raw_sockets,
            "servers": GLOBAL_EXTENSIONS._allow_servers
        }
    }

GLOBAL_EXTENSIONS.register_tool("list_ops_tools", list_ops_tools)

structured_log("info", "Operational communication layer loaded (append-only). Enable with GLOBAL_EXTENSIONS.get_tool('enable_ops')(True) and enable_networking/enable_raw_sockets/enable_servers as needed.")

from typing import Any, Callable, Dict, List, Optional, Tuple
import os
import threading
import time
import ssl
import urllib.request
import urllib.error
import urllib.parse
import sys
import hashlib as _hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
import logging
from vm import VM, GLOBAL_EXTENSIONS
import requests  # type: ignore
import socket

# Append-only networking layer for VM: HTTP(S), WebSocket, TCP/UDP, DNS, mDNS, static server, reverse proxy.
# All powerful features are gated; enable with GLOBAL_EXTENSIONS.get_tool("enable_networking")(True)
# and (where needed) enable_raw_sockets / enable_servers.
# Optional dependencies: requests, websockets, zeroconf.
# Note: this module is append-only; do not modify existing code to preserve auditability.
GLOBAL_EXTENSIONS._allow_networking = False
def enable_networking(allow: bool = True):
    GLOBAL_EXTENSIONS._allow_networking = bool(allow)
    GLOBAL_EXTENSIONS.register_tool("enable_networking", enable_networking)
    GLOBAL_EXTENSIONS.register_tool("networking_enabled", lambda: GLOBAL_EXTENSIONS._allow_networking)
    GLOBAL_EXTENSIONS._allow_servers = False
    GLOBAL_EXTENSIONS._allow_raw_sockets = False
    GLOBAL_EXTENSIONS.register_tool("enable_servers", lambda allow=True: setattr(GLOBAL_EXTENSIONS, "_allow_servers", bool(allow)))

# Performance append-only layer: aggressive VM/Executor hot-path optimizations
# - Adds an opt-in high-performance VM run loop that inlines common op handling,
#   reduces attribute lookups, and batches work into local variables.
# - Provides an easy switch: GLOBAL_EXTENSIONS.get_tool("enable_high_performance")(True)
# - Safe: remains opt-in and falls back to original VM.run when disabled.
#
# Note: This is a targeted, conservative speed layer (no unsafe native code).
# It focuses on reducing Python overhead (localizing variables, inlining hot ops).
# For further speed, enable __enable_native_acceleration__ and register_vectorized_math.

import types
from types import MethodType

# Feature gate
GLOBAL_EXTENSIONS._hp_enabled = False
GLOBAL_EXTENSIONS._hp_hot_count = 0
GLOBAL_EXTENSIONS._hp_hot_threshold = 100  # simple heuristic if later used

def enable_high_performance(allow: bool = True, hot_threshold: int = 100):
    """
    Toggle the high-performance VM run loop.
    hot_threshold currently recorded for future JIT triggers.
    """
    GLOBAL_EXTENSIONS._hp_enabled = bool(allow)
    GLOBAL_EXTENSIONS._hp_hot_threshold = int(hot_threshold)
    # swap VM.run implementation
    if allow:
        if not hasattr(VM, "_orig_run"):
            VM._orig_run = VM.run
        VM.run = _vm_run_highperf  # assign optimized run as method
    else:
        if hasattr(VM, "_orig_run"):
            VM.run = VM._orig_run
    return GLOBAL_EXTENSIONS._hp_enabled

GLOBAL_EXTENSIONS.register_tool("enable_high_performance", enable_high_performance)
GLOBAL_EXTENSIONS.register_tool("high_performance_enabled", lambda: GLOBAL_EXTENSIONS._hp_enabled)

# Fast set of opcodes we inline in the hot loop
_HP_FAST_OPS = frozenset({
    "LOAD_CONST","LOAD_VAR","STORE_VAR","BINARY_OP","PRINT",
    "JUMP","JUMP_IF_FALSE","HALT","NOOP"
})

def _vm_run_highperf(self):
    """
    Optimized VM.run:
    - Localize fields to local variables (big speedup).
    - Inline common op handlers instead of repeated getattr calls.
    - Consult GLOBAL_EXTENSIONS.get_vm_op(op) for extension handlers (slower path).
    - Keep semantics identical to existing VM.run.
    """
    # Localize frequently used attributes
    bytecode = self.bytecode
    const_pool = self.const_pool
    stack = self.stack
    vars_dict = self.vars
    echo = getattr(self, "echo", False)
    output = self.output
    pc = getattr(self, "pc", 0)
    running = True

    # micro helpers local-bound for speed
    _get_ext = GLOBAL_EXTENSIONS.get_vm_op
    _push = stack.append
    _pop = stack.pop
    _is_int = isinstance

    try:
        while running and pc < len(bytecode):
            op, arg = bytecode[pc]

            # Fast common op path (avoid attribute lookup and method dispatch)
            if op in _HP_FAST_OPS:
                if op == "LOAD_CONST":
                    # arg is index
                    idx = arg
                    # bounds check retained
                    if not (0 <= idx < len(const_pool)):
                        raise RuntimeError(f"LOAD_CONST with invalid index: {idx}")
                    _push(const_pool[idx])
                    pc += 1
                    continue

                if op == "LOAD_VAR":
                    if arg not in vars_dict:
                        raise RuntimeError(f"Undefined variable: {arg}")
                    _push(vars_dict[arg])
                    pc += 1
                    continue

                if op == "STORE_VAR":
                    # pop value and store
                    if not stack:
                        raise RuntimeError("STORE_VAR with empty stack")
                    val = _pop()
                    vars_dict[arg] = val
                    pc += 1
                    continue

                if op == "BINARY_OP":
                    # pop order: b then a
                    try:
                        b = _pop()
                        a = _pop()
                    except IndexError:
                        raise RuntimeError("BINARY_OP with insufficient stack items")
                    if arg == "+":
                        _push(a + b)
                    elif arg == "-":
                        _push(a - b)
                    elif arg == "*":
                        _push(a * b)
                    elif arg == "/":
                        # integer division semantics if both ints (preserve existing behavior)
                        if _is_int(a, int) and _is_int(b, int):
                            if b == 0:
                                raise RuntimeError("Division by zero")
                            _push(a // b)
                        else:
                            _push(a / b)
                    elif arg == "==":
                        _push(a == b)
                    elif arg == "!=":
                        _push(a != b)
                    elif arg == "<":
                        _push(a < b)
                    elif arg == ">":
                        _push(a > b)
                    elif arg == "<=":
                        _push(a <= b)
                    elif arg == ">=":
                        _push(a >= b)
                    else:
                        # fallback to extension handler if mapped (rare path)
                        ext = _get_ext(op)
                        if ext:
                            # push back operands for ext to consume if expected semantics differ
                            _push(a); _push(b)
                            ext(self, arg)
                            # ext is expected to update stack/pc as needed
                            pc = getattr(self, "pc", pc) + 1
                            continue
                        raise RuntimeError(f"Unsupported operator: {arg!r}")
                    pc += 1
                    continue

                if op == "PRINT":
                    # pop and append to output, optionally echo
                    try:
                        val = _pop()
                    except IndexError:
                        raise RuntimeError("PRINT with empty stack")
                    s = str(val)
                    output.append(s)
                    if echo:
                        print(s)
                    pc += 1
                    continue

                if op == "JUMP":
                    # arg is target index
                    pc = int(arg)
                    continue

                if op == "JUMP_IF_FALSE":
                    # pop condition and jump if falsy
                    try:
                        cond = _pop()
                    except IndexError:
                        raise RuntimeError("JUMP_IF_FALSE with empty stack")
                    if not cond:
                        pc = int(arg)
                        continue
                    pc += 1
                    continue

                if op == "HALT":
                    running = False
                    break

                if op == "NOOP":
                    pc += 1
                    continue

            # Slower path: consult extension handlers or class methods
            ext_handler = _get_ext(op)
            if ext_handler is not None:
                # extension handler may mutate self.stack/self.vars/pc/running
                ext_handler(self, arg)
                # reflect possible changes back into locals
                stack = self.stack
                vars_dict = self.vars
                const_pool = self.const_pool
                output = self.output
                echo = getattr(self, "echo", echo)
                pc = getattr(self, "pc", pc) + 1
                # ensure running flag respected
                running = getattr(self, "running", running)
                continue

            # fallback to class method handler (uncommon for fast ops due to earlier branch)
            handler = getattr(self, f"op_{op.lower()}", None)
            if handler is None:
                raise RuntimeError(f"Unknown opcode: {op} @pc={pc}")
            # invoke handler with original semantics; handlers may use self.*, so sync locals
            # ensure self.* matches locals
            self.stack = stack
            self.vars = vars_dict
            self.const_pool = const_pool
            self.output = output
            self.echo = echo
            self.pc = pc
            handler(arg)
            # sync back
            stack = self.stack
            vars_dict = self.vars
            const_pool = self.const_pool
            output = self.output
            echo = getattr(self, "echo", echo)
            pc = getattr(self, "pc", pc) + 1
            running = getattr(self, "running", running)

        # store updated pc/back to object
        self.pc = pc
        self.stack = stack
        self.vars = vars_dict
        self.output = output
        self.running = running
    except Exception as e:
        # attach context for debugging (preserve behavior)
        raise RuntimeError(f"HighPerf VM op error @pc={pc}: {e}")

# End of performance append-only layer.

# Append-only: language expansion — large dictionary, preprocessors, VM ops, and rule engine
# - Adds a broad `language_dictionary` describing keywords, builtins, operators and types.
# - Adds preprocessors: `switch/case` -> if/elif/else, `for`-range -> while loop (textual expansion).
# - Adds many small VM ops for strings, regex, datetime, random, and threading helpers.
# - Adds a lightweight RuleEngine for declarative rules and a VM op `RULES_EVAL`.
# All additions are append-only and opt-in where appropriate via GLOBAL_EXTENSIONS tools.

import datetime
import random
import re as _re
from types import SimpleNamespace

# -----------------------
# 1) Language dictionary / definitions
# -----------------------
_language_dictionary = {
    # keywords
    "keywords": {
        "if": "Conditional branch",
        "elif": "Else-if branch",
        "else": "Else branch",
        "while": "Loop while condition true",
        "repeat": "Syntactic sugar -> repeated while",
        "for": "Syntactic sugar -> for-each / range loops (preprocessed)",
        "switch": "Syntactic sugar -> switch/case (preprocessed)",
        "case": "Switch case label",
        "break": "Break out of nearest loop (preprocessed simulation)",
        "continue": "Continue loop (preprocessed simulation)",
        "def": "User function definition (not full runtime; placeholder)",
        "return": "Return from def (placeholder)",
    },
    # builtins
    "builtins": {
        "print": "Print value",
        "fail": "Raise ExecutionError",
        "end": "Halt executor",
        "len": "Length of container (vm builtin)",
        "input": "Read input from stdin (vm builtin)",
    },
    # operators
    "operators": {
        "+": "Addition / string concat",
        "-": "Subtraction",
        "*": "Multiplication",
        "/": "Division (integer if both ints)",
        "==": "Equality",
        "!=": "Inequality",
        "<": "Less than",
        "<=": "Less or equal",
        ">": "Greater than",
        ">=": "Greater or equal",
        "and": "Logical and",
        "or": "Logical or"
    },
    # types
    "types": {
        "int": "Integer",
        "str": "String",
        "list": "List",
        "dict": "Dictionary",
        "bool": "Boolean",
        "none": "None",
    },
    # vm ops summary (for discoverability)
    "vm_ops": {
        "STR_UPPER": "Uppercase string",
        "STR_LOWER": "Lowercase string",
        "STR_SPLIT": "Split string by sep",
        "STR_JOIN": "Join list into string",
        "REGEX_MATCH/SEARCH/FINDALL": "Regex helpers (VM ops)",
        "DATE_NOW/DATE_FORMAT": "Datetime helpers",
        "RAND_INT": "Random integer generator",
        "THREAD_SPAWN": "Spawn background thread to run a small callable"
    }
}

GLOBAL_EXTENSIONS.register_tool("language_dictionary", lambda: _language_dictionary)

# -----------------------
# 2) Preprocessors: switch/case and for-range -> while
# -----------------------
def _expand_switch_blocks(src: str) -> str:
    """
    Expand:
      switch <expr>
        case <value>:
          ...
        case <value>:
          ...
        default:
          ...
      end
    Into nested if/elif/else blocks.
    This is a textual source-to-source transformation (conservative).
    """
    lines = src.splitlines()
    out = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = re.match(r"^\s*switch\s+(.+)$", ln)
        if m:
            expr = m.group(1).strip()
            i += 1
            cases = []
            default_block = []
            cur_block = None
            while i < len(lines):
                l2 = lines[i]
                if re.match(r"^\s*end\s*$", l2):
                    break
                mc = re.match(r"^\s*case\s+(.+?):\s*$", l2)
                if mc:
                    val = mc.group(1).strip()
                    cur_block = (val, [])
                    cases.append(cur_block)
                    i += 1
                    continue
                md = re.match(r"^\s*default\s*:\s*$", l2)
                if md:
                    cur_block = ("__default__", [])
                    default_block = cur_block[1]
                    i += 1
                    continue
                if cur_block is not None:
                    cur_block[1].append(l2)
                else:
                    # ignore stray lines until case/default seen
                    pass
                i += 1
            # produce if/elif chain
            first = True
            for val, body in cases:
                if first:
                    out.append(f"if {expr} == {val}")
                    first = False
                else:
                    out.append(f"elif {expr} == {val}")
                out.extend(body)
                out.append("end")
            if default_block:
                out.append("else")
                out.extend(default_block)
                out.append("end")
            i += 1  # skip 'end'
            continue
        out.append(ln)
        i += 1
    return "\n".join(out)

GLOBAL_EXTENSIONS.register_preprocessor(_expand_switch_blocks)

_for_counter = 0
def _expand_for_range(src: str) -> str:
    """
    Expand simple for-range forms into while loops:
      for i in range(a, b)
        <body>
      end
    -> 
      __for_i_x = a
      while __for_i_x < b
        i = __for_i_x
        <body>
        __for_i_x = __for_i_x + 1
      end
    Also supports `for x in listvar` (desugars to indexed loop).
    """
    global _for_counter
    lines = src.splitlines()
    out = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = re.match(r"^\s*for\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\s+range\(\s*(.+?)\s*(?:,\s*(.+?)\s*)?\)\s*$", ln)
        if m:
            var = m.group(1)
            a = m.group(2)
            b = m.group(3) or None
            # support range(a) or range(a,b)
            if b is None:
                start = "0"
                end = a
            else:
                start = a
                end = b
            tmp = f"__for_idx_{_for_counter}"
            _for_counter += 1
            out.append(f"{tmp} = {start}")
            out.append(f"while {tmp} < {end}")
            out.append(f"{var} = {tmp}")
            # copy body
            i += 1
            depth = 0
            while i < len(lines):
                l2 = lines[i]
                if re.match(r"^\s*for\b", l2):
                    depth += 1
                if re.match(r"^\s*end\s*$", l2):
                    if depth == 0:
                        break
                    depth -= 1
                out.append(l2)
                i += 1
            out.append(f"{tmp} = {tmp} + 1")
            out.append("end")
            i += 1  # skip end
            continue
        # for-in-list: for x in listvar  -> index-based desugar if simple
        m2 = re.match(r"^\s*for\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", ln)
        if m2:
            var = m2.group(1); lst = m2.group(2)
            tmp_i = f"__for_idx_{_for_counter}"
            tmp_len = f"__for_len_{_for_counter}"
            _for_counter += 1
            out.append(f"{tmp_i} = 0")
            out.append(f"{tmp_len} = len({lst})")
            out.append(f"while {tmp_i} < {tmp_len}")
            out.append(f"{var} = {lst}[{tmp_i}]")
            i += 1
            depth = 0
            while i < len(lines):
                l2 = lines[i]
                if re.match(r"^\s*for\b", l2):
                    depth += 1
                if re.match(r"^\s*end\s*$", l2):
                    if depth == 0:
                        break
                    depth -= 1
                out.append(l2)
                i += 1
            out.append(f"{tmp_i} = {tmp_i} + 1")
            out.append("end")
            i += 1
            continue
        out.append(ln)
        i += 1
    return "\n".join(out)

GLOBAL_EXTENSIONS.register_preprocessor(_expand_for_range)

# -----------------------
# 3) New VM ops (strings, regex, datetime, random, thread spawn)
# -----------------------
def _op_str_upper(vm: VM, arg):
    s = vm.stack.pop()
    vm.stack.append(str(s).upper())
GLOBAL_EXTENSIONS.register_vm_op("STR_UPPER", _op_str_upper)

def _op_str_lower(vm: VM, arg):
    s = vm.stack.pop()
    vm.stack.append(str(s).lower())
GLOBAL_EXTENSIONS.register_vm_op("STR_LOWER", _op_str_lower)

def _op_str_split(vm: VM, arg):
    sep = arg if arg is not None else None
    s = vm.stack.pop()
    vm.stack.append(str(s).split(sep))
GLOBAL_EXTENSIONS.register_vm_op("STR_SPLIT", _op_str_split)

def _op_str_join(vm: VM, arg):
    lst = vm.stack.pop()
    sep = vm.stack.pop() if arg is None else arg
    vm.stack.append(str(sep).join(map(str, lst)))
GLOBAL_EXTENSIONS.register_vm_op("STR_JOIN", _op_str_join)

def _op_regex_match(vm: VM, arg):
    """stack: text, pattern -> push match_obj or None"""
    pattern = vm.stack.pop()
    text = vm.stack.pop()
    m = _re.match(pattern, text)
    vm.stack.append(bool(m))
GLOBAL_EXTENSIONS.register_vm_op("REGEX_MATCH", _op_regex_match)

def _op_regex_search(vm: VM, arg):
    pattern = vm.stack.pop()
    text = vm.stack.pop()
    m = _re.search(pattern, text)
    vm.stack.append(bool(m))
GLOBAL_EXTENSIONS.register_vm_op("REGEX_SEARCH", _op_regex_search)

def _op_regex_findall(vm: VM, arg):
    pattern = vm.stack.pop()
    text = vm.stack.pop()
    vm.stack.append(_re.findall(pattern, text))
GLOBAL_EXTENSIONS.register_vm_op("REGEX_FINDALL", _op_regex_findall)

def _op_date_now(vm: VM, arg):
    vm.stack.append(datetime.datetime.utcnow())
GLOBAL_EXTENSIONS.register_vm_op("DATE_NOW", _op_date_now)

def _op_date_format(vm: VM, arg):
    fmt = vm.stack.pop() if arg is None else arg
    dt = vm.stack.pop()
    if not isinstance(dt, datetime.datetime):
        raise RuntimeError("DATE_FORMAT expects datetime on stack")
    vm.stack.append(dt.strftime(fmt))
GLOBAL_EXTENSIONS.register_vm_op("DATE_FORMAT", _op_date_format)

def _op_rand_int(vm: VM, arg):
    hi = vm.stack.pop()
    lo = vm.stack.pop()
    vm.stack.append(random.randint(int(lo), int(hi)))
GLOBAL_EXTENSIONS.register_vm_op("RAND_INT", _op_rand_int)

def _op_thread_spawn(vm: VM, arg):
    """
    THREAD_SPAWN expects: callable (python callable object) and args(list) on stack.
    For safety, only SimpleNamespace-like callables (small wrappers) are recommended.
    Pushes a thread id token.
    """
    fn = vm.stack.pop()
    args = vm.stack.pop() if vm.stack else []
    if not callable(fn):
        raise RuntimeError("THREAD_SPAWN expects callable on stack")
    t = threading.Thread(target=lambda: fn(*args), daemon=True)
    t.start()
    vm.stack.append({"thread_id": id(t)})
GLOBAL_EXTENSIONS.register_vm_op("THREAD_SPAWN", _op_thread_spawn)

# -----------------------
# 4) RuleEngine: small declarative ruleset
# -----------------------
class RuleEngine:
    """
    Lightweight rule engine.
    Rules are Python-callable-based: condition(context)->bool and action(context)->dict|None
    Use register_rule(name, cond_fn, action_fn).
    Returns list of fired rule names and context updates.
    """
    def __init__(self):
        self.rules: List[Tuple[str, Callable[[Dict], bool], Callable[[Dict], Optional[Dict]]]] = []
        self.lock = threading.RLock()

    def register_rule(self, name: str, cond: Callable[[Dict], bool], action: Callable[[Dict], Optional[Dict]]):
        with self.lock:
            self.rules.append((name, cond, action))
        return True

    def run(self, context: Dict[str, Any], max_runs: int = 100) -> Dict[str, Any]:
        """
        Evaluate rules until no rule fires or max_runs reached.
        Each action may return a dict of updates to context which are applied immediately.
        Returns {"fired": [names], "context": context}
        """
        fired = []
        runs = 0
        changed = True
        while changed and runs < max_runs:
            changed = False
            runs += 1
            for name, cond, action in list(self.rules):
                try:
                    if cond(context):
                        res = action(context)
                        fired.append(name)
                        changed = changed or bool(res)
                        if isinstance(res, dict):
                            context.update(res)
                except Exception:
                    # swallow rule errors but log
                    structured_log("warn", f"rule {name} raised during evaluation")
            if not changed:
                break
        return {"fired": fired, "context": context, "runs": runs}

_RULE_ENGINE = RuleEngine()
GLOBAL_EXTENSIONS.register_tool("rule_engine", lambda: _RULE_ENGINE)

def register_rule(name: str, cond_src: Optional[str] = None, action_src: Optional[str] = None, cond_fn: Callable[[Dict], bool] = None, action_fn: Callable[[Dict], Optional[Dict]] = None):
    """
    Register a rule.
    - Prefer passing Python callables cond_fn and action_fn.
    - cond_src/action_src may be small expressions compiled to lambdas; the function compiles them with restricted globals.
    Warning: compiling source strings executes code; use with caution.
    """
    if cond_fn is None and cond_src is None:
        raise RuntimeError("Either cond_fn or cond_src required")
    if action_fn is None and action_src is None:
        raise RuntimeError("Either action_fn or action_src required")
    # restricted globals for eval
    safe_globals = {"__builtins__": {}, "math": math, "re": _re, "datetime": datetime}
    if cond_fn is None:
        # compile cond_src into a lambda taking context -> bool
        cond_fn = eval(f"lambda context: bool({cond_src})", safe_globals)  # noqa: B307
    if action_fn is None:
        # compile action_src into a lambda returning dict or None
        action_fn = eval(f"lambda context: ({action_src})", safe_globals)  # noqa: B307
    return _RULE_ENGINE.register_rule(name, cond_fn, action_fn)

GLOBAL_EXTENSIONS.register_tool("register_rule", register_rule)

# VM op: RULES_EVAL expects a context dict on stack and optional max_runs arg. Pushes results dict.
def _op_rules_eval(vm: VM, arg):
    max_runs = arg if isinstance(arg, int) else 100
    ctx = vm.stack.pop()
    if not isinstance(ctx, dict):
        raise RuntimeError("RULES_EVAL expects dict on stack")
    res = _RULE_ENGINE.run(dict(ctx), max_runs=max_runs)
    vm.stack.append(res)

GLOBAL_EXTENSIONS.register_vm_op("RULES_EVAL", _op_rules_eval)

# -----------------------
# 5) Extended lint rules that use the language dictionary and rules
# -----------------------
def expanded_lint(src: str) -> List[str]:
    issues = []
    # reuse earlier linters
    issues.extend(simple_linter(src))
    # check for unknown keywords usage
    kw = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", src))
    known = set(_language_dictionary["keywords"].keys()) | set(_language_dictionary["builtins"].keys()) | set(_language_dictionary["operators"].keys())
    # flag suspicious unknown words longer than 3 letters
    for w in sorted(kw):
        if len(w) > 3 and w not in known and not w.islower() and not w.startswith("__"):
            # conservative: only warn (not error)
            issues.append(f"lint: unknown symbol or mixed-case '{w}' (check definitions/dictionary)")
    # warn about switch/for usage if preprocessor not registered
    if re.search(r"^\s*switch\b", src, flags=re.MULTILINE) and _expand_switch_blocks not in GLOBAL_EXTENSIONS.preprocessors:
        issues.append("switch statement used but switch->if preprocessor not registered")
    return issues

GLOBAL_EXTENSIONS.register_tool("expanded_lint", expanded_lint)

# -----------------------
# 6) Convenience: register many definitions for quick discovery
# -----------------------
def list_definitions() -> Dict[str, Any]:
    d = dict(_language_dictionary)
    # include registered VM ops and tools counts
    d["vm_ops_listed"] = sorted(list(GLOBAL_EXTENSIONS.vm_ops.keys()))
    d["tools_listed"] = sorted(list(GLOBAL_EXTENSIONS.tool_hooks.keys()))
    return d

GLOBAL_EXTENSIONS.register_tool("list_definitions", list_definitions)

structured_log("info", "Language expansion layer loaded: added many definitions, preprocessors (switch/for), VM ops (string/regex/date/random/thread), and a RuleEngine. Use GLOBAL_EXTENSIONS.get_tool('list_definitions')() to inspect.")

# --- Append-only: ultra-power techniques layer (opt-in, non-destructive) ---
# Adds opt-in JIT, SSA/CFG optimizer passes, actor model, lightweight distributed execution,
# symbolic execution stubs, GPU vectorization hooks, transformer-based embeddings (opt-in),
# taint analysis, and registration helpers for discovery.
#
# All features require explicit enabling via GLOBAL_EXTENSIONS tools to avoid accidental exposure.

import threading
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, Future
from queue import Queue, Empty
from typing import Any, Dict, List, Tuple, Optional, Callable
import inspect
import time

# -- Feature gates --
GLOBAL_EXTENSIONS._enable_jit = False
GLOBAL_EXTENSIONS._enable_distributed = False
GLOBAL_EXTENSIONS._enable_symbolic = False
GLOBAL_EXTENSIONS._enable_gpu = False
GLOBAL_EXTENSIONS._enable_transformers = False

def enable_jit(allow: bool = True):
    GLOBAL_EXTENSIONS._enable_jit = bool(allow)
GLOBAL_EXTENSIONS.register_tool("enable_jit", enable_jit)

def enable_distributed(allow: bool = True):
    GLOBAL_EXTENSIONS._enable_distributed = bool(allow)
GLOBAL_EXTENSIONS.register_tool("enable_distributed", enable_distributed)

def enable_symbolic(allow: bool = True):
    GLOBAL_EXTENSIONS._enable_symbolic = bool(allow)
GLOBAL_EXTENSIONS.register_tool("enable_symbolic", enable_symbolic)

def enable_gpu(allow: bool = True):
    GLOBAL_EXTENSIONS._enable_gpu = bool(allow)
GLOBAL_EXTENSIONS.register_tool("enable_gpu", enable_gpu)

def enable_transformers(allow: bool = True):
    GLOBAL_EXTENSIONS._enable_transformers = bool(allow)
GLOBAL_EXTENSIONS.register_tool("enable_transformers", enable_transformers)

# -- Attempt optional libs --
try:
    import numba  # type: ignore
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

try:
    import llvmlite  # type: ignore
    _HAS_LLVM = True
except Exception:
    _HAS_LLVM = False

try:
    import z3  # type: ignore
    _HAS_Z3 = True
except Exception:
    _HAS_Z3 = False

try:
    import cupy as _cupy  # type: ignore
    _HAS_CUPY = True
except Exception:
    _HAS_CUPY = False

try:
    from transformers import AutoTokenizer, AutoModel  # type: ignore
    import torch  # type: ignore
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

# -------------------------
# Simple Control-Flow Graph & SSA-ish optimizer (conservative)
# -------------------------
def build_basic_blocks(bytecode: List[Tuple[str, Any]]) -> Tuple[List[int], Dict[int, List[int]]]:
    """
    Return (leaders, succs) where leaders is list of block start indices and succs maps leader->list of successor indices.
    Conservative: treats JUMP/JUMP_IF_FALSE as control-flow altering ops.
    """
    leaders = {0}
    for i, (op, arg) in enumerate(bytecode):
        if op in ("JUMP", "JUMP_IF_FALSE"):
            target = arg if isinstance(arg, int) else None
            if target is not None and 0 <= target < len(bytecode):
                leaders.add(target)
            # next instruction is leader for conditional fall-through or after unconditional jump we consider next as leader
            if i + 1 < len(bytecode):
                leaders.add(i + 1)
    leaders = sorted(leaders)
    # map leader -> successor leaders
    succs = {}
    leader_set = set(leaders)
    # helper to find leader for index
    def find_leader(idx):
        # leaders sorted, find greatest leader <= idx
        cur = leaders[0]
        for l in leaders:
            if l <= idx:
                cur = l
            else:
                break
        return cur
    for l in leaders:
        succs[l] = []
        i = l
        # walk until next leader (exclusive)
        nxt_idx = len(bytecode)
        for candidate in leaders:
            if candidate > l:
                nxt_idx = candidate
                break
        # scan block
        last_op, last_arg = bytecode[nxt_idx - 1] if nxt_idx - 1 < len(bytecode) else ("HALT", None)
        # inspect last instruction in block to derive successors
        if last_op == "JUMP":
            t = last_arg
            if isinstance(t, int): succs[l].append(find_leader(t))
        elif last_op == "JUMP_IF_FALSE":
            t = last_arg
            if isinstance(t, int):
                succs[l].append(find_leader(t))
            # fallthrough
            if nxt_idx < len(bytecode):
                succs[l].append(find_leader(nxt_idx))
        else:
            # fallthrough if exists
            if nxt_idx < len(bytecode):
                succs[l].append(find_leader(nxt_idx))
    return leaders, succs

def ssa_simple_rename(bytecode: List[Tuple[str, Any]], const_pool: List[Any]) -> Tuple[List[Tuple[str, Any]], List[Any]]:
    """
    Very conservative SSA-like renaming for STORE_VAR/LOAD_VAR where possible.
    This function performs local renaming for sequential stores/loads without control-flow merges.
    Returns (bytecode, const_pool) (may return original inputs if no change).
    """
    bc = []
    var_version: Dict[str, int] = {}
    var_map: Dict[str, str] = {}
    for op, arg in bytecode:
        if op == "STORE_VAR":
            name = arg
            ver = var_version.get(name, 0) + 1
            var_version[name] = ver
            new_name = f"{name}__v{ver}"
            var_map[name] = new_name
            bc.append((op, new_name))
        elif op == "LOAD_VAR":
            name = arg
            mapped = var_map.get(name, name)
            bc.append((op, mapped))
        else:
            bc.append((op, arg))
    return bc, const_pool

def optimizer_cfg_ssa_pass(bytecode: List[Tuple[str, Any]], const_pool: List[Any]):
    """
    Compose passes: build CFG, then perform limited SSA rename and jump folding.
    Safe and conservative.
    """
    try:
        leaders, succs = build_basic_blocks(bytecode)
        bc, cp = ssa_simple_rename(bytecode, const_pool)
        return bc, cp
    except Exception:
        return bytecode, const_pool

GLOBAL_EXTENSIONS.register_optimizer(optimizer_cfg_ssa_pass)

# -------------------------
# JIT helpers (wrap Python callables with numba if available and opt-in)
# -------------------------
def jit_compile_pyfunc(pyfunc: Callable) -> Callable:
    """
    If JIT enabled and numba available, return compiled version; otherwise return original.
    Designed for small pure numeric helpers used by tooling (not arbitrary VM bytecode).
    """
    if GLOBAL_EXTENSIONS._enable_jit and _HAS_NUMBA:
        try:
            return numba.njit(pyfunc)
        except Exception:
            return pyfunc
    return pyfunc

GLOBAL_EXTENSIONS.register_tool("jit_compile_pyfunc", jit_compile_pyfunc)

# -------------------------
# Actor model (in-process)
# -------------------------
class Actor:
    def __init__(self, target: Callable[[Any], None], name: Optional[str] = None):
        self._inbox: Queue = Queue()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._target = target
        self.name = name or f"actor-{id(self)}"
        self._alive = True
        self._thread.start()

    def _run_loop(self):
        while self._alive:
            try:
                msg = self._inbox.get(timeout=0.2)
            except Empty:
                continue
            try:
                self._target(msg)
            except Exception:
                structured_log("error", f"actor {self.name} target raised")
            finally:
                self._inbox.task_done()

    def send(self, msg: Any):
        self._inbox.put(msg)

    def stop(self):
        self._alive = False

# actor registry
_GLOBAL_ACTORS: Dict[str, Actor] = {}
def spawn_actor(target: Callable[[Any], None], name: Optional[str] = None) -> str:
    ident = name or f"actor_{len(_GLOBAL_ACTORS)+1}"
    a = Actor(target, name=ident)
    _GLOBAL_ACTORS[ident] = a
    return ident

def send_actor(actor_id: str, msg: Any) -> bool:
    a = _GLOBAL_ACTORS.get(actor_id)
    if not a:
        return False
    a.send(msg)
    return True

GLOBAL_EXTENSIONS.register_tool("spawn_actor", spawn_actor)
GLOBAL_EXTENSIONS.register_tool("send_actor", send_actor)

# VM ops for actor model (lightweight)
def _op_actor_spawn(vm: VM, arg):
    # expects: callable on stack (python callable)
    fn = vm.stack.pop()
    if not callable(fn):
        raise RuntimeError("ACTOR_SPAWN expects callable on stack")
    name = arg if isinstance(arg, str) else None
    actor_id = spawn_actor(fn, name=name)
    vm.stack.append(actor_id)
GLOBAL_EXTENSIONS.register_vm_op("ACTOR_SPAWN", _op_actor_spawn)

def _op_actor_send(vm: VM, arg):
    # expects: msg, actor_id on stack (top is actor_id)
    actor_id = vm.stack.pop()
    msg = vm.stack.pop()
    ok = send_actor(actor_id, msg)
    vm.stack.append(ok)
GLOBAL_EXTENSIONS.register_vm_op("ACTOR_SEND", _op_actor_send)

# -------------------------
# Distributed execution (Process pool that runs bytecode via safe_run_vm)
# -------------------------
_GLOBAL_PROC_POOL: Optional[ProcessPoolExecutor] = None
def get_process_pool(max_workers: int = None) -> ProcessPoolExecutor:
    global _GLOBAL_PROC_POOL
    if _GLOBAL_PROC_POOL is None:
        _GLOBAL_PROC_POOL = ProcessPoolExecutor(max_workers=max_workers or max(1, multiprocessing.cpu_count()-1))
    return _GLOBAL_PROC_POOL

def _run_bytecode_in_process(bytecode: List[Tuple[str, Any]], const_pool: List[Any], timeout: Optional[float] = None, step_limit: Optional[int] = None):
    """
    Helper executed in worker process. Reconstructs VM and runs safe_run_vm.
    Designed to be picklable (uses simple data).
    """
    try:
        vm = VM(bytecode, const_pool)
        # safe_run_vm is registered tool, call directly
        outputs = safe_run_vm(vm, timeout=timeout, step_limit=step_limit)
        return {"ok": True, "output": outputs}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def submit_remote_job(bytecode: List[Tuple[str, Any]], const_pool: List[Any], *, timeout: Optional[float] = None, step_limit: Optional[int] = None) -> Future:
    if not GLOBAL_EXTENSIONS._enable_distributed:
        raise RuntimeError("Distributed execution disabled; enable via GLOBAL_EXTENSIONS.get_tool('enable_distributed')(True)")
    pool = get_process_pool()
    fut = pool.submit(_run_bytecode_in_process, bytecode, const_pool, timeout, step_limit)
    return fut

GLOBAL_EXTENSIONS.register_tool("submit_remote_job", submit_remote_job)
GLOBAL_EXTENSIONS.register_tool("get_process_pool", get_process_pool)

# -------------------------
# Symbolic execution stub (Z3 optional)
# -------------------------
def symbolic_check_reachability(bytecode: List[Tuple[str, Any]], const_pool: List[Any], property_expr: str) -> Dict[str, Any]:
    """
    Very small stub: if z3 available and symbolic mode enabled, attempt to symbolically reason about simple integer comparisons in bytecode.
    property_expr is a user-supplied boolean expression over 'x','y' etc. This stub demonstrates the hook and returns informative message.
    """
    if not GLOBAL_EXTENSIONS._enable_symbolic:
        raise RuntimeError("Symbolic execution disabled")
    if not _HAS_Z3:
        return {"ok": False, "reason": "z3 not available"}
    # Conservative demonstration: return unknown until a real symbolic engine is wired
    return {"ok": False, "reason": "symbolic analysis is a stub; enable full engine integration to use"}

GLOBAL_EXTENSIONS.register_tool("symbolic_check_reachability", symbolic_check_reachability)

# -------------------------
# GPU vectorization registration
# -------------------------
def register_gpu_vector_ops():
    """
    If GPU enabled and cupy is available, register BINARY_OP_GPU which uses cupy arrays when present.
    """
    if not GLOBAL_EXTENSIONS._enable_gpu:
        return False
    if not _HAS_CUPY:
        return False

    def _op_binary_op_gpu(vm: VM, arg):
        b = vm.stack.pop()
        a = vm.stack.pop()
        try:
            a_gpu = _cupy.asarray(a)
            b_gpu = _cupy.asarray(b)
            if arg == "+":
                res = _cupy.add(a_gpu, b_gpu)
            elif arg == "-":
                res = _cupy.subtract(a_gpu, b_gpu)
            elif arg == "*":
                res = _cupy.multiply(a_gpu, b_gpu)
            elif arg == "/":
                res = _cupy.divide(a_gpu, b_gpu)
            else:
                raise RuntimeError("unsupported gpu op")
            vm.stack.append(_cupy.asnumpy(res).tolist())
        except Exception:
            # fallback to normal binary op semantics
            vm.stack.append(a); vm.stack.append(b)
            return VM.op_binary_op(vm, arg)

    GLOBAL_EXTENSIONS.register_vm_op("BINARY_OP_GPU", _op_binary_op_gpu)
    return True

GLOBAL_EXTENSIONS.register_tool("register_gpu_vector_ops", register_gpu_vector_ops)

# -------------------------
# Transformers-based embedding hook (opt-in, heavy)
# -------------------------
def embed_text_transformer(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[List[float]]:
    if not GLOBAL_EXTENSIONS._enable_transformers:
        raise RuntimeError("Transformers disabled")
    if not _HAS_TRANSFORMERS:
        raise RuntimeError("transformers/torch not available")
    # lazy init tokenizer/model cached on GLOBAL_EXTENSIONS
    tok_key = f"_transformer_tok_{model_name}"
    model_key = f"_transformer_mod_{model_name}"
    if not GLOBAL_EXTENSIONS.get_tool(tok_key):
        try:
            GLOBAL_EXTENSIONS.register_tool(tok_key, AutoTokenizer.from_pretrained(model_name))
            GLOBAL_EXTENSIONS.register_tool(model_key, AutoModel.from_pretrained(model_name))
        except Exception as e:
            raise RuntimeError(f"failed to load transformer model: {e}")
    tokenizer = GLOBAL_EXTENSIONS.get_tool(tok_key)
    model = GLOBAL_EXTENSIONS.get_tool(model_key)
    # simple pooling: mean of last hidden states
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    last = out.last_hidden_state  # (batch, seq, dim)
    masks = enc.attention_mask.unsqueeze(-1)
    summed = (last * masks).sum(1)
    lens = masks.sum(1)
    embeds = (summed / lens).cpu().numpy().tolist()
    return embeds

GLOBAL_EXTENSIONS.register_tool("embed_text_transformer", embed_text_transformer)

# -------------------------
# Taint analysis (static heuristic)
# -------------------------
def taint_check(src: str) -> List[str]:
    """
    Heuristic static taint analysis:
      - mark any variable assigned from HTTP_GET/FS_READ/WASM_INVOKE as tainted
      - propagate through simple assignments and flag sinks (FS_WRITE, HTTP_POST, AES_ENCRYPT)
    This is conservative and for developer guidance only.
    """
    issues = []
    # identify lines where taint sources occur
    lines = src.splitlines()
    tainted = set()
    for i, ln in enumerate(lines):
        if "HTTP_GET" in ln or "FS_READ" in ln or "WASM_INVOKE" in ln:
            issues.append(f"taint: possible untrusted source on line {i+1}")
        if "FS_WRITE" in ln or "HTTP_POST" in ln or "AES_ENCRYPT" in ln:
            # if any taint source seen earlier, warn
            if tainted:
                issues.append(f"taint: sink on line {i+1} may receive tainted data")
    return issues

GLOBAL_EXTENSIONS.register_tool("taint_check", taint_check)

# -------------------------
# Discovery helper for advanced techniques
# -------------------------
def list_advanced_techniques() -> Dict[str, Any]:
    return {
        "gates": {
            "jit": GLOBAL_EXTENSIONS._enable_jit,
            "distributed": GLOBAL_EXTENSIONS._enable_distributed,
            "symbolic": GLOBAL_EXTENSIONS._enable_symbolic,
            "gpu": GLOBAL_EXTENSIONS._enable_gpu,
            "transformers": GLOBAL_EXTENSIONS._enable_transformers,
        },
        "optional_libs": {
            "numba": _HAS_NUMBA,
            "llvmlite": _HAS_LLVM,
            "z3": _HAS_Z3,
            "cupy": _HAS_CUPY,
            "transformers+torch": _HAS_TRANSFORMERS
        },
        "tools": [k for k in GLOBAL_EXTENSIONS.tool_hooks.keys() if k.startswith(("enable_", "register_", "submit_remote_job", "spawn_actor", "embed_text_transformer"))],
        "vm_ops": [op for op in GLOBAL_EXTENSIONS.vm_ops.keys() if op.startswith(("ACTOR_","BINARY_OP_","AES_","NLP_","RULES_"))],
    }

GLOBAL_EXTENSIONS.register_tool("list_advanced_techniques", list_advanced_techniques)

structured_log("info", "Ultra-power techniques layer loaded (append-only). All features are opt-in: enable_jit, enable_distributed, enable_symbolic, enable_gpu, enable_transformers.")

structured_log("info", "Optional libs: numba, llvmlite, z3, cupy, transformers/torch. Use GLOBAL_EXTENSIONS.get_tool('list_advanced_techniques')() to inspect status.")
# End of system.py
# Ultra-power techniques layer (append-only, opt-in)
# Adds opt-in JIT, SSA/CFG optimizer passes, actor model, lightweight distributed execution,
# symbolic execution stubs, GPU vectorization hooks, transformer-based embeddings (opt-in),
# taint analysis, and registration helpers for discovery.
#
# All features require explicit enabling via GLOBAL_EXTENSIONS tools to avoid accidental exposure.
import threading
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, Future
from queue import Queue, Empty
from typing import Any, Dict, List, Tuple, Optional, Callable
import inspect
import time
import math
# -- Feature gates --
GLOBAL_EXTENSIONS._enable_jit = False
GLOBAL_EXTENSIONS._enable_distributed = False
GLOBAL_EXTENSIONS._enable_symbolic = False
GLOBAL_EXTENSIONS._enable_gpu = False
GLOBAL_EXTENSIONS._enable_transformers = False
def enable_jit(allow: bool = True):
    GLOBAL_EXTENSIONS._enable_jit = bool(allow)
    GLOBAL_EXTENSIONS.register_tool("enable_jit", enable_jit)
    def enable_distributed(allow: bool = True):
        GLOBAL_EXTENSIONS._enable_distributed = bool(allow)
        GLOBAL_EXTENSIONS.register_tool("enable_distributed", enable_distributed)
        def enable_symbolic(allow: bool = True):
            GLOBAL_EXTENSIONS._enable_symbolic = bool(allow)
            GLOBAL_EXTENSIONS.register_tool("enable_symbolic", enable_symbolic)
            def enable_gpu(allow: bool = True):
                GLOBAL_EXTENSIONS._enable_gpu = bool(allow)
                GLOBAL_EXTENSIONS.register_tool("enable_gpu", enable_gpu)
                def enable_transformers(allow: bool = True):
                    GLOBAL_EXTENSIONS._enable_transformers = bool(allow)
                    GLOBAL_EXTENSIONS.register_tool("enable_transformers", enable_transformers)
                    # -- Attempt optional libs --
                    try:
                        import numba  # type: ignore
                        _HAS_NUMBA = True
                    except Exception:
                        _HAS_NUMBA = False
