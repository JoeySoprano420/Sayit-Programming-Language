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

