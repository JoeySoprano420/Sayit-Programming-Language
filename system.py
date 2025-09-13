# say_lexer.py
import re
from typing import Iterator, Tuple

# === Token definitions ===
TOKENS = [
    ("START",   r"Start"),
    ("RITUAL",  r"Ritual:"),
    ("MAKE",    r"Make:"),
    ("WHILE",   r"While"),
    ("IF",      r"If"),
    ("ELIF",    r"Elif"),
    ("ELSE",    r"Else:"),
    ("FINALLY", r"Finally:"),
    ("PRINT",   r"print"),
    ("STRING",  r'"[^"]*"'),
    ("NUMBER",  r"[0-9]+"),
    ("IDENT",   r"[a-zA-Z_][a-zA-Z0-9_]*"),
    ("OP",      r"[+\-*/<>=!]+"),
    ("LPAREN",  r"\("),
    ("RPAREN",  r"\)"),
    ("LBRACK",  r"\["),
    ("RBRACK",  r"\]"),
    ("DOTS",    r"\.\.\."),
    ("COLON",   r":"),
    ("NEWLINE", r"\n"),
    ("SKIP",    r"[ \t]+"),
]

# === Lexer ===
def _build_token_list():
    """
    Return a token list based on TOKENS with a COMMENT token inserted
    (keeps original TOKENS definition unchanged for compatibility).
    """
    t = list(TOKENS)
    names = [name for name, _ in t]
    if "COMMENT" not in names:
        try:
            idx = names.index("NEWLINE")
        except ValueError:
            idx = len(t)
        t.insert(idx, ("COMMENT", r"#.*"))
    return t

def tokenize(code: str, include_pos: bool = False) -> Iterator[Tuple]:
    """
    Convert source code into a sequence of tokens.

    Backward-compatible default: yields (TOKEN_TYPE, LEXEME).
    If include_pos=True yields (TOKEN_TYPE, LEXEME, LINE, COLUMN).

    This implementation:
    - Adds simple `# comment` support (skipped).
    - Performs explicit scanning to detect illegal characters and report location.
    """
    token_list = _build_token_list()
    regex = "|".join("(?P<%s>%s)" % pair for pair in token_list)
    token_re = re.compile(regex)
    pos = 0
    length = len(code)

    while pos < length:
        m = token_re.match(code, pos)
        if not m:
            # compute line/col for error message
            line = code.count("\n", 0, pos) + 1
            last_n = code.rfind("\n", 0, pos)
            col = pos - last_n
            raise SyntaxError(f"Illegal character {code[pos]!r} at line {line} column {col}")
        typ = m.lastgroup
        val = m.group()
        pos = m.end()

        if typ in ("SKIP", "COMMENT"):
            continue

        if include_pos:
            # compute line and column from m.start()
            start = m.start()
            line = code.count("\n", 0, start) + 1
            last_n = code.rfind("\n", 0, start)
            col = start - last_n
            yield (typ, val, line, col)
        else:
            yield (typ, val)


if __name__ == "__main__":
    # Lightweight CLI: print tokens for a given file. Use --pos to include line/column.
    import argparse
    import sys
    parser = argparse.ArgumentParser(prog="say_lexer", description="Tokenize a .say source file")
    parser.add_argument("file", nargs="?", help="Source file to tokenize")
    parser.add_argument("--pos", action="store_true", help="Include token line and column")
    args = parser.parse_args()

    if not args.file:
        print("Usage: say_lexer.py <file.say> [--pos]")
        sys.exit(1)

    try:
        with open(args.file, "r", encoding="utf-8") as f:
            src = f.read()
    except Exception as e:
        print(f"Error reading {args.file}: {e}")
        sys.exit(1)

    try:
        for tok in tokenize(src, include_pos=args.pos):
            print(tok)
    except SyntaxError as e:
        print(f"Lexing error: {e}")
        sys.exit(1)

        # say_parser.py

# === AST Node Definitions ===
class Node: pass

class Program(Node):
    def __init__(self, stmts):
        self.stmts = stmts

class Print(Node):
    def __init__(self, expr):
        self.expr = expr

class Assign(Node):
    def __init__(self, ident, expr):
        self.ident, self.expr = ident, expr

class BinOp(Node):
    def __init__(self, left, op, right):
        self.left, self.op, self.right = left, op, right

class If(Node):
    def __init__(self, cond, then, elifs, els):
        self.cond, self.then, self.elifs, self.els = cond, then, elifs, els

class While(Node):
    def __init__(self, cond, body):
        self.cond, self.body = cond, body

class Literal(Node):
    def __init__(self, val):
        self.val = val

class Ident(Node):
    def __init__(self, name):
        self.name = name


# === Parser ===
class Parser:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", "")

    def consume(self, kind=None):
        tok = self.peek()
        if kind and tok[0] != kind:
            raise SyntaxError(f"Expected {kind}, got {tok}")
        self.pos += 1
        return tok

    def parse_program(self):
        stmts = []
        while self.peek()[0] != "EOF":
            stmts.append(self.parse_stmt())
        return Program(stmts)

    def parse_stmt(self):
        t = self.peek()
        if t[0] == "PRINT":
            self.consume("PRINT")
            self.consume("LPAREN")
            expr = self.parse_expr()
            self.consume("RPAREN")
            return Print(expr)

        if t[0] == "IDENT":
            name = self.consume("IDENT")[1]
            self.consume("OP")  # currently only supports '='
            expr = self.parse_expr()
            return Assign(name, expr)

        if t[0] == "IF":
            return self.parse_if()

        if t[0] == "WHILE":
            return self.parse_while()

        raise SyntaxError(f"Unexpected token {t}")

    def parse_if(self):
        self.consume("IF")
        cond = self.parse_expr()
        self.consume("COLON")
        then = [self.parse_stmt()]
        elifs, els = [], []
        while self.peek()[0] == "ELIF":
            self.consume("ELIF")
            cond2 = self.parse_expr()
            self.consume("COLON")
            body2 = [self.parse_stmt()]
            elifs.append((cond2, body2))
        if self.peek()[0] == "ELSE":
            self.consume("ELSE")
            els = [self.parse_stmt()]
        return If(cond, then, elifs, els)

    def parse_while(self):
        self.consume("WHILE")
        cond = self.parse_expr()
        self.consume("COLON")
        body = [self.parse_stmt()]
        return While(cond, body)

    def parse_expr(self):
        left = self.parse_term()
        while self.peek()[0] == "OP":
            op = self.consume("OP")[1]
            right = self.parse_term()
            left = BinOp(left, op, right)
        return left

    def parse_term(self):
        t = self.peek()
        if t[0] == "NUMBER":
            return Literal(int(self.consume("NUMBER")[1]))
        if t[0] == "STRING":
            return Literal(self.consume("STRING")[1].strip('"'))
        if t[0] == "IDENT":
            return Ident(self.consume("IDENT")[1])
        raise SyntaxError(f"Unexpected {t}")

# ---------------------------
# Utilities: parse helpers, AST pretty-printer and CLI
# ---------------------------
import argparse
import os
from say_lexer import tokenize

def parse_code(code: str) -> Program:
    """Parse source text and return a `Program` AST."""
    tokens = list(tokenize(code))
    p = Parser(tokens)
    return p.parse_program()

def parse_file(path: str) -> Program:
    """Read `path` and parse it into a `Program`."""
    with open(path, "r", encoding="utf-8") as f:
        return parse_code(f.read())

def ast_to_string(node, indent: int = 0) -> str:
    """Human-readable AST printer for debugging and tests."""
    pad = "  " * indent
    if isinstance(node, Program):
        s = pad + "Program:\n"
        for st in node.stmts:
            s += ast_to_string(st, indent + 1)
        return s
    if isinstance(node, Print):
        return pad + "Print:\n" + ast_to_string(node.expr, indent + 1)
    if isinstance(node, Assign):
        return pad + f"Assign: {node.ident}\n" + ast_to_string(node.expr, indent + 1)
    if isinstance(node, BinOp):
        s = pad + f"BinOp: {node.op}\n"
        s += ast_to_string(node.left, indent + 1)
        s += ast_to_string(node.right, indent + 1)
        return s
    if isinstance(node, If):
        s = pad + "If:\n"
        s += pad + "  Cond:\n" + ast_to_string(node.cond, indent + 2)
        s += pad + "  Then:\n"
        for st in node.then:
            s += ast_to_string(st, indent + 2)
        for cond, body in node.elifs:
            s += pad + "  Elif Cond:\n" + ast_to_string(cond, indent + 2)
            s += pad + "  Elif Body:\n"
            for st in body:
                s += ast_to_string(st, indent + 2)
        if node.els:
            s += pad + "  Else:\n"
            for st in node.els:
                s += ast_to_string(st, indent + 2)
        return s
    if isinstance(node, While):
        s = pad + "While:\n"
        s += pad + "  Cond:\n" + ast_to_string(node.cond, indent + 2)
        s += pad + "  Body:\n"
        for st in node.body:
            s += ast_to_string(st, indent + 2)
        return s
    if isinstance(node, Literal):
        return pad + f"Literal: {node.val}\n"
    if isinstance(node, Ident):
        return pad + f"Ident: {node.name}\n"
    return pad + f"<Unknown node {node}>\n"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="say_parser", description="Parse a .say file and show AST")
    parser.add_argument("file", nargs="?", help="Source file to parse")
    parser.add_argument("--tokens", action="store_true", help="Print raw tokens instead of the AST")
    args = parser.parse_args()

    if not args.file:
        print("Usage: say_parser.py <file.say> [--tokens]")
        raise SystemExit(0)

    if not os.path.isfile(args.file):
        print(f"File not found: {args.file}")
        raise SystemExit(1)

    src = open(args.file, "r", encoding="utf-8").read()

    if args.tokens:
        for tok in tokenize(src):
            print(tok)
        raise SystemExit(0)

    try:
        prog = parse_code(src)
        print(ast_to_string(prog))
    except Exception as e:
        print(f"Parse error: {e}")
        raise SystemExit(1)

    # say_codegen.py
from llvmlite import ir
import re
from typing import Dict, List, Tuple, Optional

class Codegen:
    def __init__(self, module_name="sayit"):
        # === Create LLVM Module ===
        self.module = ir.Module(name=module_name)
        self.module.triple = "x86_64-pc-linux-gnu"  # default triple
        self.module.data_layout = ""  # can be filled by target machine later

        self.func = None
        self.builder = None
        self.printf = None

    # ---------------------------
    # Function Setup
    # ---------------------------
    def emit_main(self):
        """Create a main() entry function."""
        fnty = ir.FunctionType(ir.IntType(32), [])
        self.func = ir.Function(self.module, fnty, name="main")
        block = self.func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)

        # Declare printf once
        voidptr = ir.IntType(8).as_pointer()
        printf_ty = ir.FunctionType(ir.IntType(32), [voidptr], var_arg=True)
        self.printf = ir.Function(self.module, printf_ty, name="printf")

    # ---------------------------
    # Printing Support
    # ---------------------------
    def emit_print(self, text: str):
        """Emit a printf call with a string literal."""
        if self.builder is None:
            raise RuntimeError("emit_main() must be called first")

        # Global string literal
        cstr = self.builder.global_string_ptr(text, name="str")
        self.builder.call(self.printf, [cstr])

    # ---------------------------
    # Variables (alloca/store/load)
    # ---------------------------
    def emit_var(self, name: str, init_val: int):
        """Allocate and initialize an integer variable."""
        ptr = self.builder.alloca(ir.IntType(32), name=name)
        self.builder.store(ir.Constant(ir.IntType(32), init_val), ptr)
        return ptr

    def emit_load(self, ptr):
        """Load an integer value."""
        return self.builder.load(ptr)

    def emit_binop(self, op, lhs, rhs):
        """Binary operations on integers."""
        if op == "+":
            return self.builder.add(lhs, rhs)
        if op == "-":
            return self.builder.sub(lhs, rhs)
        if op == "*":
            return self.builder.mul(lhs, rhs)
        if op == "/":
            return self.builder.sdiv(lhs, rhs)
        raise NotImplementedError(f"Unsupported binop {op}")

    # ---------------------------
    # Finisher
    # ---------------------------
    def finish(self):
        """Return from main() and dump LLVM IR as string."""
        if self.builder is not None and not self.builder.block.is_terminated:
            self.builder.ret(ir.IntType(32)(0))
        return str(self.module)

# ---------------------------
# Ritual parsing + Codegen helpers
# ---------------------------
def _parse_start_ritual_block(block_text: str) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    """
    Parse a Start Ritual block like:
      Start Ritual:
          string message = "Hello World!"
          print(message)
          end()
    Returns (strings_map, actions)
      - strings_map: name -> string value
      - actions: list of (action, arg) tuples, currently supports ('print', name)
    """
    strings: Dict[str, str] = {}
    actions: List[Tuple[str, str]] = []

    for raw in block_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # skip header/footer tokens
        if re.match(r"(?i)^start\s+ritual\s*:", line):
            continue
        if re.match(r"(?i)^end\s*\(\s*\)\s*$", line) or line.lower() == "end()":
            continue

        # string declaration: string name = "value"
        m = re.match(r'^string\s+([a-zA-Z_]\w*)\s*=\s*"([^"]*)"\s*$', line, flags=re.IGNORECASE)
        if m:
            name, val = m.group(1), m.group(2)
            strings[name] = val
            continue

        # print call: print(name)
        m2 = re.match(r'^print\s*\(\s*([a-zA-Z_]\w*)\s*\)\s*$', line, flags=re.IGNORECASE)
        if m2:
            actions.append(("print", m2.group(1)))
            continue

        # ignore unknown lines (could raise if stricter behavior desired)

    return strings, actions

def _apply_start_ritual_to_codegen(cg: Codegen, block_text: str):
    """
    Given a Codegen instance with an active builder (emit_main called),
    create global string constants for declarations and emit print calls
    for print(...) actions that reference those declared strings.
    """
    if cg.builder is None:
        raise RuntimeError("Codegen.emit_main() must be called before applying Start Ritual")

    strings, actions = _parse_start_ritual_block(block_text)
    voidptr = ir.IntType(8).as_pointer()
    ptrs: Dict[str, object] = {}

    # Create a local pointer (i8*) alloca and store a global string pointer into it.
    for name, val in strings.items():
        cptr = cg.builder.global_string_ptr(val, name=f"str_{name}")
        alloca_ptr = cg.builder.alloca(voidptr, name=name)
        cg.builder.store(cptr, alloca_ptr)
        ptrs[name] = alloca_ptr

    # Execute actions in order.
    for act, arg in actions:
        if act == "print":
            if arg in ptrs:
                loaded = cg.builder.load(ptrs[arg])
                cg.builder.call(cg.printf, [loaded])
            else:
                # fallback: if name not found, print a placeholder
                cg.emit_print(f"<{arg}>")

def example_module_from_start_ritual(block_text: str, out_file: str = None) -> str:
    """
    Convenience builder: create a module, apply the Start Ritual block, and return the IR.
    """
    cg = Codegen()
    cg.emit_main()
    _apply_start_ritual_to_codegen(cg, block_text)
    llvm_ir = cg.finish()

    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(llvm_ir)
    return llvm_ir

# ---------------------------
# Demo / helper for quick IR generation
# ---------------------------
def example_module(out_file: str = None) -> str:
    """
    Build a tiny example module and return the LLVM IR as a string.
    If out_file is provided, the IR is also written to that file.
    """
    cg = Codegen()
    cg.emit_main()
    cg.emit_print("Hello, Sayit!")
    cg.emit_print("This is an example LLVM IR from Codegen.")
    llvm_ir = cg.finish()

    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(llvm_ir)
    return llvm_ir


if __name__ == "__main__":
    # Lightweight CLI: optional `-o <file>` to write IR to disk, otherwise print to stdout.
    import sys
    out_file = None
    if len(sys.argv) >= 3 and sys.argv[1] in ("-o", "--out"):
        out_file = sys.argv[2]

    ir_text = example_module(out_file=out_file)
    if out_file is None:
        print(ir_text)
        
    from say_codegen import example_module_with_make
make_text = """Make:
    x = 1
    y = 2
"""
ir = example_module_with_make(make_text)
print(ir)

# Quick demo when running this file directly
if __name__ == "__main__" and False:
    # Example block (the user-specified block)
    ritual = """Start Ritual:
        string message = "Hello World!"
        print(message)
        end()
    """
    print(example_module_from_start_ritual(ritual))

# ---------------------------
# While-block parsing + Codegen helper
# ---------------------------

def _parse_while_header(header: str):
    """
    Parse a simple while header like: 'While z < 10:'
    Returns (var_name, op, int_const)
    """
    m = re.match(r'(?i)^\s*While\s+([a-zA-Z_]\w*)\s*([<>=!]+)\s*([0-9]+)\s*:', header)
    if not m:
        raise ValueError(f"Malformed While header: {header!r}")
    name, op, val = m.group(1), m.group(2), int(m.group(3))
    return name, op, val

def _parse_until_assignment(line: str):
    """
    Parse a line like: 'y = y + x until y == 9'
    Returns (target, lhs, op, rhs, until_lhs, until_op, until_rhs)
    """
    m = re.match(r'^\s*([a-zA-Z_]\w*)\s*=\s*([a-zA-Z_]\w*|\d+)\s*([\+\-\*/])\s*([a-zA-Z_]\w*|\d+)\s+until\s+(.+)$', line)
    if not m:
        raise ValueError(f"Malformed until-assignment: {line!r}")
    target, lhs, op, rhs, until = m.groups()
    until = until.strip()
    mu = re.match(r'^\s*([a-zA-Z_]\w*|\d+)\s*([<>=!]+)\s*([a-zA-Z_]\w*|\d+)\s*$', until)
    if not mu:
        raise ValueError(f"Malformed until condition: {until!r}")
    until_lhs, until_op, until_rhs = mu.group(1), mu.group(2), mu.group(3)
    return target, lhs, op, rhs, until_lhs, until_op, until_rhs

def _get_or_create_ptr(cg: Codegen, name: str):
    """
    Ensure an alloca exists for `name`. We create a new alloca initialized to 0 if needed.
    This is a simple approach for demo IR generation.
    """
    # Create a default-initialized alloca; Codegen does not track ptrs, so just create one.
    return cg.emit_var(name, 0)

def _cmp_ir(cg: Codegen, lhs, op: str, rhs):
    """
    Create an i1 comparison for signed integers using op string.
    lhs and rhs are LLVM i32 values or ints (we convert ints to constants).
    """
    it32 = ir.IntType(32)
    if isinstance(lhs, int):
        lhs = ir.Constant(it32, lhs)
    if isinstance(rhs, int):
        rhs = ir.Constant(it32, rhs)

    if op == "<":
        return cg.builder.icmp_signed("slt", lhs, rhs)
    if op == "<=":
        return cg.builder.icmp_signed("sle", lhs, rhs)
    if op == ">":
        return cg.builder.icmp_signed("sgt", lhs, rhs)
    if op == ">=":
        return cg.builder.icmp_signed("sge", lhs, rhs)
    if op == "==":
        return cg.builder.icmp_signed("eq", lhs, rhs)
    if op == "!=":
        return cg.builder.icmp_signed("ne", lhs, rhs)
    raise NotImplementedError(f"Unsupported comparison op {op}")

def _apply_while_block_to_codegen(cg: Codegen, block_text: str):
    """
    Parse the provided block_text containing a While header and the following body,
    then emit corresponding LLVM IR into the provided Codegen instance.

    The function understands the specific structure you provided:
      While z < 10:
          y = y + x until y == 9

          If z > 7 and z < 9:
              print("true")
          Else:
              print("maybe")

          Elif z in [3...8]:
              print("true")
          Else:
              print("false")

    Notes:
    - Variables referenced will get simple alloca initializations if not present.
    - Range `[a...b]` is treated as inclusive: z >= a && z <= b.
    - 'until' produces an inner do-while loop that repeats the assignment until the until-condition is true.
    """
    if cg.builder is None:
        raise RuntimeError("Codegen.emit_main() must be called before applying While block")

    lines = [ln for ln in block_text.splitlines() if ln.strip() != ""]
    if not lines:
        return

    # Parse header
    header = lines[0]
    var_name, var_op, var_val = _parse_while_header(header)

    # locate the assignment and the two condition blocks (we accept the exact ordering)
    assignment_line = None
    first_if_lines = []
    second_if_lines = []
    mode = None
    for ln in lines[1:]:
        s = ln.strip()
        if s.lower().startswith("if "):
            mode = "first_if"
            first_if_lines = [s]
            continue
        if s.lower().startswith("elif ") or s.lower().startswith("elif"):
            mode = "second_if"
            second_if_lines = [s]
            continue
        if mode == "first_if":
            first_if_lines.append(s)
        elif mode == "second_if":
            second_if_lines.append(s)
        elif "until" in s and "=" in s:
            assignment_line = s

    if assignment_line is None:
        raise ValueError("Missing until-assignment in While block")

    # Ensure ptrs for variables used
    # We'll collect variable names from header, assignment and conditions
    # Minimal extraction:
    needed = set()
    needed.add(var_name)
    # parse assignment to collect names
    tgt, lhs, aop, rhs, until_lhs, until_op, until_rhs = _parse_until_assignment(assignment_line)
    for name in (tgt, lhs, rhs, until_lhs, until_rhs):
        if not name.isdigit():
            needed.add(name)

    ptrs = {name: _get_or_create_ptr(cg, name) for name in needed}

    # Create loop blocks
    func = cg.func
    entry_block = cg.builder.block
    cond_bb = func.append_basic_block(name="while_cond")
    body_bb = func.append_basic_block(name="while_body")
    after_bb = func.append_basic_block(name="while_after")

    # Jump to condition
    cg.builder.branch(cond_bb)

    # Condition block
    cg.builder.position_at_end(cond_bb)
    z_val = cg.emit_load(ptrs[var_name])
    cond_i1 = _cmp_ir(cg, z_val, var_op, var_val)
    cg.builder.cbranch(cond_i1, body_bb, after_bb)

    # Body block
    cg.builder.position_at_end(body_bb)

    # Inner 'until' do-while loop: create inner blocks
    inner_body_bb = func.append_basic_block(name="inner_body")
    inner_cond_bb = func.append_basic_block(name="inner_cond")
    inner_after_bb = func.append_basic_block(name="inner_after")

    # Enter inner body
    cg.builder.branch(inner_body_bb)

    # inner body: perform assignment y = y + x (or rhs)
    cg.builder.position_at_end(inner_body_bb)
    # resolve left and right operands
    def resolve_operand(tok):
        if tok.isdigit():
            return ir.Constant(ir.IntType(32), int(tok))
        return cg.emit_load(ptrs[tok])

    left_val = resolve_operand(lhs)
    right_val = resolve_operand(rhs)
    if aop == "+":
        sumv = cg.builder.add(left_val, right_val)
    elif aop == "-":
        sumv = cg.builder.sub(left_val, right_val)
    elif aop == "*":
        sumv = cg.builder.mul(left_val, right_val)
    elif aop == "/":
        sumv = cg.builder.sdiv(left_val, right_val)
    else:
        raise NotImplementedError(f"Unsupported arithmetic op {aop}")

    # store into target
    cg.builder.store(sumv, ptrs[tgt])
    # jump to inner condition
    cg.builder.branch(inner_cond_bb)

    # inner condition
    cg.builder.position_at_end(inner_cond_bb)
    u_lhs = resolve_operand(until_lhs)
    u_rhs = resolve_operand(until_rhs)
    until_cmp = _cmp_ir(cg, u_lhs, until_op, u_rhs)  # true when until-condition satisfied
    # if until-condition is true -> exit inner loop, else repeat inner body
    cg.builder.cbranch(until_cmp, inner_after_bb, inner_body_bb)

    # continue after inner loop: proceed to first conditional
    cg.builder.position_at_end(inner_after_bb)

    # --- First If: If z > 7 and z < 9: print("true") Else: print("maybe")
    # build blocks
    first_then = func.append_basic_block(name="if1_then")
    first_else = func.append_basic_block(name="if1_else")
    first_after = func.append_basic_block(name="if1_after")

    # compute conditions
    zv = cg.emit_load(ptrs[var_name])
    cmp1 = _cmp_ir(cg, zv, ">", 7)
    cmp2 = _cmp_ir(cg, zv, "<", 9)
    both = cg.builder.and_(cmp1, cmp2)
    cg.builder.cbranch(both, first_then, first_else)

    # then
    cg.builder.position_at_end(first_then)
    cg.emit_print("true")
    cg.builder.branch(first_after)

    # else
    cg.builder.position_at_end(first_else)
    cg.emit_print("maybe")
    cg.builder.branch(first_after)

    cg.builder.position_at_end(first_after)

    # --- Second conditional: Elif z in [3...8] -> print("true") else print("false")
    second_then = func.append_basic_block(name="if2_then")
    second_else = func.append_basic_block(name="if2_else")
    second_after = func.append_basic_block(name="if2_after")

    # treat [3...8] as inclusive 3..8
    cmp_lo = _cmp_ir(cg, zv, ">=", 3)
    cmp_hi = _cmp_ir(cg, zv, "<=", 8)
    in_range = cg.builder.and_(cmp_lo, cmp_hi)
    cg.builder.cbranch(in_range, second_then, second_else)

    cg.builder.position_at_end(second_then)
    cg.emit_print("true")
    cg.builder.branch(second_after)

    cg.builder.position_at_end(second_else)
    cg.emit_print("false")
    cg.builder.branch(second_after)

    cg.builder.position_at_end(second_after)

    # end of body -> jump back to while condition
    cg.builder.branch(cond_bb)

    # position builder at after_bb so subsequent emissions append correctly
    cg.builder.position_at_end(after_bb)

# ---------------------------
# Boolean-while parsing + Codegen helper (While true / While false)
# ---------------------------
def _parse_simple_call(line: str) -> Optional[str]:
    """
    Parse a simple zero-arg call like `execute()` or `fail()`.
    Returns the function name or None if not a simple call.
    """
    m = re.match(r'^\s*([a-zA-Z_]\w*)\s*\(\s*\)\s*$', line)
    return m.group(1) if m else None

def _apply_boolean_while_block_to_codegen(cg: Codegen, block_text: str):
    """
    Handle `While true:` and `While false:` blocks where the body contains
    simple zero-argument calls such as `execute()` or `fail()`.

    - `While true:` emits an infinite loop that calls the listed functions.
    - `While false:` emits a loop whose condition is constant false (body not executed).
    """
    if cg.builder is None:
        raise RuntimeError("Codegen.emit_main() must be called before applying While block")

    lines = [ln for ln in block_text.splitlines() if ln.strip() != ""]
    if not lines:
        return

    header = lines[0].strip()
    header_low = header.lower()
    if header_low not in ("while true:", "while false:"):
        return  # not a boolean-while; caller can fall back to other handler

    # collect simple call names from the body (ignore non-matching lines)
    calls = []
    for raw in lines[1:]:
        name = _parse_simple_call(raw.strip())
        if name:
            calls.append(name)

    func = cg.func
    cond_bb = func.append_basic_block(name="bool_while_cond")
    body_bb = func.append_basic_block(name="bool_while_body")
    after_bb = func.append_basic_block(name="bool_while_after")

    # jump to condition
    cg.builder.branch(cond_bb)

    # condition block: constant true or false
    cg.builder.position_at_end(cond_bb)
    bool_val = ir.Constant(ir.IntType(1), 1 if header_low == "while true:" else 0)
    cg.builder.cbranch(bool_val, body_bb, after_bb)

    # body: call each function in order, then loop back to condition
    cg.builder.position_at_end(body_bb)
    # declare or reuse zero-arg void functions
    for name in calls:
        fnty = ir.FunctionType(ir.VoidType(), [])
        fn = ir.Function(cg.module, fnty, name=name)
        cg.builder.call(fn, [])
    # loop back
    cg.builder.branch(cond_bb)

    # continue after loop
    cg.builder.position_at_end(after_bb)

def example_module_from_while_block(block_text: str, out_file: Optional[str] = None) -> str:
    """
    Convenience function to generate a module from a While block text.
    """
    cg = Codegen()
    cg.emit_main()
    _apply_while_block_to_codegen(cg, block_text)
    llvm_ir = cg.finish()
    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(llvm_ir)
    return llvm_ir

# ---------------------------
# Finally-block parsing + Codegen helper
# ---------------------------
def _parse_finally_block(block_text: str) -> List[str]:
    """
    Parse a Finally: block and return list of simple zero-arg call names found.
    Example:
      Finally:
          end()
    """
    calls: List[str] = []
    for raw in block_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.match(r"(?i)^finally\s*:", line):
            continue
        # collect simple zero-arg calls (reuses `_parse_simple_call` if present)
        name = None
        try:
            name = _parse_simple_call(line)  # uses function defined above for simple calls
        except NameError:
            # fallback: inline parse if helper isn't available
            m = re.match(r'^\s*([a-zA-Z_]\w*)\s*\(\s*\)\s*$', line)
            name = m.group(1) if m else None
        if name:
            calls.append(name)
    return calls

def _apply_finally_to_codegen(cg: Codegen, block_text: str):
    """
    Emit the calls declared in a Finally block at the current builder position.
    Each zero-arg function is declared as an external void function and called once.
    """
    if cg.builder is None:
        raise RuntimeError("Codegen.emit_main() must be called before applying Finally block")

    calls = _parse_finally_block(block_text)
    for name in calls:
        fnty = ir.FunctionType(ir.VoidType(), [])
        fn = ir.Function(cg.module, fnty, name=name)
        cg.builder.call(fn, [])

def example_module_from_finally_block(block_text: str, out_file: Optional[str] = None) -> str:
    """
    Convenience builder: create a module, apply the Finally block, and return the IR.
    """
    cg = Codegen()
    cg.emit_main()
    _apply_finally_to_codegen(cg, block_text)
    llvm_ir = cg.finish()

    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(llvm_ir)
    return llvm_ir

# say_vm.py
"""
Enhanced VM with high-performance execution features and many built-in services:
- Bytecode compiler + stack VM for fast execution
- Constant-folding optimization
- Built-in service registry for: import/export, IO, rendering, packetizing,
  capsuling, compiling, scanning, uploading, downloading, scaling, range checks,
  modes, scoping and ping
- Execution limits (max_steps / timeout) and profiling
- Async execution and parallel runner (ThreadPool)
- Improved REPL (multi-line) and CLI integration with tracing/profiling flags

Notes:
- New services are exposed as zero-arg builtins callable by name (CALL "name").
  They use VM.vars for input/output (convention: set _arg_* or named vars).
- These implementations are pragmatic stubs meant to be easy to extend/replace.
"""
from __future__ import annotations
import argparse
import base64
import json
import os
import pathlib
import socket
import sys
import tempfile
import threading
import time
import concurrent.futures
import hashlib
import shutil
from typing import Any, Dict, List, Optional, Tuple

from say_parser import Print, Assign, BinOp, Literal, Ident, If, While, Program
from say_lexer import tokenize as _tokenize
from say_parser import Parser as _Parser

class ExecError(Exception):
    pass

class ExecVM:
    def __init__(self,
                 max_steps: Optional[int] = 10_000_000,
                 timeout: Optional[float] = None,
                 builtins: Optional[Dict[str, Any]] = None,
                 trace: bool = False,
                 profile: bool = False,
                 optimize: bool = True):
        self.vars: Dict[str, Any] = {}
        self.max_steps = max_steps
        self._steps = 0
        self.timeout = timeout
        self._start_time: Optional[float] = None
        self.trace = trace
        self.profile = profile
        self.optimize = optimize
        self._profile_counts: Dict[str, int] = {}
        self._halted = False

        # lightweight mode/scope managers
        self.mode: Optional[str] = None
        self.scope_stack: List[Dict[str, Any]] = []

        # service registry for pluggable features
        self.services: Dict[str, Any] = {}

        # builtins: zero-arg callables used by CALL opcode
        self.builtins: Dict[str, Any] = {
            # core control
            "execute": lambda: print("[builtin] execute()"),
            "fail": lambda: (_ for _ in ()).throw(RuntimeError("fail() called")),
            "end": lambda: (_ for _ in ()).throw(StopIteration("end()")),
        }
        if builtins:
            self.builtins.update(builtins)

        # register default services
        self._register_default_services()

    # ---------------------------
    # Service registration
    # ---------------------------
    def register_service(self, name: str, obj: Any):
        self.services[name] = obj

    def get_service(self, name: str):
        return self.services.get(name)

    def _register_default_services(self):
        # IO service
        self.register_service("io", {
            "import": self._svc_import,
            "export": self._svc_export,
            "input": self._svc_input,
            "read": self._svc_read_file,
            "write": self._svc_write_file,
        })

        # render service
        self.register_service("render", {
            "template": self._svc_render_template
        })

        # network service
        self.register_service("net", {
            "packetize": self._svc_packetize,
            "upload": self._svc_upload,
            "download": self._svc_download,
            "ping": self._svc_ping,
        })

        # util service
        self.register_service("util", {
            "capsule": self._svc_capsule,
            "scan": self._svc_scan,
            "compile": self._svc_compile_ast,
            "scale": self._svc_scale,
            "range_check": self._svc_range_check,
            "set_mode": self._svc_set_mode,
            "get_mode": self._svc_get_mode,
            "push_scope": self._svc_push_scope,
            "pop_scope": self._svc_pop_scope,
            "get_scope": self._svc_get_scope,
        })

        # expose convenient builtins that call services using VM.vars convention
        # These builtins are zero-arg and use pre-agreed variable names for inputs/outputs.
        self.builtins.update({
            "import_file": lambda: self._svc_import(),
            "export_file": lambda: self._svc_export(),
            "input_line": lambda: self._svc_input(),
            "render_template": lambda: self._svc_render_template(),
            "packetize": lambda: self._svc_packetize(),
            "capsule": lambda: self._svc_capsule(),
            "compile_ast": lambda: self._svc_compile_ast(),
            "scan_text": lambda: self._svc_scan(),
            "upload": lambda: self._svc_upload(),
            "download": lambda: self._svc_download(),
            "scale": lambda: self._svc_scale(),
            "range_check": lambda: self._svc_range_check(),
            "set_mode": lambda: self._svc_set_mode(),
            "get_mode": lambda: self._svc_get_mode(),
            "push_scope": lambda: self._svc_push_scope(),
            "pop_scope": lambda: self._svc_pop_scope(),
            "get_scope": lambda: self._svc_get_scope(),
            "ping": lambda: self._svc_ping(),
        })

    # ---------------------------
    # Default service implementations (pragmatic stubs)
    # Inputs/outputs are read/written to self.vars by convention:
    # - _arg_path, _arg_data, _arg_template, _arg_target, etc.
    # - outputs placed into _out_* keys.
    # ---------------------------
    # IO
    def _svc_import(self):
        path = self.vars.get("_arg_path")
        if not path:
            raise ExecError("import: _arg_path not set")
        if not os.path.isfile(path):
            raise ExecError(f"import: file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        # store raw source and parsed AST
        self.vars["_out_imported_source"] = src
        try:
            prog = _Parser(list(_tokenize(src))).parse_program()
            self.vars["_out_imported_ast"] = prog
        except Exception as e:
            self.vars["_out_import_error"] = str(e)
        return None

    def _svc_export(self):
        path = self.vars.get("_arg_path")
        data = self.vars.get("_arg_data")
        if not path:
            raise ExecError("export: _arg_path not set")
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(data))
        self.vars["_out_export_path"] = path
        return None

    def _svc_input(self):
        prompt = self.vars.get("_arg_prompt", "")
        val = input(prompt)
        self.vars["_out_input"] = val
        return None

    def _svc_read_file(self):
        path = self.vars.get("_arg_path")
        if not path:
            raise ExecError("read: _arg_path not set")
        with open(path, "r", encoding="utf-8") as f:
            self.vars["_out_read"] = f.read()
        return None

    def _svc_write_file(self):
        path = self.vars.get("_arg_path")
        data = self.vars.get("_arg_data", "")
        if not path:
            raise ExecError("write: _arg_path not set")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(data))
        self.vars["_out_write"] = path
        return None

    # Render
    def _svc_render_template(self):
        template = self.vars.get("_arg_template", "")
        # context: all keys starting with "ctx_"
        ctx = {k[4:]: v for k, v in self.vars.items() if k.startswith("ctx_")}
        try:
            rendered = template.format(**ctx)
        except Exception as e:
            rendered = f"<render-error: {e}>"
        self.vars["_out_rendered"] = rendered
        return None

    # Packetize / upload / download / ping
    def _svc_packetize(self):
        data = self.vars.get("_arg_data", "")
        chunk_size = int(self.vars.get("_arg_chunk", 1024))
        b = data.encode("utf-8") if isinstance(data, str) else (data if isinstance(data, bytes) else str(data).encode("utf-8"))
        packets = [base64.b64encode(b[i:i+chunk_size]).decode("ascii") for i in range(0, len(b), chunk_size)]
        self.vars["_out_packets"] = packets
        return None

    def _svc_upload(self):
        # pragmatic stub: copy file to .remote dir to simulate upload
        src = self.vars.get("_arg_path")
        if not src or not os.path.exists(src):
            raise ExecError("upload: _arg_path missing or does not exist")
        remote_dir = os.path.join(".remote")
        os.makedirs(remote_dir, exist_ok=True)
        dst = os.path.join(remote_dir, os.path.basename(src))
        shutil.copy(src, dst)
        self.vars["_out_uploaded"] = dst
        return None

    def _svc_download(self):
        # pragmatic stub: copy from .remote dir to destination
        name = self.vars.get("_arg_name")
        dest = self.vars.get("_arg_dest")
        remote_dir = os.path.join(".remote")
        src = os.path.join(remote_dir, name) if name else None
        if not src or not os.path.exists(src):
            raise ExecError("download: remote object not found")
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        shutil.copy(src, dest)
        self.vars["_out_downloaded"] = dest
        return None

    def _svc_ping(self):
        host = self.vars.get("_arg_host")
        port = int(self.vars.get("_arg_port", 7))
        timeout = float(self.vars.get("_arg_timeout", 1.0))
        if not host:
            raise ExecError("ping: _arg_host not set")
        try:
            s = socket.create_connection((host, port), timeout=timeout)
            s.close()
            self.vars["_out_ping"] = True
        except Exception as e:
            self.vars["_out_ping"] = False
            self.vars["_out_ping_error"] = str(e)
        return None

    # Capsule / scan / compile / scale / range / mode / scope
    def _svc_capsule(self):
        data = self.vars.get("_arg_data")
        meta = self.vars.get("_arg_meta", {})
        capsule = {"data": data, "meta": meta}
        self.vars["_out_capsule"] = capsule
        return None

    def _svc_scan(self):
        src = self.vars.get("_arg_text", "")
        toks = list(_tokenize(src))
        self.vars["_out_tokens"] = toks
        return None

    def _svc_compile_ast(self):
        # expects _arg_ast to be present (Program)
        prog = self.vars.get("_arg_ast")
        if not prog:
            raise ExecError("compile: _arg_ast not set")
        ops = self.compile_program(prog)
        self.vars["_out_compiled_ops"] = ops
        return None

    def _svc_scale(self):
        name = self.vars.get("_arg_name")
        factor = float(self.vars.get("_arg_factor", 1.0))
        if name not in self.vars:
            raise ExecError("scale: variable not found")
        val = self.vars[name]
        try:
            scaled = val * factor
        except Exception as e:
            raise ExecError(f"scale: cannot scale {val}: {e}")
        self.vars["_out_scaled"] = scaled
        return None

    def _svc_range_check(self):
        name = self.vars.get("_arg_name")
        lo = float(self.vars.get("_arg_lo"))
        hi = float(self.vars.get("_arg_hi"))
        if name not in self.vars:
            raise ExecError("range_check: variable not found")
        val = float(self.vars[name])
        ok = lo <= val <= hi
        self.vars["_out_range_ok"] = ok
        return None

    def _svc_set_mode(self):
        m = self.vars.get("_arg_mode")
        self.mode = m
        self.vars["_out_mode"] = self.mode
        return None

    def _svc_get_mode(self):
        self.vars["_out_mode"] = self.mode
        return None

    def _svc_push_scope(self):
        # push an empty scope or copy of provided dict at _arg_scope
        sc = self.vars.get("_arg_scope", {})
        self.scope_stack.append(dict(sc))
        self.vars["_out_scope_len"] = len(self.scope_stack)
        return None

    def _svc_pop_scope(self):
        if not self.scope_stack:
            raise ExecError("pop_scope: stack empty")
        popped = self.scope_stack.pop()
        self.vars["_out_popped_scope"] = popped
        return None

    def _svc_get_scope(self):
        idx = int(self.vars.get("_arg_index", -1))
        if not self.scope_stack:
            self.vars["_out_scope"] = None
            return None
        if idx == -1:
            self.vars["_out_scope"] = self.scope_stack[-1]
        else:
            self.vars["_out_scope"] = self.scope_stack[idx]
        return None

    # ---------------------------
    # Compiler: AST -> bytecode (unchanged from prior)
    # ---------------------------
    def compile_program(self, program: Program) -> List[Tuple[str, Any]]:
        self._label_counter = 0
        ops: List[Tuple[str, Any]] = []
        self._emit_statements(program.stmts, ops)
        ops.append(("HALT", None))
        return ops

    def _emit_statements(self, stmts: List, ops: List[Tuple[str, Any]]):
        for s in stmts:
            self._emit_stmt(s, ops)

    def _emit_stmt(self, stmt, ops: List[Tuple[str, Any]]):
        cls = stmt.__class__.__name__
        if cls == "Print":
            self._emit_expr(stmt.expr, ops)
            ops.append(("PRINT", None))
        elif cls == "Assign":
            self._emit_expr(stmt.expr, ops)
            ops.append(("STORE_VAR", stmt.ident))
        elif cls == "If":
            # cond, then, elifs, els
            self._emit_expr(stmt.cond, ops)
            jfalse_idx = len(ops)
            ops.append(("JUMP_IF_FALSE", None))
            self._emit_statements(stmt.then, ops)
            jend_idx = len(ops)
            ops.append(("JUMP", None))
            else_start = len(ops)
            ops[jfalse_idx] = ("JUMP_IF_FALSE", else_start)
            for (econd, ebody) in stmt.elifs:
                self._emit_expr(econd, ops)
                jf = len(ops)
                ops.append(("JUMP_IF_FALSE", None))
                self._emit_statements(ebody, ops)
                je = len(ops)
                ops.append(("JUMP", None))
                ops[jf] = ("JUMP_IF_FALSE", len(ops))
            if stmt.els:
                self._emit_statements(stmt.els, ops)
            cur = len(ops)
            for i, (op, arg) in enumerate(ops):
                if op == "JUMP" and arg is None:
                    ops[i] = ("JUMP", cur)
        elif cls == "While":
            cond_start = len(ops)
            self._emit_expr(stmt.cond, ops)
            jf_idx = len(ops)
            ops.append(("JUMP_IF_FALSE", None))
            self._emit_statements(stmt.body, ops)
            ops.append(("JUMP", cond_start))
            after = len(ops)
            ops[jf_idx] = ("JUMP_IF_FALSE", after)
        else:
            ops.append(("NOP", None))

    def _emit_expr(self, expr, ops: List[Tuple[str, Any]]):
        if self.optimize and isinstance(expr, BinOp):
            if isinstance(expr.left, Literal) and isinstance(expr.right, Literal):
                lv, rv = expr.left.val, expr.right.val
                op = expr.op
                val = self._compute_binop_value(op, lv, rv)
                ops.append(("LOAD_CONST", val))
                return
        if isinstance(expr, Literal):
            ops.append(("LOAD_CONST", expr.val))
        elif isinstance(expr, Ident):
            ops.append(("LOAD_VAR", expr.name))
        elif isinstance(expr, BinOp):
            self._emit_expr(expr.left, ops)
            self._emit_expr(expr.right, ops)
            ops.append(("BINARY_OP", expr.op))
        else:
            ops.append(("LOAD_CONST", None))

    def _compute_binop_value(self, op: str, l, r):
        if op == "+": return l + r
        if op == "-": return l - r
        if op == "*": return l * r
        if op == "/": return l // r
        if op == "==": return l == r
        if op == "!=": return l != r
        if op == ">": return l > r
        if op == "<": return l < r
        if op == ">=": return l >= r
        if op == "<=": return l <= r
        if op == "and": return bool(l and r)
        if op == "or": return bool(l or r)
        raise ExecError(f"Unsupported compile-time op {op}")

    # ---------------------------
    # Interpreter: execute bytecode (extended CALL support)
    # ---------------------------
    def run_ops(self, ops: List[Tuple[str, Any]]):
        stack: List[Any] = []
        pc = 0
        self._steps = 0
        self._start_time = time.time() if self.timeout else None
        self._halted = False

        while pc < len(ops):
            if self.max_steps is not None:
                self._steps += 1
                if self._steps > self.max_steps:
                    raise ExecError("instruction limit exceeded (possible infinite loop)")

            if self.timeout and self._start_time and (time.time() - self._start_time) > self.timeout:
                raise ExecError("execution timeout exceeded")

            op, arg = ops[pc]
            if self.profile:
                self._profile_counts[op] = self._profile_counts.get(op, 0) + 1
            if self.trace:
                print(f"[trace] pc={pc} op={op} arg={arg} stack={stack} vars={self.vars}")

            try:
                if op == "LOAD_CONST":
                    stack.append(arg)
                    pc += 1
                elif op == "LOAD_VAR":
                    stack.append(self.vars.get(arg, 0))
                    pc += 1
                elif op == "STORE_VAR":
                    val = stack.pop() if stack else None
                    self.vars[arg] = val
                    pc += 1
                elif op == "BINARY_OP":
                    rhs = stack.pop() if stack else None
                    lhs = stack.pop() if stack else None
                    res = self._apply_binop(arg, lhs, rhs)
                    stack.append(res)
                    pc += 1
                elif op == "PRINT":
                    val = stack.pop() if stack else None
                    print(val)
                    pc += 1
                elif op == "JUMP":
                    pc = arg
                elif op == "JUMP_IF_FALSE":
                    cond = stack.pop() if stack else None
                    if not cond:
                        pc = arg
                    else:
                        pc += 1
                elif op == "CALL":
                    # CALL name -> call builtin (zero-arg) or service method by naming convention
                    name = arg
                    fn = self.builtins.get(name)
                    if fn:
                        try:
                            res = fn()
                        except StopIteration:
                            self._halted = True
                            break
                        pc += 1
                        continue
                    # service call convention: "svc:service_name:action"
                    if isinstance(name, str) and name.startswith("svc:"):
                        _, svc_name, action = name.split(":", 2)
                        svc = self.get_service(svc_name)
                        if not svc:
                            raise ExecError(f"Unknown service: {svc_name}")
                        action_fn = svc.get(action) if isinstance(svc, dict) else getattr(svc, action, None)
                        if not action_fn:
                            raise ExecError(f"Service {svc_name} has no action {action}")
                        try:
                            action_fn()
                        except StopIteration:
                            self._halted = True
                            break
                        pc += 1
                        continue
                    # unknown call
                    raise ExecError(f"Unknown callable: {name}")
                elif op == "NOP":
                    pc += 1
                elif op == "HALT":
                    break
                else:
                    raise ExecError(f"Unknown opcode: {op}")
            except StopIteration:
                break
            except Exception as e:
                raise ExecError(f"Runtime error at pc={pc}, op={op}: {e}") from e

        return {
            "steps": self._steps,
            "profile": self._profile_counts.copy() if self.profile else {},
            "halted": self._halted
        }

    def _apply_binop(self, op: str, l, r):
        if op == "and":
            return bool(l and r)
        if op == "or":
            return bool(l or r)
        if op == "+":
            return l + r
        if op == "-":
            return l - r
        if op == "*":
            return l * r
        if op == "/":
            if r == 0:
                raise ExecError("division by zero")
            return l // r
        if op == "==":
            return l == r
        if op == "!=":
            return l != r
        if op == ">":
            return l > r
        if op == "<":
            return l < r
        if op == ">=":
            return l >= r
        if op == "<=":
            return l <= r
        raise ExecError(f"Unsupported binary op: {op}")

    # ---------------------------
    # High-level runtimes
    # ---------------------------
    def run_program(self, program: Program):
        ops = self.compile_program(program)
        return self.run_ops(ops)

    def run_program_async(self, program: Program, executor: Optional[concurrent.futures.Executor] = None):
        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        return executor.submit(self.run_program, program)

    @staticmethod
    def run_parallel(programs: List[Program], max_workers: int = 4, **vm_kwargs):
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = []
            for p in programs:
                vm = ExecVM(**vm_kwargs)
                futures.append(ex.submit(vm.run_program, p))
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())
        return results

# ---------------------------
# Compatibility wrapper for older simple VM API
# ---------------------------
class CompatVM:
    """
    A thin compatibility layer matching the original simple API:
    - preserves .vars and run(program) interface
    - delegates to ExecVM for execution
    """
    def __init__(self, max_steps: Optional[int] = 1_000_000, timeout: Optional[float] = None, trace: bool = False):
        self._exec_vm = ExecVM(max_steps=max_steps, timeout=timeout, trace=trace, optimize=True)
        self.vars = self._exec_vm.vars

    def run(self, program):
        res = self._exec_vm.run_program(program)
        # sync back vars (already shared)
        return res

# ---------------------------
# CLI, REPL and helpers (unchanged)
# ---------------------------
def _run_source_with_execvm(source: str, trace: bool = False, max_steps: int = 1_000_000, timeout: Optional[float] = None, profile: bool = False):
    tokens = list(_tokenize(source))
    parser = _Parser(tokens)
    program = parser.parse_program()
    vm = ExecVM(max_steps=max_steps, timeout=timeout, trace=trace, profile=profile, optimize=True)
    return vm.run_program(program)

def run_file(path: str, trace: bool = False, max_steps: int = 1_000_000, timeout: Optional[float] = None, profile: bool = False):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    return _run_source_with_execvm(src, trace=trace, max_steps=max_steps, timeout=timeout, profile=profile)

def repl(trace: bool = False, max_steps: int = 1_000_000, timeout: Optional[float] = None):
    print("Sayit REPL (type 'exit' or 'quit' to leave). Enter blank line to execute multi-line block.")
    vm = ExecVM(max_steps=max_steps, timeout=timeout, trace=trace, optimize=True)
    buffer_lines: List[str] = []
    try:
        while True:
            prompt = "say> " if not buffer_lines else " ... "
            try:
                line = input(prompt)
            except EOFError:
                print()
                break
            if line is None:
                continue
            if line.strip() in ("exit", "quit"):
                break
            # multi-line entry: blank line executes buffer
            if line.strip() == "" and buffer_lines:
                src = "\n".join(buffer_lines)
                buffer_lines = []
                try:
                    prog = _Parser(list(_tokenize(src))).parse_program()
                    vm.run_program(prog)
                except Exception as e:
                    print(f"[repl] Error: {e}")
                continue
            buffer_lines.append(line)
            # attempt single-line immediate execution
            if len(buffer_lines) == 1:
                try:
                    prog = _Parser(list(_tokenize(line))).parse_program()
                    vm.run_program(prog)
                    buffer_lines = []
                except Exception:
                    # incomplete or invalid -> wait for more lines
                    pass
    except KeyboardInterrupt:
        print("\n[repl] Interrupted.")
    return vm

# ---------------------------
# Backwards-compatible simple-run CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="say_vm", description="Run Sayit programs via the enhanced VM")
    parser.add_argument("file", nargs="?", help="Source file to run (.say)")
    parser.add_argument("--repl", action="store_true", help="Start an interactive REPL")
    parser.add_argument("--trace", action="store_true", help="Enable instruction tracing")
    parser.add_argument("--profile", action="store_true", help="Enable opcode profiling")
    parser.add_argument("--dump-vars", action="store_true", help="Dump VM variables after run")
    parser.add_argument("--demo", action="store_true", help="Run a tiny demo program")
    parser.add_argument("--max-steps", type=int, default=1_000_000, help="Instruction limit to prevent infinite loops")
    parser.add_argument("--timeout", type=float, default=None, help="Execution timeout in seconds (per run)")
    args = parser.parse_args()

    vm_instance = None

    try:
        if args.demo:
            demo_src = 'x = 1\nprint(x)\nprint("Hello from Sayit VM")\n'
            vm_instance = _run_source_with_execvm(demo_src, trace=args.trace, max_steps=args.max_steps, timeout=args.timeout, profile=args.profile)

        elif args.repl:
            repl(trace=args.trace, max_steps=args.max_steps, timeout=args.timeout)

        elif args.file:
            if not os.path.isfile(args.file):
                print(f"File not found: {args.file}")
                sys.exit(1)
            vm_instance = run_file(args.file, trace=args.trace, max_steps=args.max_steps, timeout=args.timeout, profile=args.profile)

        else:
            parser.print_usage()
            sys.exit(1)

    except ExecError as e:
        print(f"[say_vm] Execution error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[say_vm] Error: {e}")
        sys.exit(1)

    if args.dump_vars and isinstance(vm_instance, dict):
        # run_program returns profiling dict; variables available via ExecVM instance in REPL or when using API
        print("Execution result:", vm_instance)

# ---------------------------
# Advanced optimizations, services and AST/bytecode toolset
# ---------------------------
import zlib
import base64
import urllib.request
from html import escape as html_escape
from dataclasses import dataclass

# --- AST walker / visitor ----------------------------------------------------
class ASTWalker:
    """Generic AST visitor/walker with ability to transform nodes."""
    def visit(self, node):
        meth = getattr(self, f"visit_{node.__class__.__name__}", None)
        if meth:
            return meth(node)
        return self.generic_visit(node)

    def generic_visit(self, node):
        # Walk attributes that are lists or nodes
        for attr in getattr(node, "__dict__", {}):
            val = getattr(node, attr)
            if isinstance(val, list):
                for i, child in enumerate(val):
                    if hasattr(child, "__class__"):
                        new = self.visit(child)
                        if new is None:
                            continue
                        val[i] = new
            elif hasattr(val, "__class__"):
                new = self.visit(val)
                if new is not None:
                    setattr(node, attr, new)
        return node

# --- Optimization passes ----------------------------------------------------
class Pass:
    name = "generic"
    def run(self, program):  # mutate program or return new program
        return program

class ConstantFoldingPass(Pass):
    name = "constant-fold"
    def run(self, program):
        # Fold simple BinOp Literals in AST
        class CF(ASTWalker):
            def visit_BinOp(self, node):
                left = node.left
                right = node.right
                if getattr(left, "val", None) is not None and getattr(right, "val", None) is not None:
                    # compute at Python level
                    l, r = left.val, right.val
                    op = node.op
                    try:
                        if op == "+": v = l + r
                        elif op == "-": v = l - r
                        elif op == "*": v = l * r
                        elif op == "/": v = l // r
                        elif op == "==": v = l == r
                        elif op == "!=": v = l != r
                        elif op == ">": v = l > r
                        elif op == "<": v = l < r
                        else: return node
                        return Literal(v)
                    except Exception:
                        return node
                return node
        CF().visit(program)
        return program

class DeadCodeElimPass(Pass):
    name = "dead-code-elim"
    def run(self, program):
        # Remove trivial if/while bodies that are provably unreachable.
        class DCE(ASTWalker):
            def visit_If(self, node):
                # if condition is literal False => replace with els or remove
                if getattr(node.cond, "val", None) is not None:
                    if node.cond.val is False:
                        if node.els:
                            # replace with else body sequence
                            return node.els[0] if len(node.els) == 1 else node.els
                        return None  # remove
                    if node.cond.val is True:
                        # replace with then body
                        return node.then[0] if len(node.then) == 1 else node.then
                return node
            def visit_While(self, node):
                if getattr(node.cond, "val", None) is not None and node.cond.val is False:
                    return None
                return node
        DCE().visit(program)
        # prune None entries at top-level
        program.stmts = [s for s in program.stmts if s is not None]
        return program

# --- Bytecode peephole & simple loop unroll ---------------------------------
def peephole_optimize_ops(ops):
    """Simple peephole: fold consecutive constants + binary op into one const."""
    i = 0
    out = []
    while i < len(ops):
        op, arg = ops[i]
        # pattern: LOAD_CONST a, LOAD_CONST b, BINARY_OP op -> LOAD_CONST result
        if i + 2 < len(ops):
            op1, a1 = ops[i]
            op2, a2 = ops[i+1]
            op3, a3 = ops[i+2]
            if op1 == "LOAD_CONST" and op2 == "LOAD_CONST" and op3 == "BINARY_OP":
                try:
                    # compute at Python level
                    l, r = a1, a2
                    bop = a3
                    if bop == "+": v = l + r
                    elif bop == "-": v = l - r
                    elif bop == "*": v = l * r
                    elif bop == "/": v = l // r
                    elif bop == "and": v = bool(l and r)
                    elif bop == "or": v = bool(l or r)
                    elif bop == "==": v = l == r
                    elif bop == "!=": v = l != r
                    else:
                        raise Exception()
                    out.append(("LOAD_CONST", v))
                    i += 3
                    continue
                except Exception:
                    pass
        # remove NOPs
        if op == "NOP":
            i += 1
            continue
        out.append((op, arg))
        i += 1
    return out

def simple_loop_unroll(ops, max_unroll=4):
    """
    Detect simple back-edge loops and unroll a small number of iterations if
    the loop body is constant and the iteration count is small (heuristic).
    This is conservative and only handles a very limited pattern.
    """
    # Look for pattern: cond ... JUMP_IF_FALSE X ... JUMP back_to_cond
    out = []
    i = 0
    while i < len(ops):
        op, arg = ops[i]
        if op == "JUMP":
            out.append((op, arg))
            i += 1
            continue
        # find a backward jump
        if op == "JUMP" and arg < i:
            # not used here
            out.append((op, arg))
            i += 1
            continue
        # detect small loop by scanning for JUMP back
        found = False
        for j in range(i+3, min(len(ops), i+64)):
            if ops[j][0] == "JUMP" and isinstance(ops[j][1], int) and ops[j][1] <= i:
                body = ops[i:j]
                # conservative: body must not contain JUMP/JUMP_IF_FALSE
                if all(o not in ("JUMP", "JUMP_IF_FALSE") for o, _ in body):
                    # emit unrolled copies
                    for _ in range(min(max_unroll, 2)):
                        out.extend(body)
                    # skip original body and the backwards jump
                    i = j + 1
                    found = True
                    break
        if not found:
            out.append((op, arg))
            i += 1
    return out

# --- Virtual register allocator (stack->regs very simple) -------------------
@dataclass
class RegOp:
    op: str
    dst: Optional[str]
    src1: Optional[str] = None
    src2: Optional[str] = None
    imm: Optional[Any] = None

def stack_to_register_ops(ops):
    """
    Convert stack-based ops to simple register ops using temporary virtual registers.
    This is a naive translator for analysis / lower-level backends.
    """
    reg_count = 0
    stack = []
    out = []
    def new_reg():
        nonlocal reg_count
        r = f"r{reg_count}"
        reg_count += 1
        return r

    for op, arg in ops:
        if op == "LOAD_CONST":
            r = new_reg()
            out.append(RegOp("MOV_IMM", r, imm=arg, dst=r))
            stack.append(r)
        elif op == "LOAD_VAR":
            r = new_reg()
            out.append(RegOp("LOAD_VAR", r, src1=arg, dst=r))
            stack.append(r)
        elif op == "STORE_VAR":
            src = stack.pop() if stack else None
            out.append(RegOp("STORE_VAR", arg, src1=src))
        elif op == "BINARY_OP":
            r2 = stack.pop() if stack else None
            r1 = stack.pop() if stack else None
            rd = new_reg()
            out.append(RegOp("BINOP", rd, src1=r1, src2=r2, imm=arg))
            stack.append(rd)
        elif op == "PRINT":
            src = stack.pop() if stack else None
            out.append(RegOp("PRINT", None, src1=src))
        elif op in ("JUMP", "JUMP_IF_FALSE", "CALL", "NOP", "HALT"):
            out.append(RegOp(op, arg))
        else:
            out.append(RegOp(op, arg))
    return out

# --- Serialization / obfuscation / compression ------------------------------
def serialize_ops_to_bytes(ops):
    """Simple serialization: repr() then compress+base64."""
    raw = repr(ops).encode("utf-8")
    compressed = zlib.compress(raw)
    return base64.b64encode(compressed).decode("ascii")

def deserialize_ops_from_bytes(blob):
    raw = base64.b64decode(blob.encode("ascii"))
    decompressed = zlib.decompress(raw)
    return eval(decompressed.decode("utf-8"))

# --- HTML / HTTP helpers ----------------------------------------------------
def http_get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8")

def http_post(url, data: bytes, timeout=5):
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")

def render_html_fragment(template: str, **ctx):
    safe = {k: html_escape(str(v)) for k, v in ctx.items()}
    try:
        return template.format(**safe)
    except Exception as e:
        return f"<render-error:{e}>"

# --- Advanced numerics ------------------------------------------------------
def to_base12(n: int) -> str:
    """Convert integer to base-12 string using digits 0-9,A,B."""
    if n == 0:
        return "0"
    digits = "0123456789AB"
    neg = n < 0
    n = abs(n)
    out = []
    while n:
        out.append(digits[n % 12])
        n //= 12
    if neg:
        out.append("-")
    return "".join(reversed(out))

def from_base12(s: str) -> int:
    s = s.strip().upper()
    neg = s.startswith("-")
    if neg:
        s = s[1:]
    value = 0
    for ch in s:
        value = value * 12 + "0123456789AB".index(ch)
    return -value if neg else value

# --- Higher-level optimizer wrapper & ExecVMPro subclass --------------------
class PassManager:
    def __init__(self, passes: Optional[List[Pass]] = None):
        self.passes = passes or []

    def add(self, p: Pass):
        self.passes.append(p)

    def run(self, program):
        for p in self.passes:
            program = p.run(program)
        return program

class ExecVMPro(ExecVM):
    """ExecVM with optimization pipeline and advanced lowering hooks."""
    def __init__(self, *args, passes: Optional[PassManager] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pass_manager = passes or PassManager([ConstantFoldingPass(), DeadCodeElimPass()])
        self.enable_peep = True
        self.enable_unroll = True
        self.enable_regalloc = False

    def run_program(self, program: Program):
        # AST level optimizations
        program = self.pass_manager.run(program)

        # compile to ops
        ops = super().compile_program(program)

        # peephole & constant folding on ops
        if self.enable_peep:
            ops = peephole_optimize_ops(ops)
        if self.enable_unroll:
            ops = simple_loop_unroll(ops)

        # optional register lowering
        if self.enable_regalloc:
            regops = stack_to_register_ops(ops)
            # For now we do not execute regops; serialize for analysis
            self.vars["_out_regops"] = regops
            # lower back to stack ops by simple mapping (noop here)
            # fallback: run original ops
        # execute optimized ops
        return self.run_ops(ops)

# --- Glue helpers: produce NASM/WASM placeholders ---------------------------
def ops_to_nasm(ops):
    lines = ["; NASM-like placeholder"]
    for i, (op, arg) in enumerate(ops):
        lines.append(f"; {i:04}: {op} {arg!r}")
    return "\n".join(lines)

def ops_to_wasm_stub(ops):
    # produce a minimal WebAssembly text stub describing ops
    lines = [";; WASM stub - representation only"]
    for i, (op, arg) in enumerate(ops):
        lines.append(f";; {i:04}: {op} {arg!r}")
    return "\n".join(lines)

# ---------------------------
# Integration helpers (callable from REPL / scripts)
# ---------------------------
def optimize_and_serialize_program(src: str, compress: bool = True):
    prog = parse_program_from_source(src)
    pm = PassManager([ConstantFoldingPass(), DeadCodeElimPass()])
    prog = pm.run(prog)
    vm = ExecVMPro()
    ops = vm.compile_program(prog)
    ops = peephole_optimize_ops(ops)
    if compress:
        return serialize_ops_to_bytes(ops)
    return repr(ops)

def demo_base12(n):
    return to_base12(n)

# End of advanced toolset

# ---------------------------
# Editor/IDE features: automatic TOML loading, definitions, highlighting,
# tabbing/formatting, code correction, autocompletion, and prioritization
# ---------------------------
import re
from typing import Iterable

try:
    import toml as _toml  # optional dependency for robust TOML parsing
except Exception:
    _toml = None

def auto_load_toml_config(start_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Search upward from start_path (or CWD) for common TOML config files
    (sayit.toml, pyproject.toml). Parse and return a dict. Falls back to a
    tiny key=value parser if toml library is not available.
    """
    if start_path is None:
        start_path = os.getcwd()
    cur = pathlib.Path(start_path).resolve()
    candidates = ["sayit.toml", "pyproject.toml"]
    while True:
        for name in candidates:
            p = cur / name
            if p.is_file():
                try:
                    if _toml:
                        return _toml.load(str(p))
                    # fallback: very small toml-ish parser for [tool.sayit] or simple key = "value"
                    cfg: Dict[str, Any] = {}
                    with open(p, "r", encoding="utf-8") as f:
                        for ln in f:
                            ln = ln.strip()
                            if not ln or ln.startswith("#") or ln.startswith("["):
                                continue
                            if "=" in ln:
                                k, v = ln.split("=", 1)
                                k = k.strip()
                                v = v.strip().strip('"').strip("'")
                                cfg[k] = v
                    return cfg
                except Exception:
                    return {}
        if cur.parent == cur:
            break
        cur = cur.parent
    return {}

def build_definitions_from_program(program) -> Dict[str, Any]:
    """
    Walk AST program and collect simple definitions: assigned variable names,
    literal string names (if pattern like `string name = "..."` exists at parser level),
    and Print literal values. Returns a mapping name -> sample value/metadata.
    """
    defs: Dict[str, Any] = {}
    for st in getattr(program, "stmts", []):
        cls = st.__class__.__name__
        if cls == "Assign":
            defs[st.ident] = {"kind": "var", "sample": getattr(st.expr, "val", None)}
        elif cls == "Print":
            if hasattr(st.expr, "val"):
                # create an anonymous literal entry for quick lookup by value
                key = f"_lit_{str(st.expr.val)[:32]}"
                defs[key] = {"kind": "literal", "sample": st.expr.val}
        # other node types may carry definitions in future
    return defs

ANSI = {
    "reset": "\x1b[0m",
    "black": "\x1b[30m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "blue": "\x1b[34m",
    "magenta": "\x1b[35m",
    "cyan": "\x1b[36m",
    "white": "\x1b[37m",
    "bold": "\x1b[1m",
}

_TOKEN_COLOR = {
    "START": "magenta",
    "RITUAL": "magenta",
    "MAKE": "magenta",
    "WHILE": "magenta",
    "IF": "magenta",
    "ELIF": "magenta",
    "ELSE": "magenta",
    "FINALLY": "magenta",
    "PRINT": "blue",
    "STRING": "green",
    "NUMBER": "cyan",
    "IDENT": "white",
    "OP": "yellow",
    "LPAREN": "yellow",
    "RPAREN": "yellow",
    "LBRACK": "yellow",
    "RBRACK": "yellow",
    "DOTS": "yellow",
    "COLON": "yellow",
    "NEWLINE": None,
    "SKIP": None,
}

def syntax_highlight(source: str) -> str:
    """
    Return an ANSI-colored version of `source` using the project's lexer.
    Safe to call in terminals that support ANSI.
    """
    out = []
    try:
        for tok, lex in _tokenize(source):
            color = _TOKEN_COLOR.get(tok, "white")
            if color:
                out.append(ANSI[color] + lex + ANSI["reset"])
            else:
                out.append(lex)
        return "".join(out)
    except Exception:
        # fallback: escape and return raw
        return source

def normalize_tabbing(source: str, tab_size: int = 4, use_spaces: bool = True) -> str:
    """
    Normalize leading indentation to a consistent tab/space policy.
    use_spaces=True converts tabs to spaces; False converts groups of spaces to tabs.
    Also trims trailing whitespace but preserves blank lines.
    """
    lines = source.splitlines()
    out_lines = []
    for ln in lines:
        # count existing indentation in tabs/spaces
        leading = re.match(r'^[ \t]*', ln).group(0)
        rest = ln[len(leading):]
        # compute indent level in spaces
        spaces = leading.replace("\t", " " * tab_size).count(" ")
        indent_units = spaces // tab_size
        if use_spaces:
            new_indent = " " * (indent_units * tab_size)
        else:
            new_indent = "\t" * indent_units
        out_lines.append(new_indent + rest.rstrip())
    # preserve trailing newline if original had it
    trailing_newline = "\n" if source.endswith("\n") else ""
    return "\n".join(out_lines) + trailing_newline

def autocorrect_source(source: str) -> str:
    """
    Perform lightweight autocorrections:
    - balance parentheses/brackets/braces by appending missing closers
    - balance double quotes if odd count
    - ensure file ends with newline
    - ensure basic spacing around operators (naive; skips strings)
    """
    # 1) Balance quotes
    dq_count = source.count('"')
    if dq_count % 2 == 1:
        source = source + '"'

    # 2) Balance various brackets
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack = []
    for ch in source:
        if ch in pairs.keys():
            stack.append(ch)
        elif ch in pairs.values():
            if stack and pairs[stack[-1]] == ch:
                stack.pop()
            else:
                # unmatched closer - ignore (could remove but safer to leave)
                pass
    # append closers for remaining openers
    while stack:
        opener = stack.pop()
        source += pairs[opener]

    # 3) Ensure trailing newline
    if not source.endswith("\n"):
        source += "\n"

    # 4) Naive operator spacing outside quotes
    def _space_ops(line):
        if '"' in line:
            # skip complex lines that include strings
            return line
        # add spaces around common ops when adjacent to alphanum
        line = re.sub(r'(?P<a>\w)(?P<op>[+\-*/<>=])(?P<b>\w)', r'\g<a> \g<op> \g<b>', line)
        return line
    lines = [ _space_ops(ln) for ln in source.splitlines() ]
    return "\n".join(lines) + "\n"

def _extract_prefix_at(source: str, cursor: int) -> str:
    """
    Return the identifier-like prefix immediately left of cursor.
    """
    if cursor > len(source):
        cursor = len(source)
    left = source[:cursor]
    m = re.search(r'([A-Za-z_][A-Za-z0-9_]*)$', left)
    return m.group(1) if m else ""

def suggest_completions(source: str, cursor: int = None, max_suggestions: int = 12) -> List[Dict[str, Any]]:
    """
    Provide completion candidates based on current source and AST definitions,
    builtins and services. Returns list of dicts: {'name','kind','score'}.
    """
    if cursor is None:
        cursor = len(source)
    prefix = _extract_prefix_at(source, cursor)
    # parse program to collect definitions if possible
    candidates: Dict[str, Dict[str, Any]] = {}
    try:
        toks = list(_tokenize(source))
        parser = _Parser(toks)
        prog = parser.parse_program()
        defs = build_definitions_from_program(prog)
        for k, v in defs.items():
            candidates[k] = {"kind": v.get("kind"), "sample": v.get("sample")}
    except Exception:
        pass

    # builtin suggestions from ExecVM instance
    try:
        vm_tmp = ExecVM()
        for name in vm_tmp.builtins.keys():
            candidates[name] = {"kind": "builtin"}
        for svc_name, svc in vm_tmp.services.items():
            # if service is a dict, include its actions
            if isinstance(svc, dict):
                for action in svc.keys():
                    fullname = f"{svc_name}.{action}"
                    candidates[fullname] = {"kind": "service_action"}
            else:
                candidates[svc_name] = {"kind": "service"}
    except Exception:
        # ignore if ExecVM cannot be instantiated for some reason
        pass

    # also include some heuristics: keywords and operators
    for kw in ["While", "If", "Elif", "Else", "Start", "Ritual", "Make", "Finally", "print"]:
        candidates[kw] = {"kind": "keyword"}

    # build suggestion list and score
    suggestions = []
    for name, meta in candidates.items():
        score = 0
        lname = name.lower()
        lpref = prefix.lower()
        if lpref == "":
            score = 1
        elif lname.startswith(lpref):
            score = 100 + (len(lpref) * 5)
        elif lpref in lname:
            score = 50
        # boost builtins / service actions
        if meta.get("kind") == "builtin":
            score += 10
        if meta.get("kind") == "service_action":
            score += 5
        suggestions.append({"name": name, "kind": meta.get("kind"), "score": score})
    # sort and return top results
    suggestions.sort(key=lambda x: (-x["score"], x["name"]))
    return suggestions[:max_suggestions]

def prioritize_suggestions(suggestions: Iterable[Dict[str, Any]], context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Re-rank suggestions using contextual signals:
    - context may include recent identifiers, frequency map, and caret scope.
    - currently uses a simple heuristic combining score and recency.
    """
    if context is None:
        context = {}
    recent = context.get("recent", [])
    freq = context.get("freq", {})
    def score_fn(s):
        base = s.get("score", 0)
        name = s.get("name", "")
        bonus = freq.get(name, 0) * 10
        rec = 50 if name in recent else 0
        return -(base + bonus + rec)
    return sorted(list(suggestions), key=score_fn)

def editor_features_summary() -> Dict[str, Any]:
    """Return a short dictionary describing available editor features for integration."""
    return {
        "features": [
            "auto_toml_load",
            "definitions_extraction",
            "syntax_highlight",
            "tab_normalize",
            "autocorrect",
            "suggest_completions",
            "prioritize_suggestions",
        ],
        "notes": "Completions use AST + builtins + services. Autocorrect is conservative; review changes before saving."
    }

# sayc.py
"""
Enhanced sayc: CLI, engines, libraries, caching, watch mode, config, and templates.

New capabilities:
- Subcommands: run, build, repl, test, list-engines, make-engine
- Engine discovery: builtin 'vm' and 'ir', plus Python modules in ./engines or --engine-path
- Library loading (--libs), config file support (--config)
- Caching of emitted IR (.cache/<hash>.ll) with --force to bypass
- File watch mode (--watch) (simple polling)
- Verbose / dry-run / timeout / max-steps options
- Engine template generator (make-engine)
"""
from __future__ import annotations
import sys
import os
import argparse
import importlib.util
import importlib.machinery
import json
import time
import hashlib
from typing import List, Optional, Dict, Any

from say_lexer import tokenize
from say_parser import Parser
from say_vm import VM

# Keep Codegen import lazy so llvmlite is optional
# from say_codegen import Codegen


# ---------------------------
# Utilities
# ---------------------------
def load_source_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def parse_program_from_source(src: str):
    toks = tokenize(src)
    parser = Parser(toks)
    return parser.parse_program()

def sha1_of_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def ensure_cache_dir() -> str:
    d = os.path.join(".cache")
    os.makedirs(d, exist_ok=True)
    return d

def write_cache(key: str, data: str) -> str:
    d = ensure_cache_dir()
    path = os.path.join(d, f"{key}.ll")
    with open(path, "w", encoding="utf-8") as f:
        f.write(data)
    return path

def read_cache(key: str) -> Optional[str]:
    d = os.path.join(".cache")
    path = os.path.join(d, f"{key}.ll")
    if os.path.isfile(path):
        return load_source_file(path)
    return None

def verbose_print(verbose: bool, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


# ---------------------------
# Engine loading / discovery
# ---------------------------
ENGINES_DIR = os.path.join(os.path.dirname(__file__), "engines")

def discover_local_engines() -> Dict[str, str]:
    """Return mapping engine_name -> file_path for ./engines/*.py"""
    result = {}
    if not os.path.isdir(ENGINES_DIR):
        return result
    for fn in os.listdir(ENGINES_DIR):
        if fn.endswith(".py") and not fn.startswith("_"):
            name = os.path.splitext(fn)[0]
            result[name] = os.path.join(ENGINES_DIR, fn)
    return result

def load_engine_from_path(path: str):
    """Dynamically import Python module from path and return it."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Engine module not found: {path}")
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import engine module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def resolve_engine(engine: Optional[str], engine_path: Optional[str], verbose: bool = False):
    """
    Resolve engine selection:
      - builtin: 'vm' or 'ir'
      - discovered local engine name (in ./engines)
      - engine_path: direct path to a Python module
    Returns a callable runner object with either `run(program, args)` or `emit_ir(program, out)`.
    """
    local = discover_local_engines()
    verbose_print(verbose, "discovered local engines:", local)
    if engine == "vm" or engine is None:
        return {"type": "builtin", "name": "vm"}
    if engine == "ir":
        return {"type": "builtin", "name": "ir"}
    # engine may be name available in local
    if engine in local:
        mod = load_engine_from_path(local[engine])
        return {"type": "module", "module": mod}
    # engine as path
    if engine_path:
        mod = load_engine_from_path(engine_path)
        return {"type": "module", "module": mod}
    # fallback: if engine looks like a path
    if engine and os.path.isfile(engine):
        mod = load_engine_from_path(engine)
        return {"type": "module", "module": mod}
    raise ValueError(f"Unknown engine '{engine}' (no local module or path found)")

# ---------------------------
# Core behavior
# ---------------------------
def compile_to_ir_and_maybe_cache(program, out_filename: Optional[str], force: bool, verbose: bool):
    """Emit IR using say_codegen.Codegen (lazy import). Cache by program hash."""
    try:
        from say_codegen import Codegen
    except Exception as e:
        raise RuntimeError("Codegen (llvmlite) is required for IR emission: " + str(e))

    # produce canonical program text for hashing: join stringified stmts minimally
    program_text = json.dumps([str(type(s).__name__) + ":" + getattr(s, "expr", getattr(s, "ident", "")).__class__.__name__ if hasattr(s, "expr") else type(s).__name__ for s in program.stmts])
    key = sha1_of_text(program_text)
    cached = None if force else read_cache(key)
    if cached:
        verbose_print(verbose, f"[cache] Found cached IR for key {key}")
        if out_filename:
            with open(out_filename, "w", encoding="utf-8") as f:
                f.write(cached)
            return out_filename
        return None  # caller can print cached if desired

    cg = Codegen()
    cg.emit_main()
    # simple emitter: Print statements and basic Make/Start/While helpers are supported in Codegen module in repo
    for stmt in program.stmts:
        cls = stmt.__class__.__name__
        if cls == "Print":
            if hasattr(stmt.expr, "val"):
                cg.emit_print(str(stmt.expr.val))
            else:
                cg.emit_print(f"<{stmt.expr.name}>")
        else:
            # leave other statements to specialized codegen helpers in say_codegen (if present)
            # noop here to keep compatibility
            pass

    llvm_ir = cg.finish()
    cache_path = write_cache(key, llvm_ir)
    verbose_print(verbose, f"[cache] Wrote IR to {cache_path}")
    if out_filename:
        with open(out_filename, "w", encoding="utf-8") as f:
            f.write(llvm_ir)
        return out_filename
    else:
        print(llvm_ir)
        return None

def run_with_vm(program, args):
    vm = VM(max_steps=getattr(args, "max_steps", 1000000))
    vm.run(program)
    return vm

def run_with_module_engine(mod, program, args):
    """Call engine module using run(program, args) or emit_ir(program,out)."""
    if hasattr(mod, "run"):
        return mod.run(program, args)
    if hasattr(mod, "emit_ir"):
        return mod.emit_ir(program, getattr(args, "out", None))
    if hasattr(mod, "main"):
        # try convenient main(program, args)
        try:
            return mod.main(program, args)
        except TypeError:
            return mod.main(program)
    raise RuntimeError("Engine module does not expose run/emit_ir/main API")

# ---------------------------
# CLI: subcommands
# ---------------------------
def make_engine_template(path: str):
    tmpl = """# Example Sayit engine module
def run(program, args):
    \"""
    Minimal engine: receives parsed AST `program` and parsed `args`.
    Implement execution or transformation here.
    \"""
    print("[engine] run called - implement execution")
    for s in program.stmts:
        print("stmt:", type(s).__name__)
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(tmpl)
    return path

def run_tests_in_dir(tests_dir: str, args) -> int:
    """
    Discover .say files in tests_dir, run each under VM and report pass/fail.
    A test file may contain `# expect: <text>` comment lines; this runner does a simple stdout match.
    """
    import subprocess, tempfile
    passed = 0
    total = 0
    if not os.path.isdir(tests_dir):
        print(f"[test] tests dir not found: {tests_dir}")
        return 1
    for fn in sorted(os.listdir(tests_dir)):
        if not fn.endswith(".say"):
            continue
        total += 1
        path = os.path.join(tests_dir, fn)
        src = load_source_file(path)
        prog = parse_program_from_source(src)
        # run in a fresh VM and capture stdout using subprocess running this script for isolation
        cmd = [sys.executable, __file__, path]  # default behavior run using VM
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=getattr(args, "timeout", 5))
            out_text = out.decode("utf-8").strip()
            # look for expectations in comments
            expects = []
            for line in src.splitlines():
                if "# expect:" in line:
                    expects.append(line.split("# expect:", 1)[1].strip())
            ok = all(e in out_text for e in expects) if expects else True
            if ok:
                passed += 1
                print(f"[test] {fn}: PASS")
            else:
                print(f"[test] {fn}: FAIL\n  output:\n{out_text}\n  expects: {expects}")
        except Exception as e:
            print(f"[test] {fn}: ERROR: {e}")
    print(f"[test] {passed}/{total} passed")
    return 0 if passed == total else 2

def watch_and_rebuild(path: str, build_cb, poll: float = 0.5, verbose: bool = False):
    """Simple polling file watcher: call build_cb() on changes to `path` (file or directory)."""
    last_snapshot = {}
    def snapshot():
        info = {}
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for fn in files:
                    p = os.path.join(root, fn)
                    try:
                        info[p] = os.path.getmtime(p)
                    except OSError:
                        info[p] = 0
        else:
            info[path] = os.path.getmtime(path) if os.path.exists(path) else 0
        return info
    last_snapshot = snapshot()
    try:
        while True:
            time.sleep(poll)
            cur = snapshot()
            if cur != last_snapshot:
                verbose_print(verbose, "[watch] change detected, rebuilding...")
                try:
                    build_cb()
                except Exception as e:
                    print("[watch] build error:", e)
                last_snapshot = cur
    except KeyboardInterrupt:
        verbose_print(verbose, "[watch] stopped by user")

# ---------------------------
# Entrypoint
# ---------------------------
def build_and_dispatch(program, args):
    # Choose engine resolution
    resolved = resolve_engine(getattr(args, "engine", None), getattr(args, "engine_path", None), verbose=getattr(args, "verbose", False))
    if resolved["type"] == "builtin":
        if resolved["name"] == "vm":
            return run_with_vm(program, args)
        if resolved["name"] == "ir":
            return compile_to_ir_and_maybe_cache(program, getattr(args, "out", None), getattr(args, "force", False), getattr(args, "verbose", False))
    elif resolved["type"] == "module":
        return run_with_module_engine(resolved["module"], program, args)
    else:
        raise RuntimeError("Unsupported engine resolution result")

def load_config(config_path: Optional[str], verbose: bool):
    if not config_path:
        # look for sayit.json or sayit.toml (toml optional)
        if os.path.isfile("sayit.json"):
            config_path = "sayit.json"
        elif os.path.isfile("sayit.toml"):
            config_path = "sayit.toml"
    if not config_path:
        return {}
    if config_path.endswith(".json"):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            verbose_print(verbose, f"[config] loaded {config_path}")
            return cfg
    # optional toml
    try:
        import toml
        cfg = toml.load(config_path)
        verbose_print(verbose, f"[config] loaded {config_path}")
        return cfg
    except Exception:
        print(f"[config] Unsupported config format or toml not installed: {config_path}")
        return {}

def main(argv: Optional[List[str]] = None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(prog="sayc", description="Sayit multi-engine tool")
    sub = parser.add_subparsers(dest="cmd", help="subcommands")

    # run / default
    p_run = sub.add_parser("run", help="Run a .say file (default)")
    p_run.add_argument("file", nargs="?", help="Source file to run")
    p_run.add_argument("--libs", nargs="+", help="Library .say files to load first")
    p_run.add_argument("--engine", choices=["vm", "ir"] + list(discover_local_engines().keys()), help="Engine to use")
    p_run.add_argument("--engine-path", help="Path to a custom engine module")
    p_run.add_argument("--out", help="Output file for IR (when using ir engine)")
    p_run.add_argument("--force", action="store_true", help="Bypass cache when building IR")
    p_run.add_argument("--watch", action="store_true", help="Watch file/dir and rebuild on changes")
    p_run.add_argument("--verbose", action="store_true", help="Verbose logging")
    p_run.add_argument("--max-steps", type=int, default=1000000, help="VM instruction limit")
    p_run.add_argument("--config", help="Config file (json/toml)")

    # build (emit IR)
    p_build = sub.add_parser("build", help="Emit LLVM IR for a .say file")
    p_build.add_argument("file")
    p_build.add_argument("--out", help="Write IR to file")
    p_build.add_argument("--force", action="store_true")
    p_build.add_argument("--verbose", action="store_true")

    # repl
    p_repl = sub.add_parser("repl", help="Start REPL using VM engine")
    p_repl.add_argument("--verbose", action="store_true")

    # test
    p_test = sub.add_parser("test", help="Run tests in tests/ directory")
    p_test.add_argument("--tests-dir", default="tests")
    p_test.add_argument("--timeout", type=int, default=5)
    p_test.add_argument("--verbose", action="store_true")

    # list engines
    p_list = sub.add_parser("list-engines", help="List available engines")

    # make engine template
    p_tmpl = sub.add_parser("make-engine", help="Create an engine template file")
    p_tmpl.add_argument("path", nargs="?", default=os.path.join("engines", "example_engine.py"))

    args = parser.parse_args(argv)

    # If user invoked no subcommand, default to run
    cmd = args.cmd or "run"

    # Load config early
    cfg = load_config(getattr(args, "config", None), getattr(args, "verbose", False))

    if cmd == "list-engines":
        local = discover_local_engines()
        print("builtin: vm, ir")
        for k, p in local.items():
            print(f"local: {k} -> {p}")
        return

    if cmd == "make-engine":
        path = args.path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        made = make_engine_template(path)
        print(f"[sayc] Engine template written to {made}")
        return

    if cmd == "test":
        code = run_tests_in_dir(args.tests_dir, args)
        return code

    if cmd == "repl":
        # simple REPL using VM
        from say_vm import repl
        return repl(trace=getattr(args, "verbose", False))

    # For run/build, require file
    file_arg = getattr(args, "file", None)
    if not file_arg:
        print("No source file provided.")
        parser.print_help()
        return

    if not os.path.isfile(file_arg):
        print(f"File not found: {file_arg}")
        return

    src = load_source_file(file_arg)
    program = parse_program_from_source(src)

    # libs
    if getattr(args, "libs", None):
        lib_stmts = []
        for p in args.libs:
            if not os.path.isfile(p):
                print(f"[sayc] Library file not found: {p}")
                return
            lib_src = load_source_file(p)
            lib_prog = parse_program_from_source(lib_src)
            lib_stmts.extend(lib_prog.stmts)
        program.stmts = lib_stmts + program.stmts
        verbose_print(getattr(args, "verbose", False), f"[sayc] loaded {len(args.libs)} libraries")

    # build callback used by watch
    def build_cb():
        verbose_print(getattr(args, "verbose", False), f"[sayc] building {file_arg} ...")
        return build_and_dispatch(program, args)

    if getattr(args, "watch", False):
        watch_and_rebuild(file_arg if os.path.isfile(file_arg) else os.path.dirname(file_arg), build_cb, verbose=bool(getattr(args, "verbose", False)))
        return

    # dispatch build/run
    try:
        res = build_and_dispatch(program, args)
        return res
    except Exception as e:
        print("[sayc] Error:", e)
        return 1

if __name__ == "__main__":
    main()

    # ---------------------------
    # End of sayc.py
    # ---------------------------

    # system.py
    """
    Core system utilities for Sayit: safe eval, base-12 numerics, advanced VM subclass,
    optimization passes, and editor/IDE features (syntax highlight, autocorrect, completions).
    """
    import os
    import pathlib
    from typing import Any, List, Optional, Dict, Tuple
    from say_parser import Parser, _tokenize
    from say_vm import ExecVM, Program
    from say_optimizer import Pass, ConstantFoldingPass, DeadCodeElimPass, peephole_optimize_ops, simple_loop_unroll, stack_to_register_ops, serialize_ops_to_bytes
    import re
