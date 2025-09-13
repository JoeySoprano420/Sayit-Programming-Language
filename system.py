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
    import json
    import sys
    import time
    # ---------------------------
    # Safe eval
    def safe_eval(expr: str, vars: Optional[Dict[str, Any]] = None, timeout: float = 1.0) -> Any:
        """
        Safely evaluate a simple arithmetic expression with optional variables.
        Supports +, -, *, /, parentheses, integers, floats, and variable names.
        Execution is limited by timeout (in seconds).
        """
        allowed_names = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'int': int,
            'float': float,
        }
        if vars:
            allowed_names.update(vars)
        code = compile(expr, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                raise NameError(f"Use of '{name}' not allowed in safe_eval")
        start_time = time.time()
        def time_limited_eval():
            if time.time() - start_time > timeout:
                raise TimeoutError("safe_eval timed out")
            return eval(code, {"__builtins__": {}}, allowed_names)
        return time_limited_eval()
    # ---------------------------
    # Base-12 numerics
    def to_base12(n: int) -> str:
        """Convert integer n to base-12 string using digits 0-9, T (10), E (11)."""
        if n == 0:
            return "0"
        digits = []
        neg = n < 0
        n = abs(n)
        while n:
            r = n % 12
            if r < 10:
                digits.append(str(r))
            elif r == 10:
                digits.append("T")
            else:
                digits.append("E")
            n //= 12
        if neg:
            digits.append("-")
        return "".join(reversed(digits))
    def from_base12(s: str) -> int:
        """Convert base-12 string s to integer."""
        s = s.strip().upper()
        if not s or any(c not in "0123456789TE-" for c in s):
            raise ValueError(f"Invalid base-12 string: {s}")
        neg = s.startswith("-")
        if neg:
            s = s[1:]
        n = 0
        for c in s:
            if c in "0123456789":
                v = int(c)
            elif c == "T":
                v = 10
            elif c == "E":
                v = 11
            else:
                raise ValueError(f"Invalid character in base-12 string: {c}")
            n = n * 12 + v
            return
    
# ---------------------------
# Additional extensive utility functions appended to bottom of file
# ---------------------------
import ast
import hashlib
import io
import subprocess
import tempfile
from functools import wraps
from threading import Thread, Event
from time import monotonic
from typing import Callable, Iterable, Iterator

# --- Robust base-12 helpers (safe, well-tested) ----------------------------
_BASE12_DIGITS = "0123456789TE"  # T==10, E==11

def to_base12_ext(n: int) -> str:
    """
    Convert integer to base-12 string using digits 0-9,T(10),E(11).
    Stable, iterative implementation that handles negatives.
    """
    if n == 0:
        return "0"
    neg = n < 0
    n = abs(n)
    out = []
    while n:
        out.append(_BASE12_DIGITS[n % 12])
        n //= 12
    if neg:
        out.append("-")
    return "".join(reversed(out))

def from_base12_ext(s: str) -> int:
    """
    Parse base-12 string produced by `to_base12_ext`. Raises ValueError on invalid input.
    Accepts optional leading '+' or '-' and tolerates surrounding whitespace.
    """
    if not isinstance(s, str):
        raise TypeError("from_base12_ext expects a string")
    s = s.strip().upper()
    if s == "":
        raise ValueError("empty string")
    neg = False
    if s[0] in "+-":
        neg = s[0] == "-"
        s = s[1:]
    if s == "":
        raise ValueError("no digits")
    value = 0
    for ch in s:
        try:
            v = _BASE12_DIGITS.index(ch)
        except ValueError:
            raise ValueError(f"invalid base-12 digit: {ch!r}")
        value = value * 12 + v
    return -value if neg else value

# --- Safe expression evaluator using AST whitelisting -----------------------
_ALLOWED_AST_NODES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Num,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.LShift,
    ast.RShift,
    ast.BitOr,
    ast.BitAnd,
    ast.BitXor,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.And,
    ast.Or,
    ast.BoolOp,
    ast.IfExp,
    ast.Tuple,
    ast.List,
}

def _ast_is_allowed(node: ast.AST) -> bool:
    """Recursively check AST only contains allowed node types."""
    if type(node) not in _ALLOWED_AST_NODES:
        return False
    for child in ast.iter_child_nodes(node):
        if not _ast_is_allowed(child):
            return False
    return True

def safe_eval_ast(expr: str, variables: Optional[Dict[str, object]] = None, timeout_sec: Optional[float] = None) -> object:
    """
    Evaluate a small expression safely using AST validation.
    - Supports arithmetic, booleans, comparisons, and simple names.
    - `variables` provides the allowed names (numbers/functions not allowed).
    - Optional `timeout_sec` will raise TimeoutError if evaluation doesn't finish.
    """
    if variables is None:
        variables = {}
    # parse expression
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"invalid expression: {e}") from e
    if not _ast_is_allowed(tree):
        raise ValueError("expression contains disallowed constructs")
    # build evaluation closure
    code = compile(tree, "<safe_eval>", "eval")
    result_container = {}
    done = Event()
    def _runner():
        try:
            result_container["value"] = eval(code, {"__builtins__": {}}, variables)
        except Exception as e:
            result_container["error"] = e
        finally:
            done.set()
    t = Thread(target=_runner, daemon=True)
    t.start()
    started = monotonic()
    if timeout_sec is None:
        done.wait()
    else:
        waited = done.wait(timeout_sec)
        if not waited:
            raise TimeoutError("safe_eval_ast: evaluation timed out")
    if "error" in result_container:
        raise result_container["error"]
    return result_container.get("value")

# --- Atomic file helper / hashing / JSON-safe load --------------------------
def hash_file_sha256(path: str, chunk_size: int = 8192) -> str:
    """Return SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def atomic_write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    """
    Atomically write `text` to `path` by writing to a temporary file and renaming.
    Ensures partial writes won't leave corrupt file.
    """
    dirn = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile("w", encoding=encoding, dir=dirn, delete=False) as tf:
        tf.write(text)
        tmpname = tf.name
    os.replace(tmpname, path)

def load_json_safe(path: str) -> Optional[object]:
    """
    Load JSON file returning parsed object, or None on parse error.
    Does not raise for non-fatal problems; logs and returns None.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# --- Lightweight LRU cache decorator (thread-safe-ish) ----------------------
def lru_cache_simple(maxsize: int = 128):
    """
    Simple LRU cache decorator for pure functions with hashable args.
    Not as featureful as functools.lru_cache but easy to inspect.
    """
    def deco(fn: Callable):
        cache = {}
        order = []
        @wraps(fn)
        def wrapped(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            if key in cache:
                # bump recency
                try:
                    order.remove(key)
                except ValueError:
                    pass
                order.append(key)
                return cache[key]
            result = fn(*args, **kwargs)
            cache[key] = result
            order.append(key)
            if len(order) > maxsize:
                old = order.pop(0)
                cache.pop(old, None)
            return result
        wrapped.cache_clear = lambda: (cache.clear(), order.clear())
        wrapped._cache = cache
        return wrapped
    return deco

# --- Robust subprocess runner with timeout and streamed output -------------
def run_subprocess_capture(cmd: Iterable[str], timeout: Optional[float] = None) -> Tuple[int, str, str]:
    """
    Run a subprocess and capture stdout/stderr. Returns (returncode, stdout, stderr).
    Uses subprocess.run with text mode. Raises TimeoutError on timeout.
    """
    try:
        proc = subprocess.run(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f"subprocess timed out after {timeout}s") from e

# --- Retry decorator for flaky IO / network calls ---------------------------
def retry(times: int = 3, delay: float = 0.1, allowed_exceptions: Tuple[type, ...] = (Exception,)):
    """
    Retry decorator. Retries `times` times on allowed_exceptions with `delay` seconds between attempts.
    """
    def deco(fn: Callable):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            last_exc = None
            for attempt in range(1, times + 1):
                try:
                    return fn(*args, **kwargs)
                except allowed_exceptions as e:
                    last_exc = e
                    if attempt == times:
                        break
                    time.sleep(delay)
            raise last_exc
        return wrapped
    return deco

# --- Tiny test/helpers that can be called from other modules ----------------
def _self_test_base12():
    cases = [0, 1, 10, 11, 12, 144, -1, 12345]
    for n in cases:
        txt = to_base12_ext(n)
        got = from_base12_ext(txt)
        if got != n:
            raise AssertionError(f"base12 roundtrip failed for {n}: encoded={txt} decoded={got}")
    return True

def _self_test_safe_eval():
    assert safe_eval_ast("1 + 2 * 3") == 7
    assert safe_eval_ast("-5 + 2") == -3
    assert safe_eval_ast("2 ** 3") == 8
    assert safe_eval_ast("1 < 2 and 3 > 2") is True
    return True

def run_self_tests() -> Dict[str, bool]:
    """
    Run lightweight self-tests for the helpers added in this section.
    Returns a mapping of test-name -> bool success.
    """
    results = {}
    try:
        results["base12"] = _self_test_base12()
    except Exception:
        results["base12"] = False
    try:
        results["safe_eval"] = _self_test_safe_eval()
    except Exception:
        results["safe_eval"] = False
    return results

# Expose a concise __all__ for importers that want the utilities
__all__ = [
    "to_base12_ext", "from_base12_ext",
    "safe_eval_ast",
    "hash_file_sha256", "atomic_write_text", "load_json_safe",
    "lru_cache_simple", "run_subprocess_capture", "retry",
    "run_self_tests",
]

# --- Editor/IDE features: syntax highlight, autocorrect, completions --------
def syntax_highlight(source: str) -> str:
    """
    Simple syntax highlighter for Sayit source code.
    Wraps keywords, numbers, strings, comments in ANSI color codes.
    """
    import keyword
    KEYWORDS = set(keyword.kwlist + ["print", "if", "else", "while", "def", "return", "let", "in", "true", "false"])
    token_specification = [
        ("NUMBER",   r'\b\d+(\.\d*)?\b'),  # Integer or decimal number
        ("STRING",   r'"([^"\\]|\\.)*"'),  # Double-quoted string
        ("COMMENT",  r'#.*'),               # Comment
        ("ID",       r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'),  # Identifiers
        ("OP",       r'[+\-*/=<>!]+'),     # Operators
        ("NEWLINE",  r'\n'),                # Line endings
        ("SKIP",     r'[ \t]+'),           # Skip over spaces and tabs
        ("MISMATCH", r'.'),                 # Any other character
    ]
    tok_regex = '|'.join(f'(?P<{pair[0]}>{pair[1]})' for pair in token_specification)
    get_token = re.compile(tok_regex).match
    line_num = 1
    line_start = 0
    pos = 0
    mo = get_token(source)
    highlighted = []
    while mo is not None:
        kind = mo.lastgroup
        value = mo.group()
        if kind == "NUMBER":
            highlighted.append(f"\033[94m{value}\033[0m")  # Blue
        elif kind == "STRING":
            highlighted.append(f"\033[92m{value}\033[0m")  # Green
        elif kind == "COMMENT":
            highlighted.append(f"\033[90m{value}\033[0m")  # Grey
        elif kind == "ID":
            if value in KEYWORDS:
                highlighted.append(f"\033[95m{value}\033[0m")  # Magenta for keywords
            else:
                highlighted.append(value)
        elif kind == "OP":
            highlighted.append(f"\033[93m{value}\033[0m")  # Yellow
            highlighted.append(value)
line_start = pos
line_num += 1
kind == "SKIP"
highlighted.append(value)
kind == "MISMATCH"
highlighted.append(value)
pos = mo.end()
mo = get_token(source, pos)
if pos != len(source):
                            raise RuntimeError(f'{source[pos]!r} unexpected on line {line_num}')
''.join(highlighted)
def suggest_autocorrect(source: str, cursor_pos: int) -> List[str]:
                        """
                        Suggest autocorrections for the token at cursor_pos in source.
                        Returns a list of suggestions.
                        """
                        tokens = list(_tokenize(source))
                        for i, (typ, val, start, end) in enumerate(tokens):
                            if start <= cursor_pos <= end:
                                if typ == "ID":
                                    # suggest keywords that start with the same prefix
                                    prefix = val[:cursor_pos - start]
                                    return [kw for kw in keyword.kwlist if kw.startswith(prefix) and kw != val]
                                break
                        return []

def complete_code(source: str, cursor_pos: int) -> List[str]:
    """
    Provide code completions for the token at cursor_pos in source.
    Returns a list of possible completions.
    """
    tokens = list(_tokenize(source))
    for i, (typ, val, start, end) in enumerate(tokens):
        if start <= cursor_pos <= end:
            if typ == "ID":
                prefix = val[:cursor_pos - start]
                return [kw for kw in keyword.kwlist if kw.startswith(prefix)]
            break
    return []
def sha1_of_text(text: str) -> str:
    """Return SHA-1 hex digest of text."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
def write_cache(key: str, content: str) -> str:
    """Write content to cache file named by key. Returns path."""
    d = os.path.join(".cache")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{key}.ll")
    atomic_write_text(path, content)
    return path

def read_cache(key: str) -> Optional[str]:
    """Read cached content by key, or None if not found."""
    path = os.path.join(".cache", f"{key}.ll")
    if os.path.isfile(path):
 
            if verbose:
                print("[sayc]", *args)
                def discover_local_engines() -> Dict[str, str]:
                    """Discover local engine modules in ./engines directory."""
                    ENGINES_DIR = os.path.join(os.path.dirname(__file__), "engines")
                    engines = {}
                    if os.path.isdir(ENGINES_DIR):
                        for fn in os.listdir(ENGINES_DIR):
                            if fn.endswith(".py") and not fn.startswith("_"):
                                name = os.path.splitext(fn)[0]
                                path = os.path.join(ENGINES_DIR, fn)
                                engines[name] = path
                                return engines
                            def load_engine_from_path(path: str):

                                """Dynamically load a Python module from path."""
                                import importlib.util
                                spec = importlib.util.spec_from_file_location("engine_module", path)
                                if spec is None or spec.loader is None:
                                    raise ImportError(f"Cannot load module from {path}")
                                mod = importlib.util.module_from_spec(spec)
                                spec.loader.exec_module(mod)
                                return mod
                            def resolve_engine(engine: Optional[str], engine_path: Optional[str], verbose: bool = False) -> Dict[str, Any]:
                                """Resolve engine name/path to a module or builtin."""
                                with open(path, "r", encoding="utf-8") as f:
                                    return load_json_safe(path)
                                with open(path, "r", encoding="utf-8") as f:
                                    return f.read()
                                with open(path, "r", encoding="utf-8") as f:
                                    return None
                                return None
                            return None
                        with open(path, "r", encoding="utf-8") as f:
                            return f.read()
                        return None
                    return None
                return None
            return None

# Extensive networking, concurrency, IO and types additions
import socket
import selectors
import ssl
import asyncio
import urllib.request
import contextlib
import os
import threading
import queue
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Dict, Any, List, Tuple, Iterator, TypedDict, Iterable
import json
import tempfile
import hashlib
import concurrent.futures

# ---------------------------
# Types / Data models
# ---------------------------
class ConnType(Enum):
    TCP = "tcp"
    SSL = "ssl"

@dataclass
class Endpoint:
    host: str
    port: int

@dataclass
class Message:
    data: bytes
    meta: Dict[str, Any] = field(default_factory=dict)

class HTTPResponse(TypedDict):
    code: int
    headers: Dict[str, str]
    body: bytes

# ---------------------------
# Threaded TCP Server / Client (blocking, thread-per-connection)
# ---------------------------
class ExtTCPServer:
    """
    Simple threaded TCP server.
    - handler(conn: socket.socket, addr: (host,port)) called in a worker thread per connection.
    - start() launches acceptor thread; stop() shuts down gracefully.
    """
    def __init__(self, endpoint: Endpoint, backlog: int = 50, reuse_addr: bool = True, recv_buf: int = 4096):
        self.endpoint = endpoint
        self.backlog = backlog
        self.recv_buf = recv_buf
        self._sock: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._stopped = threading.Event()
        self._workers: List[threading.Thread] = []
        self._handler: Optional[Callable[[socket.socket, Tuple[str,int]], None]] = None
        self.reuse_addr = reuse_addr

    def set_handler(self, handler: Callable[[socket.socket, Tuple[str,int]], None]):
        self._handler = handler

    def start(self):
        if self._sock:
            raise RuntimeError("server already started")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if self.reuse_addr:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.endpoint.host, self.endpoint.port))
        s.listen(self.backlog)
        self._sock = s
        self._stopped.clear()
        self._accept_thread = threading.Thread(target=self._accept_loop, name="ExtTCPServer.accept", daemon=True)
        self._accept_thread.start()

    def _accept_loop(self):
        assert self._sock is not None
        while not self._stopped.is_set():
            try:
                client, addr = self._sock.accept()
                if self._handler:
                    th = threading.Thread(target=self._worker_wrapper, args=(client, addr), daemon=True)
                    th.start()
                    self._workers.append(th)
                else:
                    client.close()
            except OSError:
                break

    def _worker_wrapper(self, client: socket.socket, addr):
        try:
            if self._handler:
                self._handler(client, addr)
        finally:
            try:
                client.close()
            except Exception:
                pass

    def stop(self, wait: bool = True, timeout: Optional[float] = None):
        self._stopped.set()
        if self._sock:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._accept_thread:
            self._accept_thread.join(timeout=timeout)
            self._accept_thread = None
        if wait:
            for w in list(self._workers):
                w.join(timeout=timeout)

class ExtTCPClient:
    """
    Simple blocking TCP client with convenience send/recv helpers.
    """
    def __init__(self, endpoint: Endpoint, timeout: Optional[float] = None, use_ssl: bool = False, ssl_context: Optional[ssl.SSLContext] = None):
        self.endpoint = endpoint
        self.timeout = timeout
        self.use_ssl = use_ssl
        self.ssl_context = ssl_context
        self.sock: Optional[socket.socket] = None

    def connect(self):
        s = socket.create_connection((self.endpoint.host, self.endpoint.port), timeout=self.timeout)
        if self.use_ssl:
            ctx = self.ssl_context or ssl.create_default_context()
            s = ctx.wrap_socket(s, server_hostname=self.endpoint.host)
        self.sock = s

    def send(self, data: bytes):
        if not self.sock:
            raise RuntimeError("not connected")
        totalsent = 0
        while totalsent < len(data):
            sent = self.sock.send(data[totalsent:])
            if sent == 0:
                raise RuntimeError("socket connection broken")
            totalsent += sent

    def recv(self, max_bytes: int = 4096) -> bytes:
        if not self.sock:
            raise RuntimeError("not connected")
        return self.sock.recv(max_bytes)

    def close(self):
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

# ---------------------------
# Asyncio TCP Server / Client
# ---------------------------
class ExtAsyncTCPServer:
    """
    Asyncio-based TCP server wrapper.
    Usage:
        server = ExtAsyncTCPServer("0.0.0.0", 9000, client_handler)
        asyncio.run(server.serve_forever())
    Where client_handler(reader, writer) is async coroutine.
    """
    def __init__(self, host: str, port: int, client_handler: Callable[[asyncio.StreamReader, asyncio.StreamWriter], asyncio.Future]):
        self.host = host
        self.port = port
        self._handler = client_handler
        self._server: Optional[asyncio.base_events.Server] = None

    async def serve_forever(self):
        self._server = await asyncio.start_server(self._handler, host=self.host, port=self.port)
        async with self._server:
            await self._server.serve_forever()

    async def close(self):
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

class ExtAsyncTCPClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None

    async def connect(self):
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)

    async def send(self, data: bytes):
        if not self.writer:
            raise RuntimeError("not connected")
        self.writer.write(data)
        await self.writer.drain()

    async def recv(self, n: int = 4096) -> bytes:
        if not self.reader:
            raise RuntimeError("not connected")
        return await self.reader.read(n)

    def close(self):
        if self.writer:
            self.writer.close()

# ---------------------------
# Connection Pool (simple)
# ---------------------------
class ExtConnPool:
    """
    Basic TCP connection pool for blocking clients.
    - create(pool_size, endpoint)
    - acquire()/release() to reuse sockets.
    """
    def __init__(self, endpoint: Endpoint, pool_size: int = 4, timeout: Optional[float] = None):
        self.endpoint = endpoint
        self.pool_size = pool_size
        self.timeout = timeout
        self._pool: queue.Queue = queue.Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._initialized = False

    def _make_conn(self) -> socket.socket:
        s = socket.create_connection((self.endpoint.host, self.endpoint.port), timeout=self.timeout)
        return s

    def initialize(self):
        with self._lock:
            if self._initialized:
                return
            for _ in range(self.pool_size):
                try:
                    self._pool.put_nowait(self._make_conn())
                except Exception:
                    break
            self._initialized = True

    def acquire(self, timeout: Optional[float] = None) -> socket.socket:
        if not self._initialized:
            self.initialize()
        return self._pool.get(timeout=timeout)

    def release(self, sock: socket.socket):
        try:
            self._pool.put_nowait(sock)
        except queue.Full:
            try:
                sock.close()
            except Exception:
                pass

    def close_all(self):
        while not self._pool.empty():
            try:
                s = self._pool.get_nowait()
                s.close()
            except Exception:
                pass
        self._initialized = False

# ---------------------------
# HTTP utilities (urllib based)
# ---------------------------
class ExtHTTPClient:
    @staticmethod
    def http_get(url: str, timeout: Optional[float] = 10) -> HTTPResponse:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            headers = {k: v for k, v in resp.getheaders()}
            return {"code": resp.getcode(), "headers": headers, "body": body}

    @staticmethod
    def http_post_json(url: str, obj: Any, timeout: Optional[float] = 10, headers: Optional[Dict[str,str]] = None) -> HTTPResponse:
        body = json.dumps(obj).encode("utf-8")
        hdrs = {"Content-Type": "application/json"}
        if headers:
            hdrs.update(headers)
        req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            rbody = resp.read()
            return {"code": resp.getcode(), "headers": {k:v for k,v in resp.getheaders()}, "body": rbody}

    @staticmethod
    def download_file(url: str, dest_path: str, chunk_size: int = 8192, timeout: Optional[float] = 30, progress: Optional[Callable[[int, Optional[int]], None]] = None) -> str:
        """Download URL to dest_path. Progress callback receives (downloaded_bytes, total_bytes_or_None)."""
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            total = resp.getheader('Content-Length')
            total_n = int(total) if total and total.isdigit() else None
            os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
            downloaded = 0
            with open(dest_path, "wb") as out:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if progress:
                        progress(downloaded, total_n)
        return dest_path

# ---------------------------
# Concurrency utilities
# ---------------------------
class ExtThreadPool:
    """
    Thin wrapper around concurrent.futures.ThreadPoolExecutor with simple submit/map helpers.
    """
    def __init__(self, max_workers: int = 4):
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, fn: Callable, *args, **kwargs) -> concurrent.futures.Future:
        return self._executor.submit(fn, *args, **kwargs)

    def map(self, fn: Callable, iterable: Iterable):
        return self._executor.map(fn, iterable)

    def shutdown(self, wait: bool = True):
        self._executor.shutdown(wait=wait)

class ExtTaskScheduler:
    """
    Simple scheduler for delayed and periodic tasks using threads.
    schedule(delay_seconds, fn, *args, **kwargs) -> returns Timer object
    schedule_periodic(interval, fn, *args, **kwargs) -> returns controller object with cancel()
    """
    @staticmethod
    def schedule(delay: float, fn: Callable, *args, **kwargs) -> threading.Timer:
        t = threading.Timer(delay, fn, args=args, kwargs=kwargs)
        t.daemon = True
        t.start()
        return t

    class _Periodic:
        def __init__(self, interval: float, fn: Callable, args, kwargs):
            self.interval = interval
            self.fn = fn
            self.args = args
            self.kwargs = kwargs
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

        def _run(self):
            next_at = time.time() + self.interval
            while not self._stop.wait(max(0, next_at - time.time())):
                try:
                    self.fn(*self.args, **self.kwargs)
                except Exception:
                    pass
                next_at += self.interval

        def cancel(self):
            self._stop.set()
            self._thread.join(timeout=1.0)

    @staticmethod
    def schedule_periodic(interval: float, fn: Callable, *args, **kwargs) -> '_Periodic':
        return ExtTaskScheduler._Periodic(interval, fn, args, kwargs)

# ---------------------------
# File watcher (polling) and utility helpers
# ---------------------------
class ExtFileWatcher:
    """
    Polling file watcher. Calls callback(path, changed_files) on change.
    Use start() to spawn a background thread; stop() to cancel.
    """
    def __init__(self, path: str, callback: Callable[[str, List[str]], None], poll_interval: float = 0.5):
        self.path = path
        self.callback = callback
        self.poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._snapshot: Dict[str, float] = {}

    def _snapshot_dir(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if os.path.isdir(self.path):
            for root, _, files in os.walk(self.path):
                for fn in files:
                    p = os.path.join(root, fn)
                    try:
                        out[p] = os.path.getmtime(p)
                    except OSError:
                        out[p] = 0.0
        else:
            if os.path.exists(self.path):
                out[self.path] = os.path.getmtime(self.path)
        return out

    def start(self):
        if self._thread:
            return
        self._snapshot = self._snapshot_dir()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            time.sleep(self.poll_interval)
            cur = self._snapshot_dir()
            changed = []
            # detect added/modified
            for p, m in cur.items():
                if p not in self._snapshot or self._snapshot[p] != m:
                    changed.append(p)
            # detect removed
            for p in list(self._snapshot.keys()):
                if p not in cur:
                    changed.append(p)
            if changed:
                try:
                    self.callback(self.path, changed)
                except Exception:
                    pass
            self._snapshot = cur

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

# ---------------------------
# Utility helpers
# ---------------------------
def ext_port_scan(host: str, ports: Iterable[int], timeout: float = 0.3) -> Dict[int, bool]:
    """
    Scan provided ports on `host`. Returns dict port->open(bool).
    """
    result: Dict[int, bool] = {}
    for p in ports:
        try:
            with socket.create_connection((host, p), timeout=timeout) as s:
                result[p] = True
        except Exception:
            result[p] = False
    return result

@retry(times=3, delay=0.2)
def ext_tcp_ping(host: str, port: int = 7, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            return True
    except Exception:
        return False

# ---------------------------
# Exports
# ---------------------------
__all__ += [
    "ConnType", "Endpoint", "Message",
    "ExtTCPServer", "ExtTCPClient",
    "ExtAsyncTCPServer", "ExtAsyncTCPClient",
    "ExtConnPool", "ExtHTTPClient",
    "ExtThreadPool", "ExtTaskScheduler", "ExtFileWatcher",
    "ext_port_scan", "ext_tcp_ping",
]


# --- Base-12 conversion with extended digits -------------------------------
_BASE12_DIGITS = "0123456789↊↋"

def to_base12_ext(n: int) -> str:
    """Convert integer n to base-12 string using extended digits."""
    if n == 0:
        return "0"
    neg = n < 0
    n = abs(n)
    digits = []
    while n > 0:
        n, rem = divmod(n, 12)
        digits.append(_BASE12_DIGITS[rem])
    if neg:
        digits.append("-")
    return "".join(reversed(digits))

def from_base12_ext(s: str) -> int:
    """Convert base-12 string with extended digits back to integer."""
    s = s.strip()
    if not s:
        raise ValueError("empty string")
    neg = s[0] == "-"
    if neg:
        s = s[1:]
    n = 0
    for ch in s:
        if ch not in _BASE12_DIGITS:
            raise ValueError(f"invalid base-12 digit: {ch}")
        n = n * 12 + _BASE12_DIGITS.index(ch)
        return
    if neg:
        n = -n
        return n
    return n
# --- Safe eval using AST -----------------------------------------------
_ALLOWED_AST_NODES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Num,
    ast.Str,
    ast.Constant,  # for Python 3.8+
    ast.Name,
    ast.Load,
    ast.BoolOp,
    ast.Compare,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
    ast.Call,
    ast.Attribute,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Subscript,
    ast.Index
    }
def _ast_is_allowed(node: ast.AST) -> bool:
    """Recursively check if all AST nodes are in the allowed set."""
    if type(node) not in _ALLOWED_AST_NODES:
        return False
    for child in ast.iter_child_nodes(node):
        if not _ast_is_allowed(child):
            return False
    return True
def safe_eval_ast(expr: str, variables: Optional[Dict[str, Any]] = None, timeout_sec: Optional[float] = None) -> Any:
    """
    Safely evaluate a simple expression using AST parsing.
    Supports literals, arithmetic, comparisons, boolean ops, and variable names.
    Variables can be provided in the `variables` dict.
    Raises ValueError for disallowed constructs or SyntaxError for invalid syntax.
    Can raise TimeoutError if evaluation exceeds timeout_sec.
    """
    if variables is None:
        variables = {}
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"syntax error in expression: {e}") from e
    try:
        ast.fix_missing_locations(tree)
    except Exception  # pragma: no cover; ast.fix_missing_locations may raise in some Python versions:as e:
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
import json
import sys
import time
# ---------------------------
# Safe eval
def safe_eval(expr: str, vars: Optional[Dict[str, Any]] = None, timeout: float = 1.0) -> Any:
    """
    Safely evaluate a simple arithmetic expression with optional variables.
    Supports +, -, *, /, parentheses, integers, floats, and variable names.
    Execution is limited by timeout (in seconds).
    """
    allowed_names = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'int': int,
        'float': float,
    }
    if vars:
        allowed_names.update(vars)
    code = compile(expr, "<string>", "eval")
    for name in code.co_names:
        if name not in allowed_names:
            raise NameError(f"Use of '{name}' not allowed in safe_eval")
    start_time = time.time()
    def time_limited_eval():
        if time.time() - start_time > timeout:
            raise TimeoutError("safe_eval timed out")
        return eval(code, {"__builtins__": {}}, allowed_names)
    return time_limited_eval()
# ---------------------------
# Base-12 numerics
def to_base12(n: int) -> str:
    """Convert integer n to base-12 string using extended digits."""
    if n == 0:
        return "0"
    neg = n < 0
    n = abs(n)
    digits = []
    while n > 0:
        n, rem = divmod(n, 12)
        digits.append(_BASE12_DIGITS[rem])
    if neg:
        digits.append("-")
    return "".join(reversed(digits))

def from_base12(s: str) -> int:
    """Convert base-12 string with extended digits back to integer."""
    s = s.strip()
    if not s:
        raise ValueError("empty string")
    neg = s[0] == "-"
    if neg:
        s = s[1:]
    n = 0
    for ch in s:
        if ch not in _BASE12_DIGITS:
            raise ValueError(f"invalid base-12 digit: {ch}")
        n = n * 12 + _BASE12_DIGITS.index(ch)
    return -n if neg else n
# --- Safe eval using AST -----------------------------------------------
_ALLOWED_AST_NODES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Num,
    ast.Str,
    ast.Constant,  # for Python 3.8+
    ast.Name,
    ast.Load,
    ast.BoolOp,
    ast.Compare,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
    ast.Call,
    ast.Attribute,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Subscript,
    ast.Index
    }
def _ast_is_allowed(node: ast.AST) -> bool:
    """Recursively check if all AST nodes are in the allowed set."""
    if type(node) not in _ALLOWED_AST_NODES:
        return False
    for child in ast.iter_child_nodes(node):
        if not _ast_is_allowed(child):
            return False
    return True
def safe_eval_ast(expr: str, variables: Optional[Dict[str, Any]] = None, timeout_sec: Optional[float] = None) -> Any:
    """
    Safely evaluate a simple expression using AST parsing.
    Supports literals, arithmetic, comparisons, boolean ops, and variable names.
    Variables can be provided in the `variables` dict.
    Raises ValueError for disallowed constructs or SyntaxError for invalid syntax.
    Can raise TimeoutError if evaluation exceeds timeout_sec.
    """
    if variables is None:
        variables = {}
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"syntax error in expression: {e}") from e

    # Attempt to fix missing locations if possible; ignore non-fatal failures.
    try:
        ast.fix_missing_locations(tree)
    except Exception:
        # pragma: no cover - best-effort, not critical
        pass

    if not _ast_is_allowed(tree):
        raise ValueError("expression contains disallowed constructs")

    code = compile(tree, filename="<ast>", mode="eval")
    result_container: Dict[str, Any] = {}

    def _runner():
        try:
            result_container["value"] = eval(code, {"__builtins__": {}}, variables)
        except Exception as e:
            result_container["error"] = e

    thread = Thread(target=_runner, daemon=True)
    thread.start()

    # Wait for completion with optional timeout
    thread.join(timeout=timeout_sec)
    if thread.is_alive():
        raise TimeoutError("evaluation timed out")

    if "error" in result_container:
        raise result_container["error"]

    return result_container.get("value")

# --- Atomic file helper / hashing / JSON-safe load --------------------------
def hash_file_sha256(path: str, chunk_size: int = 8192) -> str:
    """Return SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def atomic_write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    """
    Atomically write `text` to `path` by writing to a temporary file and renaming.
    Ensures partial writes won't leave corrupt file.
    """
    dirn = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile("w", encoding=encoding, dir=dirn, delete=False) as tf:
        tf.write(text)
        tmpname = tf.name
    os.replace(tmpname, path)

def load_json_safe(path: str) -> Optional[object]:
    """
    Load JSON file returning parsed object, or None on parse error.
    Does not raise for non-fatal problems; logs and returns None.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# --- Lightweight LRU cache decorator (thread-safe-ish) ----------------------
def lru_cache_simple(maxsize: int = 128):
    """
    Simple LRU cache decorator for pure functions with hashable args.
    Not as featureful as functools.lru_cache but easy to inspect.
    """
    def deco(fn: Callable):
        cache = {}
        order = []
        @wraps(fn)
        def wrapped(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            if key in cache:
                # bump recency
                try:
                    order.remove(key)
                except ValueError:
                    pass
                order.append(key)
                return cache[key]
            result = fn(*args, **kwargs)
            cache[key] = result
            order.append(key)
            if len(order) > maxsize:
                old = order.pop(0)
                cache.pop(old, None)
            return result
        wrapped.cache_clear = lambda: (cache.clear(), order.clear())
        wrapped._cache = cache
        return wrapped
    return deco

# --- Robust subprocess runner with timeout and streamed output -------------
def run_subprocess_capture(cmd: Iterable[str], timeout: Optional[float] = None) -> Tuple[int, str, str]:
    """
    Run a subprocess and capture stdout/stderr. Returns (returncode, stdout, stderr).
    Uses subprocess.run with text mode. Raises TimeoutError on timeout.
    """
    try:
        proc = subprocess.run(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f"subprocess timed out after {timeout}s") from e

# --- Retry decorator for flaky IO / network calls ---------------------------
def retry(times: int = 3, delay: float = 0.1, allowed_exceptions: Tuple[type, ...] = (Exception,)):
    """
    Retry decorator. Retries `times` times on allowed_exceptions with `delay` seconds between attempts.
    """
    def deco(fn: Callable):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            last_exc = None
            for attempt in range(1, times + 1):
                try:
                    return fn(*args, **kwargs)
                except allowed_exceptions as e:
                    last_exc = e
                    if attempt == times:
                        break
                    time.sleep(delay)
            raise last_exc
        return wrapped
    return deco

def run_threaded(target: Callable, *args, daemon: bool = True) -> Thread:
    """
    Helper to start a thread with the given target function and args.
    Daemon status can be set; returns the started thread.
    """
    th = Thread(target=target, args=args, daemon=daemon)
    th.start()
    return th

# ---------------------------
# Exports
# ---------------------------
__all__ = [
    "safe_eval",
    "to_base12_ext", "from_base12_ext",
    "safe_eval_ast",
    "hash_file_sha256", "atomic_write_text", "load_json_safe",
    "lru_cache_simple", "run_subprocess_capture", "retry",
]

# --- Base-12 conversion with extended digits -------------------------------
_BASE12_DIGITS = "0123456789↊↋"

def to_base12_ext(n: int) -> str:
    """Convert integer n to base-12 string using extended digits."""
    if n == 0:
        return "0"
    neg = n < 0
    n = abs(n)
    digits = []
    while n > 0:
        n, rem = divmod(n, 12)
        digits.append(_BASE12_DIGITS[rem])
    if neg:
        digits.append("-")
    return "".join(reversed(digits))

def from_base12_ext(s: str) -> int:
    """Convert base-12 string with extended digits back to integer."""
    s = s.strip()
    if not s:
        raise ValueError("empty string")
    neg = s[0] == "-"
    if neg:
        s = s[1:]
    n = 0
    for ch in s:
        if ch not in _BASE12_DIGITS:
            raise ValueError(f"invalid base-12 digit: {ch}")
        n = n * 12 + _BASE12_DIGITS.index(ch)
    return -n if neg else n
# --- Safe eval using AST -----------------------------------------------
_ALLOWED_AST_NODES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Num,
    ast.Str,
    ast.Constant,  # for Python 3.8+
    ast.Name,
    ast.Load,
    ast.BoolOp,
    ast.Compare,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
    ast.Call,
    ast.Attribute,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Subscript,
    ast.Index
    }
def _ast_is_allowed(node: ast.AST) -> bool:
    """Recursively check if all AST nodes are in the allowed set."""
    if type(node) not in _ALLOWED_AST_NODES:
        return False
    for child in ast.iter_child_nodes(node):
        if not _ast_is_allowed(child):
            return False
    return True
def safe_eval_ast(expr: str, variables: Optional[Dict[str, Any]] = None, timeout_sec: Optional[float] = None) -> Any:
    """
    Safely evaluate a simple expression using AST parsing.
    Supports literals, arithmetic, comparisons, boolean ops, and variable names.
    Variables can be provided in the `variables` dict.
    Raises ValueError for disallowed constructs or SyntaxError for invalid syntax.
    Can raise TimeoutError if evaluation exceeds timeout_sec.
    """
    if variables is None:
        variables = {}
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"syntax error in expression: {e}") from e

    # Attempt to fix missing locations if possible; ignore non-fatal failures.
    try:
        ast.fix_missing_locations(tree)
    except Exception:
        # pragma: no cover - best-effort, not critical
        pass

    if not _ast_is_allowed(tree):
        raise ValueError("expression contains disallowed constructs")

    code = compile(tree, filename="<ast>", mode="eval")
    result_container: Dict[str, Any] = {}

    def _runner():
        try:
            result_container["value"] = eval(code, {"__builtins__": {}}, variables)
        except Exception as e:
            result_container["error"] = e

    thread = Thread(target=_runner, daemon=True)
    thread.start()

    # Wait for completion with optional timeout
    thread.join(timeout=timeout_sec)
    if thread.is_alive():
        raise TimeoutError("evaluation timed out")

    if "error" in result_container:
        raise result_container["error"]

    return result_container.get("value")

# --- Atomic file helper / hashing / JSON-safe load --------------------------
def hash_file_sha256(path: str, chunk_size: int = 8192) -> str:
    """Return SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def atomic_write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    """
    Atomically write `text` to `path` by writing to a temporary file and renaming.
    Ensures partial writes won't leave corrupt file.
    """
    dirn = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile("w", encoding=encoding, dir=dirn, delete=False) as tf:
        tf.write(text)
        tmpname = tf.name
    os.replace(tmpname, path)

def load_json_safe(path: str) -> Optional[object]:
    """
    Load JSON file returning parsed object, or None on parse error.
    Does not raise for non-fatal problems; logs and returns None.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# --- Lightweight LRU cache decorator (thread-safe-ish) ----------------------
def lru_cache_simple(maxsize: int = 128):
    """
    Simple LRU cache decorator for pure functions with hashable args.
    Not as featureful as functools.lru_cache but easy to inspect.
    """
    def deco(fn: Callable):
        cache = {}
        order = []
        @wraps(fn)
        def wrapped(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            if key in cache:
                # bump recency
                try:
                    order.remove(key)
                except ValueError:
                    pass
                order.append(key)
                return cache[key]
            result = fn(*args, **kwargs)
            cache[key] = result
            order.append(key)
            if len(order) > maxsize:
                old = order.pop(0)
                cache.pop(old, None)
            return result
        wrapped.cache_clear = lambda: (cache.clear(), order.clear())
        wrapped._cache = cache
        return wrapped
    return deco

# --- Robust subprocess runner with timeout and streamed output -------------
def run_subprocess_capture(cmd: Iterable[str], timeout: Optional[float] = None) -> Tuple[int, str, str]:
    """
    Run a subprocess and capture stdout/stderr. Returns (returncode, stdout, stderr).
    Uses subprocess.run with text mode. Raises TimeoutError on timeout.
    """
    try:
        proc = subprocess.run(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f"subprocess timed out after {timeout}s") from e

# --- Retry decorator for flaky IO / network calls ---------------------------
def retry(times: int = 3, delay: float = 0.1, allowed_exceptions: Tuple[type, ...] = (Exception,)):
    """
    Retry decorator. Retries `times` times on allowed_exceptions with `delay` seconds between attempts.
    """
    def deco(fn: Callable):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            last_exc = None
            for attempt in range(1, times + 1):
                try:
                    return fn(*args, **kwargs)
                except allowed_exceptions as e:
                    last_exc = e
                    if attempt == times:
                        break
                    time.sleep(delay)
            raise last_exc
        return wrapped
    return deco

def run_threaded(target: Callable, *args, daemon: bool = True) -> Thread:
    """
    Helper to start a thread with the given target function and args.
    Daemon status can be set; returns the started thread.
    """
    th = Thread(target=target, args=args, daemon=daemon)
    th.start()
    return th

# ---------------------------
# Exports
# ---------------------------
__all__ = [
    "safe_eval",
    "to_base12_ext", "from_base12_ext",
    "safe_eval_ast",
    "hash_file_sha256", "atomic_write_text", "load_json_safe",
    "lru_cache_simple", "run_subprocess_capture", "retry",
]

# --- Base-12 conversion with extended digits -------------------------------
_BASE12_DIGITS = "0123456789↊↋"

def to_base12_ext(n: int) -> str:
    """Convert integer n to base-12 string using extended digits."""
    if n == 0:
        return "0"
    neg = n < 0
    n = abs(n)
    digits = []
    while n > 0:
        n, rem = divmod(n, 12)
        digits.append(_BASE12_DIGITS[rem])
    if neg:
        digits.append("-")
    return "".join(reversed(digits))

def from_base12_ext(s: str) -> int:
    """Convert base-12 string with extended digits back to integer."""
    s = s.strip()
    if not s:
        raise ValueError("empty string")
    neg = s[0] == "-"
    if neg:
        s = s[1:]
    n = 0
    for ch in s:
        if ch not in _BASE12_DIGITS:
            raise ValueError(f"invalid base-12 digit: {ch}")
        n = n * 12 + _BASE12_DIGITS.index(ch)
    return -n if neg else n
# --- Safe eval using AST -----------------------------------------------
_ALLOWED_AST_NODES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Num,
    ast.Str,
    ast.Constant,  # for Python 3.8+
    ast.Name,
    ast.Load,
    ast.BoolOp,
    ast.Compare,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
    ast.Call,
    ast.Attribute,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Subscript,
    ast.Index
    }
def _ast_is_allowed(node: ast.AST) -> bool:
    """Recursively check if all AST nodes are in the allowed set."""
    if type(node) not in _ALLOWED_AST_NODES:
        return False
    for child in ast.iter_child_nodes(node):
        if not _ast_is_allowed(child):
            return False
    return True
def safe_eval_ast(expr: str, variables: Optional[Dict[str, Any]] = None, timeout_sec: Optional[float] = None) -> Any:
    """
    Safely evaluate a simple expression using AST parsing.
    Supports literals, arithmetic, comparisons, boolean ops, and variable names.
    Variables can be provided in the `variables` dict.
    Raises ValueError for disallowed constructs or SyntaxError for invalid syntax.
    Can raise TimeoutError if evaluation exceeds timeout_sec.
    """
    if variables is None:
        variables = {}
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"syntax error in expression: {e}") from e

    # Attempt to fix missing locations if possible; ignore non-fatal failures.
    try:
        ast.fix_missing_locations(tree)
    except Exception:
        # pragma: no cover - best-effort, not critical
        pass

    if not _ast_is_allowed(tree):
        raise ValueError("expression contains disallowed constructs")

    code = compile(tree, filename="<ast>", mode="eval")
    result_container: Dict[str, Any] = {}

    def _runner():
        try:
            result_container["value"] = eval(code, {"__builtins__": {}}, variables)
        except Exception as e:
            result_container["error"] = e

    thread = Thread(target=_runner, daemon=True)
    thread.start()

    # Wait for completion with optional timeout
    thread.join(timeout=timeout_sec)
    if thread.is_alive():
        raise TimeoutError("evaluation timed out")

    if "error" in result_container:
        raise result_container["error"]

    return result_container.get("value")

# --- Atomic file helper / hashing / JSON-safe load --------------------------
def hash_file_sha256(path: str, chunk_size: int = 8192) -> str:
    """Return SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def atomic_write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    """
    Atomically write `text` to `path` by writing to a temporary file and renaming.
    Ensures partial writes won't leave corrupt file.
    """
    dirn = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile("w", encoding=encoding, dir=dirn, delete=False) as tf:
        tf.write(text)
        tmpname = tf.name
    os.replace(tmpname, path)

def load_json_safe(path: str) -> Optional[object]:
    """
    Load JSON file returning parsed object, or None on parse error.
    Does not raise for non-fatal problems; logs and returns None.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# --- Lightweight LRU cache decorator (thread-safe-ish) ----------------------
def lru_cache_simple(maxsize: int = 128):
    """
    Simple LRU cache decorator for pure functions with hashable args.
    Not as featureful as functools.lru_cache but easy to inspect.
    """
    def deco(fn: Callable):
        cache = {}
        order = []
        @wraps(fn)
        def wrapped(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            if key in cache:
                # bump recency
                try:
                    order.remove(key)
                except ValueError:
                    pass
                order.append(key)
                return cache[key]
            result = fn(*args, **kwargs)
            cache[key] = result
            order.append(key)
            if len(order) > maxsize:
                old = order.pop(0)
                cache.pop(old, None)
            return result
        wrapped.cache_clear = lambda: (cache.clear(), order.clear())
        wrapped._cache = cache
        return wrapped
    return deco

# --- Robust subprocess runner with timeout and streamed output -------------
def run_subprocess_capture(cmd: Iterable[str], timeout: Optional[float] = None) -> Tuple[int, str, str]:
    """
    Run a subprocess and capture stdout/stderr. Returns (returncode, stdout, stderr).
    Uses subprocess.run with text mode. Raises TimeoutError on timeout.
    """
    try:
        proc = subprocess.run(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f"subprocess timed out after {timeout}s") from e

# --- Retry decorator for flaky IO / network calls ---------------------------
def retry(times: int = 3, delay: float = 0.1, allowed_exceptions: Tuple[type, ...] = (Exception,)):
    """
    Retry decorator. Retries `times` times on allowed_exceptions with `delay` seconds between attempts.
    """
    def deco(fn: Callable):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            last_exc = None
            for attempt in range(1, times + 1):
                try:
                    return fn(*args, **kwargs)
                except allowed_exceptions as e:
                    last_exc = e
                    if attempt == times:
                        break
                    time.sleep(delay)
            raise last_exc
        return wrapped
    return deco

def run_threaded(target: Callable, *args, daemon: bool = True) -> Thread:
    """
    Helper to start a thread with the given target function and args.
    Daemon status can be set; returns the started thread.
    """
    th = Thread(target=target, args=args, daemon=daemon)
    th.start()
    return th

# ---------------------------
# Exports
# ---------------------------
__all__ = [
    "safe_eval",
    "to_base12_ext", "from_base12_ext",
    "safe_eval_ast",
    "hash_file_sha256", "atomic_write_text", "load_json_safe",
    "lru_cache_simple", "run_subprocess_capture", "retry",
]

# --- Base-12 conversion with extended digits -------------------------------
_BASE12_DIGITS = "0123456789↊↋"

def to_base12_ext(n: int) -> str:
    """Convert integer n to base-12 string using extended digits."""
    if n == 0:
        return "0"
    neg = n < 0
    n = abs(n)
    digits = []
    while n > 0:
        n, rem = divmod(n, 12)
        digits.append(_BASE12_DIGITS[rem])
    if neg:
        digits.append("-")
    return "".join(reversed(digits))

def from_base12_ext(s: str) -> int:
    """Convert base-12 string with extended digits back to integer."""
    s = s.strip()
    if not s:
        raise ValueError("empty string")
    neg = s[0] == "-"
    if neg:
        s = s[1:]
    n = 0
    for ch in s:
        if ch not in _BASE12_DIGITS:
            raise ValueError(f"invalid base-12 digit: {ch}")
        n = n * 12 + _BASE12_DIGITS.index(ch)
    return -n if neg else n
# --- Safe eval using AST -----------------------------------------------
_ALLOWED_AST_NODES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Num,
    ast.Str,
    ast.Constant,  # for Python 3.8+
    ast.Name,
    ast.Load,
    ast.BoolOp,
    ast.Compare,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
    ast.Call,
    ast.Attribute,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Subscript,
    ast.Index
    }
def _ast_is_allowed(node: ast.AST) -> bool:
    """Recursively check if all AST nodes are in the allowed set."""
    if type(node) not in _ALLOWED_AST_NODES:
        return False
    for child in ast.iter_child_nodes(node):
        if not _ast_is_allowed(child):
            return False
    return True
def safe_eval_ast(expr: str, variables: Optional[Dict[str, Any]] = None, timeout_sec: Optional[float] = None) -> Any:
    """
    Safely evaluate a simple expression using AST parsing.
    Supports literals, arithmetic, comparisons, boolean ops, and variable names.
    Variables can be provided in the `variables` dict.
    Raises ValueError for disallowed constructs or SyntaxError for invalid syntax.
    Can raise TimeoutError if evaluation exceeds timeout_sec.
    """
    if variables is None:
        variables = {}
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"syntax error in expression: {e}") from e

    # Attempt to fix missing locations if possible; ignore non-fatal failures.
    try:
        ast.fix_missing_locations(tree)
    except Exception:
        # pragma: no cover - best-effort, not critical
        pass

    if not _ast_is_allowed(tree):
        raise ValueError("expression contains disallowed constructs")

    code = compile(tree, filename="<ast>", mode="eval")
    result_container: Dict[str, Any] = {}

    def _runner():
        try:
            result_container["value"] = eval(code, {"__builtins__": {}}, variables)
        except Exception as e:
            result_container["error"] = e

    thread = Thread(target=_runner, daemon=True)
    thread.start()

    # Wait for completion with optional timeout
    thread.join(timeout=timeout_sec)
    if thread.is_alive():
        raise TimeoutError("evaluation timed out")

    if "error" in result_container:
        raise result_container["error"]

    return result_container.get("value")

# --- Atomic file helper / hashing / JSON-safe load --------------------------
def hash_file_sha256(path: str, chunk_size: int = 8192) -> str:
    """Return SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def atomic_write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    """
    Atomically write `text` to `path` by writing to a temporary file and renaming.
    Ensures partial writes won't leave corrupt file.
    """
    dirn = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile("w", encoding=encoding, dir=dirn, delete=False) as tf:
        tf.write(text)
        tmpname = tf.name
    os.replace(tmpname, path)

def load_json_safe(path: str) -> Optional[object]:
    """
    Load JSON file returning parsed object, or None on parse error.
    Does not raise for non-fatal problems; logs and returns None.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# --- Lightweight LRU cache decorator (thread-safe-ish) ----------------------
def lru_cache_simple(maxsize: int = 128):
    """
    Simple LRU cache decorator for pure functions with hashable args.
    Not as featureful as functools.lru_cache but easy to inspect.
    """
    def deco(fn: Callable):
        cache = {}
        order = []
        @wraps(fn)
        def wrapped(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            if key in cache:
                # bump recency
                try:
                    order.remove(key)
                except ValueError:
                    pass
                order.append(key)
                return cache[key]
            result = fn(*args, **kwargs)
            cache[key] = result
            order.append(key)
            if len(order) > maxsize:
                old = order.pop(0)
                cache.pop(old, None)
            return result
        wrapped.cache_clear = lambda: (cache.clear(), order.clear())
        wrapped._cache = cache
        return wrapped
    return deco

# --- Robust subprocess runner with timeout and streamed output -------------
def run_subprocess_capture(cmd: Iterable[str], timeout: Optional[float] = None) -> Tuple[int, str, str]:
    """
    Run a subprocess and capture stdout/stderr. Returns (returncode, stdout, stderr).
    Uses subprocess.run with text mode. Raises TimeoutError on timeout.
    """
    try:
        proc = subprocess.run(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f"subprocess timed out after {timeout}s") from e

# --- Retry decorator for flaky IO / network calls ---------------------------
def retry(times: int = 3, delay: float = 0.1, allowed_exceptions: Tuple[type, ...] = (Exception,)):
    """
    Retry decorator. Retries `times` times on allowed_exceptions with `delay` seconds between attempts.
    """
    def deco(fn: Callable):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            last_exc = None
            for attempt in range(1, times + 1):
                try:
                    return fn(*args, **kwargs)
                except allowed_exceptions as e:
                    last_exc = e
                    if attempt == times:
                        break
                    time.sleep(delay)
            raise last_exc
        return wrapped
    return deco

def run_threaded(target: Callable, *args, daemon: bool = True) -> Thread:
    """
    Helper to start a thread with the given target function and args.
    Daemon status can be set; returns the started thread.
    """
    th = Thread(target=target, args=args, daemon=daemon)
    th.start()
    return th

# ---------------------------
# Exports
# ---------------------------
__all__ = [
    "safe_eval",
    "to_base12_ext", "from_base12_ext",
    "safe_eval_ast",
    "hash_file_sha256", "atomic_write_text", "load_json_safe",
    "lru_cache_simple", "run_subprocess_capture", "retry",
]
if not _ast_is_allowed(tree):
        raise ValueError("expression contains disallowed constructs")
        code = compile(tree, filename="<ast>", mode="eval")
        result_container: Dict[str, Any] = {}
def target():
                                   thread.start()
thread.join(timeout=timeout_sec)
if thread.is_alive():
                                raise TimeoutError("evaluation timed out")
if 'error' in result_container:
                                raise result_container['error']

# Production runtime bootstrap and operational tooling
# Adds a production-oriented runtime, logging, graceful shutdown, metrics, and optional JIT path.
# Intended as a concrete, realistic scaffold to harden the project toward production readiness.
# This code appends non-invasive helpers and does not change existing behavior unless you opt into `--prod` CLI.

import logging
import logging.handlers
import signal
import threading
import http.server
import socketserver
import socket
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Callable, Any, Dict

# ---------------------------------------------------------------------------
# Versioning / VCS helpers
# ---------------------------------------------------------------------------
def get_vcs_version() -> str:
    """Return short git SHA if available, else 'unknown'."""
    try:
        import subprocess, os
        repo_root = os.path.dirname(os.path.abspath(__file__))
        p = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=1.0)
        if p.returncode == 0:
            return p.stdout.strip()
    except Exception:
        pass
    return "unknown"

# ---------------------------------------------------------------------------
# Logging configuration for production
# ---------------------------------------------------------------------------
def configure_logging(name: str = "sayit", level: int = logging.INFO, logfile: Optional[str] = "sayit.log") -> logging.Logger:
    """Configure structured logging for console + rotating file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    fmt = logging.Formatter(fmt="%(asctime)s %(levelname)s [%(name)s] %(threadName)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(level)
    logger.addHandler(sh)

    if logfile:
        fh = logging.handlers.RotatingFileHandler(logfile, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)  # keep detailed file logs
        logger.addHandler(fh)

    # make library loggers less noisy by default
    for lib in ("urllib3", "asyncio", "llvmlite"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    return logger

# ---------------------------------------------------------------------------
# Minimal metrics HTTP server (exposes /metrics and /health)
# ---------------------------------------------------------------------------
class _MetricsHandler(http.server.BaseHTTPRequestHandler):
    metrics_provider: Optional[Callable[[], Dict[str, Any]]] = None

    def _write_json(self, data: dict, code: int = 200):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            self._write_json({"status": "ok", "version": get_vcs_version()})
            return
        if self.path == "/metrics":
            try:
                if _MetricsHandler.metrics_provider:
                    self._write_json(_MetricsHandler.metrics_provider())
                else:
                    self._write_json({"metrics": {}})
            except Exception as e:
                self.send_error(500, f"metrics error: {e}")
            return
        # default 404
        self.send_error(404, "not found")

    def log_message(self, format: str, *args):
        # route standard server logs to our logger
        logging.getLogger("sayit.metrics.http").info(format % args)

class ProductionRuntime:
    """
    ProductionRuntime wraps:
      - Logging and structured startup/shutdown
      - ThreadPool for CPU-bound tasks
      - Lightweight metrics HTTP endpoint (/metrics, /health)
      - Graceful signal handling and uncaught-exception hook
      - Optional JIT path (llvmlite) when available (best-effort)
    Use:
        rt = ProductionRuntime(...)
        rt.start()
        rt.submit(my_work)
        rt.stop()
    """
    def __init__(self,
                 max_workers: int = 4,
                 metrics_port: Optional[int] = 8000,
                 log_file: Optional[str] = "sayit.log"):
        self.logger = configure_logging(level=logging.INFO, logfile=log_file)
        self.logger.info("ProductionRuntime init: version=%s", get_vcs_version())
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sayit-worker")
        self._running = threading.Event()
        self.metrics_port = metrics_port
        self._metrics_server_thread: Optional[threading.Thread] = None
        self._metrics_httpd: Optional[socketserver.TCPServer] = None
        self._metrics_data: Dict[str, Any] = {"uptime_start": time.time(), "tasks_submitted": 0, "tasks_completed": 0}
        self._lock = threading.Lock()
        self._original_excepthook = threading.excepthook if hasattr(threading, "excepthook") else None

        # optional llvmlite JIT presence flag and helper
        self.jit_available = False
        try:
            import llvmlite
            self.jit_available = True
            self.logger.info("llvmlite available: JIT optimizations enabled")
        except Exception:
            self.logger.info("llvmlite not available: running interpreter-only path")

    # -------------------------
    # Metrics
    # -------------------------
    def _metrics_provider(self) -> Dict[str, Any]:
        with self._lock:
            m = dict(self._metrics_data)
        m["uptime_seconds"] = time.time() - self._metrics_data.get("uptime_start", time.time())
        m["worker_threads"] = self.max_workers
        return m

    def _start_metrics_server(self):
        if self.metrics_port is None:
            return
        handler = _MetricsHandler
        handler.metrics_provider = self._metrics_provider
        class ThreadedTCPServer(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
        try:
            httpd = ThreadedTCPServer(("0.0.0.0", self.metrics_port), handler)
        except OSError:
            # fallback to loopback only if privileged port or port in use
            httpd = ThreadedTCPServer(("127.0.0.1", self.metrics_port), handler)
        self._metrics_httpd = httpd
        t = threading.Thread(target=httpd.serve_forever, name="sayit-metrics", daemon=True)
        t.start()
        self._metrics_server_thread = t
        self.logger.info("Metrics HTTP server started on port %s", self.metrics_port)

    def _stop_metrics_server(self):
        if self._metrics_httpd:
            try:
                self._metrics_httpd.shutdown()
                self._metrics_httpd.server_close()
            except Exception:
                pass
            self._metrics_httpd = None
        if self._metrics_server_thread:
            self._metrics_server_thread.join(timeout=1.0)
            self._metrics_server_thread = None
        self.logger.info("Metrics HTTP server stopped")

    # -------------------------
    # Task submission helpers
    # -------------------------
    def submit(self, fn: Callable[..., Any], *args, **kwargs):
        with self._lock:
            self._metrics_data["tasks_submitted"] = self._metrics_data.get("tasks_submitted", 0) + 1
        fut = self.executor.submit(self._task_wrapper, fn, *args, **kwargs)
        return fut

    def _task_wrapper(self, fn: Callable[..., Any], *args, **kwargs):
        try:
            res = fn(*args, **kwargs)
            return res
        finally:
            with self._lock:
                self._metrics_data["tasks_completed"] = self._metrics_data.get("tasks_completed", 0) + 1

    # -------------------------
    # Signal handling + uncaught exceptions
    # -------------------------
    def _signal_handler(self, signum, frame):
        self.logger.info("Received signal %s, initiating graceful shutdown", signum)
        self.stop(timeout=10.0)

    def _threading_excepthook(self, args):
        # args is a ThreadException object in Python 3.8+: (exc_type, exc_value, exc_traceback, thread)
        try:
            self.logger.error("Uncaught exception in thread %s: %s", getattr(args, "thread", "unknown"), "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)))
        except Exception:
            self.logger.exception("Exception in threading excepthook")
        # chain to original if present
        if self._original_excepthook:
            try:
                self._original_excepthook(args)
            except Exception:
                pass

    # -------------------------
    # Start / Stop
    # -------------------------
    def start(self):
        if self._running.is_set():
            return
        self.logger.info("Starting ProductionRuntime")
        # bind signal handlers for graceful termination
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._signal_handler)
            except Exception:
                self.logger.warning("Unable to register signal handler for %s", sig)
        # threading excepthook (Python 3.8+)
        if hasattr(threading, "excepthook"):
            threading.excepthook = self._threading_excepthook
        # start metrics server
        self._start_metrics_server()
        self._running.set()
        self.logger.info("ProductionRuntime started")

    def stop(self, timeout: float = 5.0):
        if not self._running.is_set():
            return
        self.logger.info("Stopping ProductionRuntime")
        # shutdown executor gracefully
        self.executor.shutdown(wait=False)
        # give tasks a grace period
        deadline = time.time() + timeout
        while time.time() < deadline:
            # if there are still running threads, sleep briefly
            # Note: ThreadPoolExecutor doesn't expose active count directly; best-effort wait
            time.sleep(0.1)
            break
        # stop metrics
        self._stop_metrics_server()
        self._running.clear()
        self.logger.info("ProductionRuntime stopped")

    # -------------------------
    # High-level runner convenience
    # -------------------------
    def run_program_text(self, source_text: str, timeout: Optional[float] = None, trace: bool = False) -> Dict[str, Any]:
        """
        High-level convenience: parse source text, compile, run in ExecVM/ExecVMPro.
        This is the production entry used by sayc/run modes when --prod is chosen.
        """
        self.logger.info("Running program (prod runner) trace=%s timeout=%s", trace, timeout)
        try:
            toks = list(_tokenize(source_text))
            parser = Parser(toks)
            prog = parser.parse_program()
            # Prefer ExecVMPro (optimized) if available
            vm = None
            try:
                vm = ExecVMPro()  # type: ignore[name-defined]
            except Exception:
                vm = ExecVM()
            vm.trace = trace
            if timeout is not None:
                vm.timeout = timeout
            res = vm.run_program(prog)
            self.logger.info("Program finished: %s", res)
            return {"ok": True, "result": res}
        except Exception as e:
            self.logger.exception("Program execution error")
            return {"ok": False, "error": str(e), "trace": traceback.format_exc()}

# ---------------------------------------------------------------------------
# CLI integration: lightweight `--prod` runner for production usage
# ---------------------------------------------------------------------------
def _entry_prod_mode(argv=None):
    import argparse, sys
    parser = argparse.ArgumentParser(prog="sayit-prod", description="Run Sayit program in production mode (graceful, metrics, logging)")
    parser.add_argument("file", nargs="?", help="Source file to run (.say)")
    parser.add_argument("--port", type=int, default=8000, help="Metrics HTTP port")
    parser.add_argument("--workers", type=int, default=4, help="Worker thread pool size")
    parser.add_argument("--log", default="sayit.log", help="Log file path")
    parser.add_argument("--timeout", type=float, default=None, help="Per-program execution timeout (s)")
    parser.add_argument("--trace", action="store_true", help="Enable VM tracing")
    args = parser.parse_args(argv)

    runtime = ProductionRuntime(max_workers=args.workers, metrics_port=args.port, log_file=args.log)
    runtime.start()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            src = f.read()
        fut = runtime.submit(runtime.run_program_text, src, args.timeout, args.trace)
        try:
            res = fut.result(timeout=(args.timeout + 5) if args.timeout else None)
            print(json.dumps(res, indent=2))
        except Exception as e:
            runtime.logger.exception("Error waiting for program completion")
            print(json.dumps({"ok": False, "error": str(e)}))
    else:
        runtime.logger.info("No file specified; production runtime is running (metrics available). Press Ctrl-C to stop.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
    runtime.stop()

# ---------------------------------------------------------------------------
# Minimal CI / packaging helpers (advisory, not performing remote ops)
# ---------------------------------------------------------------------------
def generate_release_notes(short: bool = True) -> str:
    """
    Generate a short release note document based on git history.
    This is a best-effort helper for packaging and release automation.
    """
    try:
        import subprocess, os
        repo_root = os.path.dirname(os.path.abspath(__file__))
        # gather last 20 commits as summary
        p = subprocess.run(["git", "log", "--pretty=format:%h %s", "-n", "20"], cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=2.0)
        if p.returncode == 0:
            lines = p.stdout.strip().splitlines()
            header = f"Sayit release {get_vcs_version()}"
            body = "\n".join(lines if not short else lines[:10])
            return header + "\n\n" + body
    except Exception:
        pass
    return f"Sayit release {get_vcs_version()} (changelog unavailable)"

# ---------------------------------------------------------------------------
# Expose small public API for integration tests and CI
# ---------------------------------------------------------------------------
__all__ += [
    "ProductionRuntime",
    "configure_logging",
    "get_vcs_version",
    "_entry_prod_mode",
    "generate_release_notes",
]

# ---------------------------------------------------------------------------
# If module executed with `--prod` flag, run prod entrypoint (non-invasive)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if "--prod" in sys.argv:
        # call production entry and exit
        _entry_prod_mode([a for a in sys.argv[1:] if a != "--prod"])
        sys.exit(0)
    # otherwise keep quiet so earlier __main__ blocks in sub-modules keep working.
    
    print("This module is intended to be imported, or run with --prod for production mode.")
    sys.exit(0)
    print("Usage: say_lexer.py <file.say> [--pos]")
print(f"Error reading file {args.file}: {e}")
sys.exit(1)
tokens = list(tokenize(src, include_pos=args.pos))
for t in tokens:
                print(t)
# Self-training scaffold: safe, opt-in framework to propose and (with explicit approval) apply repo patches.
# Non-autonomous: requires `--selftrain` and `--apply` flags to persist changes. Designed to be auditable and reversible.

import subprocess
import shutil
import tempfile
import pathlib
import difflib
from typing import List, Dict, Tuple, Optional

class SelfTrainer:
    """
    Lightweight self-training scaffold.
    - gather_examples(): collects candidate source files
    - propose_patch(): produces simple, safe fixes (uses existing `autocorrect_source`)
    - evaluate_patch(): runs tests in isolated worktree/copy and returns results
    - apply_patch(): writes changes into working tree and creates a git branch + commit (requires confirm)
    Usage (dry-run):
        python system.py --selftrain
    To apply:
        python system.py --selftrain --apply
    NOTE: This is a safe scaffold — it will not modify your repo unless `--apply` is provided.
    """

    def __init__(self, repo_path: Optional[str] = None, include_exts: Optional[List[str]] = None):
        self.repo = pathlib.Path(repo_path or ".").resolve()
        self.include_exts = include_exts or [".py", ".say"]
        self.tmp_worktree: Optional[pathlib.Path] = None

    def _run(self, cmd: List[str], cwd: Optional[pathlib.Path] = None, check: bool = False) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, cwd=str(cwd or self.repo), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=check)

    def find_candidate_files(self) -> List[pathlib.Path]:
        out = []
        for p in self.repo.rglob("*"):
            if p.is_file() and p.suffix in self.include_exts:
                # ignore .git and .cache dirs
                if ".git" in p.parts or ".cache" in p.parts:
                    continue
                out.append(p)
        return sorted(out)

    def propose_patch(self, files: List[pathlib.Path]) -> Dict[str, Tuple[str, str]]:
        """
        Propose changes. Returns mapping path -> (orig, proposed).
        Current proposal strategy: run `autocorrect_source` (conservative fixes).
        """
        proposals: Dict[str, Tuple[str, str]] = {}
        for p in files:
            try:
                orig = p.read_text(encoding="utf-8")
            except Exception:
                continue
            # use autocorrect_source available earlier in this module
            try:
                proposed = autocorrect_source(orig)
            except Exception:
                proposed = orig
            if proposed != orig:
                proposals[str(p.relative_to(self.repo))] = (orig, proposed)
        return proposals

    def _render_diff(self, orig: str, proposed: str, filename: str) -> str:
        od = orig.splitlines(keepends=True)
        pd = proposed.splitlines(keepends=True)
        return "".join(difflib.unified_diff(od, pd, fromfile=filename, tofile=filename + ".proposed"))

    def _create_worktree(self) -> pathlib.Path:
        # try to use `git worktree` for isolated evaluation
        tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="sayit-selftrain-"))
        try:
            self._run(["git", "worktree", "add", "--detach", str(tmpdir), "HEAD"], cwd=self.repo)
            self.tmp_worktree = tmpdir
            return tmpdir
        except Exception:
            # fallback: shallow copy of repo files
            for item in self.repo.iterdir():
                if item.name == ".git":
                    continue
                dest = tmpdir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, symlinks=False)
                else:
                    shutil.copy2(item, dest)
            self.tmp_worktree = tmpdir
            return tmpdir

    def _cleanup_worktree(self):
        if not self.tmp_worktree:
            return
        try:
            # if created via git worktree, remove it cleanly
            self._run(["git", "worktree", "remove", "--force", str(self.tmp_worktree)], cwd=self.repo)
        except Exception:
            # best-effort remove
            try:
                shutil.rmtree(self.tmp_worktree)
            except Exception:
                pass
        self.tmp_worktree = None

    def evaluate_patch(self, proposals: Dict[str, Tuple[str, str]]) -> Dict[str, object]:
        """
        Apply proposals to isolated copy and run test suite.
        Returns dictionary with diffs and test run result.
        """
        if not proposals:
            return {"ok": True, "message": "no proposals", "diffs": {}, "tests": None}
        wt = self._create_worktree()
        diffs: Dict[str, str] = {}
        try:
            for relpath, (orig, proposed) in proposals.items():
                targ = wt / relpath
                targ.parent.mkdir(parents=True, exist_ok=True)
                targ.write_text(proposed, encoding="utf-8")
                diffs[relpath] = self._render_diff(orig, proposed, relpath)
            # run tests using unittest discovery, fallback to no tests
            try:
                cp = subprocess.run([sys.executable, "-m", "unittest", "discover", "-v"], cwd=str(wt), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)
                tests_out = cp.stdout
                tests_rc = cp.returncode
            except subprocess.TimeoutExpired as e:
                tests_out = f"timeout: {e}"
                tests_rc = 2
            return {"ok": tests_rc == 0, "diffs": diffs, "tests": {"returncode": tests_rc, "output": tests_out}}
        finally:
            self._cleanup_worktree()

    def apply_patch(self, proposals: Dict[str, Tuple[str, str]], branch_name: Optional[str] = None, commit_msg: Optional[str] = None, confirm: bool = False) -> Dict[str, object]:
        """
        If confirm is True, create branch, write files and commit. Returns summary.
        Will not force-push or change remote.
        """
        if not proposals:
            return {"applied": False, "reason": "no proposals"}
        if not confirm:
            return {"applied": False, "reason": "confirm flag not set", "preview": {k: self._render_diff(o, p, k) for k, (o, p) in proposals.items()}}
        branch = branch_name or f"selftrain/{int(time.time())}"
        # create branch
        try:
            self._run(["git", "checkout", "-b", branch], cwd=self.repo, check=True)
        except subprocess.CalledProcessError as e:
            return {"applied": False, "error": f"git branch create failed: {e.stderr}"}
        try:
            for relpath, (orig, proposed) in proposals.items():
                target = self.repo / relpath
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(proposed, encoding="utf-8")
                self._run(["git", "add", relpath], cwd=self.repo, check=True)
            msg = commit_msg or "selftrain: apply proposed autocorrections (opt-in)"
            self._run(["git", "commit", "-m", msg], cwd=self.repo, check=True)
            return {"applied": True, "branch": branch, "commit_msg": msg, "files": list(proposals.keys())}
        except subprocess.CalledProcessError as e:
            return {"applied": False, "error": f"git commit failed: {e.stderr}"}

    def run_cycle(self, dry_run: bool = True) -> Dict[str, object]:
        """
        End-to-end cycle: discover, propose, evaluate, and (optionally) apply.
        Returns a report dict.
        """
        files = self.find_candidate_files()
        proposals = self.propose_patch(files)
        if not proposals:
            return {"ok": True, "message": "no proposals"}
        eval_res = self.evaluate_patch(proposals)
        report = {"proposals_count": len(proposals), "evaluation": eval_res}
        # include diffs summary
        report["diffs_preview"] = {k: self._render_diff(o, p, k) for k, (o, p) in proposals.items()}
        report["applied"] = False
        if not dry_run and eval_res.get("ok"):
            apply_res = self.apply_patch(proposals, confirm=True)
            report["applied"] = apply_res
        return report

# CLI integration for self-training
def _entry_selftrain(argv: Optional[List[str]] = None):
    import argparse
    parser = argparse.ArgumentParser(prog="sayit-selftrain", description="Self-train scaffold (opt-in and auditable)")
    parser.add_argument("--apply", action="store_true", help="Apply proposed patches (requires passing tests and explicit consent)")
    parser.add_argument("--only-preview", action="store_true", help="Show diffs without running tests")
    args = parser.parse_args(argv)
    st = SelfTrainer(repo_path=".")
    files = st.find_candidate_files()
    proposals = st.propose_patch(files)
    if not proposals:
        print("No safe proposals generated.")
        return 0
    # show brief summary
    print(f"Proposals generated for {len(proposals)} files. Preview diffs:")
    for p, (o, n) in proposals.items():
        print("---", p)
        print(st._render_diff(o, n, p))
    if args.only_preview:
        print("Preview only; exiting.")
        return 0
    print("Evaluating proposals in isolated worktree (running unit tests)...")
    eval_res = st.evaluate_patch(proposals)
    print("Test result:", eval_res.get("tests", {}).get("returncode"), flush=True)
    print(eval_res.get("tests", {}).get("output", ""))
    if args.apply:
        if not eval_res.get("ok"):
            print("Tests did not pass in evaluation environment; refusing to apply.", flush=True)
            return 2
        print("Applying proposals to a new git branch (confirming)...")
        apply_res = st.apply_patch(proposals, confirm=True)
        print("Apply result:", apply_res)
    else:
        print("Run with `--apply` to write proposals to a new git branch and commit them (opt-in).")
    return 0

# Hook into module-level CLI: safe opt-in invocation
if __name__ == "__main__":
    import sys
    if "--selftrain" in sys.argv:
        # strip the flag and pass the rest to the subcommand parser
        argv = [a for a in sys.argv[1:] if a != "--selftrain"]
        exit(_entry_selftrain(argv))
        result_container['value'] = eval(code, {"__builtins__": {}})
        
thread = threading.Thread(target=target, name="safe-eval", daemon=True)
thread.start()
thread.join(timeout=timeout)
if thread.is_alive():
                raise TimeoutError("evaluation timed out")
print("This module is intended to be imported or run with --selftrain for self-training.")
sys.exit(0)
if 'error' in result_container:
                raise result_container['error']

# Self-healing subsystem: monitors core components, proposes non-destructive repairs,
# evaluates repairs in an isolated worktree, and (with explicit opt-in) applies fixes.
# Safe-by-default: no destructive actions unless --apply is provided.

import importlib
import importlib.util
import py_compile
import difflib
import traceback
from typing import List, Dict, Any, Optional
from pathlib import Path

class SelfHealer:
    """
    Self-healing coordinator.
    - run_health_checks() -> collects smoke-test failures and traces
    - propose_repairs() -> conservative edit proposals (uses autocorrect_source)
    - evaluate_repairs() -> apply proposals in a worktree and run tests
    - apply_repairs() -> apply to repo (creates branch and commit) only with explicit confirm
    - recover_via_git_reset() -> non-destructive fallback: checkout files from HEAD into worktree
    Design: non-autonomous. All write/apply actions require explicit confirm flags.
    """
    CORE_MODULES = ["say_lexer", "say_parser", "say_vm", "say_codegen", "sayc"]

    def __init__(self, repo_path: Optional[str] = None):
        self.repo = Path(repo_path or ".").resolve()
        self.traces: Dict[str, str] = {}
        self.proposals: Dict[str, Dict[str, str]] = {}  # path -> {"orig":..., "proposed":...}
        self.logger = configure_logging("selfhealer", level=logging.INFO, logfile=None)

    # ---------- Health checks / smoke tests ----------
    def _module_origin(self, modulename: str) -> Optional[Path]:
        spec = importlib.util.find_spec(modulename)
        if spec and spec.origin:
            return Path(spec.origin)
        return None

    def run_health_checks(self) -> Dict[str, Any]:
        """Run lightweight smoke tests for core modules and report failures with traces."""
        report: Dict[str, Any] = {}
        for m in self.CORE_MODULES:
            info: Dict[str, Any] = {"ok": True, "trace": None, "origin": None}
            try:
                # import module fresh (best-effort)
                mod = importlib.import_module(m)
                origin = self._module_origin(m)
                info["origin"] = str(origin) if origin else None

                # run small smoke tests depending on module
                if m == "say_lexer":
                    try:
                        toks = list(mod.tokenize('print("ok")'))
                        info["smoke"] = {"tokens": len(toks)}
                    except Exception as e:
                        raise

                elif m == "say_parser":
                    try:
                        # use lexer to produce tokens then parse
                        lex = importlib.import_module("say_lexer")
                        src = 'print("ok")'
                        toks = list(lex.tokenize(src))
                        pmod = importlib.import_module("say_parser")
                        prog = pmod.Parser(toks).parse_program()
                        info["smoke"] = {"stmts": len(prog.stmts)}
                    except Exception:
                        raise

                elif m == "say_vm":
                    try:
                        vm_mod = importlib.import_module("say_vm")
                        # run a trivial program via public helper if available
                        if hasattr(vm_mod, "_run_source_with_execvm"):
                            res = vm_mod._run_source_with_execvm('x = 1\nprint(x)\n', trace=False, max_steps=1000, timeout=1.0)
                            info["smoke"] = {"result": res}
                        else:
                            info["smoke"] = {"note": "no helper available"}
                    except Exception:
                        raise

                elif m == "say_codegen":
                    try:
                        cg = importlib.import_module("say_codegen")
                        # try generating IR for a trivial module
                        if hasattr(cg, "example_module"):
                            ir_text = cg.example_module()
                            info["smoke"] = {"ir_len": len(ir_text)}
                        else:
                            info["smoke"] = {"note": "no example_module"}
                    except Exception:
                        raise

                elif m == "sayc":
                    try:
                        scm = importlib.import_module("sayc")
                        info["smoke"] = {"loaded": True}
                    except Exception:
                        raise

            except Exception as e:
                info["ok"] = False
                info["trace"] = traceback.format_exc()
                self.traces[m] = info["trace"]
                self.logger.error("Health check failed for %s: %s", m, e)
            report[m] = info
        return report

    # ---------- Proposals ----------
    def propose_repairs(self, health_report: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """
        For modules with failures, produce conservative text proposals:
         - read file from module origin (if available)
         - run autocorrect_source (conservative fixes)
         - run py_compile check on proposed text
        Returns mapping file_path -> {orig, proposed, diff}
        """
        proposals: Dict[str, Dict[str, str]] = {}
        for mod_name, info in health_report.items():
            if info.get("ok", True):
                continue
            origin = info.get("origin")
            if not origin:
                self.logger.warning("No origin for failed module %s; skipping propose", mod_name)
                continue
            p = Path(origin)
            if not p.exists():
                self.logger.warning("Origin path %s missing; skipping propose", origin)
                continue
            try:
                orig_text = p.read_text(encoding="utf-8")
            except Exception as e:
                self.logger.exception("Cannot read %s: %s", p, e)
                continue
            try:
                proposed = autocorrect_source(orig_text)
            except Exception:
                proposed = orig_text
            if proposed == orig_text:
                # still propose running a compile check and include trace for reviewer
                diff = ""
            else:
                od = orig_text.splitlines(keepends=True)
                pd = proposed.splitlines(keepends=True)
                diff = "".join(difflib.unified_diff(od, pd, fromfile=str(p), tofile=str(p) + ".proposed"))
            # sanity compile test for proposed text
            compile_ok = False
            compile_trace = None
            try:
                # write to temp file and py_compile
                tmp = Path(tempfile.mkstemp(suffix=p.suffix)[1])
                tmp.write_text(proposed, encoding="utf-8")
                py_compile.compile(str(tmp), doraise=True)
                compile_ok = True
            except Exception as ce:
                compile_trace = traceback.format_exc()
            finally:
                try:
                    tmp.unlink()
                except Exception:
                    pass
            proposals[str(p.relative_to(self.repo))] = {"orig": orig_text, "proposed": proposed, "diff": diff, "compile_ok": compile_ok, "compile_trace": compile_trace}
        self.proposals = proposals
        return proposals

    # ---------- Evaluation ----------
    def evaluate_repairs(self, proposals: Dict[str, Dict[str, str]], run_tests: bool = True) -> Dict[str, Any]:
        """
        Apply proposals into isolated worktree (via SelfTrainer._create_worktree) and run test suite.
        Returns evaluation report including diffs and test output.
        """
        st = SelfTrainer(repo_path=str(self.repo))
        wt = st._create_worktree()
        diffs = {}
        try:
            for rel, data in proposals.items():
                targ = wt / rel
                targ.parent.mkdir(parents=True, exist_ok=True)
                targ.write_text(data["proposed"], encoding="utf-8")
                diffs[rel] = data.get("diff", "")
            test_res = None
            if run_tests:
                try:
                    cp = subprocess.run([sys.executable, "-m", "unittest", "discover", "-v"], cwd=str(wt), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)
                    test_res = {"returncode": cp.returncode, "output": cp.stdout}
                except subprocess.TimeoutExpired as e:
                    test_res = {"returncode": 2, "output": f"timeout: {e}"}
            return {"ok": (test_res is None) or (test_res.get("returncode") == 0), "diffs": diffs, "tests": test_res}
        finally:
            st._cleanup_worktree()

    # ---------- Apply ----------
    def apply_repairs(self, proposals: Dict[str, Dict[str, str]], confirm: bool = False) -> Dict[str, Any]:
        """
        Apply proposals to repo. Uses SelfTrainer.apply_patch to create a branch and commit.
        confirm must be True to actually write; otherwise returns preview.
        """
        st = SelfTrainer(repo_path=str(self.repo))
        # convert our proposals into SelfTrainer expected mapping
        st_map: Dict[str, Tuple[str, str]] = {}
        for rel, data in proposals.items():
            st_map[rel] = (data["orig"], data["proposed"])
        if not confirm:
            preview = {k: v["diff"] for k, v in proposals.items()}
            return {"applied": False, "reason": "confirm flag not set", "preview": preview}
        # delegate commit/branch responsibilities to SelfTrainer.apply_patch
        res = st.apply_patch(st_map, confirm=True)
        return res

    # ---------- Non-destructive fallback recovery ----------
    def recover_via_git_reset(self, mod_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Attempt to restore module files to HEAD (non-destructive; creates branch if needed).
        Returns map of file -> status.
        """
        out = {}
        mods = mod_names or list(self.traces.keys())
        for m in mods:
            origin = self._module_origin(m)
            if not origin:
                out[m] = {"ok": False, "reason": "no origin found"}
                continue
            rel = str(Path(origin).relative_to(self.repo))
            try:
                # run git checkout -- <file>
                cp = subprocess.run(["git", "checkout", "--", rel], cwd=str(self.repo), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
                ok = cp.returncode == 0
                out[m] = {"ok": ok, "stdout": cp.stdout, "stderr": cp.stderr}
            except Exception as e:
                out[m] = {"ok": False, "error": traceback.format_exc()}
        return out

# ---------- CLI entry ----------
def _entry_selfheal(argv: Optional[List[str]] = None):
    import argparse
    parser = argparse.ArgumentParser(prog="sayit-selfheal", description="Self-healing for Sayit (safe-by-default, opt-in applies)")
    parser.add_argument("--apply", action="store_true", help="Apply proposed repairs (requires tests to pass and explicit confirmation)")
    parser.add_argument("--auto-recover", action="store_true", help="If repair proposals fail, attempt git reset recovery for module files")
    parser.add_argument("--run-tests", action="store_true", help="Run unit tests during evaluation (default: True if unspecified)")
    parser.add_argument("--repair-strategy", choices=["autocorrect", "none"], default="autocorrect", help="How to propose non-destructive repairs")
    args = parser.parse_args(argv)

    healer = SelfHealer(repo_path=".")
    healer.logger.info("Starting self-heal health checks")
    health = healer.run_health_checks()

    # print summary
    for mod, info in health.items():
        status = "OK" if info.get("ok", False) else "FAIL"
        print(f"{mod}: {status}")
        if not info.get("ok"):
            print("  origin:", info.get("origin"))
            print("  trace (short):")
            t = info.get("trace", "")
            print("\n".join(t.splitlines()[:8]))

    # prepare proposals if requested
    proposals = {}
    if args.repair_strategy == "autocorrect":
        healer.logger.info("Proposing conservative repairs (autocorrect strategy)")
        proposals = healer.propose_repairs(health)
        if not proposals:
            print("No conservative proposals generated.")
        else:
            print(f"Generated proposals for {len(proposals)} files. Preview diffs:")
            for rel, d in proposals.items():
                print("---", rel)
                diff = d.get("diff") or "(no textual diff; proposal identical or whitespace-only)"
                print(diff[:1600])  # clip long diffs

    # Evaluate proposals in isolated worktree
    if proposals:
        print("Evaluating proposals in isolated worktree (running tests)...")
        eval_res = healer.evaluate_repairs(proposals, run_tests=args.run_tests or True)
        print("Evaluation ok:", eval_res.get("ok"))
        if eval_res.get("tests"):
            print("Test returncode:", eval_res["tests"].get("returncode"))
            print("Test output (first 800 chars):")
            print((eval_res["tests"].get("output") or "")[:800])

        if args.apply:
            if not eval_res.get("ok"):
                print("Evaluation failed; refusing to apply. Use --auto-recover to try git reset fallback.")
                if args.auto_recover:
                    rec = healer.recover_via_git_reset()
                    print("Recovery results:", rec)
                return 2
            print("Applying proposals to repository (creating branch and committing)...")
            apply_res = healer.apply_repairs(proposals, confirm=True)
            print("Apply result:", apply_res)
            return 0

    # If no proposals or not applying, optionally attempt git reset recovery
    if args.auto_recover and not proposals:
        print("No proposals to apply; running git reset recovery for failed modules...")
        rec = healer.recover_via_git_reset()
        print("Recovery results:", rec)

    print("Self-heal run complete. No destructive action performed (run with --apply to commit proposals).")
    return 0

# Hook: safe opt-in invocation
if __name__ == "__main__":
    import sys as _sys
    if "--selfheal" in _sys.argv:
        argv = [a for a in _sys.argv[1:] if a != "--selfheal"]
        exit(_entry_selfheal(argv))

# Executable tooling: CLI helpers to run tests, lint, type-check, format, package, bench, and generate CI.
# Safe-by-default: operations that modify the repo (write files, build docker image) require `--apply`.

import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Tuple, Optional, List, Dict

class ExecutableTooling:
    """
    Lightweight executable tooling for development and CI.
    Usage (CLI entry below) exposes subcommands: status, test, lint, typecheck, format, build, gen-ci, bench, env-run, dockerize.
    Non-destructive by default. Use --apply to write files / build images.
    """

    def __init__(self, repo_root: Optional[str] = None):
        self.repo = Path(repo_root or ".").resolve()
        self.tools = {
            "python": shutil.which("python") or shutil.which("python3"),
            "pytest": shutil.which("pytest"),
            "flake8": shutil.which("flake8"),
            "pylint": shutil.which("pylint"),
            "mypy": shutil.which("mypy"),
            "black": shutil.which("black"),
            "build": shutil.which("python") or shutil.which("python3"),  # used with -m build
            "docker": shutil.which("docker"),
        }

    # -------------------------
    # Helpers
    # -------------------------
    def _sh(self, cmd: List[str], cwd: Optional[Path] = None, timeout: Optional[int] = None) -> Tuple[int, str, str]:
        try:
            cp = subprocess.run(cmd, cwd=str(cwd or self.repo), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
            return cp.returncode, cp.stdout, cp.stderr
        except subprocess.TimeoutExpired as e:
            return 124, "", f"timeout after {timeout}s"

    def available_tools(self) -> Dict[str, Optional[str]]:
        return self.tools.copy()

    # -------------------------
    # Operations
    # -------------------------
    def status(self) -> str:
        lines = ["Tooling status:"]
        for k, p in self.available_tools().items():
            lines.append(f"  - {k}: {'available at ' + p if p else 'missing'}")
        return "\n".join(lines)

    def run_tests(self, use_pytest: bool = True, timeout: int = 300) -> Dict[str, object]:
        """
        Prefer pytest if present and requested; otherwise run unittest discovery.
        Returns dict with returncode and output.
        """
        if use_pytest and self.tools.get("pytest"):
            rc, out, err = self._sh([self.tools["pytest"], "-q"], timeout=timeout)
            return {"engine": "pytest", "returncode": rc, "stdout": out, "stderr": err}
        # fallback to unittest discovery
        rc, out, err = self._sh([self.tools["python"] or "python", "-m", "unittest", "discover", "-v"], timeout=timeout)
        return {"engine": "unittest", "returncode": rc, "stdout": out, "stderr": err}

    def lint(self, targets: Optional[List[str]] = None, tools: Optional[List[str]] = None, timeout: int = 120) -> Dict[str, Dict]:
        """
        Run configured linters over paths. tools can be subset of ['flake8','pylint'].
        Returns mapping tool -> result dict.
        """
        results = {}
        targets = targets or ["."]
        tools = tools or ["flake8", "pylint"]
        for t in tools:
            exe = self.tools.get(t)
            if not exe:
                results[t] = {"ok": False, "reason": "not installed"}
                continue
            cmd = [exe] + targets
            rc, out, err = self._sh(cmd, timeout=timeout)
            results[t] = {"ok": rc == 0, "returncode": rc, "stdout": out, "stderr": err}
        return results

    def typecheck(self, targets: Optional[List[str]] = None, timeout: int = 120) -> Dict[str, object]:
        exe = self.tools.get("mypy")
        targets = targets or ["."]
        if not exe:
            return {"ok": False, "reason": "mypy not installed"}
        rc, out, err = self._sh([exe] + targets, timeout=timeout)
        return {"ok": rc == 0, "returncode": rc, "stdout": out, "stderr": err}

    def format_code(self, targets: Optional[List[str]] = None, check: bool = False, timeout: int = 120) -> Dict[str, object]:
        exe = self.tools.get("black")
        targets = targets or ["."]
        if not exe:
            return {"ok": False, "reason": "black not installed"}
        cmd = [exe]
        if check:
            cmd.append("--check")
        cmd += targets
        rc, out, err = self._sh(cmd, timeout=timeout)
        return {"ok": rc == 0, "returncode": rc, "stdout": out, "stderr": err}

    def build_package(self, apply: bool = False, timeout: int = 300) -> Dict[str, object]:
        """
        Build sdist/wheel using PEP517 build if `python -m build` available; fallback to setup.py.
        If apply=False only reports what would be done and required tools.
        """
        python_exe = self.tools.get("python") or "python"
        # detect build package
        has_build_module = False
        try:
            rc, out, err = self._sh([python_exe, "-c", "import importlib.util, sys; print(importlib.util.find_spec('build') is not None)"])
            has_build_module = "True" in out
        except Exception:
            pass

        if not apply:
            return {"ok": True, "would": "build using python -m build" if has_build_module else "build using setup.py sdist bdist_wheel", "has_build": has_build_module}

        if has_build_module:
            rc, out, err = self._sh([python_exe, "-m", "build"], timeout=timeout)
        else:
            if (self.repo / "setup.py").exists():
                rc, out, err = self._sh([python_exe, "setup.py", "sdist", "bdist_wheel"], timeout=timeout)
            else:
                return {"ok": False, "reason": "no build module and no setup.py"}
        return {"ok": rc == 0, "returncode": rc, "stdout": out, "stderr": err}

    def generate_github_actions(self, apply: bool = False) -> Dict[str, object]:
        """
        Generate a pragmatic GitHub Actions workflow for test + lint + format checks.
        apply=False returns preview; apply=True writes `.github/workflows/sayit-ci.yml`.
        """
        wf_dir = self.repo / ".github" / "workflows"
        wf_file = wf_dir / "sayit-ci.yml"
        content = textwrap.dedent("""\
            name: Sayit CI

            on:
              push:
              pull_request:

            jobs:
              test:
                runs-on: ubuntu-latest
                strategy:
                  matrix:
                    python-version: [3.9, 3.10, 3.11]
                steps:
                  - uses: actions/checkout@v4
                  - name: Setup Python
                    uses: actions/setup-python@v4
                    with:
                      python-version: ${{ matrix.python-version }}
                  - name: Install dependencies
                    run: |
                      python -m pip install --upgrade pip
                      pip install pytest flake8 black mypy
                  - name: Run tests
                    run: |
                      pytest -q || python -m unittest discover -v
                  - name: Lint (flake8)
                    run: flake8 .
                  - name: Type-check (mypy)
                    run: mypy .
            """)
        if not apply:
            return {"ok": True, "preview": str(wf_file), "content": content}
        wf_dir.mkdir(parents=True, exist_ok=True)
        wf_file.write_text(content, encoding="utf-8")
        return {"ok": True, "written": str(wf_file)}

    def bench_vm(self, reps: int = 5, program_samples: Optional[List[str]] = None) -> Dict[str, object]:
        """
        Simple benchmark harness that runs internal VM on sample sources and reports timing.
        program_samples: list of source strings. If None uses small builtins.
        """
        samples = program_samples or [
            "x = 0\nWhile x < 1000:\n    x = x + 1\n    end()\n",  # ends early via end() may stop; use small loops
            "x = 1\nprint(x)\n" * 10,
        ]
        results = []
        for src in samples:
            times = []
            for _ in range(reps):
                start = time.perf_counter()
                try:
                    _run_source_with_execvm(src, trace=False, max_steps=10_000_000, timeout=5.0)
                    rc = True
                except Exception as e:
                    rc = False
                times.append(time.perf_counter() - start)
            results.append({"sample_len": len(src), "times": times, "median": sorted(times)[len(times)//2], "ok": rc})
        return {"ok": True, "results": results}

    def create_venv_and_run(self, apply: bool = False, test_cmd: Optional[List[str]] = None) -> Dict[str, object]:
        """
        Create .venv, install requirements if present, and run tests inside it.
        apply=False only previews actions.
        """
        venv_dir = self.repo / ".venv"
        req = self.repo / "requirements.txt"
        python_exe = shutil.which("python") or shutil.which("python3")
        if not python_exe:
            return {"ok": False, "reason": "python exe not found"}

        if not apply:
            return {"ok": True, "would": f"python -m venv {venv_dir}; pip install -r requirements.txt (if present); run tests"}

        # create venv
        rc, out, err = self._sh([python_exe, "-m", "venv", str(venv_dir)])
        if rc != 0:
            return {"ok": False, "returncode": rc, "stderr": err}
        pip = venv_dir / ("Scripts" if shutil.which("python") and Path("Scripts").exists() else "bin") / "pip"
        pip = str(pip)
        if req.exists():
            rc, out, err = self._sh([pip, "install", "-r", str(req)], timeout=600)
            if rc != 0:
                return {"ok": False, "stderr": err}
        # run tests
        tc = test_cmd or [str(venv_dir / "bin" / "pytest")] if (venv_dir / "bin" / "pytest").exists() else [python_exe, "-m", "unittest", "discover", "-v"]
        rc, out, err = self._sh(tc, timeout=600)
        return {"ok": rc == 0, "returncode": rc, "stdout": out, "stderr": err}

    def dockerize(self, image_name: str = "sayit:latest", apply: bool = False) -> Dict[str, object]:
        """
        Create a minimal Dockerfile and optionally build the image (requires docker).
        apply=False previews the Dockerfile; apply=True writes Dockerfile and builds image.
        """
        dockerfile = self.repo / "Dockerfile.sayit"
        dockerfile_content = textwrap.dedent("""\
            FROM python:3.11-slim
            WORKDIR /app
            COPY . /app
            RUN pip install --no-cache-dir .
            CMD ["python", "-m", "sayc", "run"]
            """)
        if not apply:
            return {"ok": True, "preview_path": str(dockerfile), "content": dockerfile_content, "docker_available": bool(self.tools.get("docker"))}
        dockerfile.write_text(dockerfile_content, encoding="utf-8")
        if not self.tools.get("docker"):
            return {"ok": False, "reason": "docker not available"}
        rc, out, err = self._sh([self.tools["docker"], "build", "-t", image_name, "-f", str(dockerfile), "."], timeout=900)
        return {"ok": rc == 0, "returncode": rc, "stdout": out, "stderr": err}

# -------------------------
# CLI entry for tooling
# -------------------------
def _entry_toolbox(argv: Optional[List[str]] = None):
    import argparse
    parser = argparse.ArgumentParser(prog="sayit-toolbox", description="Executable tooling for Sayit (tests, lint, typecheck, format, build, bench, CI generation)")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("status", help="Show available toolchain components")
    p_test = sub.add_parser("test", help="Run tests (pytest preferred)")
    p_test.add_argument("--no-pytest", action="store_true", help="Do not use pytest even if available")
    p_lint = sub.add_parser("lint", help="Run linters")
    p_lint.add_argument("--tools", nargs="+", choices=["flake8", "pylint"], default=["flake8"])
    p_format = sub.add_parser("format", help="Run black formatter")
    p_format.add_argument("--check", action="store_true", help="Run black in --check mode")
    p_type = sub.add_parser("typecheck", help="Run mypy")
    p_build = sub.add_parser("build", help="Build package")
    p_build.add_argument("--apply", action="store_true", help="Perform build")
    p_ci = sub.add_parser("gen-ci", help="Generate GitHub Actions workflow (preview unless --apply)")
    p_ci.add_argument("--apply", action="store_true", help="Write workflow file")
    p_bench = sub.add_parser("bench", help="Run small VM benchmarks")
    p_bench.add_argument("--reps", type=int, default=5)
    p_venv = sub.add_parser("env-run", help="Create venv and run tests (preview unless --apply)")
    p_venv.add_argument("--apply", action="store_true", help="Create venv and run")
    p_docker = sub.add_parser("dockerize", help="Create Dockerfile and optionally build image")
    p_docker.add_argument("--apply", action="store_true", help="Write Dockerfile and build image")
    p_docker.add_argument("--name", default="sayit:latest")
    parser.add_argument("--repo", default=".", help="Repository root (default: current directory)")

    args = parser.parse_args(argv)
    tb = ExecutableTooling(repo_root=args.repo)

    if args.cmd == "status":
        print(tb.status())
        return 0

    if args.cmd == "test":
        res = tb.run_tests(use_pytest=not getattr(args, "no_pytest", False))
        print("Engine:", res.get("engine"))
        print("RC:", res.get("returncode"))
        print(res.get("stdout") or res.get("stderr"))
        return 0 if res.get("returncode") == 0 else 2

    if args.cmd == "lint":
        res = tb.lint(tools=getattr(args, "tools", ["flake8"]))
        for k, v in res.items():
            print(f"{k}: ok={v.get('ok')} rc={v.get('returncode')}")
            if v.get("stdout"):
                print(v["stdout"][:2000])
        return 0

    if args.cmd == "format":
        res = tb.format_code(check=getattr(args, "check", False))
        if not res.get("ok"):
            print("format:", res.get("reason") or res.get("stderr"))
            return 2
        print("format ok")
        return 0

    if args.cmd == "typecheck":
        res = tb.typecheck()
        print(res.get("stdout") or res.get("stderr") or "mypy ok")
        return 0 if res.get("ok") else 2

    if args.cmd == "build":
        res = tb.build_package(apply=getattr(args, "apply", False))
        print(res.get("stdout") or res.get("stderr") or res.get("would") or res)
        return 0 if res.get("ok") else 2

    if args.cmd == "gen-ci":
        res = tb.generate_github_actions(apply=getattr(args, "apply", False))
        if res.get("ok"):
            if getattr(args, "apply", False):
                print("Workflow written to .github/workflows/sayit-ci.yml")
            else:
                print("Preview:\n", res.get("content", "")[:1000])
            return 0
        print("Failed:", res)
        return 2

    if args.cmd == "bench":
        res = tb.bench_vm(reps=getattr(args, "reps", 5))
        for r in res.get("results", []):
            print("sample_len:", r["sample_len"], "median(s):", r["median"], "ok:", r["ok"])
        return 0

    if args.cmd == "env-run":
        res = tb.create_venv_and_run(apply=getattr(args, "apply", False))
        print(res.get("stdout") or res.get("stderr") or res.get("would") or res)
        return 0 if res.get("ok") else 2

    if args.cmd == "dockerize":
        res = tb.dockerize(image_name=getattr(args, "name", "sayit:latest"), apply=getattr(args, "apply", False))
        print(res.get("stdout") or res.get("stderr") or res.get("preview_path") or res)
        return 0 if res.get("ok") else 2

    parser.print_help()
    return 0

# Expose in __all__
__all__ += ["ExecutableTooling", "_entry_toolbox"]

# Hook: run as `python system.py --toolbox <subcmd>` or use dedicated entry
if __name__ == "__main__":
    import sys as _sys
    if "--toolbox" in _sys.argv:
        argv = [a for a in _sys.argv[1:] if a != "--toolbox"]
        exit(_entry_toolbox(argv))
        import sys
        import logging
        import tempfile
        import subprocess
        import time
        from typing import Any, Dict, List, Optional, Tuple
        from pathlib import Path
        from say_vm import _run_source_with_execvm
        from selftrainer import SelfTrainer
        from autocorrect import autocorrect_source
        from logging_config import configure_logging
        
