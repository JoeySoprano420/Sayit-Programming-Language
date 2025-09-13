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
        raise SystemExit(1)

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
