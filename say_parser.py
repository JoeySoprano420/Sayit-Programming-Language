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
