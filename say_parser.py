# say_parser.py
class Node: pass

class Program(Node):
    def __init__(self, blocks): self.blocks = blocks

class Print(Node):
    def __init__(self, expr): self.expr = expr

class Assign(Node):
    def __init__(self, ident, expr): self.ident, self.expr = ident, expr

class BinOp(Node):
    def __init__(self, left, op, right): self.left, self.op, self.right = left, op, right

# Parser driver
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
