# say_vm.py
class VM:
    def __init__(self):
        self.vars = {}

    def run(self, node):
        if isinstance(node, Print):
            print(self.eval(node.expr))
        elif isinstance(node, Assign):
            self.vars[node.ident] = self.eval(node.expr)

    def eval(self, expr):
        if isinstance(expr, str):
            return expr.strip('"')
        if isinstance(expr, int):
            return expr
        if isinstance(expr, BinOp):
            l, r = self.eval(expr.left), self.eval(expr.right)
            if expr.op == "+": return l + r
            if expr.op == "==": return l == r
        return None

from say_parser import Print, Assign, BinOp, Literal, Ident, If, While

class VM:
    def __init__(self):
        self.vars = {}

    def run(self, program):
        for stmt in program.stmts:
            self.exec_stmt(stmt)

    def exec_stmt(self, stmt):
        if isinstance(stmt, Print):
            print(self.eval(stmt.expr))
        elif isinstance(stmt, Assign):
            self.vars[stmt.ident] = self.eval(stmt.expr)
        elif isinstance(stmt, If):
            if self.eval(stmt.cond):
                for s in stmt.then: self.exec_stmt(s)
            else:
                done = False
                for cond, body in stmt.elifs:
                    if self.eval(cond):
                        for s in body: self.exec_stmt(s)
                        done = True
                        break
                if not done and stmt.els:
                    for s in stmt.els: self.exec_stmt(s)
        elif isinstance(stmt, While):
            while self.eval(stmt.cond):
                for s in stmt.body:
                    self.exec_stmt(s)

    def eval(self, expr):
        if isinstance(expr, Literal):
            return expr.val
        if isinstance(expr, Ident):
            return self.vars.get(expr.name, 0)
        if isinstance(expr, BinOp):
            l, r = self.eval(expr.left), self.eval(expr.right)
            if expr.op == "+": return l + r
            if expr.op == "-": return l - r
            if expr.op == "*": return l * r
            if expr.op == "/": return l // r
            if expr.op == "==": return l == r
            if expr.op == "!=": return l != r
            if expr.op == ">": return l > r
            if expr.op == "<": return l < r
        return None
