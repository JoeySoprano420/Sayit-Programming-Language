# say_vm.py
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
                for s in stmt.then:
                    self.exec_stmt(s)
            else:
                executed = False
                for cond, body in stmt.elifs:
                    if self.eval(cond):
                        for s in body:
                            self.exec_stmt(s)
                        executed = True
                        break
                if not executed and stmt.els:
                    for s in stmt.els:
                        self.exec_stmt(s)

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
            if expr.op == ">=": return l >= r
            if expr.op == "<=": return l <= r
            if expr.op == "and": return bool(l and r)
            if expr.op == "or": return bool(l or r)

        return None
