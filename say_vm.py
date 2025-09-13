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
