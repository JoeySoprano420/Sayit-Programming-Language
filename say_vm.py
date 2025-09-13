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
# ---------------------------
# CLI, REPL and tracing helpers for the VM
# ---------------------------
import argparse
import os
import sys
from say_lexer import tokenize as _tokenize
from say_parser import Parser as _Parser

class TracingVM(VM):
    """Lightweight VM that prints trace info for each statement evaluation."""
    def exec_stmt(self, stmt):
        print(f"[trace] exec_stmt: {stmt.__class__.__name__}")
        return super().exec_stmt(stmt)

    def eval(self, expr):
        result = super().eval(expr)
        print(f"[trace] eval: {expr.__class__.__name__} -> {result}")
        return result

def _run_source(source: str, trace: bool = False):
    """Parse source text and run it on a VM (optionally traced)."""
    tokens = list(_tokenize(source))
    parser = _Parser(tokens)
    program = parser.parse_program()
    vm = TracingVM() if trace else VM()
    vm.run(program)
    return vm

def run_file(path: str, trace: bool = False):
    """Read a file and execute it with the VM (returns the VM instance)."""
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    return _run_source(src, trace=trace)

def repl(trace: bool = False):
    """Simple REPL: enter single-line statements (prints and assignments supported)."""
    print("Sayit REPL (type 'exit' or 'quit' to leave).")
    vm = TracingVM() if trace else VM()
    buffer = ""
    try:
        while True:
            try:
                line = input("say> ")
            except EOFError:
                print()
                break
            if not line:
                continue
            if line.strip() in ("exit", "quit"):
                break

            # Try to parse and execute this single line.
            try:
                prog = _Parser(list(_tokenize(line))).parse_program()
                vm.run(prog)
            except Exception as e:
                # Try to give helpful feedback without crashing the REPL
                print(f"[repl] Error: {e}")
    except KeyboardInterrupt:
        print("\n[repl] Interrupted.")
    return vm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="say_vm", description="Run Sayit programs via the VM")
    parser.add_argument("file", nargs="?", help="Source file to run (.say)")
    parser.add_argument("--repl", action="store_true", help="Start an interactive REPL")
    parser.add_argument("--trace", action="store_true", help="Enable execution tracing")
    parser.add_argument("--dump-vars", action="store_true", help="Dump VM variables after run")
    parser.add_argument("--demo", action="store_true", help="Run a tiny demo program")
    args = parser.parse_args()

    vm_instance = None

    if args.demo:
        demo_src = 'x = 1\nprint(x)\nprint("Hello from Sayit VM")\n'
        vm_instance = _run_source(demo_src, trace=args.trace)

    elif args.repl:
        vm_instance = repl(trace=args.trace)

    elif args.file:
        if not os.path.isfile(args.file):
            print(f"File not found: {args.file}")
            sys.exit(1)
        try:
            vm_instance = run_file(args.file, trace=args.trace)
        except Exception as e:
            print(f"[say_vm] Error: {e}")
            sys.exit(1)

    else:
        parser.print_usage()
        sys.exit(1)

    if args.dump_vars and vm_instance is not None:
        print("VM variables:")
        for k, v in sorted(vm_instance.vars.items()):
            print(f"  {k} = {v}")
