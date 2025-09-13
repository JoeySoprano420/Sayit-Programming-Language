# sayc.py
import sys
from say_lexer import tokenize
from say_parser import Parser
from say_vm import VM

def main():
    if len(sys.argv) < 2:
        print("Usage: sayc <file.say>")
        return
    code = open(sys.argv[1]).read()
    tokens = tokenize(code)
    parser = Parser(tokens)
    ast = parser.parse_program()
    vm = VM()
    vm.run(ast)

if __name__ == "__main__":
    main()

import sys
from say_lexer import tokenize
from say_parser import Parser
from say_vm import VM
from say_codegen import Codegen

def main():
    if len(sys.argv) < 2:
        print("Usage: sayc <file.say> [--ir]")
        return

    code = open(sys.argv[1]).read()
    tokens = tokenize(code)
    parser = Parser(tokens)
    program = parser.parse_program()

    if "--ir" in sys.argv:
        cg = Codegen()
        cg.emit_main()
        for stmt in program.stmts:
            if stmt.__class__.__name__ == "Print":
                cg.emit_print(stmt.expr.val)
        print(cg.finish())
    else:
        vm = VM()
        vm.run(program)

if __name__ == "__main__":
    main()
