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
