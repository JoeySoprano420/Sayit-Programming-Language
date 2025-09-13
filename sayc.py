# sayc.py
import sys
from say_lexer import tokenize
from say_parser import Parser
from say_vm import VM
from say_codegen import Codegen

def main():
    if len(sys.argv) < 2:
        print("Usage: sayc <file.say> [--ir] [--out <output.ll>]")
        return

    filename = sys.argv[1]
    code = open(filename).read()
    tokens = tokenize(code)
    parser = Parser(tokens)
    program = parser.parse_program()

    # IR generation mode
    if "--ir" in sys.argv:
        cg = Codegen()
        cg.emit_main()
        for stmt in program.stmts:
            if stmt.__class__.__name__ == "Print":
                # For now, just handle literal/identifier printing
                if hasattr(stmt.expr, "val"):   # literal
                    cg.emit_print(str(stmt.expr.val))
                else:   # fallback for identifiers
                    cg.emit_print(f"<{stmt.expr.name}>")
        llvm_ir = cg.finish()

        # output file if requested
        if "--out" in sys.argv:
            out_index = sys.argv.index("--out") + 1
            if out_index < len(sys.argv):
                with open(sys.argv[out_index], "w") as f:
                    f.write(llvm_ir)
                print(f"[sayc] LLVM IR written to {sys.argv[out_index]}")
            else:
                print("[sayc] Missing filename after --out")
        else:
            print(llvm_ir)

    # VM execution mode (default)
    else:
        vm = VM()
        vm.run(program)

if __name__ == "__main__":
    main()
