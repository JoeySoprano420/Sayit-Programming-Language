# sayc.py
import sys
import os
import argparse
from say_lexer import tokenize
from say_parser import Parser
from say_vm import VM
# NOTE: do not import Codegen at module level to avoid hard dependency on llvmlite
# from say_codegen import Codegen

def compile_to_ir(program, out_filename=None):
    try:
        from say_codegen import Codegen
    except Exception as e:
        print("[sayc] Error: Failed to import Codegen (llvmlite may be missing).")
        print(f"[sayc] Import error: {e}")
        print("[sayc] Install llvmlite into the interpreter used to run this tool, or run without --ir.")
        raise

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

    if out_filename:
        with open(out_filename, "w", encoding="utf-8") as f:
            f.write(llvm_ir)
        print(f"[sayc] LLVM IR written to {out_filename}")
    else:
        print(llvm_ir)

def run_vm(program):
    vm = VM()
    vm.run(program)

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(prog="sayc", description="Sayit compiler / VM runner")
    parser.add_argument("file", nargs="?", help="Source file (.say) to compile/run")
    parser.add_argument("--ir", action="store_true", help="Emit LLVM IR instead of running VM")
    parser.add_argument("--out", metavar="OUTPUT", help="Write emitted LLVM IR to file")
    parser.add_argument("--version", action="store_true", help="Show version info")
    args = parser.parse_args(argv)

    if args.version:
        print("sayc 0.1")
        return

    if not args.file:
        parser.print_usage()
        print("Try: sayc <file.say> [--ir] [--out <output.ll>]")
        return

    filename = args.file
    if not os.path.isfile(filename):
        print(f"[sayc] Error: File not found: {filename}")
        return

    try:
        code = open(filename, "r", encoding="utf-8").read()
    except Exception as e:
        print(f"[sayc] Error reading {filename}: {e}")
        return

    try:
        tokens = tokenize(code)
        parser_obj = Parser(tokens)
        program = parser_obj.parse_program()
    except Exception as e:
        print(f"[sayc] Error while parsing {filename}: {e}")
        return

    if args.ir:
        try:
            compile_to_ir(program, out_filename=args.out)
        except Exception:
            # compile_to_ir already prints helpful messages
            return
    else:
        try:
            run_vm(program)
        except Exception as e:
            print(f"[sayc] Runtime error: {e}")
            return

if __name__ == "__main__":
    main()


