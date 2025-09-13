# say_codegen.py
from llvmlite import ir

class Codegen:
    def __init__(self, module_name="sayit"):
        # === Create LLVM Module ===
        self.module = ir.Module(name=module_name)
        self.module.triple = "x86_64-pc-linux-gnu"  # default triple
        self.module.data_layout = ""  # can be filled by target machine later

        self.func = None
        self.builder = None
        self.printf = None

    # ---------------------------
    # Function Setup
    # ---------------------------
    def emit_main(self):
        """Create a main() entry function."""
        fnty = ir.FunctionType(ir.IntType(32), [])
        self.func = ir.Function(self.module, fnty, name="main")
        block = self.func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)

        # Declare printf once
        voidptr = ir.IntType(8).as_pointer()
        printf_ty = ir.FunctionType(ir.IntType(32), [voidptr], var_arg=True)
        self.printf = ir.Function(self.module, printf_ty, name="printf")

    # ---------------------------
    # Printing Support
    # ---------------------------
    def emit_print(self, text: str):
        """Emit a printf call with a string literal."""
        if self.builder is None:
            raise RuntimeError("emit_main() must be called first")

        # Global string literal
        cstr = self.builder.global_string_ptr(text, name="str")
        self.builder.call(self.printf, [cstr])

    # ---------------------------
    # Variables (alloca/store/load)
    # ---------------------------
    def emit_var(self, name: str, init_val: int):
        """Allocate and initialize an integer variable."""
        ptr = self.builder.alloca(ir.IntType(32), name=name)
        self.builder.store(ir.Constant(ir.IntType(32), init_val), ptr)
        return ptr

    def emit_load(self, ptr):
        """Load an integer value."""
        return self.builder.load(ptr)

    def emit_binop(self, op, lhs, rhs):
        """Binary operations on integers."""
        if op == "+":
            return self.builder.add(lhs, rhs)
        if op == "-":
            return self.builder.sub(lhs, rhs)
        if op == "*":
            return self.builder.mul(lhs, rhs)
        if op == "/":
            return self.builder.sdiv(lhs, rhs)
        raise NotImplementedError(f"Unsupported binop {op}")

    # ---------------------------
    # Finisher
    # ---------------------------
    def finish(self):
        """Return from main() and dump LLVM IR as string."""
        if self.builder is not None and not self.builder.block.is_terminated:
            self.builder.ret(ir.IntType(32)(0))
        return str(self.module)
