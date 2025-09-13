# say_codegen.py
from llvmlite import ir

class Codegen:
    def __init__(self):
        self.module = ir.Module(name="sayit")
        self.func = None
        self.builder = None

    def emit_main(self):
        """Create main() function and entry block"""
        fnty = ir.FunctionType(ir.IntType(32), [])
        self.func = ir.Function(self.module, fnty, name="main")
        block = self.func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)

    def emit_print(self, text: str):
        """Emit a printf call with a string literal"""
        voidptr = ir.IntType(8).as_pointer()
        printf_ty = ir.FunctionType(ir.IntType(32), [voidptr], var_arg=True)

        # Reuse or create printf declaration
        if "printf" in self.module.globals:
            printf = self.module.globals["printf"]
        else:
            printf = ir.Function(self.module, printf_ty, name="printf")

        # Create global string pointer
        cstr = self.builder.global_string_ptr(text, name="str")
        self.builder.call(printf, [cstr])

    def finish(self):
        """Return 0 and emit the final LLVM IR"""
        if self.builder.block.is_terminated is False:
            self.builder.ret(ir.IntType(32)(0))
        return str(self.module)
