# say_codegen.py
from llvmlite import ir

class Codegen:
    def __init__(self):
        self.module = ir.Module("sayit")
        self.func = None
        self.builder = None

    def emit_main(self):
        fnty = ir.FunctionType(ir.IntType(32), [])
        self.func = ir.Function(self.module, fnty, name="main")
        block = self.func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)

    def emit_print(self, msg):
        voidptr = ir.IntType(8).as_pointer()
        printf_ty = ir.FunctionType(ir.IntType(32), [voidptr], var_arg=True)
        printf = self.module.globals.get("printf") or \
                 ir.Function(self.module, printf_ty, name="printf")
        cstr = self.builder.global_string(msg, name="str")
        self.builder.call(printf, [cstr])

from llvmlite import ir

class Codegen:
    def __init__(self):
        self.module = ir.Module(name="sayit")
        self.func = None
        self.builder = None

    def emit_main(self):
        fnty = ir.FunctionType(ir.IntType(32), [])
        self.func = ir.Function(self.module, fnty, name="main")
        block = self.func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)

    def emit_print(self, text):
        voidptr = ir.IntType(8).as_pointer()
        printf_ty = ir.FunctionType(ir.IntType(32), [voidptr], var_arg=True)
        printf = self.module.globals.get("printf") or \
                 ir.Function(self.module, printf_ty, name="printf")
        cstr = self.builder.global_string_ptr(text)
        self.builder.call(printf, [cstr])

    def finish(self):
        self.builder.ret(ir.IntType(32)(0))
        return str(self.module)
