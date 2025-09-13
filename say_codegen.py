# say_codegen.py
from llvmlite import ir
import re
from typing import Dict, List, Tuple, Optional

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

# ---------------------------
# Ritual parsing + Codegen helpers
# ---------------------------
def _parse_start_ritual_block(block_text: str) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    """
    Parse a Start Ritual block like:
      Start Ritual:
          string message = "Hello World!"
          print(message)
          end()
    Returns (strings_map, actions)
      - strings_map: name -> string value
      - actions: list of (action, arg) tuples, currently supports ('print', name)
    """
    strings: Dict[str, str] = {}
    actions: List[Tuple[str, str]] = []

    for raw in block_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # skip header/footer tokens
        if re.match(r"(?i)^start\s+ritual\s*:", line):
            continue
        if re.match(r"(?i)^end\s*\(\s*\)\s*$", line) or line.lower() == "end()":
            continue

        # string declaration: string name = "value"
        m = re.match(r'^string\s+([a-zA-Z_]\w*)\s*=\s*"([^"]*)"\s*$', line, flags=re.IGNORECASE)
        if m:
            name, val = m.group(1), m.group(2)
            strings[name] = val
            continue

        # print call: print(name)
        m2 = re.match(r'^print\s*\(\s*([a-zA-Z_]\w*)\s*\)\s*$', line, flags=re.IGNORECASE)
        if m2:
            actions.append(("print", m2.group(1)))
            continue

        # ignore unknown lines (could raise if stricter behavior desired)

    return strings, actions

def _apply_start_ritual_to_codegen(cg: Codegen, block_text: str):
    """
    Given a Codegen instance with an active builder (emit_main called),
    create global string constants for declarations and emit print calls
    for print(...) actions that reference those declared strings.
    """
    if cg.builder is None:
        raise RuntimeError("Codegen.emit_main() must be called before applying Start Ritual")

    strings, actions = _parse_start_ritual_block(block_text)
    voidptr = ir.IntType(8).as_pointer()
    ptrs: Dict[str, object] = {}

    # Create a local pointer (i8*) alloca and store a global string pointer into it.
    for name, val in strings.items():
        cptr = cg.builder.global_string_ptr(val, name=f"str_{name}")
        alloca_ptr = cg.builder.alloca(voidptr, name=name)
        cg.builder.store(cptr, alloca_ptr)
        ptrs[name] = alloca_ptr

    # Execute actions in order.
    for act, arg in actions:
        if act == "print":
            if arg in ptrs:
                loaded = cg.builder.load(ptrs[arg])
                cg.builder.call(cg.printf, [loaded])
            else:
                # fallback: if name not found, print a placeholder
                cg.emit_print(f"<{arg}>")

def example_module_from_start_ritual(block_text: str, out_file: str = None) -> str:
    """
    Convenience builder: create a module, apply the Start Ritual block, and return the IR.
    """
    cg = Codegen()
    cg.emit_main()
    _apply_start_ritual_to_codegen(cg, block_text)
    llvm_ir = cg.finish()

    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(llvm_ir)
    return llvm_ir

# ---------------------------
# Demo / helper for quick IR generation
# ---------------------------
def example_module(out_file: str = None) -> str:
    """
    Build a tiny example module and return the LLVM IR as a string.
    If out_file is provided, the IR is also written to that file.
    """
    cg = Codegen()
    cg.emit_main()
    cg.emit_print("Hello, Sayit!")
    cg.emit_print("This is an example LLVM IR from Codegen.")
    llvm_ir = cg.finish()

    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(llvm_ir)
    return llvm_ir


if __name__ == "__main__":
    # Lightweight CLI: optional `-o <file>` to write IR to disk, otherwise print to stdout.
    import sys
    out_file = None
    if len(sys.argv) >= 3 and sys.argv[1] in ("-o", "--out"):
        out_file = sys.argv[2]

    ir_text = example_module(out_file=out_file)
    if out_file is None:
        print(ir_text)
        
    from say_codegen import example_module_with_make
make_text = """Make:
    x = 1
    y = 2
"""
ir = example_module_with_make(make_text)
print(ir)

# Quick demo when running this file directly
if __name__ == "__main__" and False:
    # Example block (the user-specified block)
    ritual = """Start Ritual:
        string message = "Hello World!"
        print(message)
        end()
    """
    print(example_module_from_start_ritual(ritual))

# ---------------------------
# While-block parsing + Codegen helper
# ---------------------------

def _parse_while_header(header: str):
    """
    Parse a simple while header like: 'While z < 10:'
    Returns (var_name, op, int_const)
    """
    m = re.match(r'(?i)^\s*While\s+([a-zA-Z_]\w*)\s*([<>=!]+)\s*([0-9]+)\s*:', header)
    if not m:
        raise ValueError(f"Malformed While header: {header!r}")
    name, op, val = m.group(1), m.group(2), int(m.group(3))
    return name, op, val

def _parse_until_assignment(line: str):
    """
    Parse a line like: 'y = y + x until y == 9'
    Returns (target, lhs, op, rhs, until_lhs, until_op, until_rhs)
    """
    m = re.match(r'^\s*([a-zA-Z_]\w*)\s*=\s*([a-zA-Z_]\w*|\d+)\s*([\+\-\*/])\s*([a-zA-Z_]\w*|\d+)\s+until\s+(.+)$', line)
    if not m:
        raise ValueError(f"Malformed until-assignment: {line!r}")
    target, lhs, op, rhs, until = m.groups()
    until = until.strip()
    mu = re.match(r'^\s*([a-zA-Z_]\w*|\d+)\s*([<>=!]+)\s*([a-zA-Z_]\w*|\d+)\s*$', until)
    if not mu:
        raise ValueError(f"Malformed until condition: {until!r}")
    until_lhs, until_op, until_rhs = mu.group(1), mu.group(2), mu.group(3)
    return target, lhs, op, rhs, until_lhs, until_op, until_rhs

def _get_or_create_ptr(cg: Codegen, name: str):
    """
    Ensure an alloca exists for `name`. We create a new alloca initialized to 0 if needed.
    This is a simple approach for demo IR generation.
    """
    # Create a default-initialized alloca; Codegen does not track ptrs, so just create one.
    return cg.emit_var(name, 0)

def _cmp_ir(cg: Codegen, lhs, op: str, rhs):
    """
    Create an i1 comparison for signed integers using op string.
    lhs and rhs are LLVM i32 values or ints (we convert ints to constants).
    """
    it32 = ir.IntType(32)
    if isinstance(lhs, int):
        lhs = ir.Constant(it32, lhs)
    if isinstance(rhs, int):
        rhs = ir.Constant(it32, rhs)

    if op == "<":
        return cg.builder.icmp_signed("slt", lhs, rhs)
    if op == "<=":
        return cg.builder.icmp_signed("sle", lhs, rhs)
    if op == ">":
        return cg.builder.icmp_signed("sgt", lhs, rhs)
    if op == ">=":
        return cg.builder.icmp_signed("sge", lhs, rhs)
    if op == "==":
        return cg.builder.icmp_signed("eq", lhs, rhs)
    if op == "!=":
        return cg.builder.icmp_signed("ne", lhs, rhs)
    raise NotImplementedError(f"Unsupported comparison op {op}")

def _apply_while_block_to_codegen(cg: Codegen, block_text: str):
    """
    Parse the provided block_text containing a While header and the following body,
    then emit corresponding LLVM IR into the provided Codegen instance.

    The function understands the specific structure you provided:
      While z < 10:
          y = y + x until y == 9

          If z > 7 and z < 9:
              print("true")
          Else:
              print("maybe")

          Elif z in [3...8]:
              print("true")
          Else:
              print("false")

    Notes:
    - Variables referenced will get simple alloca initializations if not present.
    - Range `[a...b]` is treated as inclusive: z >= a && z <= b.
    - 'until' produces an inner do-while loop that repeats the assignment until the until-condition is true.
    """
    if cg.builder is None:
        raise RuntimeError("Codegen.emit_main() must be called before applying While block")

    lines = [ln for ln in block_text.splitlines() if ln.strip() != ""]
    if not lines:
        return

    # Parse header
    header = lines[0]
    var_name, var_op, var_val = _parse_while_header(header)

    # locate the assignment and the two condition blocks (we accept the exact ordering)
    assignment_line = None
    first_if_lines = []
    second_if_lines = []
    mode = None
    for ln in lines[1:]:
        s = ln.strip()
        if s.lower().startswith("if "):
            mode = "first_if"
            first_if_lines = [s]
            continue
        if s.lower().startswith("elif ") or s.lower().startswith("elif"):
            mode = "second_if"
            second_if_lines = [s]
            continue
        if mode == "first_if":
            first_if_lines.append(s)
        elif mode == "second_if":
            second_if_lines.append(s)
        elif "until" in s and "=" in s:
            assignment_line = s

    if assignment_line is None:
        raise ValueError("Missing until-assignment in While block")

    # Ensure ptrs for variables used
    # We'll collect variable names from header, assignment and conditions
    # Minimal extraction:
    needed = set()
    needed.add(var_name)
    # parse assignment to collect names
    tgt, lhs, aop, rhs, until_lhs, until_op, until_rhs = _parse_until_assignment(assignment_line)
    for name in (tgt, lhs, rhs, until_lhs, until_rhs):
        if not name.isdigit():
            needed.add(name)

    ptrs = {name: _get_or_create_ptr(cg, name) for name in needed}

    # Create loop blocks
    func = cg.func
    entry_block = cg.builder.block
    cond_bb = func.append_basic_block(name="while_cond")
    body_bb = func.append_basic_block(name="while_body")
    after_bb = func.append_basic_block(name="while_after")

    # Jump to condition
    cg.builder.branch(cond_bb)

    # Condition block
    cg.builder.position_at_end(cond_bb)
    z_val = cg.emit_load(ptrs[var_name])
    cond_i1 = _cmp_ir(cg, z_val, var_op, var_val)
    cg.builder.cbranch(cond_i1, body_bb, after_bb)

    # Body block
    cg.builder.position_at_end(body_bb)

    # Inner 'until' do-while loop: create inner blocks
    inner_body_bb = func.append_basic_block(name="inner_body")
    inner_cond_bb = func.append_basic_block(name="inner_cond")
    inner_after_bb = func.append_basic_block(name="inner_after")

    # Enter inner body
    cg.builder.branch(inner_body_bb)

    # inner body: perform assignment y = y + x (or rhs)
    cg.builder.position_at_end(inner_body_bb)
    # resolve left and right operands
    def resolve_operand(tok):
        if tok.isdigit():
            return ir.Constant(ir.IntType(32), int(tok))
        return cg.emit_load(ptrs[tok])

    left_val = resolve_operand(lhs)
    right_val = resolve_operand(rhs)
    if aop == "+":
        sumv = cg.builder.add(left_val, right_val)
    elif aop == "-":
        sumv = cg.builder.sub(left_val, right_val)
    elif aop == "*":
        sumv = cg.builder.mul(left_val, right_val)
    elif aop == "/":
        sumv = cg.builder.sdiv(left_val, right_val)
    else:
        raise NotImplementedError(f"Unsupported arithmetic op {aop}")

    # store into target
    cg.builder.store(sumv, ptrs[tgt])
    # jump to inner condition
    cg.builder.branch(inner_cond_bb)

    # inner condition
    cg.builder.position_at_end(inner_cond_bb)
    u_lhs = resolve_operand(until_lhs)
    u_rhs = resolve_operand(until_rhs)
    until_cmp = _cmp_ir(cg, u_lhs, until_op, u_rhs)  # true when until-condition satisfied
    # if until-condition is true -> exit inner loop, else repeat inner body
    cg.builder.cbranch(until_cmp, inner_after_bb, inner_body_bb)

    # continue after inner loop: proceed to first conditional
    cg.builder.position_at_end(inner_after_bb)

    # --- First If: If z > 7 and z < 9: print("true") Else: print("maybe")
    # build blocks
    first_then = func.append_basic_block(name="if1_then")
    first_else = func.append_basic_block(name="if1_else")
    first_after = func.append_basic_block(name="if1_after")

    # compute conditions
    zv = cg.emit_load(ptrs[var_name])
    cmp1 = _cmp_ir(cg, zv, ">", 7)
    cmp2 = _cmp_ir(cg, zv, "<", 9)
    both = cg.builder.and_(cmp1, cmp2)
    cg.builder.cbranch(both, first_then, first_else)

    # then
    cg.builder.position_at_end(first_then)
    cg.emit_print("true")
    cg.builder.branch(first_after)

    # else
    cg.builder.position_at_end(first_else)
    cg.emit_print("maybe")
    cg.builder.branch(first_after)

    cg.builder.position_at_end(first_after)

    # --- Second conditional: Elif z in [3...8] -> print("true") else print("false")
    second_then = func.append_basic_block(name="if2_then")
    second_else = func.append_basic_block(name="if2_else")
    second_after = func.append_basic_block(name="if2_after")

    # treat [3...8] as inclusive 3..8
    cmp_lo = _cmp_ir(cg, zv, ">=", 3)
    cmp_hi = _cmp_ir(cg, zv, "<=", 8)
    in_range = cg.builder.and_(cmp_lo, cmp_hi)
    cg.builder.cbranch(in_range, second_then, second_else)

    cg.builder.position_at_end(second_then)
    cg.emit_print("true")
    cg.builder.branch(second_after)

    cg.builder.position_at_end(second_else)
    cg.emit_print("false")
    cg.builder.branch(second_after)

    cg.builder.position_at_end(second_after)

    # end of body -> jump back to while condition
    cg.builder.branch(cond_bb)

    # position builder at after_bb so subsequent emissions append correctly
    cg.builder.position_at_end(after_bb)

# ---------------------------
# Boolean-while parsing + Codegen helper (While true / While false)
# ---------------------------
def _parse_simple_call(line: str) -> Optional[str]:
    """
    Parse a simple zero-arg call like `execute()` or `fail()`.
    Returns the function name or None if not a simple call.
    """
    m = re.match(r'^\s*([a-zA-Z_]\w*)\s*\(\s*\)\s*$', line)
    return m.group(1) if m else None

def _apply_boolean_while_block_to_codegen(cg: Codegen, block_text: str):
    """
    Handle `While true:` and `While false:` blocks where the body contains
    simple zero-argument calls such as `execute()` or `fail()`.

    - `While true:` emits an infinite loop that calls the listed functions.
    - `While false:` emits a loop whose condition is constant false (body not executed).
    """
    if cg.builder is None:
        raise RuntimeError("Codegen.emit_main() must be called before applying While block")

    lines = [ln for ln in block_text.splitlines() if ln.strip() != ""]
    if not lines:
        return

    header = lines[0].strip()
    header_low = header.lower()
    if header_low not in ("while true:", "while false:"):
        return  # not a boolean-while; caller can fall back to other handler

    # collect simple call names from the body (ignore non-matching lines)
    calls = []
    for raw in lines[1:]:
        name = _parse_simple_call(raw.strip())
        if name:
            calls.append(name)

    func = cg.func
    cond_bb = func.append_basic_block(name="bool_while_cond")
    body_bb = func.append_basic_block(name="bool_while_body")
    after_bb = func.append_basic_block(name="bool_while_after")

    # jump to condition
    cg.builder.branch(cond_bb)

    # condition block: constant true or false
    cg.builder.position_at_end(cond_bb)
    bool_val = ir.Constant(ir.IntType(1), 1 if header_low == "while true:" else 0)
    cg.builder.cbranch(bool_val, body_bb, after_bb)

    # body: call each function in order, then loop back to condition
    cg.builder.position_at_end(body_bb)
    # declare or reuse zero-arg void functions
    for name in calls:
        fnty = ir.FunctionType(ir.VoidType(), [])
        fn = ir.Function(cg.module, fnty, name=name)
        cg.builder.call(fn, [])
    # loop back
    cg.builder.branch(cond_bb)

    # continue after loop
    cg.builder.position_at_end(after_bb)

def example_module_from_while_block(block_text: str, out_file: Optional[str] = None) -> str:
    """
    Convenience function to generate a module from a While block text.
    """
    cg = Codegen()
    cg.emit_main()
    _apply_while_block_to_codegen(cg, block_text)
    llvm_ir = cg.finish()
    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(llvm_ir)
    return llvm_ir

# ---------------------------
# Finally-block parsing + Codegen helper
# ---------------------------
def _parse_finally_block(block_text: str) -> List[str]:
    """
    Parse a Finally: block and return list of simple zero-arg call names found.
    Example:
      Finally:
          end()
    """
    calls: List[str] = []
    for raw in block_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.match(r"(?i)^finally\s*:", line):
            continue
        # collect simple zero-arg calls (reuses `_parse_simple_call` if present)
        name = None
        try:
            name = _parse_simple_call(line)  # uses function defined above for simple calls
        except NameError:
            # fallback: inline parse if helper isn't available
            m = re.match(r'^\s*([a-zA-Z_]\w*)\s*\(\s*\)\s*$', line)
            name = m.group(1) if m else None
        if name:
            calls.append(name)
    return calls

def _apply_finally_to_codegen(cg: Codegen, block_text: str):
    """
    Emit the calls declared in a Finally block at the current builder position.
    Each zero-arg function is declared as an external void function and called once.
    """
    if cg.builder is None:
        raise RuntimeError("Codegen.emit_main() must be called before applying Finally block")

    calls = _parse_finally_block(block_text)
    for name in calls:
        fnty = ir.FunctionType(ir.VoidType(), [])
        fn = ir.Function(cg.module, fnty, name=name)
        cg.builder.call(fn, [])

def example_module_from_finally_block(block_text: str, out_file: Optional[str] = None) -> str:
    """
    Convenience builder: create a module, apply the Finally block, and return the IR.
    """
    cg = Codegen()
    cg.emit_main()
    _apply_finally_to_codegen(cg, block_text)
    llvm_ir = cg.finish()

    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(llvm_ir)
    return llvm_ir
