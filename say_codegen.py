"""
say_codegen.py - Pure Python "Codegen" for Sayit language
========================================================

This version does NOT depend on LLVM, llvmlite, Clang, or MSVC.
It builds and executes a tiny IR-like representation entirely in Python.
"""

import re
from typing import Dict, List, Tuple, Optional


class Codegen:
    def __init__(self, module_name="sayit"):
        self.module_name = module_name
        self.variables: Dict[str, str] = {}
        self.output: List[str] = []
        self.running = False

    # ---------------------------
    # Function Setup
    # ---------------------------
    def emit_main(self):
        """Initialize state for execution."""
        self.running = True
        self.output.append(f";; Begin module {self.module_name}")

    # ---------------------------
    # Printing Support
    # ---------------------------
    def emit_print(self, text: str):
        """Emit a simulated print call."""
        if not self.running:
            raise RuntimeError("emit_main() must be called first")

        if text in self.variables:
            val = self.variables[text]
        else:
            val = text.strip('"')
        self.output.append(f"PRINT {val}")
        print(val)  # immediate execution

    # ---------------------------
    # Variables
    # ---------------------------
    def emit_var(self, name: str, init_val):
        """Store a variable in the interpreter state."""
        self.variables[name] = init_val
        self.output.append(f"VAR {name} = {init_val}")
        return name

    def emit_binop(self, op, lhs, rhs):
        """Perform binary operations."""
        try:
            lhs_val = int(self.variables.get(lhs, lhs))
            rhs_val = int(self.variables.get(rhs, rhs))
        except ValueError:
            raise RuntimeError(f"Non-integer operands: {lhs}, {rhs}")

        if op == "+":
            return lhs_val + rhs_val
        if op == "-":
            return lhs_val - rhs_val
        if op == "*":
            return lhs_val * rhs_val
        if op == "/":
            return lhs_val // rhs_val
        raise NotImplementedError(f"Unsupported binop {op}")

    # ---------------------------
    # Finisher
    # ---------------------------
    def finish(self):
        """Return a pseudo-IR dump as string."""
        self.output.append(";; End module")
        return "\n".join(self.output)


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
    """
    strings: Dict[str, str] = {}
    actions: List[Tuple[str, str]] = []

    for raw in block_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.match(r"(?i)^start\s+ritual\s*:", line):
            continue
        if re.match(r"(?i)^end\s*\(\s*\)\s*$", line) or line.lower() == "end()":
            continue

        # string declaration
        m = re.match(r'^string\s+([a-zA-Z_]\w*)\s*=\s*"([^"]*)"\s*$', line, flags=re.IGNORECASE)
        if m:
            strings[m.group(1)] = m.group(2)
            continue

        # print call
        m2 = re.match(r'^print\s*\(\s*([a-zA-Z_]\w*|"[^"]*")\s*\)\s*$', line, flags=re.IGNORECASE)
        if m2:
            actions.append(("print", m2.group(1)))
            continue

    return strings, actions


def _apply_start_ritual_to_codegen(cg: Codegen, block_text: str):
    """Apply Start Ritual block to Codegen instance."""
    if not cg.running:
        raise RuntimeError("emit_main() must be called first")

    strings, actions = _parse_start_ritual_block(block_text)

    for name, val in strings.items():
        cg.emit_var(name, val)

    for act, arg in actions:
        if act == "print":
            cg.emit_print(arg)


def example_module_from_start_ritual(block_text: str, out_file: Optional[str] = None) -> str:
    """Convenience builder for Start Ritual blocks."""
    cg = Codegen()
    cg.emit_main()
    _apply_start_ritual_to_codegen(cg, block_text)
    ir = cg.finish()
    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(ir)
    return ir


def example_module(out_file: Optional[str] = None) -> str:
    """Tiny example module builder."""
    cg = Codegen()
    cg.emit_main()
    cg.emit_print("Hello, Sayit!")
    cg.emit_print("This is a pure Python Codegen run.")
    ir = cg.finish()
    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(ir)
    return ir


# ---------------------------
# Demo CLI
# ---------------------------
if __name__ == "__main__":
    import sys

    out_file = None
    if len(sys.argv) >= 3 and sys.argv[1] in ("-o", "--out"):
        out_file = sys.argv[2]

    ir_text = example_module(out_file=out_file)
    if out_file is None:
        print(ir_text)
