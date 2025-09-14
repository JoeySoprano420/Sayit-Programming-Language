"""
say_codegen.py - Pure Python implementation of Sayit Codegen
============================================================

This implementation does NOT require LLVM, llvmlite, or Visual Studio
compiler integration. It is a pure Python backend that:
 - Keeps the same Codegen API shape as the LLVM-based version
 - Produces a pseudo-IR log (string form of emitted ops)
 - Immediately executes print and variable ops in Python
 - Provides block parsers for Start Ritual, Make, While, If/Else, Finally

It can run .say scripts directly without any external tools.
"""

import re
from typing import Dict, List, Tuple, Optional, Any


# ============================================================
# Core Codegen class (pseudo-IR backend)
# ============================================================

class Codegen:
    def __init__(self, module_name="sayit"):
        self.module_name = module_name
        self.variables: Dict[str, Any] = {}
        self.output: List[str] = []
        self.running = False

    # ---------------------------
    # Function Setup
    # ---------------------------
    def emit_main(self):
        """Initialize module execution."""
        self.running = True
        self.output.append(f";; Begin module {self.module_name}")

    # ---------------------------
    # Printing Support
    # ---------------------------
    def emit_print(self, text: str):
        """Emit a print call with immediate execution."""
        if not self.running:
            raise RuntimeError("emit_main() must be called first")

        val = text
        # Resolve variable if reference
        if text in self.variables:
            val = self.variables[text]
        # Strip quotes if string literal
        elif text.startswith('"') and text.endswith('"'):
            val = text.strip('"')

        self.output.append(f"PRINT {val}")
        print(val)

    # ---------------------------
    # Variables
    # ---------------------------
    def emit_var(self, name: str, init_val: Any):
        """Create or update a variable."""
        self.variables[name] = init_val
        self.output.append(f"VAR {name} = {init_val}")
        return name

    def emit_load(self, name: str):
        """Load variable value."""
        if name not in self.variables:
            raise RuntimeError(f"Variable {name} not defined")
        return self.variables[name]

    def emit_binop(self, op: str, lhs: Any, rhs: Any):
        """Perform binary arithmetic and return result."""
        lval = self._resolve_value(lhs)
        rval = self._resolve_value(rhs)

        if op == "+": res = lval + rval
        elif op == "-": res = lval - rval
        elif op == "*": res = lval * rval
        elif op == "/": res = lval // rval
        else:
            raise NotImplementedError(f"Unsupported binop {op}")

        self.output.append(f"BINOP {lhs} {op} {rhs} = {res}")
        return res

    def _resolve_value(self, tok: Any):
        """Helper to turn variable names or literals into values."""
        if isinstance(tok, int):
            return tok
        if isinstance(tok, str):
            if tok.isdigit():
                return int(tok)
            if tok in self.variables:
                return self.variables[tok]
            if tok.startswith('"') and tok.endswith('"'):
                return tok.strip('"')
        return tok

    # ---------------------------
    # Finisher
    # ---------------------------
    def finish(self):
        """Finish and return pseudo-IR trace."""
        self.output.append(";; End module")
        return "\n".join(self.output)


# ============================================================
# Block Parsers
# ============================================================

# --- Start Ritual ---
def _parse_start_ritual_block(block_text: str) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
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
        m = re.match(r'^string\s+([a-zA-Z_]\w*)\s*=\s*"([^"]*)"\s*$', line)
        if m:
            strings[m.group(1)] = m.group(2)
            continue

        # print call
        m2 = re.match(r'^print\s*\(\s*([a-zA-Z_]\w*|"[^"]*")\s*\)\s*$', line)
        if m2:
            actions.append(("print", m2.group(1)))
            continue

    return strings, actions


def _apply_start_ritual_to_codegen(cg: Codegen, block_text: str):
    if not cg.running:
        raise RuntimeError("emit_main() must be called first")

    strings, actions = _parse_start_ritual_block(block_text)

    for name, val in strings.items():
        cg.emit_var(name, val)

    for act, arg in actions:
        if act == "print":
            cg.emit_print(arg)


# --- Make block ---
def _apply_make_block_to_codegen(cg: Codegen, block_text: str):
    for raw in block_text.splitlines():
        line = raw.strip()
        if not line or line.lower() == "make:":
            continue
        m = re.match(r'^([a-zA-Z_]\w*)\s*=\s*(\d+)$', line)
        if m:
            cg.emit_var(m.group(1), int(m.group(2)))


# --- While block ---
def _parse_while_header(header: str):
    m = re.match(r'(?i)^\s*while\s+([a-zA-Z_]\w*)\s*([<>=!]+)\s*([0-9]+)\s*:', header)
    if not m:
        raise ValueError(f"Malformed While header: {header!r}")
    return m.group(1), m.group(2), int(m.group(3))


def _apply_while_block_to_codegen(cg: Codegen, block_text: str):
    lines = [ln for ln in block_text.splitlines() if ln.strip()]
    header = lines[0]
    var, op, val = _parse_while_header(header)

    while True:
        lhs_val = cg._resolve_value(var)
        rhs_val = cg._resolve_value(val)

        cond = False
        if op == "<": cond = lhs_val < rhs_val
        elif op == "<=": cond = lhs_val <= rhs_val
        elif op == ">": cond = lhs_val > rhs_val
        elif op == ">=": cond = lhs_val >= rhs_val
        elif op == "==": cond = lhs_val == rhs_val
        elif op == "!=": cond = lhs_val != rhs_val

        if not cond:
            break

        # execute body lines
        for ln in lines[1:]:
            s = ln.strip()
            if not s:
                continue
            if "=" in s and "until" not in s:
                tgt, expr = [x.strip() for x in s.split("=", 1)]
                cg.emit_var(tgt, eval(expr, {}, cg.variables))
            elif s.lower().startswith("print("):
                arg = s[s.find("(")+1:s.rfind(")")]
                cg.emit_print(arg)


# --- If / Elif / Else ---
def _apply_if_block_to_codegen(cg: Codegen, block_text: str):
    lines = [ln.strip() for ln in block_text.splitlines() if ln.strip()]
    first = lines[0]
    cond = first[2:].strip().rstrip(":")

    true_lines, false_lines = [], []
    mode = "true"
    for ln in lines[1:]:
        if ln.lower().startswith("else"):
            mode = "false"
            continue
        if mode == "true":
            true_lines.append(ln)
        else:
            false_lines.append(ln)

    cond_val = eval(cond, {}, cg.variables)
    chosen = true_lines if cond_val else false_lines

    for ln in chosen:
        if ln.startswith("print("):
            arg = ln[ln.find("(")+1:ln.rfind(")")]
            cg.emit_print(arg)
        elif "=" in ln:
            tgt, expr = [x.strip() for x in ln.split("=", 1)]
            cg.emit_var(tgt, eval(expr, {}, cg.variables))


# --- Finally block ---
def _apply_finally_to_codegen(cg: Codegen, block_text: str):
    for raw in block_text.splitlines():
        line = raw.strip()
        if not line or line.lower().startswith("finally:"):
            continue
        if line.startswith("print("):
            arg = line[line.find("(")+1:line.rfind(")")]
            cg.emit_print(arg)


# ============================================================
# Convenience Module Builders
# ============================================================

def example_module_from_start_ritual(block_text: str, out_file: Optional[str] = None) -> str:
    cg = Codegen()
    cg.emit_main()
    _apply_start_ritual_to_codegen(cg, block_text)
    ir = cg.finish()
    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(ir)
    return ir


def example_module(out_file: Optional[str] = None) -> str:
    cg = Codegen()
    cg.emit_main()
    cg.emit_print("Hello, Sayit!")
    cg.emit_print("This is a pure Python Codegen run.")
    ir = cg.finish()
    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(ir)
    return ir


# ============================================================
# Demo CLI
# ============================================================

if __name__ == "__main__":
    import sys
    out_file = None
    if len(sys.argv) >= 3 and sys.argv[1] in ("-o", "--out"):
        out_file = sys.argv[2]

    ir_text = example_module(out_file=out_file)
    if out_file is None:
        print(ir_text)
