# say_vm.py
"""
Enhanced VM with high-performance execution features and many built-in services:
- Bytecode compiler + stack VM for fast execution
- Constant-folding optimization
- Built-in service registry for: import/export, IO, rendering, packetizing,
  capsuling, compiling, scanning, uploading, downloading, scaling, range checks,
  modes, scoping and ping
- Execution limits (max_steps / timeout) and profiling
- Async execution and parallel runner (ThreadPool)
- Improved REPL (multi-line) and CLI integration with tracing/profiling flags

Notes:
- New services are exposed as zero-arg builtins callable by name (CALL "name").
  They use VM.vars for input/output (convention: set _arg_* or named vars).
- These implementations are pragmatic stubs meant to be easy to extend/replace.
"""
from __future__ import annotations
import argparse
import base64
import json
import os
import pathlib
import socket
import sys
import tempfile
import threading
import time
import concurrent.futures
import hashlib
import shutil
from typing import Any, Dict, List, Optional, Tuple

from say_parser import Print, Assign, BinOp, Literal, Ident, If, While, Program
from say_lexer import tokenize as _tokenize
from say_parser import Parser as _Parser

class ExecError(Exception):
    pass

class ExecVM:
    def __init__(self,
                 max_steps: Optional[int] = 10_000_000,
                 timeout: Optional[float] = None,
                 builtins: Optional[Dict[str, Any]] = None,
                 trace: bool = False,
                 profile: bool = False,
                 optimize: bool = True):
        self.vars: Dict[str, Any] = {}
        self.max_steps = max_steps
        self._steps = 0
        self.timeout = timeout
        self._start_time: Optional[float] = None
        self.trace = trace
        self.profile = profile
        self.optimize = optimize
        self._profile_counts: Dict[str, int] = {}
        self._halted = False

        # lightweight mode/scope managers
        self.mode: Optional[str] = None
        self.scope_stack: List[Dict[str, Any]] = []

        # service registry for pluggable features
        self.services: Dict[str, Any] = {}

        # builtins: zero-arg callables used by CALL opcode
        self.builtins: Dict[str, Any] = {
            # core control
            "execute": lambda: print("[builtin] execute()"),
            "fail": lambda: (_ for _ in ()).throw(RuntimeError("fail() called")),
            "end": lambda: (_ for _ in ()).throw(StopIteration("end()")),
        }
        if builtins:
            self.builtins.update(builtins)

        # register default services
        self._register_default_services()

    # ---------------------------
    # Service registration
    # ---------------------------
    def register_service(self, name: str, obj: Any):
        self.services[name] = obj

    def get_service(self, name: str):
        return self.services.get(name)

    def _register_default_services(self):
        # IO service
        self.register_service("io", {
            "import": self._svc_import,
            "export": self._svc_export,
            "input": self._svc_input,
            "read": self._svc_read_file,
            "write": self._svc_write_file,
        })

        # render service
        self.register_service("render", {
            "template": self._svc_render_template
        })

        # network service
        self.register_service("net", {
            "packetize": self._svc_packetize,
            "upload": self._svc_upload,
            "download": self._svc_download,
            "ping": self._svc_ping,
        })

        # util service
        self.register_service("util", {
            "capsule": self._svc_capsule,
            "scan": self._svc_scan,
            "compile": self._svc_compile_ast,
            "scale": self._svc_scale,
            "range_check": self._svc_range_check,
            "set_mode": self._svc_set_mode,
            "get_mode": self._svc_get_mode,
            "push_scope": self._svc_push_scope,
            "pop_scope": self._svc_pop_scope,
            "get_scope": self._svc_get_scope,
        })

        # expose convenient builtins that call services using VM.vars convention
        # These builtins are zero-arg and use pre-agreed variable names for inputs/outputs.
        self.builtins.update({
            "import_file": lambda: self._svc_import(),
            "export_file": lambda: self._svc_export(),
            "input_line": lambda: self._svc_input(),
            "render_template": lambda: self._svc_render_template(),
            "packetize": lambda: self._svc_packetize(),
            "capsule": lambda: self._svc_capsule(),
            "compile_ast": lambda: self._svc_compile_ast(),
            "scan_text": lambda: self._svc_scan(),
            "upload": lambda: self._svc_upload(),
            "download": lambda: self._svc_download(),
            "scale": lambda: self._svc_scale(),
            "range_check": lambda: self._svc_range_check(),
            "set_mode": lambda: self._svc_set_mode(),
            "get_mode": lambda: self._svc_get_mode(),
            "push_scope": lambda: self._svc_push_scope(),
            "pop_scope": lambda: self._svc_pop_scope(),
            "get_scope": lambda: self._svc_get_scope(),
            "ping": lambda: self._svc_ping(),
        })

    # ---------------------------
    # Default service implementations (pragmatic stubs)
    # Inputs/outputs are read/written to self.vars by convention:
    # - _arg_path, _arg_data, _arg_template, _arg_target, etc.
    # - outputs placed into _out_* keys.
    # ---------------------------
    # IO
    def _svc_import(self):
        path = self.vars.get("_arg_path")
        if not path:
            raise ExecError("import: _arg_path not set")
        if not os.path.isfile(path):
            raise ExecError(f"import: file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        # store raw source and parsed AST
        self.vars["_out_imported_source"] = src
        try:
            prog = _Parser(list(_tokenize(src))).parse_program()
            self.vars["_out_imported_ast"] = prog
        except Exception as e:
            self.vars["_out_import_error"] = str(e)
        return None

    def _svc_export(self):
        path = self.vars.get("_arg_path")
        data = self.vars.get("_arg_data")
        if not path:
            raise ExecError("export: _arg_path not set")
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(data))
        self.vars["_out_export_path"] = path
        return None

    def _svc_input(self):
        prompt = self.vars.get("_arg_prompt", "")
        val = input(prompt)
        self.vars["_out_input"] = val
        return None

    def _svc_read_file(self):
        path = self.vars.get("_arg_path")
        if not path:
            raise ExecError("read: _arg_path not set")
        with open(path, "r", encoding="utf-8") as f:
            self.vars["_out_read"] = f.read()
        return None

    def _svc_write_file(self):
        path = self.vars.get("_arg_path")
        data = self.vars.get("_arg_data", "")
        if not path:
            raise ExecError("write: _arg_path not set")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(data))
        self.vars["_out_write"] = path
        return None

    # Render
    def _svc_render_template(self):
        template = self.vars.get("_arg_template", "")
        # context: all keys starting with "ctx_"
        ctx = {k[4:]: v for k, v in self.vars.items() if k.startswith("ctx_")}
        try:
            rendered = template.format(**ctx)
        except Exception as e:
            rendered = f"<render-error: {e}>"
        self.vars["_out_rendered"] = rendered
        return None

    # Packetize / upload / download / ping
    def _svc_packetize(self):
        data = self.vars.get("_arg_data", "")
        chunk_size = int(self.vars.get("_arg_chunk", 1024))
        b = data.encode("utf-8") if isinstance(data, str) else (data if isinstance(data, bytes) else str(data).encode("utf-8"))
        packets = [base64.b64encode(b[i:i+chunk_size]).decode("ascii") for i in range(0, len(b), chunk_size)]
        self.vars["_out_packets"] = packets
        return None

    def _svc_upload(self):
        # pragmatic stub: copy file to .remote dir to simulate upload
        src = self.vars.get("_arg_path")
        if not src or not os.path.exists(src):
            raise ExecError("upload: _arg_path missing or does not exist")
        remote_dir = os.path.join(".remote")
        os.makedirs(remote_dir, exist_ok=True)
        dst = os.path.join(remote_dir, os.path.basename(src))
        shutil.copy(src, dst)
        self.vars["_out_uploaded"] = dst
        return None

    def _svc_download(self):
        # pragmatic stub: copy from .remote dir to destination
        name = self.vars.get("_arg_name")
        dest = self.vars.get("_arg_dest")
        remote_dir = os.path.join(".remote")
        src = os.path.join(remote_dir, name) if name else None
        if not src or not os.path.exists(src):
            raise ExecError("download: remote object not found")
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        shutil.copy(src, dest)
        self.vars["_out_downloaded"] = dest
        return None

    def _svc_ping(self):
        host = self.vars.get("_arg_host")
        port = int(self.vars.get("_arg_port", 7))
        timeout = float(self.vars.get("_arg_timeout", 1.0))
        if not host:
            raise ExecError("ping: _arg_host not set")
        try:
            s = socket.create_connection((host, port), timeout=timeout)
            s.close()
            self.vars["_out_ping"] = True
        except Exception as e:
            self.vars["_out_ping"] = False
            self.vars["_out_ping_error"] = str(e)
        return None

    # Capsule / scan / compile / scale / range / mode / scope
    def _svc_capsule(self):
        data = self.vars.get("_arg_data")
        meta = self.vars.get("_arg_meta", {})
        capsule = {"data": data, "meta": meta}
        self.vars["_out_capsule"] = capsule
        return None

    def _svc_scan(self):
        src = self.vars.get("_arg_text", "")
        toks = list(_tokenize(src))
        self.vars["_out_tokens"] = toks
        return None

    def _svc_compile_ast(self):
        # expects _arg_ast to be present (Program)
        prog = self.vars.get("_arg_ast")
        if not prog:
            raise ExecError("compile: _arg_ast not set")
        ops = self.compile_program(prog)
        self.vars["_out_compiled_ops"] = ops
        return None

    def _svc_scale(self):
        name = self.vars.get("_arg_name")
        factor = float(self.vars.get("_arg_factor", 1.0))
        if name not in self.vars:
            raise ExecError("scale: variable not found")
        val = self.vars[name]
        try:
            scaled = val * factor
        except Exception as e:
            raise ExecError(f"scale: cannot scale {val}: {e}")
        self.vars["_out_scaled"] = scaled
        return None

    def _svc_range_check(self):
        name = self.vars.get("_arg_name")
        lo = float(self.vars.get("_arg_lo"))
        hi = float(self.vars.get("_arg_hi"))
        if name not in self.vars:
            raise ExecError("range_check: variable not found")
        val = float(self.vars[name])
        ok = lo <= val <= hi
        self.vars["_out_range_ok"] = ok
        return None

    def _svc_set_mode(self):
        m = self.vars.get("_arg_mode")
        self.mode = m
        self.vars["_out_mode"] = self.mode
        return None

    def _svc_get_mode(self):
        self.vars["_out_mode"] = self.mode
        return None

    def _svc_push_scope(self):
        # push an empty scope or copy of provided dict at _arg_scope
        sc = self.vars.get("_arg_scope", {})
        self.scope_stack.append(dict(sc))
        self.vars["_out_scope_len"] = len(self.scope_stack)
        return None

    def _svc_pop_scope(self):
        if not self.scope_stack:
            raise ExecError("pop_scope: stack empty")
        popped = self.scope_stack.pop()
        self.vars["_out_popped_scope"] = popped
        return None

    def _svc_get_scope(self):
        idx = int(self.vars.get("_arg_index", -1))
        if not self.scope_stack:
            self.vars["_out_scope"] = None
            return None
        if idx == -1:
            self.vars["_out_scope"] = self.scope_stack[-1]
        else:
            self.vars["_out_scope"] = self.scope_stack[idx]
        return None

    # ---------------------------
    # Compiler: AST -> bytecode (unchanged from prior)
    # ---------------------------
    def compile_program(self, program: Program) -> List[Tuple[str, Any]]:
        self._label_counter = 0
        ops: List[Tuple[str, Any]] = []
        self._emit_statements(program.stmts, ops)
        ops.append(("HALT", None))
        return ops

    def _emit_statements(self, stmts: List, ops: List[Tuple[str, Any]]):
        for s in stmts:
            self._emit_stmt(s, ops)

    def _emit_stmt(self, stmt, ops: List[Tuple[str, Any]]):
        cls = stmt.__class__.__name__
        if cls == "Print":
            self._emit_expr(stmt.expr, ops)
            ops.append(("PRINT", None))
        elif cls == "Assign":
            self._emit_expr(stmt.expr, ops)
            ops.append(("STORE_VAR", stmt.ident))
        elif cls == "If":
            # cond, then, elifs, els
            self._emit_expr(stmt.cond, ops)
            jfalse_idx = len(ops)
            ops.append(("JUMP_IF_FALSE", None))
            self._emit_statements(stmt.then, ops)
            jend_idx = len(ops)
            ops.append(("JUMP", None))
            else_start = len(ops)
            ops[jfalse_idx] = ("JUMP_IF_FALSE", else_start)
            for (econd, ebody) in stmt.elifs:
                self._emit_expr(econd, ops)
                jf = len(ops)
                ops.append(("JUMP_IF_FALSE", None))
                self._emit_statements(ebody, ops)
                je = len(ops)
                ops.append(("JUMP", None))
                ops[jf] = ("JUMP_IF_FALSE", len(ops))
            if stmt.els:
                self._emit_statements(stmt.els, ops)
            cur = len(ops)
            for i, (op, arg) in enumerate(ops):
                if op == "JUMP" and arg is None:
                    ops[i] = ("JUMP", cur)
        elif cls == "While":
            cond_start = len(ops)
            self._emit_expr(stmt.cond, ops)
            jf_idx = len(ops)
            ops.append(("JUMP_IF_FALSE", None))
            self._emit_statements(stmt.body, ops)
            ops.append(("JUMP", cond_start))
            after = len(ops)
            ops[jf_idx] = ("JUMP_IF_FALSE", after)
        else:
            ops.append(("NOP", None))

    def _emit_expr(self, expr, ops: List[Tuple[str, Any]]):
        if self.optimize and isinstance(expr, BinOp):
            if isinstance(expr.left, Literal) and isinstance(expr.right, Literal):
                lv, rv = expr.left.val, expr.right.val
                op = expr.op
                val = self._compute_binop_value(op, lv, rv)
                ops.append(("LOAD_CONST", val))
                return
        if isinstance(expr, Literal):
            ops.append(("LOAD_CONST", expr.val))
        elif isinstance(expr, Ident):
            ops.append(("LOAD_VAR", expr.name))
        elif isinstance(expr, BinOp):
            self._emit_expr(expr.left, ops)
            self._emit_expr(expr.right, ops)
            ops.append(("BINARY_OP", expr.op))
        else:
            ops.append(("LOAD_CONST", None))

    def _compute_binop_value(self, op: str, l, r):
        if op == "+": return l + r
        if op == "-": return l - r
        if op == "*": return l * r
        if op == "/": return l // r
        if op == "==": return l == r
        if op == "!=": return l != r
        if op == ">": return l > r
        if op == "<": return l < r
        if op == ">=": return l >= r
        if op == "<=": return l <= r
        if op == "and": return bool(l and r)
        if op == "or": return bool(l or r)
        raise ExecError(f"Unsupported compile-time op {op}")

    # ---------------------------
    # Interpreter: execute bytecode (extended CALL support)
    # ---------------------------
    def run_ops(self, ops: List[Tuple[str, Any]]):
        stack: List[Any] = []
        pc = 0
        self._steps = 0
        self._start_time = time.time() if self.timeout else None
        self._halted = False

        while pc < len(ops):
            if self.max_steps is not None:
                self._steps += 1
                if self._steps > self.max_steps:
                    raise ExecError("instruction limit exceeded (possible infinite loop)")

            if self.timeout and self._start_time and (time.time() - self._start_time) > self.timeout:
                raise ExecError("execution timeout exceeded")

            op, arg = ops[pc]
            if self.profile:
                self._profile_counts[op] = self._profile_counts.get(op, 0) + 1
            if self.trace:
                print(f"[trace] pc={pc} op={op} arg={arg} stack={stack} vars={self.vars}")

            try:
                if op == "LOAD_CONST":
                    stack.append(arg)
                    pc += 1
                elif op == "LOAD_VAR":
                    stack.append(self.vars.get(arg, 0))
                    pc += 1
                elif op == "STORE_VAR":
                    val = stack.pop() if stack else None
                    self.vars[arg] = val
                    pc += 1
                elif op == "BINARY_OP":
                    rhs = stack.pop() if stack else None
                    lhs = stack.pop() if stack else None
                    res = self._apply_binop(arg, lhs, rhs)
                    stack.append(res)
                    pc += 1
                elif op == "PRINT":
                    val = stack.pop() if stack else None
                    print(val)
                    pc += 1
                elif op == "JUMP":
                    pc = arg
                elif op == "JUMP_IF_FALSE":
                    cond = stack.pop() if stack else None
                    if not cond:
                        pc = arg
                    else:
                        pc += 1
                elif op == "CALL":
                    # CALL name -> call builtin (zero-arg) or service method by naming convention
                    name = arg
                    fn = self.builtins.get(name)
                    if fn:
                        try:
                            res = fn()
                        except StopIteration:
                            self._halted = True
                            break
                        pc += 1
                        continue
                    # service call convention: "svc:service_name:action"
                    if isinstance(name, str) and name.startswith("svc:"):
                        _, svc_name, action = name.split(":", 2)
                        svc = self.get_service(svc_name)
                        if not svc:
                            raise ExecError(f"Unknown service: {svc_name}")
                        action_fn = svc.get(action) if isinstance(svc, dict) else getattr(svc, action, None)
                        if not action_fn:
                            raise ExecError(f"Service {svc_name} has no action {action}")
                        try:
                            action_fn()
                        except StopIteration:
                            self._halted = True
                            break
                        pc += 1
                        continue
                    # unknown call
                    raise ExecError(f"Unknown callable: {name}")
                elif op == "NOP":
                    pc += 1
                elif op == "HALT":
                    break
                else:
                    raise ExecError(f"Unknown opcode: {op}")
            except StopIteration:
                break
            except Exception as e:
                raise ExecError(f"Runtime error at pc={pc}, op={op}: {e}") from e

        return {
            "steps": self._steps,
            "profile": self._profile_counts.copy() if self.profile else {},
            "halted": self._halted
        }

    def _apply_binop(self, op: str, l, r):
        if op == "and":
            return bool(l and r)
        if op == "or":
            return bool(l or r)
        if op == "+":
            return l + r
        if op == "-":
            return l - r
        if op == "*":
            return l * r
        if op == "/":
            if r == 0:
                raise ExecError("division by zero")
            return l // r
        if op == "==":
            return l == r
        if op == "!=":
            return l != r
        if op == ">":
            return l > r
        if op == "<":
            return l < r
        if op == ">=":
            return l >= r
        if op == "<=":
            return l <= r
        raise ExecError(f"Unsupported binary op: {op}")

    # ---------------------------
    # High-level runtimes
    # ---------------------------
    def run_program(self, program: Program):
        ops = self.compile_program(program)
        return self.run_ops(ops)

    def run_program_async(self, program: Program, executor: Optional[concurrent.futures.Executor] = None):
        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        return executor.submit(self.run_program, program)

    @staticmethod
    def run_parallel(programs: List[Program], max_workers: int = 4, **vm_kwargs):
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = []
            for p in programs:
                vm = ExecVM(**vm_kwargs)
                futures.append(ex.submit(vm.run_program, p))
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())
        return results

# ---------------------------
# Compatibility wrapper for older simple VM API
# ---------------------------
class CompatVM:
    """
    A thin compatibility layer matching the original simple API:
    - preserves .vars and run(program) interface
    - delegates to ExecVM for execution
    """
    def __init__(self, max_steps: Optional[int] = 1_000_000, timeout: Optional[float] = None, trace: bool = False):
        self._exec_vm = ExecVM(max_steps=max_steps, timeout=timeout, trace=trace, optimize=True)
        self.vars = self._exec_vm.vars

    def run(self, program):
        res = self._exec_vm.run_program(program)
        # sync back vars (already shared)
        return res

# ---------------------------
# CLI, REPL and helpers (unchanged)
# ---------------------------
def _run_source_with_execvm(source: str, trace: bool = False, max_steps: int = 1_000_000, timeout: Optional[float] = None, profile: bool = False):
    tokens = list(_tokenize(source))
    parser = _Parser(tokens)
    program = parser.parse_program()
    vm = ExecVM(max_steps=max_steps, timeout=timeout, trace=trace, profile=profile, optimize=True)
    return vm.run_program(program)

def run_file(path: str, trace: bool = False, max_steps: int = 1_000_000, timeout: Optional[float] = None, profile: bool = False):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    return _run_source_with_execvm(src, trace=trace, max_steps=max_steps, timeout=timeout, profile=profile)

def repl(trace: bool = False, max_steps: int = 1_000_000, timeout: Optional[float] = None):
    print("Sayit REPL (type 'exit' or 'quit' to leave). Enter blank line to execute multi-line block.")
    vm = ExecVM(max_steps=max_steps, timeout=timeout, trace=trace, optimize=True)
    buffer_lines: List[str] = []
    try:
        while True:
            prompt = "say> " if not buffer_lines else " ... "
            try:
                line = input(prompt)
            except EOFError:
                print()
                break
            if line is None:
                continue
            if line.strip() in ("exit", "quit"):
                break
            # multi-line entry: blank line executes buffer
            if line.strip() == "" and buffer_lines:
                src = "\n".join(buffer_lines)
                buffer_lines = []
                try:
                    prog = _Parser(list(_tokenize(src))).parse_program()
                    vm.run_program(prog)
                except Exception as e:
                    print(f"[repl] Error: {e}")
                continue
            buffer_lines.append(line)
            # attempt single-line immediate execution
            if len(buffer_lines) == 1:
                try:
                    prog = _Parser(list(_tokenize(line))).parse_program()
                    vm.run_program(prog)
                    buffer_lines = []
                except Exception:
                    # incomplete or invalid -> wait for more lines
                    pass
    except KeyboardInterrupt:
        print("\n[repl] Interrupted.")
    return vm

# ---------------------------
# Backwards-compatible simple-run CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="say_vm", description="Run Sayit programs via the enhanced VM")
    parser.add_argument("file", nargs="?", help="Source file to run (.say)")
    parser.add_argument("--repl", action="store_true", help="Start an interactive REPL")
    parser.add_argument("--trace", action="store_true", help="Enable instruction tracing")
    parser.add_argument("--profile", action="store_true", help="Enable opcode profiling")
    parser.add_argument("--dump-vars", action="store_true", help="Dump VM variables after run")
    parser.add_argument("--demo", action="store_true", help="Run a tiny demo program")
    parser.add_argument("--max-steps", type=int, default=1_000_000, help="Instruction limit to prevent infinite loops")
    parser.add_argument("--timeout", type=float, default=None, help="Execution timeout in seconds (per run)")
    args = parser.parse_args()

    vm_instance = None

    try:
        if args.demo:
            demo_src = 'x = 1\nprint(x)\nprint("Hello from Sayit VM")\n'
            vm_instance = _run_source_with_execvm(demo_src, trace=args.trace, max_steps=args.max_steps, timeout=args.timeout, profile=args.profile)

        elif args.repl:
            repl(trace=args.trace, max_steps=args.max_steps, timeout=args.timeout)

        elif args.file:
            if not os.path.isfile(args.file):
                print(f"File not found: {args.file}")
                sys.exit(1)
            vm_instance = run_file(args.file, trace=args.trace, max_steps=args.max_steps, timeout=args.timeout, profile=args.profile)

        else:
            parser.print_usage()
            sys.exit(1)

    except ExecError as e:
        print(f"[say_vm] Execution error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[say_vm] Error: {e}")
        sys.exit(1)

    if args.dump_vars and isinstance(vm_instance, dict):
        # run_program returns profiling dict; variables available via ExecVM instance in REPL or when using API
        print("Execution result:", vm_instance)
