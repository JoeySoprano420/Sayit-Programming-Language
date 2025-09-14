# sayc.py
"""
Enhanced sayc: CLI, engines, libraries, caching, watch mode, config, and templates.

New capabilities:
- Subcommands: run, build, repl, test, list-engines, make-engine
- Engine discovery: builtin 'vm' and 'ir', plus Python modules in ./engines or --engine-path
- Library loading (--libs), config file support (--config)
- Caching of emitted IR (.cache/<hash>.ll) with --force to bypass
- File watch mode (--watch) (simple polling)
- Verbose / dry-run / timeout / max-steps options
- Engine template generator (make-engine)
"""
from __future__ import annotations
import sys
import os
import argparse
import importlib.util
import importlib.machinery
import json
import time
import hashlib
from typing import List, Optional, Dict, Any

from say_lexer import tokenize
from say_parser import Parser

# Try to import a VM class in a backward-compatible way:
# - Prefer `VM` if a direct compatibility class is provided.
# - Fall back to `CompatVM` if present.
# - Otherwise wrap `ExecVM` with a thin compatibility shim exposing `.run(program)` and `.vars`.
try:
    from say_vm import VM  # type: ignore
except Exception:
    try:
        from say_vm import CompatVM as VM  # type: ignore
    except Exception:
        from say_vm import ExecVM as _ExecVM  # type: ignore

        class VM:
            """
            Minimal compatibility wrapper around ExecVM to provide the old `.run(program)` API
            and expose `.vars`. This keeps `sayc` working even if upstream renamed VM types.
            """
            def __init__(self, max_steps: Optional[int] = 1_000_000, timeout: Optional[float] = None, trace: bool = False, **kwargs):
                # mirror common ExecVM constructor arguments
                self._exec = _ExecVM(max_steps=max_steps, timeout=timeout, trace=trace, **kwargs)
                self.vars = self._exec.vars

            def run(self, program):
                return self._exec.run_program(program)


# Keep Codegen import lazy so llvmlite is optional
# from say_codegen import Codegen


# ---------------------------
# Utilities
# ---------------------------
def load_source_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def parse_program_from_source(src: str):
    toks = tokenize(src)
    parser = Parser(toks)
    return parser.parse_program()

def sha1_of_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def ensure_cache_dir() -> str:
    d = os.path.join(".cache")
    os.makedirs(d, exist_ok=True)
    return d

def write_cache(key: str, data: str) -> str:
    d = ensure_cache_dir()
    path = os.path.join(d, f"{key}.ll")
    with open(path, "w", encoding="utf-8") as f:
        f.write(data)
    return path

def read_cache(key: str) -> Optional[str]:
    d = os.path.join(".cache")
    path = os.path.join(d, f"{key}.ll")
    if os.path.isfile(path):
        return load_source_file(path)
    return None

def verbose_print(verbose: bool, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


# ---------------------------
# Engine loading / discovery
# ---------------------------
ENGINES_DIR = os.path.join(os.path.dirname(__file__), "engines")

def discover_local_engines() -> Dict[str, str]:
    """Return mapping engine_name -> file_path for ./engines/*.py"""
    result = {}
    if not os.path.isdir(ENGINES_DIR):
        return result
    for fn in os.listdir(ENGINES_DIR):
        if fn.endswith(".py") and not fn.startswith("_"):
            name = os.path.splitext(fn)[0]
            result[name] = os.path.join(ENGINES_DIR, fn)
    return result

def load_engine_from_path(path: str):
    """Dynamically import Python module from path and return it."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Engine module not found: {path}")
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import engine module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def resolve_engine(engine: Optional[str], engine_path: Optional[str], verbose: bool = False):
    """
    Resolve engine selection:
      - builtin: 'vm' or 'ir'
      - discovered local engine name (in ./engines)
      - engine_path: direct path to a Python module
    Returns a callable runner object with either `run(program, args)` or `emit_ir(program, out)`.
    """
    local = discover_local_engines()
    verbose_print(verbose, "discovered local engines:", local)
    if engine == "vm" or engine is None:
        return {"type": "builtin", "name": "vm"}
    if engine == "ir":
        return {"type": "builtin", "name": "ir"}
    # engine may be name available in local
    if engine in local:
        mod = load_engine_from_path(local[engine])
        return {"type": "module", "module": mod}
    # engine as path
    if engine_path:
        mod = load_engine_from_path(engine_path)
        return {"type": "module", "module": mod}
    # fallback: if engine looks like a path
    if engine and os.path.isfile(engine):
        mod = load_engine_from_path(engine)
        return {"type": "module", "module": mod}
    raise ValueError(f"Unknown engine '{engine}' (no local module or path found)")

# ---------------------------
# Core behavior
# ---------------------------
def compile_to_ir_and_maybe_cache(program, out_filename: Optional[str], force: bool, verbose: bool):
    """Emit IR using say_codegen.Codegen (lazy import). Cache by program hash."""
    try:
        from say_codegen import Codegen
    except Exception as e:
        raise RuntimeError("Codegen (llvmlite) is required for IR emission: " + str(e))

    # produce canonical program text for hashing: join stringified stmts minimally
    program_text = json.dumps([str(type(s).__name__) + ":" + getattr(s, "expr", getattr(s, "ident", "")).__class__.__name__ if hasattr(s, "expr") else type(s).__name__ for s in program.stmts])
    key = sha1_of_text(program_text)
    cached = None if force else read_cache(key)
    if cached:
        verbose_print(verbose, f"[cache] Found cached IR for key {key}")
        if out_filename:
            with open(out_filename, "w", encoding="utf-8") as f:
                f.write(cached)
            return out_filename
        return None  # caller can print cached if desired

    cg = Codegen()
    cg.emit_main()
    # simple emitter: Print statements and basic Make/Start/While helpers are supported in Codegen module in repo
    for stmt in program.stmts:
        cls = stmt.__class__.__name__
        if cls == "Print":
            if hasattr(stmt.expr, "val"):
                cg.emit_print(str(stmt.expr.val))
            else:
                cg.emit_print(f"<{stmt.expr.name}>")
        else:
            # leave other statements to specialized codegen helpers in say_codegen (if present)
            # noop here to keep compatibility
            pass

    llvm_ir = cg.finish()
    cache_path = write_cache(key, llvm_ir)
    verbose_print(verbose, f"[cache] Wrote IR to {cache_path}")
    if out_filename:
        with open(out_filename, "w", encoding="utf-8") as f:
            f.write(llvm_ir)
        return out_filename
    else:
        print(llvm_ir)
        return None

def run_with_vm(program, args):
    # Create VM instance with backward-compatible parameter; call its run method if available.
    vm = VM(max_steps=getattr(args, "max_steps", 1000000))
    # Some VM variants return a result from run; call it for side-effects and ignore the return
    # to preserve previous behavior where we returned the VM object.
    if hasattr(vm, "run"):
        try:
            vm.run(program)
        except TypeError:
            # fallback: if run expects different args, try run_program
            if hasattr(vm, "run_program"):
                vm.run_program(program)
    elif hasattr(vm, "run_program"):
        vm.run_program(program)
    return vm

def run_with_module_engine(mod, program, args):
    """Call engine module using run(program, args) or emit_ir(program,out)."""
    if hasattr(mod, "run"):
        return mod.run(program, args)
    if hasattr(mod, "emit_ir"):
        return mod.emit_ir(program, getattr(args, "out", None))
    if hasattr(mod, "main"):
        # try convenient main(program, args)
        try:
            return mod.main(program, args)
        except TypeError:
            return mod.main(program)
    raise RuntimeError("Engine module does not expose run/emit_ir/main API")

# ---------------------------
# CLI: subcommands
# ---------------------------
def make_engine_template(path: str):
    tmpl = """# Example Sayit engine module
def run(program, args):
    \"""
    Minimal engine: receives parsed AST `program` and parsed `args`.
    Implement execution or transformation here.
    \"""
    print("[engine] run called - implement execution")
    for s in program.stmts:
        print("stmt:", type(s).__name__)
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(tmpl)
    return path

def run_tests_in_dir(tests_dir: str, args) -> int:
    """
    Discover .say files in tests_dir, run each under VM and report pass/fail.
    A test file may contain `# expect: <text>` comment lines; this runner does a simple stdout match.
    """
    import subprocess, tempfile
    passed = 0
    total = 0
    if not os.path.isdir(tests_dir):
        print(f"[test] tests dir not found: {tests_dir}")
        return 1
    for fn in sorted(os.listdir(tests_dir)):
        if not fn.endswith(".say"):
            continue
        total += 1
        path = os.path.join(tests_dir, fn)
        src = load_source_file(path)
        prog = parse_program_from_source(src)
        # run in a fresh VM and capture stdout using subprocess running this script for isolation
        cmd = [sys.executable, __file__, path]  # default behavior run using VM
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=getattr(args, "timeout", 5))
            out_text = out.decode("utf-8").strip()
            # look for expectations in comments
            expects = []
            for line in src.splitlines():
                if "# expect:" in line:
                    expects.append(line.split("# expect:", 1)[1].strip())
            ok = all(e in out_text for e in expects) if expects else True
            if ok:
                passed += 1
                print(f"[test] {fn}: PASS")
            else:
                print(f"[test] {fn}: FAIL\n  output:\n{out_text}\n  expects: {expects}")
        except Exception as e:
            print(f"[test] {fn}: ERROR: {e}")
    print(f"[test] {passed}/{total} passed")
    return 0 if passed == total else 2

def watch_and_rebuild(path: str, build_cb, poll: float = 0.5, verbose: bool = False):
    """Simple polling file watcher: call build_cb() on changes to `path` (file or directory)."""
    last_snapshot = {}
    def snapshot():
        info = {}
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for fn in files:
                    p = os.path.join(root, fn)
                    try:
                        info[p] = os.path.getmtime(p)
                    except OSError:
                        info[p] = 0
        else:
            info[path] = os.path.getmtime(path) if os.path.exists(path) else 0
        return info
    last_snapshot = snapshot()
    try:
        while True:
            time.sleep(poll)
            cur = snapshot()
            if cur != last_snapshot:
                verbose_print(verbose, "[watch] change detected, rebuilding...")
                try:
                    build_cb()
                except Exception as e:
                    print("[watch] build error:", e)
                last_snapshot = cur
    except KeyboardInterrupt:
        verbose_print(verbose, "[watch] stopped by user")

# ---------------------------
# Entrypoint
# ---------------------------
def build_and_dispatch(program, args):
    # Choose engine resolution
    resolved = resolve_engine(getattr(args, "engine", None), getattr(args, "engine_path", None), verbose=getattr(args, "verbose", False))
    if resolved["type"] == "builtin":
        if resolved["name"] == "vm":
            return run_with_vm(program, args)
        if resolved["name"] == "ir":
            return compile_to_ir_and_maybe_cache(program, getattr(args, "out", None), getattr(args, "force", False), getattr(args, "verbose", False))
    elif resolved["type"] == "module":
        return run_with_module_engine(resolved["module"], program, args)
    else:
        raise RuntimeError("Unsupported engine resolution result")

def load_config(config_path: Optional[str], verbose: bool):
    if not config_path:
        # look for sayit.json or sayit.toml (toml optional)
        if os.path.isfile("sayit.json"):
            config_path = "sayit.json"
        elif os.path.isfile("sayit.toml"):
            config_path = "sayit.toml"
    if not config_path:
        return {}
    if config_path.endswith(".json"):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            verbose_print(verbose, f"[config] loaded {config_path}")
            return cfg
    # optional toml
    try:
        import toml
        cfg = toml.load(config_path)
        verbose_print(verbose, f"[config] loaded {config_path}")
        return cfg
    except Exception:
        print(f"[config] Unsupported config format or toml not installed: {config_path}")
        return {}

def main(argv: Optional[List[str]] = None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(prog="sayc", description="Sayit multi-engine tool")
    sub = parser.add_subparsers(dest="cmd", help="subcommands")

    # run / default
    p_run = sub.add_parser("run", help="Run a .say file (default)")
    p_run.add_argument("file", nargs="?", help="Source file to run")
    p_run.add_argument("--libs", nargs="+", help="Library .say files to load first")
    p_run.add_argument("--engine", choices=["vm", "ir"] + list(discover_local_engines().keys()), help="Engine to use")
    p_run.add_argument("--engine-path", help="Path to a custom engine module")
    p_run.add_argument("--out", help="Output file for IR (when using ir engine)")
    p_run.add_argument("--force", action="store_true", help="Bypass cache when building IR")
    p_run.add_argument("--watch", action="store_true", help="Watch file/dir and rebuild on changes")
    p_run.add_argument("--verbose", action="store_true", help="Verbose logging")
    p_run.add_argument("--max-steps", type=int, default=1000000, help="VM instruction limit")
    p_run.add_argument("--config", help="Config file (json/toml)")

    # build (emit IR)
    p_build = sub.add_parser("build", help="Emit LLVM IR for a .say file")
    p_build.add_argument("file")
    p_build.add_argument("--out", help="Write IR to file")
    p_build.add_argument("--force", action="store_true")
    p_build.add_argument("--verbose", action="store_true")

    # repl
    p_repl = sub.add_parser("repl", help="Start REPL using VM engine")
    p_repl.add_argument("--verbose", action="store_true")

    # test
    p_test = sub.add_parser("test", help="Run tests in tests/ directory")
    p_test.add_argument("--tests-dir", default="tests")
    p_test.add_argument("--timeout", type=int, default=5)
    p_test.add_argument("--verbose", action="store_true")

    # list engines
    p_list = sub.add_parser("list-engines", help="List available engines")

    # make engine template
    p_tmpl = sub.add_parser("make-engine", help="Create an engine template file")
    p_tmpl.add_argument("path", nargs="?", default=os.path.join("engines", "example_engine.py"))

    args = parser.parse_args(argv)

    # If user invoked no subcommand, default to run
    cmd = args.cmd or "run"

    # Load config early
    cfg = load_config(getattr(args, "config", None), getattr(args, "verbose", False))

    if cmd == "list-engines":
        local = discover_local_engines()
        print("builtin: vm, ir")
        for k, p in local.items():
            print(f"local: {k} -> {p}")
        return

    if cmd == "make-engine":
        path = args.path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        made = make_engine_template(path)
        print(f"[sayc] Engine template written to {made}")
        return

    if cmd == "test":
        code = run_tests_in_dir(args.tests_dir, args)
        return code

    if cmd == "repl":
        # simple REPL using VM
        from say_vm import repl
        return repl(trace=getattr(args, "verbose", False))

    # For run/build, require file
    file_arg = getattr(args, "file", None)
    if not file_arg:
        print("No source file provided.")
        parser.print_help()
        return

    if not os.path.isfile(file_arg):
        print(f"File not found: {file_arg}")
        return

    src = load_source_file(file_arg)
    program = parse_program_from_source(src)

    # libs
    if getattr(args, "libs", None):
        lib_stmts = []
        for p in args.libs:
            if not os.path.isfile(p):
                print(f"[sayc] Library file not found: {p}")
                return
            lib_src = load_source_file(p)
            lib_prog = parse_program_from_source(lib_src)
            lib_stmts.extend(lib_prog.stmts)
        program.stmts = lib_stmts + program.stmts
        verbose_print(getattr(args, "verbose", False), f"[sayc] loaded {len(args.libs)} libraries")

    # build callback used by watch
    def build_cb():
        verbose_print(getattr(args, "verbose", False), f"[sayc] building {file_arg} ...")
        return build_and_dispatch(program, args)

    if getattr(args, "watch", False):
        watch_and_rebuild(file_arg if os.path.isfile(file_arg) else os.path.dirname(file_arg), build_cb, verbose=bool(getattr(args, "verbose", False)))
        return

    # dispatch build/run
    try:
        res = build_and_dispatch(program, args)
        return res
    except Exception as e:
        print("[sayc] Error:", e)
        return 1

if __name__ == "__main__":
    main()
