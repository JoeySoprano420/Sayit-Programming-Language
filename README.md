# 📜 Sayit Programming Language – Full-Scale Overview

---

## 🌌 Vision and Philosophy

Sayit is not just another scripting language.
It’s a **ritual-styled, human-centric DSL** designed to be read like storytelling yet executed like a real programming language. At its core:

* **Readable like prose.** Code is written in phrases like `Start Ritual:`, `Make:`, `While:`, `Finally:` instead of bare syntax.
* **Executable like a VM.** Programs compile to a bytecode or pseudo-IR, run efficiently under a stack-based virtual machine.
* **Hackable like Python.** Pure-Python implementation ensures immediate portability, extension, and embedding.
* **Symbolic like liturgy.** Constructs use ceremonial keywords (`Start Ritual`, `End`, `Make`, `Ritual`) to reinforce the “spell-like” feeling of code execution.

This unique philosophy makes Sayit a hybrid between **educational DSL**, **experimental VM playground**, and **cultural programming artifact**.

---

## 🧩 Major Components

The Sayit toolchain currently includes the following core subsystems:

### 1. **CLI Frontend (`sayc.py`)**

* Provides the user entrypoint: `sayc run`, `sayc build`, `sayc repl`, `sayc test`, `sayc list-engines`.
* Handles:

  * File loading (`.say` sources, libraries, configs).
  * Engine selection (`--engine vm` vs `--engine ir` vs custom engines).
  * IR caching (`.cache/<hash>.ll` with `--force`).
  * Watch mode (`--watch`) for continuous development.
  * Config loading (`sayit.json`, `sayit.toml`).
  * Library imports (`--libs`).
* Extensible with **engine templates** (`make-engine`) so developers can roll their own backends.

Think of `sayc.py` as the **driver**, unifying all engines under one workflow.

---

### 2. **Lexer (`say_lexer.py`)**

* Converts source text → token stream.
* Supports:

  * Core ritual keywords (`Start`, `Ritual`, `Make`, `While`, `If`, `Else`, `Finally`).
  * Literals (`STRING`, `NUMBER`).
  * Identifiers and operators.
  * Comments (`# ...`).
* Error-reporting includes **line + column**.
* CLI utility: `python say_lexer.py hello.say --pos` prints tokens with positions.

This is the **scanner**, responsible for first contact with source text.

---

### 3. **Parser (`say_parser.py`)**

* Consumes tokens into a proper **AST (Abstract Syntax Tree)**.
* Node classes: `Program`, `Print`, `Assign`, `BinOp`, `If`, `While`, `Literal`, `Ident`.
* Implements recursive-descent parsing:

  * `Print(expr)`
  * `Assign(ident, expr)`
  * `If(cond, then, elifs, els)`
  * `While(cond, body)`
* Includes utilities:

  * `parse_code(str)` / `parse_file(path)` for easy entrypoints.
  * `ast_to_string()` pretty-printer.
  * CLI mode to print either **tokens** or **AST** from `.say` files.

The parser is the **grammar engine**, transforming story-like phrases into structured programs.

---

### 4. **Virtual Machine (`say_vm.py`)**

* A **stack-based bytecode VM** with:

  * Bytecode compiler (AST → ops).
  * Constant folding optimization.
  * Execution limits (`max_steps`, `timeout`).
  * Profiling (`--profile`) and tracing (`--trace`).
  * Parallel runner (`run_parallel` on multiple programs).
  * Async execution (`run_program_async`).
* Services framework (pluggable sub-systems):

  * **IO**: import/export files, read/write, input.
  * **Render**: template rendering with variable substitution.
  * **Net**: packetize, upload/download, ping.
  * **Util**: capsule, scan, compile, scaling, range check, modes, scoping.
* Builtins exposed as zero-arg CALLables (`CALL "import_file"`, `CALL "packetize"`).

The VM is the beating heart — it makes Sayit run like a real language.

---

### 5. **Pure-Python Codegen (`say_codegen.py`)**

* A fully Python backend (no LLVM, no MSVC).
* Provides a pseudo-IR log (human readable).
* Executes statements immediately (side-effectful).
* Supports block-level parsers:

  * **Start Ritual** – define strings, print values.
  * **Make** – declare variables with constants.
  * **While** – repeated execution with inline evaluation.
  * **If/Elif/Else** – branching logic.
  * **Finally** – guaranteed block execution.
* Can dump pseudo-IR trace to file or stdout.

This is the **second engine**: instead of VM bytecode, it gives an IR log and direct execution.

---

## 🧮 Workflow and Execution Path

1. **Source Input**

   ```say
   Start Ritual:
       string msg = "Hello World!"
       print(msg)
   end()
   ```

2. **Lexing** → Tokens like `[START, RITUAL, IDENT(msg), '=', STRING("Hello World!"), PRINT(...)]`.

3. **Parsing** → AST:

   ```
   Program:
     Assign msg -> "Hello World!"
     Print msg
   ```

4. **Backend Choice**

   * **VM Engine** → Compile to bytecode, run on stack machine.
   * **IR Engine** → Generate pseudo-IR log, execute print directly.

5. **Execution**

   ```
   Hello World!
   ```

---

## ⚡ Features Beyond Basics

* **Caching system** for IR (`.cache/<hash>.ll`).
* **Watch mode**: live rebuild when source changes.
* **Config files**: auto-loads `sayit.json` or `sayit.toml`.
* **Pluggable engines**: drop a `.py` into `./engines` or point to `--engine-path`.
* **Library imports**: stitch `.say` libraries before main program.
* **Testing harness**: run `.say` files in `tests/`, check for `# expect:` comments.
* **REPL**: interactive, multi-line aware.

---

## 🏗️ Development Experience

* **Zero external dependencies** (LLVM removed, pure Python only).
* **Cross-platform** (runs anywhere Python 3.9+ is available).
* **Hack-friendly**: engine templates make it trivial to add new execution backends.
* **Readable tests**: `.say` scripts double as human documentation.
* **CLI parity**: `sayc.py` is a true driver with consistent subcommands.

---

## 🧑‍💻 Target Users

* **Language hackers** → experiment with VM, bytecode, IR design.
* **Educators** → show how compilers work (lex → parse → IR → VM).
* **Writers / Ritual coders** → treat programming as ceremonial storytelling.
* **Hackathon builders** → drop Sayit into projects as a mini scripting DSL.
* **Compiler engineers** → prototype new backend engines quickly.

---

## 🏭 Use Cases and Domains

* **Education** → teach compilers and interpreters with approachable syntax.
* **Games** → embed Sayit as a cutscene scripting language.
* **Automation** → lightweight ritual DSL for task scripting.
* **Storytelling** → combine narrative style with actual execution.
* **Experimentation** → test service abstractions (network/file/render).

---

## ⚖️ Comparisons

* **vs Python** → more ritualistic, less general.
* **vs Lua** → less minimal, more symbolic.
* **vs DSLs (Make, SQL, Regex)** → Sayit is general-purpose, but stylized.
* **vs LLVM-based languages** → lighter weight, zero dependencies, pure Python.

---

## 🚀 Future Directions

1. **Full AST → Codegen mapping** (currently `sayc.py` only handles `Print` directly).
2. **Optimized bytecode VM** with JIT via PyPy or Numba.
3. **More block types** (`For`, `Try/Catch`, `Switch`).
4. **Module system** for library packaging.
5. **Formal grammar spec** (EBNF / ANTLR).
6. **WebAssembly backend** to run Sayit in the browser.
7. **Desktop app launcher** with REPL + IDE plugin (syntax highlighting).
8. **Community engine gallery**: share custom engines as plugins.

---

## 🏁 Leadership Pitch

Sayit represents a **new category** of programming language:

* It bridges **storytelling** and **execution**.
* It demonstrates **compiler engineering concepts** in a tangible way.
* It is **hackable, pluggable, and lightweight** — no heavy toolchain.
* It has clear **educational, experimental, and cultural value**.

---



---

🔥 In summary: **Sayit is a ritual-infused scripting language, executed by a flexible VM or IR backend, packaged with a robust CLI, powered entirely in Python, and designed to feel both human and magical.**

---


