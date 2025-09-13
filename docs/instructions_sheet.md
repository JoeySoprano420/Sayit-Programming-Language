# 📘 **Sayit Programming Language — Official Instruction Sheet**

**Version:** v1.0.0
**Tagline:** *Sayit — Code as Ritual, Logic as Ceremony.*

---

## 1. Overview

**Sayit (.say)** is a **general-purpose programming language** designed for readability, safety, and performance.

It features:

* **Ritualistic program structure** (`Start Ritual`, `Finally`).
* **Readable control flow** (`If … Elif … Else`, `While … until`).
* **Deterministic safety constructs** (`fail()`, `end()`).
* **LLVM/NASM dual backend** for optimized binaries.
* **Lightweight VM interpreter** for prototyping.

---

## 2. System Requirements

* **Supported Operating Systems:**

  * Linux (Ubuntu 20.04+, Arch, Fedora)
  * macOS (12+)
  * Windows 10/11

* **Runtime Dependencies:**

  * Python ≥ 3.11 (for VM & development builds)
  * LLVM/Clang (for native compilation)

* **Optional Tools:**

  * Git (to clone repo)
  * CMake (for extended builds)

---

## 3. Download & Install

### Option 1 — Prebuilt Binaries (Recommended)

* Visit the [GitHub Releases page](https://github.com/YOUR-ORG/sayit/releases).
* Download the package for your platform:

  * **Linux:** `sayit-ubuntu-latest.tar.gz`
  * **macOS:** `sayit-macos-latest.tar.gz`
  * **Windows:** `sayit-windows-latest.zip`
* Extract the package:

  ```bash
  tar -xzf sayit-ubuntu-latest.tar.gz
  cd sayit
  ```
* Add to your `$PATH`:

  ```bash
  export PATH=$PWD:$PATH
  ```

### Option 2 — Build from Source

```bash
git clone https://github.com/YOUR-ORG/sayit.git
cd sayit
pip install -r requirements.txt
```

Run the compiler:

```bash
python sayc.py hello.say
```

---

## 4. Quick Start

Create `hello.say`:

```say
Start Ritual:
    string message = "Hello, World!"
    print(message)
Finally:
    end()
```

Run in VM:

```bash
sayc hello.say
```

Compile to LLVM IR:

```bash
sayc hello.say --ir --out hello.ll
clang hello.ll -o hello
./hello
```

---

## 5. Language Structure

### 5.1 Ritual Invocation

Every program must begin and end ceremonially:

```say
Start Ritual:
    ...code...
Finally:
    end()
```

---

### 5.2 Initialization

Declare and initialize variables:

```say
Make:
    x = 1
    y = 2
```

---

### 5.3 Control Flow

**Loops of Becoming:**

```say
While z < 10:
    y = y + x until y == 9
```

**Threshold Checks:**

```say
If z > 7 and z < 9:
    print("true")
Elif z in [3...8]:
    print("maybe")
Else:
    print("false")
```

**Eternal Conditions:**

```say
While true:
    execute()
While false:
    fail()
```

---

### 5.4 Expressions & Operators

Supported binary operators:
`+`, `-`, `*`, `/`, `==`, `!=`, `<`, `>`, `<=`, `>=`, `and`, `or`

Ranges:

```say
If x in [1...5]:
    print("inside range")
```

---

## 6. Compiler & VM

* **VM Mode (default):**
  Interprets `.say` files. Best for quick testing.

* **LLVM IR Mode:**

  ```bash
  sayc program.say --ir --out program.ll
  clang program.ll -o program
  ./program
  ```

* **NASM Hooks:**
  Inline assembly supported via `asm { ... }` (advanced).

---

## 7. Standard Library (v1.0)

* **IO**: `print()`, `read()`
* **Math**: `+ - * /` plus extended math module
* **Control**: `fail()`, `end()`
* **Data Types**: string, int, bool, list

---

## 8. IDE & Editor Support

* **VS Code Extension:** Syntax highlighting (`.say`).
* **Vim/Emacs:** Highlight definitions available.
* **CLI Tools:**

  * `sayc` — compiler
  * `sayvm` — VM interpreter
  * `saydoc` — doc generator

---

## 9. Learning Curve

* **Beginner:** Ritual-like structure is self-descriptive.
* **Intermediate:** Range conditions and `until` loops add power.
* **Advanced:** Inline NASM and LLVM backend expose full system control.

---

## 10. Safety & Security

* **Explicit Exit Conditions:** `until` replaces unsafe breakpoints.
* **Dead-Code Signaling:** `While false:` is explicit, not accidental.
* **Fail-Fast Philosophy:** `fail()` prevents silent crashes.
* **LLVM Verification:** all IR passes safety checks.

---

## 11. Roadmap

* **v1.1:** richer collections (dicts, tuples).
* **v2.0:** concurrency primitives (`spawn`, `channel`).
* **v3.0:** WASM runtime for cloud deployments.

---

## 12. Why Choose Sayit?

* **Readable like Python**
* **Safe like Rust**
* **Fast like C**
* **Expressive like a story**

Sayit redefines programming as a **ritual of clarity and precision**.

---

## 13. Support & Contribution

* **Docs:** [https://sayit.dev/docs](https://sayit.dev/docs)
* **Issues:** [https://github.com/YOUR-ORG/sayit/issues](https://github.com/YOUR-ORG/sayit/issues)
* **Contribute:** Fork, branch, PR.
* **Community:** Discord/Slack channels available.

---

# ✅ TL;DR — Quick Install & Run

```bash
# Download
wget https://github.com/YOUR-ORG/sayit/releases/download/v1.0.0/sayit-ubuntu-latest.tar.gz
tar -xzf sayit-ubuntu-latest.tar.gz
export PATH=$PWD:$PATH

# Write a program
echo 'Start Ritual:\n    print("Hello")\nFinally:\n    end()' > hello.say

# Run
sayc hello.say
```

---

👉 Would you like me to prepare a **ready-to-publish PDF guide** (with title page, chapters, examples, diagrams) for distribution alongside the GitHub release?
