# 📖 Sayit Instruction Sheet

*A Language of Ritual and Execution*

---

## 🌑 1. Core Syntax

Sayit programs are **ritual-based scripts**. Each program is made up of **blocks** that resemble ceremonial text.

* **Keywords**: `Start`, `Ritual`, `Make`, `While`, `If`, `Elif`, `Else`, `Finally`, `print`, `end()`
* **Comments**: Begin with `#`, ignored by the lexer.
* **Case**: Keywords are case-insensitive (`Start Ritual` = `start ritual`).
* **Termination**: Blocks are closed by `end()` or another block keyword.

Example:

```say
Start Ritual:
    string msg = "Hello World!"
    print(msg)
end()
```

---

## 🔮 2. Blocks & Rituals

### **Start Ritual**

Defines the opening ceremony of a program.
Inside, you may declare variables and perform actions.

```say
Start Ritual:
    string greeting = "Blessings!"
    print(greeting)
end()
```

---

### **Make**

Used to declare and initialize variables in a formal way.

```say
Make:
    x = 10
    y = 25
```

---

### **While**

Loops until a condition is false.
Condition format: `While <var> <op> <number>:`

```say
Make:
    counter = 0

While counter < 3:
    print("Looping...")
    counter = counter + 1
```

---

### **If / Elif / Else**

Branching control flow.

```say
Make:
    n = 7

If n > 10:
    print("Greater than 10")
Elif n == 10:
    print("Equal to 10")
Else:
    print("Less than 10")
```

---

### **Finally**

Runs after everything else, regardless of earlier branches.

```say
Finally:
    print("End of program reached.")
```

---

## ⚖️ 3. Expressions & Operators

Supported operators:

* Arithmetic: `+  -  *  /`
* Comparison: `==  !=  <  <=  >  >=`
* Boolean: `and  or`

Examples:

```say
x = 10 + 5
y = x * 2
If y >= 30:
    print("Big number")
```

---

## 📦 4. Variables & Types

Sayit uses **dynamic typing** with lightweight declarations.

* `string` for text:

  ```say
  string name = "Eartha"
  ```
* `number` for integers (implicit when assigning):

  ```say
  score = 42
  ```
* **Identifiers**: letters, numbers, underscores. Must not start with a digit.

Variables live in the **ritual scope** and can be updated.

---

## 🔁 5. Control Flow

* **Sequential execution**: runs top to bottom.
* **While loops**: repetition until condition false.
* **If/Else**: conditional branching.
* **Finally**: guaranteed final step.

Nested structures are supported.

---

## 🛠️ 6. Builtins & Services

### **Core builtins**

* `print(x)` → print variable or string.
* `end()` → terminate ritual.

### **VM services (via CALL or builtins)**

* **I/O**: `import_file`, `export_file`, `input_line`
* **Net**: `packetize`, `upload`, `download`, `ping`
* **Util**: `capsule`, `scan_text`, `compile_ast`, `scale`, `range_check`
* **Scopes**: `push_scope`, `pop_scope`, `get_scope`

Example (pseudo):

```say
# Save data
_arg_path = "out.txt"
_arg_data = "Hello file"
CALL "export_file"
```

---

## 🖥️ 7. CLI Usage

`sayc.py` is the master CLI driver.

```bash
# Run a program
python sayc.py run hello.say

# Build IR (pure Python backend)
python sayc.py build hello.say --engine ir

# Start REPL
python sayc.py repl

# Run all tests in /tests
python sayc.py test
```

Options:

* `--libs file1.say file2.say` → preload libraries
* `--watch` → watch mode
* `--engine vm|ir|custom` → select engine
* `--force` → ignore IR cache
* `--verbose` → debug logs

---

## 🧪 8. Testing & Debugging

Tests live in the `tests/` directory.
Use **inline expectations**:

```say
print("Hello World!")  # expect: Hello World!
```

Run all tests:

```bash
python sayc.py test
```

Debugging aids:

* `--trace` → show each instruction in VM
* `--profile` → count opcode usage
* `--dump-vars` → show VM variable state

---

## 🌌 9. Philosophy & Style

* Code is **ceremonial**: every block feels like a ritual.
* Programs are **narrative**: they read like storytelling, not machinery.
* Execution is **pragmatic**: though poetic, everything runs deterministically.
* The design is **hackable**: pure Python backends, pluggable engines, IR caching.

Sayit is a blend of **myth and machine** — a language where you *write ceremonies, but run software*.

---

✅ That’s the **complete instruction sheet** — both a reference manual and a narrative guide.

