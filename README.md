# Sayiy Programming Language

A modern, expressive programming language designed for simplicity and power.

## Features

- Clean, intuitive syntax
- Dynamic typing with optional static typing
- Built-in data structures (lists, dictionaries, sets)
- Functional programming support
- Object-oriented programming capabilities
- Interactive REPL environment
- Comprehensive standard library

## Getting Started

### Installation

Clone the repository:
```bash
git clone https://github.com/JoeySoprano420/Sayiy-Programming-Language.git
cd Sayiy-Programming-Language
```

### Running Sayiy

Run the interpreter:
```bash
python sayiy.py
```

Start the REPL:
```bash
python sayiy.py --repl
```

Run a Sayiy program:
```bash
python sayiy.py examples/hello.sayiy
```

## Language Syntax

### Variables and Basic Types
```sayiy
// Variable declarations
let name = "Sayiy"
let version = 1.0
let is_awesome = true

// Arrays
let numbers = [1, 2, 3, 4, 5]
let mixed = [1, "hello", true]

// Objects
let person = {
    name: "Alice",
    age: 30,
    city: "New York"
}
```

### Functions
```sayiy
// Function definition
fn greet(name) {
    return "Hello, " + name + "!"
}

// Arrow functions
let square = (x) => x * x

// Higher-order functions
let numbers = [1, 2, 3, 4, 5]
let doubled = numbers.map((x) => x * 2)
```

### Control Flow
```sayiy
// If statements
if condition {
    // do something
} else if other_condition {
    // do something else
} else {
    // default action
}

// Loops
for i in 0..10 {
    print(i)
}

while condition {
    // loop body
}
```

## Project Structure

```
Sayiy-Programming-Language/
├── src/
│   ├── lexer.py          # Tokenizer
│   ├── parser.py         # Parser
│   ├── ast_nodes.py      # AST definitions
│   ├── interpreter.py    # Interpreter
│   └── stdlib/           # Standard library
├── examples/             # Example programs
├── tests/               # Test suite
├── docs/                # Documentation
├── sayiy.py             # Main entry point
└── README.md            # This file
```

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.