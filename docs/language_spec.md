# Sayiy Programming Language Specification

## Version 1.0.0

### Overview

Sayiy is a dynamically typed, interpreted programming language designed for simplicity and expressiveness. It combines features from functional and object-oriented programming paradigms while maintaining an intuitive syntax.

## Language Features

### Data Types

#### Primitive Types
- **Numbers**: Integers and floating-point numbers
  ```sayiy
  let integer = 42
  let float = 3.14159
  ```

- **Strings**: UTF-8 encoded text
  ```sayiy
  let message = "Hello, World!"
  let name = 'Sayiy'
  ```

- **Booleans**: `true` and `false`
  ```sayiy
  let is_valid = true
  let is_empty = false
  ```

- **Null**: Represents absence of value
  ```sayiy
  let empty = null
  ```

#### Composite Types
- **Arrays**: Ordered collections of values
  ```sayiy
  let numbers = [1, 2, 3, 4, 5]
  let mixed = [42, "hello", true, null]
  ```

- **Objects**: Key-value mappings
  ```sayiy
  let person = {
      name: "Alice",
      age: 30,
      email: "alice@example.com"
  }
  ```

### Variables

Variables are declared using the `let` keyword:

```sayiy
let name = "Sayiy"
let count = 0
let items = []
```

### Functions

#### Function Declarations
```sayiy
fn greet(name) {
    return "Hello, " + name + "!"
}
```

#### Function Expressions
```sayiy
let add = fn(a, b) {
    return a + b
}
```

#### Arrow Functions
```sayiy
let square = (x) => x * x
let multiply = (a, b) => a * b
```

### Control Flow

#### Conditional Statements
```sayiy
if condition {
    // code
} else if other_condition {
    // code
} else {
    // code
}
```

#### Loops

**While Loop:**
```sayiy
while condition {
    // code
}
```

**For Loop:**
```sayiy
for item in iterable {
    // code
}

for i in 0..10 {
    // code
}
```

#### Control Statements
- `break`: Exit from loop
- `continue`: Skip to next iteration
- `return`: Return from function

### Operators

#### Arithmetic Operators
- `+`: Addition
- `-`: Subtraction
- `*`: Multiplication
- `/`: Division
- `%`: Modulo
- `^`: Exponentiation

#### Comparison Operators
- `==`: Equal
- `!=`: Not equal
- `<`: Less than
- `<=`: Less than or equal
- `>`: Greater than
- `>=`: Greater than or equal

#### Logical Operators
- `and`: Logical AND
- `or`: Logical OR
- `not`: Logical NOT

#### Assignment Operators
- `=`: Assignment
- `+=`: Add and assign
- `-=`: Subtract and assign

### Built-in Functions

#### I/O Functions
- `print(...)`: Print values to stdout
- `input(prompt)`: Get input from user

#### Type Functions
- `type(value)`: Get type of value
- `len(object)`: Get length of object
- `str(value)`: Convert to string
- `int(value)`: Convert to integer
- `float(value)`: Convert to float
- `bool(value)`: Convert to boolean

#### Array Functions
- `push(array, ...items)`: Add items to array
- `pop(array)`: Remove last item
- `shift(array)`: Remove first item
- `unshift(array, ...items)`: Add items to beginning
- `slice(array, start, end)`: Extract slice
- `join(array, separator)`: Join elements
- `sort(array)`: Sort array
- `reverse(array)`: Reverse array
- `map(array, function)`: Apply function to each element
- `filter(array, function)`: Filter elements
- `reduce(array, function, initial)`: Reduce to single value

#### Object Functions
- `keys(object)`: Get object keys
- `values(object)`: Get object values
- `entries(object)`: Get key-value pairs

#### String Functions
- `split(string, separator)`: Split string
- `upper(string)`: Convert to uppercase
- `lower(string)`: Convert to lowercase
- `trim(string)`: Remove whitespace
- `replace(string, old, new)`: Replace occurrences
- `contains(string, substring)`: Check if contains
- `starts_with(string, prefix)`: Check if starts with
- `ends_with(string, suffix)`: Check if ends with

#### Math Functions
- `abs(number)`: Absolute value
- `min(...numbers)`: Minimum value
- `max(...numbers)`: Maximum value
- `sum(array)`: Sum of elements
- `floor(number)`: Floor function
- `ceil(number)`: Ceiling function
- `round(number, digits)`: Round number
- `sqrt(number)`: Square root
- `pow(base, exponent)`: Power function

#### Utility Functions
- `time()`: Current timestamp
- `sleep(seconds)`: Sleep for duration
- `range(start, stop, step)`: Create range

### Syntax Rules

#### Comments
```sayiy
// Single-line comment
```

#### Statements
- Statements are separated by newlines
- Semicolons are optional
- Blocks are defined by curly braces `{}`

#### Identifiers
- Must start with letter or underscore
- Can contain letters, numbers, and underscores
- Case-sensitive

#### String Literals
- Single quotes: `'text'`
- Double quotes: `"text"`
- Escape sequences: `\n`, `\t`, `\r`, `\\`, `\"`, `\'`

### Grammar (EBNF)

```ebnf
program = statement*

statement = variable_declaration
          | function_declaration  
          | if_statement
          | while_statement
          | for_statement
          | return_statement
          | break_statement
          | continue_statement
          | block_statement
          | expression_statement

variable_declaration = "let" IDENTIFIER ("=" expression)?

function_declaration = "fn" IDENTIFIER "(" parameter_list? ")" block_statement

if_statement = "if" expression block_statement ("else" (if_statement | block_statement))?

while_statement = "while" expression block_statement

for_statement = "for" IDENTIFIER "in" expression block_statement

return_statement = "return" expression?

break_statement = "break"

continue_statement = "continue"

block_statement = "{" statement* "}"

expression_statement = expression

expression = assignment

assignment = logical_or (("=" | "+=" | "-=") assignment)?

logical_or = logical_and ("or" logical_and)*

logical_and = equality ("and" equality)*

equality = comparison (("==" | "!=") comparison)*

comparison = range_expr ((">" | ">=" | "<" | "<=") range_expr)*

range_expr = term (".." term)?

term = factor (("-" | "+") factor)*

factor = unary (("/" | "*" | "%") unary)*

unary = ("not" | "-") unary | power

power = call ("^" unary)?

call = primary (("(" argument_list? ")" | "." IDENTIFIER | "[" expression "]"))*

primary = NUMBER | STRING | "true" | "false" | "null" | IDENTIFIER
        | "(" expression ")"
        | "[" expression_list? "]"
        | "{" property_list? "}"
        | "fn" IDENTIFIER? "(" parameter_list? ")" block_statement
        | "(" parameter_list? ")" "=>" (expression | block_statement)

parameter_list = IDENTIFIER ("," IDENTIFIER)*

argument_list = expression ("," expression)*

expression_list = expression ("," expression)*

property_list = (IDENTIFIER | STRING) ":" expression ("," (IDENTIFIER | STRING) ":" expression)*
```