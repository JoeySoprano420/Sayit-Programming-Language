# Contributing to Sayiy Programming Language

Thank you for your interest in contributing to Sayiy! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites
- Python 3.7 or higher
- Git

### Setting Up Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/JoeySoprano420/Sayiy-Programming-Language.git
   cd Sayiy-Programming-Language
   ```

2. Test the installation:
   ```bash
   python sayiy.py --version
   python tests/test_sayiy.py
   ```

## Development Workflow

### Branch Structure
- `main`: Stable release branch
- `develop`: Development branch for new features
- `feature/*`: Feature branches
- `bugfix/*`: Bug fix branches
- `hotfix/*`: Critical bug fixes for production

### Making Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the coding standards below

3. Run tests to ensure everything works:
   ```bash
   python tests/test_sayiy.py
   ```

4. Test with example programs:
   ```bash
   python sayiy.py examples/hello.sayiy
   python sayiy.py examples/functions.sayiy
   ```

5. Commit your changes:
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

6. Push to your branch:
   ```bash
   git push origin feature/your-feature-name
   ```

7. Create a Pull Request

## Coding Standards

### Python Code Style
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to classes and functions
- Maximum line length: 100 characters

### Code Organization
- Keep files focused on single responsibilities
- Use type hints where appropriate
- Handle errors gracefully with proper exception handling

### Example Code Style:
```python
def parse_expression(self, source: str) -> Expression:
    """
    Parse a source string into an expression AST node.
    
    Args:
        source: The source code to parse
        
    Returns:
        Expression: The parsed expression
        
    Raises:
        ParseError: If the source cannot be parsed
    """
    try:
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        return parser.expression()
    except Exception as e:
        raise ParseError(f"Failed to parse expression: {e}")
```

## Types of Contributions

### Bug Reports
When reporting bugs, please include:
- Description of the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- Sayiy version and Python version
- Sample code that demonstrates the issue

### Feature Requests
When requesting features, please include:
- Clear description of the feature
- Use cases and motivation
- Examples of how it would work
- Any relevant examples from other languages

### Code Contributions

#### Language Features
- New syntax features
- Built-in functions
- Standard library extensions
- Performance improvements

#### Development Tools
- IDE integrations
- Debugging tools
- Build system improvements
- Documentation tools

#### Testing
- Unit tests for new features
- Integration tests
- Performance benchmarks
- Example programs

## Testing Guidelines

### Test Categories
1. **Unit Tests**: Test individual components (lexer, parser, interpreter)
2. **Integration Tests**: Test complete programs
3. **Example Tests**: Ensure all examples work correctly

### Writing Tests
- Add tests for all new features
- Test both success and error cases
- Use descriptive test names
- Keep tests focused and independent

### Running Tests
```bash
# Run all tests
python tests/test_sayiy.py

# Run with verbose output
python tests/test_sayiy.py -v

# Run specific test class
python -m unittest tests.test_sayiy.TestLexer
```

## Documentation Guidelines

### Code Documentation
- Document all public APIs
- Include examples in docstrings
- Update language specification for syntax changes

### User Documentation
- Update README.md for user-facing changes
- Add examples for new features
- Update language specification
- Create tutorials for complex features

## Release Process

### Version Numbering
- Follow Semantic Versioning (MAJOR.MINOR.PATCH)
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Release Checklist
1. Update version number in relevant files
2. Update CHANGELOG.md
3. Run full test suite
4. Test example programs
5. Update documentation
6. Create release tag
7. Update README with new features

## Architecture Overview

### Core Components

1. **Lexer** (`src/lexer.py`):
   - Tokenizes source code
   - Handles keywords, operators, literals
   - Error reporting for invalid tokens

2. **Parser** (`src/parser.py`):
   - Recursive descent parser
   - Builds Abstract Syntax Tree (AST)
   - Error recovery and reporting

3. **AST Nodes** (`src/ast_nodes.py`):
   - Defines all AST node types
   - Expressions and statements
   - Utility functions for AST manipulation

4. **Interpreter** (`src/interpreter.py`):
   - Tree-walking interpreter
   - Environment management for variables
   - Built-in function execution

5. **Standard Library** (`src/stdlib/`):
   - Built-in functions
   - Utility functions
   - Future: modules and packages

6. **REPL** (`src/repl.py`):
   - Interactive shell
   - Multi-line input handling
   - Debug features

### Design Principles

1. **Simplicity**: Keep the language simple and intuitive
2. **Consistency**: Maintain consistent syntax and behavior
3. **Extensibility**: Design for easy addition of new features
4. **Performance**: Optimize for reasonable performance
5. **Error Handling**: Provide clear, helpful error messages

## Getting Help

- Check existing issues and documentation
- Ask questions in discussions
- Join our community chat (when available)
- Contact maintainers for major architectural questions

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to Sayiy Programming Language!