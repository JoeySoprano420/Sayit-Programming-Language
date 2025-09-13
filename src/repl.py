"""
Sayiy Programming Language REPL
Interactive Read-Eval-Print Loop
"""

import sys
from typing import Optional
from lexer import Lexer, TokenType
from parser import Parser, ParseError
from interpreter import Interpreter, RuntimeError as SayiyRuntimeError
from ast_nodes import pretty_print_ast

class REPL:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.interpreter = Interpreter(debug=debug)
        self.multiline_input = ""
        self.prompt = "sayiy> "
        self.continuation_prompt = "...   "
        
    def run(self):
        """Start the REPL"""
        print("Sayiy Programming Language v1.0.0")
        print("Type 'help' for help, 'exit' to quit.")
        print()
        
        while True:
            try:
                if self.multiline_input:
                    line = input(self.continuation_prompt)
                else:
                    line = input(self.prompt)
                
                # Handle special commands
                if not self.multiline_input and line.strip() in ['exit', 'quit']:
                    print("Goodbye!")
                    break
                elif not self.multiline_input and line.strip() == 'help':
                    self.show_help()
                    continue
                elif not self.multiline_input and line.strip() == 'clear':
                    print("\033[2J\033[H")  # Clear screen
                    continue
                elif not self.multiline_input and line.strip().startswith('debug'):
                    parts = line.strip().split()
                    if len(parts) > 1 and parts[1] in ['on', 'off']:
                        self.debug = parts[1] == 'on'
                        self.interpreter.debug = self.debug
                        print(f"Debug mode {'enabled' if self.debug else 'disabled'}")
                    else:
                        print(f"Debug mode is {'enabled' if self.debug else 'disabled'}")
                    continue
                
                # Accumulate input for multiline statements
                self.multiline_input += line + "\n"
                
                # Check if input is complete
                if self.is_complete_input(self.multiline_input):
                    self.evaluate_input(self.multiline_input.strip())
                    self.multiline_input = ""
                
            except KeyboardInterrupt:
                print("\nKeyboardInterrupt")
                self.multiline_input = ""
            except EOFError:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                self.multiline_input = ""
    
    def is_complete_input(self, input_text: str) -> bool:
        """Check if the input is a complete statement"""
        try:
            # Try to tokenize and parse
            lexer = Lexer(input_text)
            tokens = lexer.tokenize()
            
            # Check for unmatched braces
            brace_count = 0
            paren_count = 0
            bracket_count = 0
            
            for token in tokens:
                if token.type == TokenType.LEFT_BRACE:
                    brace_count += 1
                elif token.type == TokenType.RIGHT_BRACE:
                    brace_count -= 1
                elif token.type == TokenType.LEFT_PAREN:
                    paren_count += 1
                elif token.type == TokenType.RIGHT_PAREN:
                    paren_count -= 1
                elif token.type == TokenType.LEFT_BRACKET:
                    bracket_count += 1
                elif token.type == TokenType.RIGHT_BRACKET:
                    bracket_count -= 1
            
            # If we have unmatched brackets, we need more input
            if brace_count > 0 or paren_count > 0 or bracket_count > 0:
                return False
            
            # Try to parse - if it succeeds, input is complete
            parser = Parser(tokens)
            parser.parse()
            return True
            
        except ParseError:
            # If parsing fails due to unexpected EOF, we might need more input
            return False
        except:
            # Other errors mean the input is malformed but complete
            return True
    
    def evaluate_input(self, input_text: str):
        """Evaluate the input and print the result"""
        try:
            # Tokenize
            lexer = Lexer(input_text)
            tokens = lexer.tokenize()
            
            if self.debug:
                print("Tokens:")
                for token in tokens:
                    if token.type not in [TokenType.NEWLINE, TokenType.EOF, TokenType.COMMENT]:
                        print(f"  {token.type.name}: {repr(token.value)}")
                print()
            
            # Parse
            parser = Parser(tokens)
            ast = parser.parse()
            
            if self.debug:
                print("AST:")
                print(pretty_print_ast(ast))
                print()
            
            # Interpret
            result = self.interpreter.interpret(ast)
            
            # Print result if it's not None
            if result is not None:
                self.print_result(result)
                
        except ParseError as e:
            print(f"Parse Error: {e}")
        except SayiyRuntimeError as e:
            print(f"Runtime Error: {e}")
        except Exception as e:
            print(f"Error: {e}")
    
    def print_result(self, result):
        """Print the result in a user-friendly format"""
        if result is None:
            print("null")
        elif isinstance(result, bool):
            print("true" if result else "false")
        elif isinstance(result, str):
            # Don't add quotes for string results in REPL
            print(result)
        elif isinstance(result, (list, dict)):
            import json
            print(json.dumps(result, indent=2, default=str))
        else:
            print(result)
    
    def show_help(self):
        """Show help information"""
        help_text = """
Sayiy Programming Language REPL Help

Commands:
  help          - Show this help
  exit, quit    - Exit the REPL
  clear         - Clear the screen
  debug on/off  - Enable/disable debug mode

Language Features:
  Variables:    let x = 42
                let name = "Sayiy"
                let numbers = [1, 2, 3]
                let person = {name: "Alice", age: 30}

  Functions:    fn greet(name) {
                  return "Hello, " + name + "!"
                }
                
                let square = (x) => x * x

  Control:      if condition {
                  // code
                } else {
                  // code
                }
                
                while condition {
                  // code
                }
                
                for item in array {
                  // code
                }

  Built-ins:    print("Hello, World!")
                len([1, 2, 3])
                type(42)
                str(123)
                int("456")
                
Examples:
  sayiy> let x = 10
  sayiy> let y = x * 2
  sayiy> print("Result:", y)
  Result: 20
  
  sayiy> fn factorial(n) {
  ...      if n <= 1 {
  ...        return 1
  ...      } else {
  ...        return n * factorial(n - 1)
  ...      }
  ...    }
  sayiy> factorial(5)
  120
"""
        print(help_text)