#!/usr/bin/env python3
"""
Sayiy Programming Language
Main entry point for the interpreter
"""

import sys
import argparse
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from lexer import Lexer
from parser import Parser
from interpreter import Interpreter
from repl import REPL

def main():
    parser = argparse.ArgumentParser(description='Sayiy Programming Language Interpreter')
    parser.add_argument('file', nargs='?', help='Sayiy source file to execute')
    parser.add_argument('--repl', action='store_true', help='Start interactive REPL')
    parser.add_argument('--version', action='version', version='Sayiy 1.0.0')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    if args.repl or not args.file:
        # Start REPL
        repl = REPL(debug=args.debug)
        repl.run()
    else:
        # Execute file
        try:
            with open(args.file, 'r') as f:
                source = f.read()
            
            # Tokenize
            lexer = Lexer(source)
            tokens = lexer.tokenize()
            
            if args.debug:
                print("Tokens:", tokens)
            
            # Parse
            parser = Parser(tokens)
            ast = parser.parse()
            
            if args.debug:
                print("AST:", ast)
            
            # Interpret
            interpreter = Interpreter(debug=args.debug)
            result = interpreter.interpret(ast)
            
            if result is not None:
                print(result)
                
        except FileNotFoundError:
            print(f"Error: File '{args.file}' not found")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

if __name__ == '__main__':
    main()