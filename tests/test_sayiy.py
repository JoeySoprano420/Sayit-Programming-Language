#!/usr/bin/env python3
"""
Sayiy Programming Language Tests
Test suite for lexer, parser, and interpreter
"""

import unittest
import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lexer import Lexer, TokenType
from parser import Parser
from interpreter import Interpreter
from ast_nodes import *

class TestLexer(unittest.TestCase):
    
    def test_numbers(self):
        source = "42 3.14 0 123.456"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        
        self.assertEqual(tokens[0].type, TokenType.NUMBER)
        self.assertEqual(tokens[0].value, "42")
        self.assertEqual(tokens[1].type, TokenType.NUMBER)
        self.assertEqual(tokens[1].value, "3.14")
    
    def test_strings(self):
        source = '"hello" \'world\' "nested \\"quotes\\""'
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        
        self.assertEqual(tokens[0].type, TokenType.STRING)
        self.assertEqual(tokens[0].value, "hello")
        self.assertEqual(tokens[1].type, TokenType.STRING)
        self.assertEqual(tokens[1].value, "world")
        self.assertEqual(tokens[2].type, TokenType.STRING)
        self.assertEqual(tokens[2].value, 'nested "quotes"')
    
    def test_keywords(self):
        source = "let fn if else while for true false null"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        
        expected_types = [
            TokenType.LET, TokenType.FN, TokenType.IF, TokenType.ELSE,
            TokenType.WHILE, TokenType.FOR, TokenType.TRUE, TokenType.FALSE, TokenType.NULL
        ]
        
        for i, expected_type in enumerate(expected_types):
            self.assertEqual(tokens[i].type, expected_type)
    
    def test_operators(self):
        source = "+ - * / % ^ == != <= >= < > and or not"
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        
        expected_types = [
            TokenType.PLUS, TokenType.MINUS, TokenType.MULTIPLY, TokenType.DIVIDE, 
            TokenType.MODULO, TokenType.POWER, TokenType.EQUAL, TokenType.NOT_EQUAL,
            TokenType.LESS_EQUAL, TokenType.GREATER_EQUAL, TokenType.LESS, 
            TokenType.GREATER, TokenType.AND, TokenType.OR, TokenType.NOT
        ]
        
        for i, expected_type in enumerate(expected_types):
            self.assertEqual(tokens[i].type, expected_type)
    
    def test_comments(self):
        source = '// This is a comment\nlet x = 42 // Another comment'
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        
        # Should have: COMMENT, NEWLINE, LET, IDENTIFIER, ASSIGN, NUMBER, COMMENT, EOF
        self.assertEqual(tokens[0].type, TokenType.COMMENT)
        self.assertEqual(tokens[1].type, TokenType.NEWLINE)
        self.assertEqual(tokens[2].type, TokenType.LET)

class TestParser(unittest.TestCase):
    
    def parse_expression(self, source):
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()
        return program.statements[0].expression
    
    def parse_program(self, source):
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        return parser.parse()
    
    def test_number_literal(self):
        expr = self.parse_expression("42")
        self.assertIsInstance(expr, NumberLiteral)
        self.assertEqual(expr.value, 42)
    
    def test_string_literal(self):
        expr = self.parse_expression('"hello"')
        self.assertIsInstance(expr, StringLiteral)
        self.assertEqual(expr.value, "hello")
    
    def test_binary_operation(self):
        expr = self.parse_expression("3 + 4 * 2")
        self.assertIsInstance(expr, BinaryOperation)
        self.assertEqual(expr.operator, "+")
        self.assertIsInstance(expr.left, NumberLiteral)
        self.assertIsInstance(expr.right, BinaryOperation)
        self.assertEqual(expr.right.operator, "*")
    
    def test_function_call(self):
        expr = self.parse_expression("print(42, 'hello')")
        self.assertIsInstance(expr, FunctionCall)
        self.assertIsInstance(expr.function, Identifier)
        self.assertEqual(expr.function.name, "print")
        self.assertEqual(len(expr.arguments), 2)
    
    def test_variable_declaration(self):
        program = self.parse_program("let x = 42")
        stmt = program.statements[0]
        self.assertIsInstance(stmt, VariableDeclaration)
        self.assertEqual(stmt.name, "x")
        self.assertIsInstance(stmt.value, NumberLiteral)
    
    def test_function_declaration(self):
        program = self.parse_program("""
        fn greet(name) {
            return "Hello, " + name
        }
        """)
        stmt = program.statements[0]
        self.assertIsInstance(stmt, FunctionDeclaration)
        self.assertEqual(stmt.name, "greet")
        self.assertEqual(stmt.parameters, ["name"])

class TestInterpreter(unittest.TestCase):
    
    def setUp(self):
        self.interpreter = Interpreter()
    
    def evaluate(self, source):
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        return self.interpreter.interpret(ast)
    
    def test_number_evaluation(self):
        result = self.evaluate("42")
        self.assertEqual(result, 42)
    
    def test_arithmetic(self):
        result = self.evaluate("3 + 4 * 2")
        self.assertEqual(result, 11)
        
        result = self.evaluate("10 / 2 - 3")
        self.assertEqual(result, 2)
        
        result = self.evaluate("2 ^ 3")
        self.assertEqual(result, 8)
    
    def test_string_concatenation(self):
        result = self.evaluate('"Hello, " + "World!"')
        self.assertEqual(result, "Hello, World!")
    
    def test_variable_assignment(self):
        self.evaluate("let x = 42")
        result = self.evaluate("x")
        self.assertEqual(result, 42)
    
    def test_function_declaration_and_call(self):
        self.evaluate("""
        fn double(x) {
            return x * 2
        }
        """)
        result = self.evaluate("double(21)")
        self.assertEqual(result, 42)
    
    def test_if_statement(self):
        result = self.evaluate("""
        let x = 10
        if x > 5 {
            x = x * 2
        }
        x
        """)
        self.assertEqual(result, 20)
    
    def test_while_loop(self):
        result = self.evaluate("""
        let i = 0
        let sum = 0
        while i < 5 {
            sum = sum + i
            i = i + 1
        }
        sum
        """)
        self.assertEqual(result, 10)  # 0 + 1 + 2 + 3 + 4
    
    def test_array_operations(self):
        result = self.evaluate("[1, 2, 3]")
        self.assertEqual(result, [1, 2, 3])
        
        # Test array indexing
        self.evaluate("let arr = [1, 2, 3]")
        result = self.evaluate("arr[1]")
        self.assertEqual(result, 2)
        
        result = self.evaluate("len([1, 2, 3, 4])")
        self.assertEqual(result, 4)
    
    def test_object_operations(self):
        # Test object member access
        self.evaluate('let obj = {name: "Alice", age: 30}')
        result = self.evaluate('obj.name')
        self.assertEqual(result, "Alice")
        
        # Test object index access
        self.evaluate('let obj2 = {x: 10, y: 20}')
        result = self.evaluate('obj2["x"]')
        self.assertEqual(result, 10)
    
    def test_builtin_functions(self):
        # Test type function
        result = self.evaluate('type(42)')
        self.assertEqual(result, "integer")
        
        result = self.evaluate('type("hello")')
        self.assertEqual(result, "string")
        
        # Test str function
        result = self.evaluate('str(42)')
        self.assertEqual(result, "42")
        
        # Test math functions
        result = self.evaluate('abs(-42)')
        self.assertEqual(result, 42)
        
        result = self.evaluate('max(1, 5, 3)')
        self.assertEqual(result, 5)

class TestIntegration(unittest.TestCase):
    
    def setUp(self):
        self.interpreter = Interpreter()
    
    def run_example(self, filename):
        """Run an example file and return whether it executed without errors"""
        try:
            example_path = Path(__file__).parent.parent / 'examples' / filename
            with open(example_path, 'r') as f:
                source = f.read()
            
            lexer = Lexer(source)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            ast = parser.parse()
            self.interpreter.interpret(ast)
            return True
        except Exception as e:
            print(f"Error running {filename}: {e}")
            return False
    
    def test_hello_example(self):
        self.assertTrue(self.run_example('hello.sayiy'))
    
    def test_functions_example(self):
        self.assertTrue(self.run_example('functions.sayiy'))

if __name__ == '__main__':
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestLexer))
    suite.addTests(loader.loadTestsFromTestCase(TestParser))
    suite.addTests(loader.loadTestsFromTestCase(TestInterpreter))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with non-zero code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)