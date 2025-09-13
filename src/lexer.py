"""
Sayiy Programming Language Lexer
Tokenizes source code into a stream of tokens
"""

import re
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, Iterator

class TokenType(Enum):
    # Literals
    NUMBER = auto()
    STRING = auto()
    BOOLEAN = auto()
    IDENTIFIER = auto()
    
    # Keywords
    LET = auto()
    FN = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    FOR = auto()
    IN = auto()
    RETURN = auto()
    BREAK = auto()
    CONTINUE = auto()
    TRUE = auto()
    FALSE = auto()
    NULL = auto()
    
    # Operators
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    MODULO = auto()
    POWER = auto()
    
    # Comparison
    EQUAL = auto()
    NOT_EQUAL = auto()
    LESS = auto()
    LESS_EQUAL = auto()
    GREATER = auto()
    GREATER_EQUAL = auto()
    
    # Logical
    AND = auto()
    OR = auto()
    NOT = auto()
    
    # Assignment
    ASSIGN = auto()
    PLUS_ASSIGN = auto()
    MINUS_ASSIGN = auto()
    
    # Punctuation
    SEMICOLON = auto()
    COMMA = auto()
    DOT = auto()
    COLON = auto()
    ARROW = auto()
    RANGE = auto()
    
    # Brackets
    LEFT_PAREN = auto()
    RIGHT_PAREN = auto()
    LEFT_BRACE = auto()
    RIGHT_BRACE = auto()
    LEFT_BRACKET = auto()
    RIGHT_BRACKET = auto()
    
    # Special
    NEWLINE = auto()
    EOF = auto()
    COMMENT = auto()

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int

class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens = []
        
        # Keywords mapping
        self.keywords = {
            'let': TokenType.LET,
            'fn': TokenType.FN,
            'if': TokenType.IF,
            'else': TokenType.ELSE,
            'while': TokenType.WHILE,
            'for': TokenType.FOR,
            'in': TokenType.IN,
            'return': TokenType.RETURN,
            'break': TokenType.BREAK,
            'continue': TokenType.CONTINUE,
            'true': TokenType.TRUE,
            'false': TokenType.FALSE,
            'null': TokenType.NULL,
            'and': TokenType.AND,
            'or': TokenType.OR,
            'not': TokenType.NOT,
        }
        
        # Multi-character operators
        self.operators = {
            '==': TokenType.EQUAL,
            '!=': TokenType.NOT_EQUAL,
            '<=': TokenType.LESS_EQUAL,
            '>=': TokenType.GREATER_EQUAL,
            '+=': TokenType.PLUS_ASSIGN,
            '-=': TokenType.MINUS_ASSIGN,
            '=>': TokenType.ARROW,
            '..': TokenType.RANGE,
        }
    
    def current_char(self) -> Optional[str]:
        if self.position >= len(self.source):
            return None
        return self.source[self.position]
    
    def peek_char(self, offset: int = 1) -> Optional[str]:
        peek_pos = self.position + offset
        if peek_pos >= len(self.source):
            return None
        return self.source[peek_pos]
    
    def advance(self) -> Optional[str]:
        char = self.current_char()
        self.position += 1
        if char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return char
    
    def skip_whitespace(self):
        while self.current_char() and self.current_char() in ' \t\r':
            self.advance()
    
    def read_string(self) -> str:
        quote_char = self.advance()  # Skip opening quote
        value = ''
        
        while self.current_char() and self.current_char() != quote_char:
            char = self.current_char()
            if char == '\\':
                self.advance()
                next_char = self.current_char()
                if next_char == 'n':
                    value += '\n'
                elif next_char == 't':
                    value += '\t'
                elif next_char == 'r':
                    value += '\r'
                elif next_char == '\\':
                    value += '\\'
                elif next_char == quote_char:
                    value += quote_char
                else:
                    value += next_char or ''
                self.advance()
            else:
                value += char
                self.advance()
        
        if self.current_char() == quote_char:
            self.advance()  # Skip closing quote
        else:
            raise SyntaxError(f"Unterminated string at line {self.line}")
        
        return value
    
    def read_number(self) -> str:
        value = ''
        has_dot = False
        
        while (self.current_char() and 
               (self.current_char().isdigit() or self.current_char() == '.')):
            if self.current_char() == '.':
                if has_dot or self.peek_char() == '.':  # Handle range operator
                    break
                has_dot = True
            value += self.advance()
        
        return value
    
    def read_identifier(self) -> str:
        value = ''
        while (self.current_char() and 
               (self.current_char().isalnum() or self.current_char() == '_')):
            value += self.advance()
        return value
    
    def read_comment(self) -> str:
        comment = ''
        self.advance()  # Skip first '/'
        self.advance()  # Skip second '/'
        
        while self.current_char() and self.current_char() != '\n':
            comment += self.advance()
        
        return comment.strip()
    
    def tokenize(self) -> List[Token]:
        while self.current_char():
            start_line = self.line
            start_column = self.column
            
            self.skip_whitespace()
            
            if not self.current_char():
                break
            
            char = self.current_char()
            
            # Newlines
            if char == '\n':
                self.tokens.append(Token(TokenType.NEWLINE, char, start_line, start_column))
                self.advance()
                continue
            
            # Comments
            if char == '/' and self.peek_char() == '/':
                comment = self.read_comment()
                self.tokens.append(Token(TokenType.COMMENT, comment, start_line, start_column))
                continue
            
            # Strings
            if char in '"\'':
                string_value = self.read_string()
                self.tokens.append(Token(TokenType.STRING, string_value, start_line, start_column))
                continue
            
            # Numbers
            if char.isdigit():
                number = self.read_number()
                self.tokens.append(Token(TokenType.NUMBER, number, start_line, start_column))
                continue
            
            # Identifiers and keywords
            if char.isalpha() or char == '_':
                identifier = self.read_identifier()
                token_type = self.keywords.get(identifier, TokenType.IDENTIFIER)
                self.tokens.append(Token(token_type, identifier, start_line, start_column))
                continue
            
            # Multi-character operators
            two_char = char + (self.peek_char() or '')
            if two_char in self.operators:
                self.advance()
                self.advance()
                self.tokens.append(Token(self.operators[two_char], two_char, start_line, start_column))
                continue
            
            # Single character tokens
            single_char_tokens = {
                '+': TokenType.PLUS,
                '-': TokenType.MINUS,
                '*': TokenType.MULTIPLY,
                '/': TokenType.DIVIDE,
                '%': TokenType.MODULO,
                '^': TokenType.POWER,
                '=': TokenType.ASSIGN,
                '<': TokenType.LESS,
                '>': TokenType.GREATER,
                '!': TokenType.NOT,
                ';': TokenType.SEMICOLON,
                ',': TokenType.COMMA,
                '.': TokenType.DOT,
                ':': TokenType.COLON,
                '(': TokenType.LEFT_PAREN,
                ')': TokenType.RIGHT_PAREN,
                '{': TokenType.LEFT_BRACE,
                '}': TokenType.RIGHT_BRACE,
                '[': TokenType.LEFT_BRACKET,
                ']': TokenType.RIGHT_BRACKET,
            }
            
            if char in single_char_tokens:
                self.tokens.append(Token(single_char_tokens[char], char, start_line, start_column))
                self.advance()
                continue
            
            # Unknown character
            raise SyntaxError(f"Unexpected character '{char}' at line {self.line}, column {self.column}")
        
        # Add EOF token
        self.tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        return self.tokens