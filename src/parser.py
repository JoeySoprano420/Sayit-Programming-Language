"""
Sayiy Programming Language Parser
Recursive descent parser that builds an Abstract Syntax Tree (AST)
"""

from typing import List, Optional, Union
from lexer import Token, TokenType
from ast_nodes import *

class ParseError(Exception):
    def __init__(self, message: str, token: Token):
        self.message = message
        self.token = token
        super().__init__(f"Parse error at line {token.line}, column {token.column}: {message}")

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current = 0
    
    def is_at_end(self) -> bool:
        return self.peek().type == TokenType.EOF
    
    def peek(self) -> Token:
        return self.tokens[self.current]
    
    def previous(self) -> Token:
        return self.tokens[self.current - 1]
    
    def advance(self) -> Token:
        if not self.is_at_end():
            self.current += 1
        return self.previous()
    
    def check(self, token_type: TokenType) -> bool:
        if self.is_at_end():
            return False
        return self.peek().type == token_type
    
    def match(self, *types: TokenType) -> bool:
        for token_type in types:
            if self.check(token_type):
                self.advance()
                return True
        return False
    
    def consume(self, token_type: TokenType, message: str) -> Token:
        if self.check(token_type):
            return self.advance()
        
        current_token = self.peek()
        raise ParseError(message, current_token)
    
    def skip_newlines(self):
        while self.match(TokenType.NEWLINE, TokenType.COMMENT):
            pass
    
    def parse(self) -> Program:
        statements = []
        
        while not self.is_at_end():
            self.skip_newlines()
            if not self.is_at_end():
                stmt = self.statement()
                if stmt:
                    statements.append(stmt)
        
        return Program(statements)
    
    def statement(self) -> Optional[Statement]:
        try:
            if self.match(TokenType.LET):
                return self.variable_declaration()
            if self.match(TokenType.FN):
                return self.function_declaration()
            if self.match(TokenType.IF):
                return self.if_statement()
            if self.match(TokenType.WHILE):
                return self.while_statement()
            if self.match(TokenType.FOR):
                return self.for_statement()
            if self.match(TokenType.RETURN):
                return self.return_statement()
            if self.match(TokenType.BREAK):
                return BreakStatement()
            if self.match(TokenType.CONTINUE):
                return ContinueStatement()
            if self.match(TokenType.LEFT_BRACE):
                return self.block_statement()
            
            return self.expression_statement()
        except ParseError:
            self.synchronize()
            return None
    
    def variable_declaration(self) -> VariableDeclaration:
        name = self.consume(TokenType.IDENTIFIER, "Expected variable name").value
        
        value = None
        if self.match(TokenType.ASSIGN):
            value = self.expression()
        
        return VariableDeclaration(name, value)
    
    def function_declaration(self) -> FunctionDeclaration:
        name = self.consume(TokenType.IDENTIFIER, "Expected function name").value
        
        self.consume(TokenType.LEFT_PAREN, "Expected '(' after function name")
        
        parameters = []
        if not self.check(TokenType.RIGHT_PAREN):
            parameters.append(self.consume(TokenType.IDENTIFIER, "Expected parameter name").value)
            while self.match(TokenType.COMMA):
                parameters.append(self.consume(TokenType.IDENTIFIER, "Expected parameter name").value)
        
        self.consume(TokenType.RIGHT_PAREN, "Expected ')' after parameters")
        
        self.consume(TokenType.LEFT_BRACE, "Expected '{' before function body")
        body = self.block_statement()
        
        return FunctionDeclaration(name, parameters, body)
    
    def if_statement(self) -> IfStatement:
        condition = self.expression()
        
        self.consume(TokenType.LEFT_BRACE, "Expected '{' after if condition")
        then_branch = self.block_statement()
        
        else_branch = None
        if self.match(TokenType.ELSE):
            if self.match(TokenType.IF):
                else_branch = self.if_statement()
            else:
                self.consume(TokenType.LEFT_BRACE, "Expected '{' after else")
                else_branch = self.block_statement()
        
        return IfStatement(condition, then_branch, else_branch)
    
    def while_statement(self) -> WhileLoop:
        condition = self.expression()
        
        self.consume(TokenType.LEFT_BRACE, "Expected '{' after while condition")
        body = self.block_statement()
        
        return WhileLoop(condition, body)
    
    def for_statement(self) -> ForLoop:
        variable = self.consume(TokenType.IDENTIFIER, "Expected variable name").value
        self.consume(TokenType.IN, "Expected 'in' after for variable")
        iterable = self.expression()
        
        self.consume(TokenType.LEFT_BRACE, "Expected '{' after for expression")
        body = self.block_statement()
        
        return ForLoop(variable, iterable, body)
    
    def return_statement(self) -> ReturnStatement:
        value = None
        if not self.check(TokenType.NEWLINE) and not self.check(TokenType.RIGHT_BRACE) and not self.is_at_end():
            value = self.expression()
        
        return ReturnStatement(value)
    
    def block_statement(self) -> Block:
        statements = []
        
        while not self.check(TokenType.RIGHT_BRACE) and not self.is_at_end():
            self.skip_newlines()
            if not self.check(TokenType.RIGHT_BRACE):
                stmt = self.statement()
                if stmt:
                    statements.append(stmt)
        
        self.consume(TokenType.RIGHT_BRACE, "Expected '}' after block")
        return Block(statements)
    
    def expression_statement(self) -> ExpressionStatement:
        expr = self.expression()
        return ExpressionStatement(expr)
    
    def expression(self) -> Expression:
        return self.assignment()
    
    def assignment(self) -> Expression:
        expr = self.logical_or()
        
        if self.match(TokenType.ASSIGN, TokenType.PLUS_ASSIGN, TokenType.MINUS_ASSIGN):
            operator = self.previous().value
            value = self.assignment()
            
            if isinstance(expr, (Identifier, MemberAccess, IndexAccess)):
                return Assignment(expr, value, operator)
            
            raise ParseError("Invalid assignment target", self.previous())
        
        return expr
    
    def logical_or(self) -> Expression:
        expr = self.logical_and()
        
        while self.match(TokenType.OR):
            operator = self.previous().value
            right = self.logical_and()
            expr = BinaryOperation(expr, operator, right)
        
        return expr
    
    def logical_and(self) -> Expression:
        expr = self.equality()
        
        while self.match(TokenType.AND):
            operator = self.previous().value
            right = self.equality()
            expr = BinaryOperation(expr, operator, right)
        
        return expr
    
    def equality(self) -> Expression:
        expr = self.comparison()
        
        while self.match(TokenType.EQUAL, TokenType.NOT_EQUAL):
            operator = self.previous().value
            right = self.comparison()
            expr = BinaryOperation(expr, operator, right)
        
        return expr
    
    def comparison(self) -> Expression:
        expr = self.range_expr()
        
        while self.match(TokenType.GREATER, TokenType.GREATER_EQUAL, 
                         TokenType.LESS, TokenType.LESS_EQUAL):
            operator = self.previous().value
            right = self.range_expr()
            expr = BinaryOperation(expr, operator, right)
        
        return expr
    
    def range_expr(self) -> Expression:
        expr = self.term()
        
        if self.match(TokenType.RANGE):
            end = self.term()
            return RangeExpression(expr, end)
        
        return expr
    
    def term(self) -> Expression:
        expr = self.factor()
        
        while self.match(TokenType.MINUS, TokenType.PLUS):
            operator = self.previous().value
            right = self.factor()
            expr = BinaryOperation(expr, operator, right)
        
        return expr
    
    def factor(self) -> Expression:
        expr = self.unary()
        
        while self.match(TokenType.DIVIDE, TokenType.MULTIPLY, TokenType.MODULO):
            operator = self.previous().value
            right = self.unary()
            expr = BinaryOperation(expr, operator, right)
        
        return expr
    
    def unary(self) -> Expression:
        if self.match(TokenType.NOT, TokenType.MINUS):
            operator = self.previous().value
            right = self.unary()
            return UnaryOperation(operator, right)
        
        return self.power()
    
    def power(self) -> Expression:
        expr = self.call()
        
        if self.match(TokenType.POWER):
            right = self.unary()  # Right associative
            expr = BinaryOperation(expr, '^', right)
        
        return expr
    
    def call(self) -> Expression:
        expr = self.primary()
        
        while True:
            if self.match(TokenType.LEFT_PAREN):
                expr = self.finish_call(expr)
            elif self.match(TokenType.DOT):
                name = self.consume(TokenType.IDENTIFIER, "Expected property name").value
                expr = MemberAccess(expr, name)
            elif self.match(TokenType.LEFT_BRACKET):
                index = self.expression()
                self.consume(TokenType.RIGHT_BRACKET, "Expected ']' after index")
                expr = IndexAccess(expr, index)
            else:
                break
        
        return expr
    
    def finish_call(self, callee: Expression) -> FunctionCall:
        arguments = []
        
        if not self.check(TokenType.RIGHT_PAREN):
            arguments.append(self.expression())
            while self.match(TokenType.COMMA):
                arguments.append(self.expression())
        
        self.consume(TokenType.RIGHT_PAREN, "Expected ')' after arguments")
        return FunctionCall(callee, arguments)
    
    def primary(self) -> Expression:
        if self.match(TokenType.TRUE):
            return BooleanLiteral(True)
        
        if self.match(TokenType.FALSE):
            return BooleanLiteral(False)
        
        if self.match(TokenType.NULL):
            return NullLiteral()
        
        if self.match(TokenType.NUMBER):
            value = self.previous().value
            if '.' in value:
                return NumberLiteral(float(value))
            else:
                return NumberLiteral(int(value))
        
        if self.match(TokenType.STRING):
            return StringLiteral(self.previous().value)
        
        if self.match(TokenType.IDENTIFIER):
            return Identifier(self.previous().value)
        
        if self.match(TokenType.LEFT_PAREN):
            # Function expression or grouped expression
            if self.check(TokenType.IDENTIFIER) or self.check(TokenType.RIGHT_PAREN):
                # Could be arrow function parameters
                checkpoint = self.current
                try:
                    params = self.parse_arrow_function_params()
                    if self.match(TokenType.RIGHT_PAREN) and self.match(TokenType.ARROW):
                        body = self.arrow_function_body()
                        return ArrowFunction(params, body)
                except:
                    pass
                
                # Reset and parse as grouped expression
                self.current = checkpoint
            
            expr = self.expression()
            self.consume(TokenType.RIGHT_PAREN, "Expected ')' after expression")
            return expr
        
        if self.match(TokenType.LEFT_BRACKET):
            return self.array_literal()
        
        if self.match(TokenType.LEFT_BRACE):
            return self.object_literal()
        
        if self.match(TokenType.FN):
            return self.function_expression()
        
        raise ParseError(f"Unexpected token '{self.peek().value}'", self.peek())
    
    def parse_arrow_function_params(self) -> List[str]:
        params = []
        if not self.check(TokenType.RIGHT_PAREN):
            params.append(self.consume(TokenType.IDENTIFIER, "Expected parameter name").value)
            while self.match(TokenType.COMMA):
                params.append(self.consume(TokenType.IDENTIFIER, "Expected parameter name").value)
        return params
    
    def arrow_function_body(self) -> Union[Expression, Block]:
        if self.match(TokenType.LEFT_BRACE):
            return self.block_statement()
        else:
            return self.expression()
    
    def function_expression(self) -> FunctionExpression:
        name = None
        if self.check(TokenType.IDENTIFIER):
            name = self.advance().value
        
        self.consume(TokenType.LEFT_PAREN, "Expected '(' after 'fn'")
        
        parameters = []
        if not self.check(TokenType.RIGHT_PAREN):
            parameters.append(self.consume(TokenType.IDENTIFIER, "Expected parameter name").value)
            while self.match(TokenType.COMMA):
                parameters.append(self.consume(TokenType.IDENTIFIER, "Expected parameter name").value)
        
        self.consume(TokenType.RIGHT_PAREN, "Expected ')' after parameters")
        self.consume(TokenType.LEFT_BRACE, "Expected '{' before function body")
        body = self.block_statement()
        
        return FunctionExpression(parameters, body, name)
    
    def array_literal(self) -> ArrayLiteral:
        elements = []
        
        if not self.check(TokenType.RIGHT_BRACKET):
            elements.append(self.expression())
            while self.match(TokenType.COMMA):
                elements.append(self.expression())
        
        self.consume(TokenType.RIGHT_BRACKET, "Expected ']' after array elements")
        return ArrayLiteral(elements)
    
    def object_literal(self) -> ObjectLiteral:
        properties = {}
        
        while not self.check(TokenType.RIGHT_BRACE) and not self.is_at_end():
            if self.match(TokenType.IDENTIFIER, TokenType.STRING):
                key = self.previous().value
                self.consume(TokenType.COLON, "Expected ':' after object key")
                value = self.expression()
                properties[key] = value
                
                if not self.match(TokenType.COMMA):
                    break
        
        self.consume(TokenType.RIGHT_BRACE, "Expected '}' after object properties")
        return ObjectLiteral(properties)
    
    def synchronize(self):
        """Recover from parse error by finding next statement boundary"""
        self.advance()
        
        while not self.is_at_end():
            if self.previous().type == TokenType.NEWLINE:
                return
            
            if self.peek().type in [
                TokenType.FN, TokenType.LET, TokenType.FOR, TokenType.IF,
                TokenType.WHILE, TokenType.RETURN
            ]:
                return
            
            self.advance()