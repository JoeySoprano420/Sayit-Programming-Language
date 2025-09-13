"""
Sayiy Programming Language AST Nodes
Abstract Syntax Tree node definitions
"""

from abc import ABC, abstractmethod
from typing import List, Any, Optional, Dict, Union
from dataclasses import dataclass

class ASTNode(ABC):
    """Base class for all AST nodes"""
    pass

class Expression(ASTNode):
    """Base class for expressions"""
    pass

class Statement(ASTNode):
    """Base class for statements"""
    pass

# Literal Expressions
@dataclass
class NumberLiteral(Expression):
    value: Union[int, float]

@dataclass
class StringLiteral(Expression):
    value: str

@dataclass
class BooleanLiteral(Expression):
    value: bool

@dataclass
class NullLiteral(Expression):
    pass

@dataclass
class ArrayLiteral(Expression):
    elements: List[Expression]

@dataclass
class ObjectLiteral(Expression):
    properties: Dict[str, Expression]

# Variable and Identifier
@dataclass
class Identifier(Expression):
    name: str

@dataclass
class MemberAccess(Expression):
    object: Expression
    property: str

@dataclass
class IndexAccess(Expression):
    object: Expression
    index: Expression

# Binary Operations
@dataclass
class BinaryOperation(Expression):
    left: Expression
    operator: str
    right: Expression

@dataclass
class UnaryOperation(Expression):
    operator: str
    operand: Expression

# Function related
@dataclass
class FunctionCall(Expression):
    function: Expression
    arguments: List[Expression]

@dataclass
class FunctionExpression(Expression):
    parameters: List[str]
    body: 'Block'
    name: Optional[str] = None

@dataclass
class ArrowFunction(Expression):
    parameters: List[str]
    body: Union[Expression, 'Block']

# Control Flow Expressions
@dataclass
class ConditionalExpression(Expression):
    condition: Expression
    true_expr: Expression
    false_expr: Expression

@dataclass
class RangeExpression(Expression):
    start: Expression
    end: Expression
    inclusive: bool = False

# Statements
@dataclass
class ExpressionStatement(Statement):
    expression: Expression

@dataclass
class VariableDeclaration(Statement):
    name: str
    value: Optional[Expression] = None
    is_const: bool = False

@dataclass
class Assignment(Statement):
    target: Expression
    value: Expression
    operator: str = '='

@dataclass
class Block(Statement):
    statements: List[Statement]

@dataclass
class IfStatement(Statement):
    condition: Expression
    then_branch: Statement
    else_branch: Optional[Statement] = None

@dataclass
class WhileLoop(Statement):
    condition: Expression
    body: Statement

@dataclass
class ForLoop(Statement):
    variable: str
    iterable: Expression
    body: Statement

@dataclass
class FunctionDeclaration(Statement):
    name: str
    parameters: List[str]
    body: Block

@dataclass
class ReturnStatement(Statement):
    value: Optional[Expression] = None

@dataclass
class BreakStatement(Statement):
    pass

@dataclass
class ContinueStatement(Statement):
    pass

# Program root
@dataclass
class Program(ASTNode):
    statements: List[Statement]

# Helper functions for AST manipulation
def ast_to_dict(node: ASTNode) -> Dict[str, Any]:
    """Convert AST node to dictionary representation"""
    if isinstance(node, list):
        return [ast_to_dict(item) for item in node]
    
    if not isinstance(node, ASTNode):
        return node
    
    result = {'type': node.__class__.__name__}
    
    for field, value in node.__dict__.items():
        if isinstance(value, ASTNode):
            result[field] = ast_to_dict(value)
        elif isinstance(value, list):
            result[field] = [ast_to_dict(item) if isinstance(item, ASTNode) else item for item in value]
        elif isinstance(value, dict):
            result[field] = {k: ast_to_dict(v) if isinstance(v, ASTNode) else v for k, v in value.items()}
        else:
            result[field] = value
    
    return result

def pretty_print_ast(node: ASTNode, indent: int = 0) -> str:
    """Pretty print AST for debugging"""
    spaces = '  ' * indent
    
    if isinstance(node, list):
        result = '[\n'
        for item in node:
            result += f'{spaces}  {pretty_print_ast(item, indent + 1)},\n'
        result += f'{spaces}]'
        return result
    
    if not isinstance(node, ASTNode):
        return repr(node)
    
    result = f'{node.__class__.__name__}('
    
    fields = []
    for field, value in node.__dict__.items():
        if isinstance(value, (ASTNode, list)):
            field_str = f'{field}={pretty_print_ast(value, indent + 1)}'
        else:
            field_str = f'{field}={repr(value)}'
        fields.append(field_str)
    
    if fields:
        if len(fields) == 1 and not isinstance(list(node.__dict__.values())[0], (ASTNode, list)):
            result += fields[0]
        else:
            result += '\n'
            for field in fields:
                result += f'{spaces}  {field},\n'
            result += f'{spaces}'
    
    result += ')'
    return result