"""
Sayiy Programming Language Interpreter
Evaluates the Abstract Syntax Tree (AST) and executes the program
"""

from typing import Any, Dict, List, Optional, Callable, Union
from ast_nodes import *
from stdlib.builtin_functions import get_builtin_functions

class SayiyFunction:
    def __init__(self, declaration: Union[FunctionDeclaration, FunctionExpression, ArrowFunction], 
                 closure: 'Environment'):
        self.declaration = declaration
        self.closure = closure
        
        if isinstance(declaration, FunctionDeclaration):
            self.name = declaration.name
            self.parameters = declaration.parameters
            self.body = declaration.body
        elif isinstance(declaration, FunctionExpression):
            self.name = declaration.name
            self.parameters = declaration.parameters
            self.body = declaration.body
        elif isinstance(declaration, ArrowFunction):
            self.name = None
            self.parameters = declaration.parameters
            self.body = declaration.body
    
    def call(self, interpreter: 'Interpreter', arguments: List[Any]) -> Any:
        environment = Environment(self.closure)
        
        # Bind parameters
        for i, param in enumerate(self.parameters):
            value = arguments[i] if i < len(arguments) else None
            environment.define(param, value)
        
        try:
            if isinstance(self.body, Block):
                interpreter.execute_block(self.body.statements, environment)
            else:
                # Arrow function with expression body
                return interpreter.evaluate_with_environment(self.body, environment)
        except ReturnValue as return_val:
            return return_val.value
        
        return None

class ReturnValue(Exception):
    def __init__(self, value: Any):
        self.value = value

class BreakException(Exception):
    pass

class ContinueException(Exception):
    pass

class RuntimeError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

class Environment:
    def __init__(self, enclosing: Optional['Environment'] = None):
        self.enclosing = enclosing
        self.values: Dict[str, Any] = {}
    
    def define(self, name: str, value: Any):
        self.values[name] = value
    
    def get(self, name: str) -> Any:
        if name in self.values:
            return self.values[name]
        
        if self.enclosing:
            return self.enclosing.get(name)
        
        raise RuntimeError(f"Undefined variable '{name}'")
    
    def assign(self, name: str, value: Any):
        if name in self.values:
            self.values[name] = value
            return
        
        if self.enclosing:
            self.enclosing.assign(name, value)
            return
        
        raise RuntimeError(f"Undefined variable '{name}'")

class Interpreter:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.globals = Environment()
        self.environment = self.globals
        
        # Define built-in functions
        for name, func in get_builtin_functions().items():
            self.globals.define(name, func)
    
    def interpret(self, program: Program) -> Any:
        try:
            result = None
            for statement in program.statements:
                result = self.execute(statement)
            return result
        except RuntimeError as error:
            print(f"Runtime Error: {error.message}")
            return None
    
    def execute(self, statement: Statement) -> Any:
        return self.visit(statement)
    
    def evaluate(self, expression: Expression) -> Any:
        return self.visit(expression)
    
    def evaluate_with_environment(self, expression: Expression, environment: Environment) -> Any:
        previous = self.environment
        try:
            self.environment = environment
            return self.evaluate(expression)
        finally:
            self.environment = previous
    
    def execute_block(self, statements: List[Statement], environment: Environment):
        previous = self.environment
        try:
            self.environment = environment
            
            for statement in statements:
                self.execute(statement)
        finally:
            self.environment = previous
    
    def visit(self, node: ASTNode) -> Any:
        method_name = f'visit_{node.__class__.__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)
    
    def generic_visit(self, node: ASTNode):
        raise RuntimeError(f"No visit method for {node.__class__.__name__}")
    
    # Expression visitors
    def visit_NumberLiteral(self, node: NumberLiteral) -> Union[int, float]:
        return node.value
    
    def visit_StringLiteral(self, node: StringLiteral) -> str:
        return node.value
    
    def visit_BooleanLiteral(self, node: BooleanLiteral) -> bool:
        return node.value
    
    def visit_NullLiteral(self, node: NullLiteral) -> None:
        return None
    
    def visit_ArrayLiteral(self, node: ArrayLiteral) -> List[Any]:
        return [self.evaluate(element) for element in node.elements]
    
    def visit_ObjectLiteral(self, node: ObjectLiteral) -> Dict[str, Any]:
        result = {}
        for key, value_expr in node.properties.items():
            result[key] = self.evaluate(value_expr)
        return result
    
    def visit_Identifier(self, node: Identifier) -> Any:
        return self.environment.get(node.name)
    
    def visit_BinaryOperation(self, node: BinaryOperation) -> Any:
        left = self.evaluate(node.left)
        right = self.evaluate(node.right)
        
        operator = node.operator
        
        # Arithmetic operators
        if operator == '+':
            return left + right
        elif operator == '-':
            return left - right
        elif operator == '*':
            return left * right
        elif operator == '/':
            if right == 0:
                raise RuntimeError("Division by zero")
            return left / right
        elif operator == '%':
            return left % right
        elif operator == '^':
            return left ** right
        
        # Comparison operators
        elif operator == '==':
            return left == right
        elif operator == '!=':
            return left != right
        elif operator == '<':
            return left < right
        elif operator == '<=':
            return left <= right
        elif operator == '>':
            return left > right
        elif operator == '>=':
            return left >= right
        
        # Logical operators
        elif operator == 'and':
            return self.is_truthy(left) and self.is_truthy(right)
        elif operator == 'or':
            return self.is_truthy(left) or self.is_truthy(right)
        
        raise RuntimeError(f"Unknown binary operator: {operator}")
    
    def visit_UnaryOperation(self, node: UnaryOperation) -> Any:
        operand = self.evaluate(node.operand)
        
        if node.operator == '-':
            return -operand
        elif node.operator == 'not':
            return not self.is_truthy(operand)
        
        raise RuntimeError(f"Unknown unary operator: {node.operator}")
    
    def visit_FunctionCall(self, node: FunctionCall) -> Any:
        callee = self.evaluate(node.function)
        
        arguments = []
        for arg in node.arguments:
            arguments.append(self.evaluate(arg))
        
        if callable(callee):
            # Built-in function
            try:
                return callee(*arguments)
            except TypeError as e:
                raise RuntimeError(f"Function call error: {str(e)}")
        elif isinstance(callee, SayiyFunction):
            # User-defined function
            if len(arguments) != len(callee.parameters):
                raise RuntimeError(f"Expected {len(callee.parameters)} arguments but got {len(arguments)}")
            return callee.call(self, arguments)
        else:
            raise RuntimeError("Can only call functions")
    
    def visit_MemberAccess(self, node: MemberAccess) -> Any:
        obj = self.evaluate(node.object)
        
        if isinstance(obj, dict):
            return obj.get(node.property)
        elif isinstance(obj, list):
            # Built-in list methods
            if node.property == 'length':
                return len(obj)
            elif node.property == 'push':
                def push(*items):
                    obj.extend(items)
                    return len(obj)
                return push
            elif node.property == 'pop':
                def pop():
                    return obj.pop() if obj else None
                return pop
        
        raise RuntimeError(f"Property '{node.property}' not found")
    
    def visit_IndexAccess(self, node: IndexAccess) -> Any:
        obj = self.evaluate(node.object)
        index = self.evaluate(node.index)
        
        try:
            return obj[index]
        except (IndexError, KeyError, TypeError):
            raise RuntimeError(f"Invalid index access")
    
    def visit_RangeExpression(self, node: RangeExpression) -> List[int]:
        start = self.evaluate(node.start)
        end = self.evaluate(node.end)
        
        if not isinstance(start, int) or not isinstance(end, int):
            raise RuntimeError("Range bounds must be integers")
        
        return list(range(start, end + (1 if node.inclusive else 0)))
    
    def visit_FunctionExpression(self, node: FunctionExpression) -> SayiyFunction:
        return SayiyFunction(node, self.environment)
    
    def visit_ArrowFunction(self, node: ArrowFunction) -> SayiyFunction:
        return SayiyFunction(node, self.environment)
    
    # Statement visitors
    def visit_ExpressionStatement(self, node: ExpressionStatement) -> Any:
        return self.evaluate(node.expression)
    
    def visit_VariableDeclaration(self, node: VariableDeclaration) -> None:
        value = None
        if node.value:
            value = self.evaluate(node.value)
        
        self.environment.define(node.name, value)
    
    def visit_Assignment(self, node: Assignment) -> Any:
        value = self.evaluate(node.value)
        
        if isinstance(node.target, Identifier):
            if node.operator == '=':
                self.environment.assign(node.target.name, value)
            elif node.operator == '+=':
                current = self.environment.get(node.target.name)
                self.environment.assign(node.target.name, current + value)
            elif node.operator == '-=':
                current = self.environment.get(node.target.name)
                self.environment.assign(node.target.name, current - value)
        elif isinstance(node.target, IndexAccess):
            obj = self.evaluate(node.target.object)
            index = self.evaluate(node.target.index)
            obj[index] = value
        elif isinstance(node.target, MemberAccess):
            obj = self.evaluate(node.target.object)
            if isinstance(obj, dict):
                obj[node.target.property] = value
            else:
                raise RuntimeError("Cannot assign to property")
        
        return value
    
    def visit_Block(self, node: Block) -> None:
        self.execute_block(node.statements, Environment(self.environment))
    
    def visit_IfStatement(self, node: IfStatement) -> Any:
        condition_value = self.evaluate(node.condition)
        
        if self.is_truthy(condition_value):
            return self.execute(node.then_branch)
        elif node.else_branch:
            return self.execute(node.else_branch)
    
    def visit_WhileLoop(self, node: WhileLoop) -> None:
        try:
            while self.is_truthy(self.evaluate(node.condition)):
                try:
                    self.execute(node.body)
                except ContinueException:
                    continue
        except BreakException:
            pass
    
    def visit_ForLoop(self, node: ForLoop) -> None:
        iterable = self.evaluate(node.iterable)
        
        if not hasattr(iterable, '__iter__'):
            raise RuntimeError("Object is not iterable")
        
        try:
            for item in iterable:
                try:
                    # Create new environment for loop variable
                    loop_env = Environment(self.environment)
                    loop_env.define(node.variable, item)
                    
                    previous = self.environment
                    self.environment = loop_env
                    try:
                        self.execute(node.body)
                    finally:
                        self.environment = previous
                        
                except ContinueException:
                    continue
        except BreakException:
            pass
    
    def visit_FunctionDeclaration(self, node: FunctionDeclaration) -> None:
        function = SayiyFunction(node, self.environment)
        self.environment.define(node.name, function)
    
    def visit_ReturnStatement(self, node: ReturnStatement) -> None:
        value = None
        if node.value:
            value = self.evaluate(node.value)
        
        raise ReturnValue(value)
    
    def visit_BreakStatement(self, node: BreakStatement) -> None:
        raise BreakException()
    
    def visit_ContinueStatement(self, node: ContinueStatement) -> None:
        raise ContinueException()
    
    def visit_Program(self, node: Program) -> Any:
        result = None
        for statement in node.statements:
            result = self.execute(statement)
        return result
    
    def is_truthy(self, value: Any) -> bool:
        """Determine if a value is truthy in Sayiy"""
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return len(value) > 0
        if isinstance(value, (list, dict)):
            return len(value) > 0
        return True