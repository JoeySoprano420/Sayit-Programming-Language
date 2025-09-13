"""
Sayiy Programming Language Standard Library
Built-in functions and utilities
"""

import sys
import time
import math
import json
from typing import Any, Dict, List, Callable, Optional

def sayiy_print(*args) -> None:
    """Print values to stdout"""
    output = ' '.join(str(arg) for arg in args)
    print(output)

def sayiy_input(prompt: str = "") -> str:
    """Get input from user"""
    return input(prompt)

def sayiy_len(obj: Any) -> int:
    """Get length of object"""
    if hasattr(obj, '__len__'):
        return len(obj)
    raise TypeError(f"Object of type {type(obj).__name__} has no len()")

def sayiy_type(obj: Any) -> str:
    """Get type of object"""
    if obj is None:
        return "null"
    elif isinstance(obj, bool):
        return "boolean"
    elif isinstance(obj, int):
        return "integer"
    elif isinstance(obj, float):
        return "float"
    elif isinstance(obj, str):
        return "string"
    elif isinstance(obj, list):
        return "array"
    elif isinstance(obj, dict):
        return "object"
    elif callable(obj):
        return "function"
    else:
        return "unknown"

def sayiy_str(obj: Any) -> str:
    """Convert object to string"""
    if obj is None:
        return "null"
    elif isinstance(obj, bool):
        return "true" if obj else "false"
    elif isinstance(obj, (list, dict)):
        return json.dumps(obj, default=str)
    else:
        return str(obj)

def sayiy_int(obj: Any) -> int:
    """Convert object to integer"""
    if isinstance(obj, str):
        try:
            return int(float(obj))  # Handle "3.14" -> 3
        except ValueError:
            raise ValueError(f"Cannot convert '{obj}' to integer")
    elif isinstance(obj, bool):
        return 1 if obj else 0
    elif isinstance(obj, (int, float)):
        return int(obj)
    else:
        raise TypeError(f"Cannot convert {type(obj).__name__} to integer")

def sayiy_float(obj: Any) -> float:
    """Convert object to float"""
    if isinstance(obj, str):
        try:
            return float(obj)
        except ValueError:
            raise ValueError(f"Cannot convert '{obj}' to float")
    elif isinstance(obj, bool):
        return 1.0 if obj else 0.0
    elif isinstance(obj, (int, float)):
        return float(obj)
    else:
        raise TypeError(f"Cannot convert {type(obj).__name__} to float")

def sayiy_bool(obj: Any) -> bool:
    """Convert object to boolean"""
    if obj is None:
        return False
    elif isinstance(obj, bool):
        return obj
    elif isinstance(obj, (int, float)):
        return obj != 0
    elif isinstance(obj, str):
        return len(obj) > 0
    elif isinstance(obj, (list, dict)):
        return len(obj) > 0
    else:
        return True

# Array functions
def sayiy_array(*args) -> List[Any]:
    """Create array from arguments"""
    return list(args)

def sayiy_push(arr: List[Any], *items) -> int:
    """Push items to array and return new length"""
    if not isinstance(arr, list):
        raise TypeError("First argument must be an array")
    arr.extend(items)
    return len(arr)

def sayiy_pop(arr: List[Any]) -> Any:
    """Remove and return last item from array"""
    if not isinstance(arr, list):
        raise TypeError("Argument must be an array")
    return arr.pop() if arr else None

def sayiy_shift(arr: List[Any]) -> Any:
    """Remove and return first item from array"""
    if not isinstance(arr, list):
        raise TypeError("Argument must be an array")
    return arr.pop(0) if arr else None

def sayiy_unshift(arr: List[Any], *items) -> int:
    """Add items to beginning of array and return new length"""
    if not isinstance(arr, list):
        raise TypeError("First argument must be an array")
    for i, item in enumerate(items):
        arr.insert(i, item)
    return len(arr)

def sayiy_slice(arr: List[Any], start: int = 0, end: Optional[int] = None) -> List[Any]:
    """Return slice of array"""
    if not isinstance(arr, list):
        raise TypeError("First argument must be an array")
    return arr[start:end]

def sayiy_join(arr: List[Any], separator: str = ",") -> str:
    """Join array elements with separator"""
    if not isinstance(arr, list):
        raise TypeError("First argument must be an array")
    return separator.join(str(item) for item in arr)

def sayiy_sort(arr: List[Any]) -> List[Any]:
    """Sort array in place"""
    if not isinstance(arr, list):
        raise TypeError("Argument must be an array")
    arr.sort(key=lambda x: (type(x).__name__, x))
    return arr

def sayiy_reverse(arr: List[Any]) -> List[Any]:
    """Reverse array in place"""
    if not isinstance(arr, list):
        raise TypeError("Argument must be an array")
    arr.reverse()
    return arr

def sayiy_map(arr: List[Any], func: Callable) -> List[Any]:
    """Apply function to each element and return new array"""
    if not isinstance(arr, list):
        raise TypeError("First argument must be an array")
    if not callable(func):
        raise TypeError("Second argument must be a function")
    
    result = []
    for item in arr:
        result.append(func(item))
    return result

def sayiy_filter(arr: List[Any], func: Callable) -> List[Any]:
    """Filter array elements using function"""
    if not isinstance(arr, list):
        raise TypeError("First argument must be an array")
    if not callable(func):
        raise TypeError("Second argument must be a function")
    
    result = []
    for item in arr:
        if func(item):
            result.append(item)
    return result

def sayiy_reduce(arr: List[Any], func: Callable, initial: Any = None) -> Any:
    """Reduce array to single value using function"""
    if not isinstance(arr, list):
        raise TypeError("First argument must be an array")
    if not callable(func):
        raise TypeError("Second argument must be a function")
    
    if not arr and initial is None:
        raise TypeError("Reduce of empty array with no initial value")
    
    iterator = iter(arr)
    if initial is None:
        result = next(iterator)
    else:
        result = initial
    
    for item in iterator:
        result = func(result, item)
    
    return result

# Object functions
def sayiy_keys(obj: Dict[str, Any]) -> List[str]:
    """Get object keys"""
    if not isinstance(obj, dict):
        raise TypeError("Argument must be an object")
    return list(obj.keys())

def sayiy_values(obj: Dict[str, Any]) -> List[Any]:
    """Get object values"""
    if not isinstance(obj, dict):
        raise TypeError("Argument must be an object")
    return list(obj.values())

def sayiy_entries(obj: Dict[str, Any]) -> List[List[Any]]:
    """Get object entries as [key, value] pairs"""
    if not isinstance(obj, dict):
        raise TypeError("Argument must be an object")
    return [[k, v] for k, v in obj.items()]

# String functions
def sayiy_split(s: str, separator: str = " ") -> List[str]:
    """Split string by separator"""
    if not isinstance(s, str):
        raise TypeError("First argument must be a string")
    return s.split(separator)

def sayiy_upper(s: str) -> str:
    """Convert string to uppercase"""
    if not isinstance(s, str):
        raise TypeError("Argument must be a string")
    return s.upper()

def sayiy_lower(s: str) -> str:
    """Convert string to lowercase"""
    if not isinstance(s, str):
        raise TypeError("Argument must be a string")
    return s.lower()

def sayiy_trim(s: str) -> str:
    """Remove whitespace from string ends"""
    if not isinstance(s, str):
        raise TypeError("Argument must be a string")
    return s.strip()

def sayiy_replace(s: str, old: str, new: str) -> str:
    """Replace occurrences in string"""
    if not isinstance(s, str):
        raise TypeError("First argument must be a string")
    return s.replace(old, new)

def sayiy_contains(s: str, substring: str) -> bool:
    """Check if string contains substring"""
    if not isinstance(s, str):
        raise TypeError("First argument must be a string")
    return substring in s

def sayiy_starts_with(s: str, prefix: str) -> bool:
    """Check if string starts with prefix"""
    if not isinstance(s, str):
        raise TypeError("First argument must be a string")
    return s.startswith(prefix)

def sayiy_ends_with(s: str, suffix: str) -> bool:
    """Check if string ends with suffix"""
    if not isinstance(s, str):
        raise TypeError("First argument must be a string")
    return s.endswith(suffix)

# Math functions
def sayiy_abs(x: float) -> float:
    """Absolute value"""
    return abs(x)

def sayiy_min(*args) -> Any:
    """Minimum value"""
    if len(args) == 1 and isinstance(args[0], list):
        return min(args[0])
    return min(args)

def sayiy_max(*args) -> Any:
    """Maximum value"""
    if len(args) == 1 and isinstance(args[0], list):
        return max(args[0])
    return max(args)

def sayiy_sum(arr: List[Any]) -> Any:
    """Sum of array elements"""
    if not isinstance(arr, list):
        raise TypeError("Argument must be an array")
    return sum(arr)

def sayiy_floor(x: float) -> int:
    """Floor of number"""
    return math.floor(x)

def sayiy_ceil(x: float) -> int:
    """Ceiling of number"""
    return math.ceil(x)

def sayiy_round(x: float, digits: int = 0) -> float:
    """Round number to digits"""
    return round(x, digits)

def sayiy_sqrt(x: float) -> float:
    """Square root"""
    return math.sqrt(x)

def sayiy_pow(base: float, exponent: float) -> float:
    """Power function"""
    return math.pow(base, exponent)

# Utility functions
def sayiy_time() -> float:
    """Get current time as timestamp"""
    return time.time()

def sayiy_sleep(seconds: float) -> None:
    """Sleep for specified seconds"""
    time.sleep(seconds)

def sayiy_range(start: int, stop: Optional[int] = None, step: int = 1) -> List[int]:
    """Create range of numbers"""
    if stop is None:
        stop = start
        start = 0
    return list(range(start, stop, step))

def get_builtin_functions() -> Dict[str, Callable]:
    """Return dictionary of all built-in functions"""
    return {
        # I/O functions
        'print': sayiy_print,
        'input': sayiy_input,
        
        # Type functions
        'len': sayiy_len,
        'type': sayiy_type,
        'str': sayiy_str,
        'int': sayiy_int,
        'float': sayiy_float,
        'bool': sayiy_bool,
        
        # Array functions
        'array': sayiy_array,
        'push': sayiy_push,
        'pop': sayiy_pop,
        'shift': sayiy_shift,
        'unshift': sayiy_unshift,
        'slice': sayiy_slice,
        'join': sayiy_join,
        'sort': sayiy_sort,
        'reverse': sayiy_reverse,
        'map': sayiy_map,
        'filter': sayiy_filter,
        'reduce': sayiy_reduce,
        
        # Object functions
        'keys': sayiy_keys,
        'values': sayiy_values,
        'entries': sayiy_entries,
        
        # String functions
        'split': sayiy_split,
        'upper': sayiy_upper,
        'lower': sayiy_lower,
        'trim': sayiy_trim,
        'replace': sayiy_replace,
        'contains': sayiy_contains,
        'starts_with': sayiy_starts_with,
        'ends_with': sayiy_ends_with,
        
        # Math functions
        'abs': sayiy_abs,
        'min': sayiy_min,
        'max': sayiy_max,
        'sum': sayiy_sum,
        'floor': sayiy_floor,
        'ceil': sayiy_ceil,
        'round': sayiy_round,
        'sqrt': sayiy_sqrt,
        'pow': sayiy_pow,
        
        # Utility functions
        'time': sayiy_time,
        'sleep': sayiy_sleep,
        'range': sayiy_range,
    }