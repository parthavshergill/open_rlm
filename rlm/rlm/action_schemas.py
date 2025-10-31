"""
Action schemas and parser for RLM agent.

This module defines the structured actions that the RLM agent can emit,
and provides a strict parser to extract them from model responses.
"""

from dataclasses import dataclass
from typing import Union
import re


@dataclass
class PyAction:
    """Execute Python code in the REPL environment.
    
    Example: PY(\"\"\"print(context['document'][:100])\"\"\")
    """
    code: str


@dataclass
class FinalAction:
    """Return a final answer directly as a string.
    
    Example: FINAL(\"\"\"The answer is 42\"\"\")
    """
    answer: str


@dataclass
class FinalVarAction:
    """Return the value of a REPL variable as the final answer.
    
    Example: FINAL_VAR(var_name="result")
    """
    var_name: str


# Union type for all possible actions
Action = Union[PyAction, FinalAction, FinalVarAction]


def parse_action(text: str) -> Action | None:
    """Parse text into one of the three action types.
    
    The parser attempts to match actions in this priority order:
    1. FINAL_VAR(var_name="...") - Most specific pattern
    2. PY(\"\"\"...\"\"\") - Triple-quoted string
    3. FINAL(\"\"\"...\"\"\") - Triple-quoted string
    
    Args:
        text: The text to parse (typically model output)
        
    Returns:
        An Action instance if parsing succeeds, None otherwise.
        
    Examples:
        >>> parse_action('PY(\"\"\"x = 1 + 1\"\"\")')
        PyAction(code='x = 1 + 1')
        
        >>> parse_action('FINAL(\"\"\"The answer is 42\"\"\")')
        FinalAction(answer='The answer is 42')
        
        >>> parse_action('FINAL_VAR(var_name="result")')
        FinalVarAction(var_name='result')
    """
    if not text or not isinstance(text, str):
        return None
    
    # Try FINAL_VAR first (most specific pattern)
    final_var_match = re.search(r'FINAL_VAR\(var_name=["\']([^"\']+)["\']\)', text)
    if final_var_match:
        return FinalVarAction(var_name=final_var_match.group(1))
    
    # Try PY with triple quotes
    py_match = re.search(r'PY\("""(.*?)"""\)', text, re.DOTALL)
    if py_match:
        return PyAction(code=py_match.group(1))
    
    # Try FINAL with triple quotes
    final_match = re.search(r'FINAL\("""(.*?)"""\)', text, re.DOTALL)
    if final_match:
        return FinalAction(answer=final_match.group(1))
    
    return None


def format_action(action: Action) -> str:
    """Format an Action object back into its string representation.
    
    This is useful for generating training data or debugging.
    
    Args:
        action: An Action instance
        
    Returns:
        The formatted action string
        
    Examples:
        >>> format_action(PyAction(code="x = 1"))
        'PY(\"\"\"x = 1\"\"\")'
    """
    if isinstance(action, PyAction):
        return f'PY("""{action.code}""")'
    elif isinstance(action, FinalAction):
        return f'FINAL("""{action.answer}""")'
    elif isinstance(action, FinalVarAction):
        return f'FINAL_VAR(var_name="{action.var_name}")'
    else:
        raise ValueError(f"Unknown action type: {type(action)}")
