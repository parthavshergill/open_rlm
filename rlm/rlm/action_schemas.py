"""Action schemas for RLM agent structured outputs.

This module defines the four action types that the RLM agent can emit:
1. PY("""...""") - Execute Python code in the REPL
2. CALL_SUBMODEL(query="...", context_var="...") - Query a sub-model
3. FINAL("""...""") - Return final answer directly
4. FINAL_VAR(var_name="...") - Return value of a REPL variable
"""

from dataclasses import dataclass
from typing import Union
import re


@dataclass
class PyAction:
    """Execute Python code in the REPL environment."""
    code: str


@dataclass
class CallSubmodelAction:
    """Call a sub-model with a query and context variable."""
    query: str
    context_var: str


@dataclass
class FinalAction:
    """Return a final answer directly as a string."""
    answer: str


@dataclass
class FinalVarAction:
    """Return the value of a REPL variable as the final answer."""
    var_name: str


Action = Union[PyAction, CallSubmodelAction, FinalAction, FinalVarAction]


def parse_action(text: str) -> Action | None:
    """Parse text into one of the four action types.
    
    Args:
        text: The text output from the model to parse
        
    Returns:
        An Action object if a valid action is found, None otherwise
        
    Examples:
        >>> parse_action('PY("""print("hello")""")')
        PyAction(code='print("hello")')
        
        >>> parse_action('CALL_SUBMODEL(query="What is X?", context_var="doc")')
        CallSubmodelAction(query='What is X?', context_var='doc')
        
        >>> parse_action('FINAL("""42""")')
        FinalAction(answer='42')
        
        >>> parse_action('FINAL_VAR(var_name="result")')
        FinalVarAction(var_name='result')
    """
    # Try to match PY("""...""")
    py_match = re.search(r'PY\("""(.*?)"""\)', text, re.DOTALL)
    if py_match:
        return PyAction(code=py_match.group(1))
    
    # Try to match CALL_SUBMODEL(query="...", context_var="...")
    # Handle both """ and " quotes for query
    submodel_match = re.search(
        r'CALL_SUBMODEL\(query="""(.*?)""",\s*context_var="(.*?)"\)',
        text, re.DOTALL
    )
    if submodel_match:
        return CallSubmodelAction(
            query=submodel_match.group(1),
            context_var=submodel_match.group(2)
        )
    
    # Try alternate format with single quotes for query
    submodel_match = re.search(
        r'CALL_SUBMODEL\(query="(.*?)",\s*context_var="(.*?)"\)',
        text, re.DOTALL
    )
    if submodel_match:
        return CallSubmodelAction(
            query=submodel_match.group(1),
            context_var=submodel_match.group(2)
        )
    
    # Try to match FINAL_VAR(var_name="...")
    final_var_match = re.search(r'FINAL_VAR\(var_name="(.*?)"\)', text)
    if final_var_match:
        return FinalVarAction(var_name=final_var_match.group(1))
    
    # Try to match FINAL("""...""")
    final_match = re.search(r'FINAL\("""(.*?)"""\)', text, re.DOTALL)
    if final_match:
        return FinalAction(answer=final_match.group(1))
    
    return None
