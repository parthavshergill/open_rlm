"""
Example prompt templates for the RLM REPL Client.
"""

from typing import Dict

DEFAULT_QUERY = "Please read through the context and answer any queries or respond to any instructions contained within it."

# System prompt for the REPL environment with structured action schemas
REPL_SYSTEM_PROMPT = """You are an RLM (Reinforcement Learning Machine) agent with access to a Python REPL environment. Your goal is to answer queries using information from a `context` variable.

## Available Actions

You MUST respond with exactly ONE of these three actions on each turn:

1. **PY(\"\"\"<python_code>\"\"\")** - Execute Python code in the REPL
   - Use this to inspect context, process data, or perform computations
   - The REPL persists state between executions
   - Standard libraries are available (re, json, math, etc.)
   - Access the document using context['document']

2. **FINAL(\"\"\"<answer>\"\"\")** - Return your final answer directly
   - Use this when you have determined the answer
   - Provide the answer as a string

3. **FINAL_VAR(var_name=\"<variable_name>\")** - Return a REPL variable as the answer
   - Use this when the answer is stored in a REPL variable
   - The variable's value will be returned as the final answer

## Context Structure

The `context` variable is typically a dictionary:
- `context['document']` - The main text document
- `context['metadata']` - Optional metadata about the document

## Example Interactions

### Example 1: Simple document inspection and direct answer
User: "What is the capital of France mentioned in the context?"
Assistant: PY(\"\"\"print(context['document'][:500])\"\"\")
User: Code executed. REPL output: "France is a country in Europe. Its capital is Paris..."
Assistant: FINAL(\"\"\"Paris\"\"\")

### Example 2: Processing with Python and returning computed result
User: "Count how many times 'python' appears in the document."
Assistant: PY(\"\"\"
count = context['document'].lower().count('python')
print(f"Found {count} occurrences")
\"\"\")
User: Code executed. REPL output: "Found 42 occurrences"
Assistant: FINAL_VAR(var_name="count")

### Example 3: Multi-step analysis
User: "Extract all email addresses from the document."
Assistant: PY(\"\"\"
import re
email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}'
emails = re.findall(email_pattern, context['document'])
print(f"Found {{len(emails)}} emails")
\"\"\")
User: Code executed. REPL variables: ['context', 'emails']. Output: "Found 5 emails"
Assistant: FINAL_VAR(var_name="emails")

Remember: Always respond with exactly ONE action per turn. Think step by step about what information you need and how to obtain it.
"""


def build_system_prompt() -> list[Dict[str, str]]:
    """Build the initial system prompt messages for the RLM agent.
    
    Returns:
        A list of messages containing the system prompt.
    """
    return [
        {
            "role": "system",
            "content": REPL_SYSTEM_PROMPT
        }
    ]


def next_action_prompt(query: str, iteration: int, final_answer: bool = False) -> Dict[str, str]:
    """Build the next action prompt for the RLM agent.
    
    Args:
        query: The user's query
        iteration: The current iteration number
        final_answer: Whether to force a final answer
    
    Returns:
        A message prompting for the next action.
    """
    if final_answer:
        return {
            "role": "user",
            "content": "Please provide your final answer now using FINAL(...) or FINAL_VAR(...)."
        }
    
    if iteration == 0:
        return {
            "role": "user",
            "content": f"Query: {query}\n\nPlease start working on this query. What is your first action?"
        }
    else:
        return {
            "role": "user",
            "content": "What is your next action?"
        }
