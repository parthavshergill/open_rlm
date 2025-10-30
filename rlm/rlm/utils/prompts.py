"""
Example prompt templates for the RLM REPL Client.
"""

from typing import Dict

DEFAULT_QUERY = "Please read through the context and answer any queries or respond to any instructions contained within it."

# System prompt for the REPL environment
REPL_SYSTEM_PROMPT = """You are tasked with answering a query using associated context available in a REPL environment.

The REPL environment provides:
1. A `context` variable containing information relevant to your query
   - The context is typically a dictionary with the structure: {'document': 'text...', 'metadata': {...}}
   - Access the document text using context['document']
2. A `llm_query(prompt)` function to query sub-LLMs (can handle ~100K characters)

When you have completed your task, provide a final answer using:
- FINAL(your answer here) to provide the answer directly, or
- FINAL_VAR(variable_name) to return a variable from the REPL environment
"""

def build_system_prompt() -> list[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": REPL_SYSTEM_PROMPT
        },
    ]


# Prompt at every step to query root LM to make a decision
USER_PROMPT = """Continue working on the query: \"{query}\"\n\nUse the REPL environment and sub-LLM queries as needed. Your next action:""" 

FIRST_ITERATION_PROMPT = """The context is available in the REPL environment. Start by inspecting the context structure, then work on answering: \"{query}\"\n\nYour next action:"""

def next_action_prompt(query: str, iteration: int = 0, final_answer: bool = False) -> Dict[str, str]:
    if final_answer:
        return {"role": "user", "content": "Based on all the information you have, provide a final answer to the user's query."}
    if iteration == 0:
        return {"role": "user", "content": FIRST_ITERATION_PROMPT.format(query=query)}
    else:
        return {"role": "user", "content": "The history before is your previous interactions with the REPL environment. " + USER_PROMPT.format(query=query)}
