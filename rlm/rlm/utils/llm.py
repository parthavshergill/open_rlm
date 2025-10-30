"""
OpenAI Client wrapper specifically for GPT-5 models.
"""

import os
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5-mini-2025-08-07"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.model = model
        self.client = OpenAI(api_key=self.api_key)

        # Implement cost tracking logic here.
    
    def completion(
        self,
        messages: list[dict[str, str]] | str,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            elif isinstance(messages, dict):
                messages = [messages]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"Error generating completion: {str(e)}")


def get_llm_client(model: str, api_key: Optional[str] = None):
    """
    Factory function to get the appropriate LLM client based on model name.
    
    Args:
        model: Model identifier (e.g., 'gpt-4', 'gemini-1.5-pro')
        api_key: Optional API key (will use env vars if not provided)
    
    Returns:
        LLM client instance (OpenAIClient or GeminiClient)
    """
    model_lower = model.lower()
    
    # Determine which client to use based on model name
    if 'gemini' in model_lower:
        from rlm.utils.gemini_client import GeminiClient
        return GeminiClient(api_key=api_key, model=model)
    else:
        # Default to OpenAI for gpt-* and other models
        return OpenAIClient(api_key=api_key, model=model)