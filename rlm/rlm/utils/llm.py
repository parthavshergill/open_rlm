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
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"Error generating completion: {str(e)}")


class LocalModelClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "local-model", base_url: str = "http://127.0.0.1:1234/v1"):
        """
        Client for local models with OpenAI-compatible API.
        
        Args:
            api_key: API key (optional for local models, defaults to "dummy")
            model: Model name to use
            base_url: Base URL for the local model server
        """
        self.api_key = api_key or "dummy"  # Local models often don't need a real API key
        self.model = model
        self.base_url = base_url
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        # Implement cost tracking logic here if needed.
    
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
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"Error generating completion: {str(e)}")


def get_llm_client(model: str, api_key: Optional[str] = None):
    """
    Factory function to get the appropriate LLM client based on model name.
    
    Args:
        model: Model identifier (e.g., 'gpt-4', 'gemini-1.5-pro', 'local-model')
        api_key: Optional API key (will use env vars if not provided)
    
    Returns:
        LLM client instance (OpenAIClient, GeminiClient, or LocalModelClient)
    """
    model_lower = model.lower()
    
    # Determine which client to use based on model name
    if 'gemini' in model_lower:
        from rlm.utils.gemini_client import GeminiClient
        return GeminiClient(api_key=api_key, model=model)
    elif 'local' in model_lower:
        return LocalModelClient(api_key=api_key, model=model)
    else:
        # Default to OpenAI for gpt-* and other models
        return OpenAIClient(api_key=api_key, model=model)