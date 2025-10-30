"""
Gemini Client wrapper for Google's Gemini models.
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class GeminiClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-pro"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        
        self.model = model
        
        # Import google.generativeai
        try:
            import google.generativeai as genai
            self.genai = genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)
        except ImportError:
            raise ImportError(
                "google-generativeai package is required for Gemini support. "
                "Install it with: pip install google-generativeai"
            )
    
    def completion(
        self,
        messages: list[dict[str, str]] | str,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        try:
            # Convert messages to Gemini format
            if isinstance(messages, str):
                prompt = messages
            elif isinstance(messages, dict):
                prompt = messages.get("content", str(messages))
            elif isinstance(messages, list):
                # Convert OpenAI-style messages to Gemini format
                prompt_parts = []
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    
                    if role == "system":
                        prompt_parts.append(f"System: {content}")
                    elif role == "user":
                        prompt_parts.append(f"User: {content}")
                    elif role == "assistant":
                        prompt_parts.append(f"Assistant: {content}")
                    else:
                        prompt_parts.append(content)
                
                prompt = "\n\n".join(prompt_parts)
            else:
                prompt = str(messages)
            
            # Configure generation settings
            generation_config = {}
            if max_tokens:
                generation_config["max_output_tokens"] = max_tokens
            
            # Generate response
            response = self.client.generate_content(
                prompt,
                generation_config=generation_config if generation_config else None
            )
            
            return response.text

        except Exception as e:
            raise RuntimeError(f"Error generating Gemini completion: {str(e)}")

