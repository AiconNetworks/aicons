"""
Gemini LLM Module

Implements the Gemini model interface.
"""

import os
from typing import AsyncGenerator, Dict, Any, Optional
import google.generativeai as genai
from dotenv import load_dotenv
from .base import BaseLLM

# Load environment variables
load_dotenv()

class GeminiLLM(BaseLLM):
    """Gemini model implementation."""
    
    MODELS = {
        "gemini-1.5-flash": {
            "context_window": 1_000_000,  # 1M tokens
            "description": "Fast and versatile performance across a diverse variety of tasks"
        },
        "gemini-1.5-pro": {
            "context_window": 2_000_000,  # 2M tokens
            "description": "Complex reasoning tasks requiring more intelligence"
        }
    }
    
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """Initialize Gemini model.
        
        Args:
            model_name: Name of the model to use
        """
        if model_name not in self.MODELS:
            raise ValueError(f"Invalid model name. Must be one of: {list(self.MODELS.keys())}")
        
        config = self.MODELS[model_name]
        super().__init__(
            model_name=model_name,
            context_window=config["context_window"],
            description=config["description"]
        )
        self.model = None
    
    def _initialize(self) -> None:
        """Initialize Gemini client."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using Gemini's token counter.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return self.model.count_tokens(text).total_tokens
    
    async def generate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate text using Gemini API.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (temperature, top_p, etc.)
            
        Returns:
            AsyncGenerator yielding response chunks
        """
        response = self.model.generate_content(prompt, **kwargs)
        yield response.text 