"""
DeepSeek LLM Module

Implements the DeepSeek R1 model interface.
"""

import json
import aiohttp
from typing import AsyncGenerator
from .base import BaseLLM

class DeepSeekLLM(BaseLLM):
    """DeepSeek R1 model implementation."""
    
    def __init__(self):
        """Initialize DeepSeek model."""
        super().__init__(
            model_name="deepseek-r1:7b",
            context_window=8_192  # 8K tokens
        )
        self.api_url = "http://localhost:11434/api/generate"
    
    def _initialize(self) -> None:
        """No special initialization needed for local DeepSeek."""
        pass
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count for DeepSeek.
        
        Uses a conservative estimate of 1 token per 4 characters.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated number of tokens
        """
        return len(text) // 4
    
    async def generate(self, prompt: str) -> AsyncGenerator[str, None]:
        """Generate text using DeepSeek API.
        
        Args:
            prompt: Input prompt
            
        Returns:
            AsyncGenerator yielding response chunks
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                json={"prompt": prompt, "model": self.model_name}
            ) as response:
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line)
                            if data.get("response"):
                                yield data["response"]
                        except json.JSONDecodeError:
                            continue 