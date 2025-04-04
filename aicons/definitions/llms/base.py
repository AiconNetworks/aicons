"""
Base LLM Module

Defines the base interface for Language Models in the AIcon system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from collections.abc import AsyncGenerator

class BaseLLM(ABC):
    """Base class for all Language Models."""
    
    def __init__(self, model_name: str, context_window: int):
        self.model_name = model_name
        self.context_window = context_window
        self._initialize()
    
    @abstractmethod
    def _initialize(self) -> None:
        """Initialize model-specific components."""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass
    
    @abstractmethod
    async def generate(self, prompt: str) -> AsyncGenerator[str, None]:
        """Generate text from prompt."""
        pass 