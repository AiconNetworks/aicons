"""
LLM Module

Exposes Language Model implementations.
"""

from .base import BaseLLM
from .deepseek import DeepSeekLLM
from .gemini import GeminiLLM

__all__ = ['BaseLLM', 'DeepSeekLLM', 'GeminiLLM']

# Factory function to create LLM instances
def create_llm(model_name: str) -> BaseLLM:
    """Create an LLM instance based on model name.
    
    Args:
        model_name: Name of the model to create
        
    Returns:
        An instance of BaseLLM
        
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name == "deepseek-r1:7b":
        return DeepSeekLLM()
    elif model_name.startswith("gemini"):
        return GeminiLLM(model_name)
    else:
        raise ValueError(f"Invalid model name. Must be one of: ['deepseek-r1:7b']") 