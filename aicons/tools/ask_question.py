"""
Ask Question Tool

This tool allows an AIcon to ask questions by printing them to the console.
"""

from typing import Optional

class AskQuestionTool:
    """Tool for asking questions by printing them to the console."""
    
    def __init__(self):
        """Initialize the ask question tool."""
        self.name = "ask_question"
        self.description = "Ask a question by printing it to the console"
    
    def execute(self, question: str, context: Optional[dict] = None) -> bool:
        """
        Execute the tool by printing the question.
        
        Args:
            question: The question to ask
            context: Optional context for the question
            
        Returns:
            True if the question was asked successfully
        """
        try:
            # Format the question with context if provided
            formatted_question = question
            if context:
                formatted_question = f"{question} (Context: {context})"
            
            # Print the question
            print(f"\nQuestion: {formatted_question}")
            return True
            
        except Exception as e:
            print(f"Error asking question: {e}")
            return False 