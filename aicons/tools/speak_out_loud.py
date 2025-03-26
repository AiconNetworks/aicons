"""
Speak Out Loud Tool

This tool allows an AIcon to speak statements by printing them to the console.
"""

from typing import Optional

class SpeakOutLoudTool:
    """Tool for speaking statements by printing them to the console."""
    
    def __init__(self):
        """Initialize the speak out loud tool."""
        self.name = "speak_out_loud"
        self.description = "Speak a statement by printing it to the console"
    
    def execute(self, statement: str, context: Optional[dict] = None) -> bool:
        """
        Execute the tool by printing the statement.
        
        Args:
            statement: The statement to speak
            context: Optional context for the statement
            
        Returns:
            True if the statement was spoken successfully
        """
        try:
            # Format the statement with context if provided
            formatted_statement = statement
            if context:
                formatted_statement = f"{statement} (Context: {context})"
            
            # Print the statement
            print(f"\nStatement: {formatted_statement}")
            return True
            
        except Exception as e:
            print(f"Error speaking statement: {e}")
            return False 