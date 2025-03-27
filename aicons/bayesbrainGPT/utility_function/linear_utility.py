"""
Linear utility functions for BayesBrainGPT.

This module provides utility functions that compute linear combinations
of action values with specified weights.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Optional, Union, List

from .utility_base import UtilityFunction
from ..decision_making.action_space import ActionSpace


class LinearUtility(UtilityFunction):
    """
    A utility function that computes a weighted sum of action values.
    
    This utility function takes a set of weights and computes the weighted sum
    of the corresponding action values. It supports both named and unnamed action spaces.
    
    Args:
        name: Name of the utility function
        weights: Dictionary mapping action names to weights, or list of weights
        description: Optional description of the utility function
        action_space: Optional action space to connect with the utility function
    """
    
    def __init__(
        self,
        name: str,
        weights: Union[Dict[str, float], List[float]],
        description: Optional[str] = None,
        action_space: Optional[ActionSpace] = None
    ):
        super().__init__(name, description, action_space)
        
        # Store weights
        self.weights = weights
        
        # Convert weights to tensor if it's a dictionary
        if isinstance(weights, dict):
            # Get action names from action space if available
            if action_space is not None and action_space.dimensions is not None:
                # Extract names from ActionDimension objects
                action_names = [dim.name for dim in action_space.dimensions]
                # Create list of weights in same order as action names
                self.weights = [weights.get(name, 0.0) for name in action_names]
            else:
                # If no action space, use dictionary values
                self.weights = list(weights.values())
        
        # Convert to tensor
        self.weights = tf.convert_to_tensor(self.weights, dtype=tf.float32)
    
    def __str__(self) -> str:
        """Show exactly what this utility function computes."""
        if isinstance(self.weights, dict):
            weights_str = ", ".join(f"{k}: {v}" for k, v in self.weights.items())
        else:
            weights_str = ", ".join(str(w) for w in self.weights)
            
        return f"LinearUtility: Σ(weights * values) where weights = [{weights_str}]"
    
    def evaluate_tf(
        self,
        action_values: tf.Tensor,
        state_samples: Optional[tf.Tensor] = None,
        **kwargs
    ) -> tf.Tensor:
        """
        Evaluate the utility function using TensorFlow.
        
        For marketing campaigns, this computes:
        U(Ad) = Expected Revenue - Cost + λ ⋅ Brand Impact
        
        Args:
            action_values: Tensor of action values (budgets)
            state_samples: Optional tensor of state samples (not used in linear utility)
            **kwargs: Additional arguments (not used)
            
        Returns:
            Tensor of utility values
        """
        # For now, we'll use a simple linear model where:
        # Expected Revenue = budget * conversion_rate
        # Cost = budget
        # Brand Impact = budget * brand_factor
        
        # These would come from historical data or LLM inference
        conversion_rate = 0.1  # Example: 10% conversion rate
        brand_factor = 0.05   # Example: 5% brand impact per dollar
        
        # Compute components
        expected_revenue = action_values * conversion_rate
        cost = action_values
        brand_impact = action_values * brand_factor
        
        # Compute final utility
        utility = expected_revenue - cost + brand_impact
        
        # Apply weights to each campaign's utility
        return tf.reduce_sum(utility * self.weights, axis=-1) 