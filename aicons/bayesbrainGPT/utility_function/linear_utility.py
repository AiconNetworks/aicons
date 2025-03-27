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
    
    def evaluate_tf(
        self,
        action_values: tf.Tensor,
        state_samples: Optional[tf.Tensor] = None,
        **kwargs
    ) -> tf.Tensor:
        """
        Evaluate the utility function using TensorFlow.
        
        Args:
            action_values: Tensor of action values
            state_samples: Optional tensor of state samples (not used in linear utility)
            **kwargs: Additional arguments (not used)
            
        Returns:
            Tensor of utility values
        """
        # Compute weighted sum
        return tf.reduce_sum(action_values * self.weights, axis=-1) 