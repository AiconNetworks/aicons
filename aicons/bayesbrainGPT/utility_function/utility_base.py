"""
Base classes for utility functions in BayesBrainGPT.

This module defines the core interfaces for utility functions used in Bayesian 
decision making. Utility functions evaluate the "goodness" of an action given a
specific state and are used to compute expected utility by averaging over posterior samples.
"""

import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Optional, Callable


class UtilityFunction(ABC):
    """
    Abstract base class for utility functions used in Bayesian decision making.
    
    Utility functions evaluate the "goodness" of an action given a specific state.
    They are used to compute expected utility by averaging over posterior samples.
    All implementations use TensorFlow for efficient computation and gradient-based optimization.
    """
    
    def __init__(self, name: str, description: str = "", action_space=None):
        """
        Initialize the utility function.
        
        Args:
            name: Name of the utility function
            description: Description of what this utility function measures
            action_space: Optional action space that this utility function will evaluate
        """
        self.name = name
        self.description = description
        
        # Store dimensions if action space is provided
        if action_space is not None:
            self.dimensions = action_space.dimensions if hasattr(action_space, 'dimensions') else None
    
    def set_action_space(self, action_space):
        """Set the action space for this utility function."""
        if action_space is not None:
            self.dimensions = action_space.dimensions if hasattr(action_space, 'dimensions') else None
            return True
        return False
    
    @abstractmethod
    def evaluate_tf(self, action: tf.Tensor, 
                   state_samples: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Evaluate the utility of an action for all state samples using TensorFlow.
        
        Args:
            action: Tensor containing action values
            state_samples: Dictionary mapping state factor names to tensors
                (represents multiple samples from the posterior)
        
        Returns:
            Tensor of utility values, one for each sample
        """
        pass
    
    def evaluate(self, action: Dict[str, Any], state_sample: Dict[str, Any]) -> float:
        """
        Evaluate the utility of an action for a specific state sample.
        This is a wrapper around evaluate_tf for single-sample evaluation.
        
        Args:
            action: Dictionary mapping action dimension names to values
            state_sample: Dictionary mapping state factor names to values
                (represents a single sample from the posterior)
        
        Returns:
            float: Utility value (higher is better)
        """
        # Convert action to tensor
        if hasattr(self, 'dimensions') and self.dimensions is not None:
            action_tensor = tf.constant([action.get(dim.name, 0.0) for dim in self.dimensions])
        else:
            # Try to identify budget values from keys
            budget_values = [v for k, v in action.items() if k.endswith('_budget')]
            if budget_values:
                action_tensor = tf.constant(budget_values)
            else:
                # Fallback to all numeric values in the action
                numeric_values = [v for k, v in action.items() 
                                 if isinstance(v, (int, float))]
                action_tensor = tf.constant(numeric_values) if numeric_values else tf.constant([0.0])
        
        # Convert state sample to tensor format
        state_tensors = {k: tf.constant([v]) for k, v in state_sample.items()}
        
        # Evaluate using TensorFlow
        utility_tensor = self.evaluate_tf(action_tensor, state_tensors)
        
        # Return scalar value
        return float(utility_tensor[0])
    
    def batch_evaluate(self, action: Dict[str, Any], 
                      state_samples: Dict[str, List[Any]]) -> List[float]:
        """
        Evaluate the utility of an action for multiple state samples.
        
        Args:
            action: Dictionary mapping action dimension names to values
            state_samples: Dictionary mapping state factor names to lists of values
                (represents multiple samples from the posterior)
        
        Returns:
            List of utility values, one for each sample
        """
        # Get the number of samples
        if not state_samples:
            return []
        
        first_factor = next(iter(state_samples.values()))
        n_samples = len(first_factor)
        
        # Convert action to tensor
        if hasattr(self, 'dimensions') and self.dimensions is not None:
            action_tensor = tf.constant([action.get(dim.name, 0.0) for dim in self.dimensions])
        else:
            budget_values = [v for k, v in action.items() if k.endswith('_budget')]
            if budget_values:
                action_tensor = tf.constant(budget_values)
            else:
                numeric_values = [v for k, v in action.items() 
                                 if isinstance(v, (int, float))]
                action_tensor = tf.constant(numeric_values) if numeric_values else tf.constant([0.0])
        
        # Convert state samples to tensor format
        state_tensors = {k: tf.constant(v) for k, v in state_samples.items()}
        
        # Evaluate using TensorFlow
        utility_tensor = self.evaluate_tf(action_tensor, state_tensors)
        
        # Return list of values
        return utility_tensor.numpy().tolist()
    
    def expected_utility(self, action: Dict[str, Any], 
                        state_samples: Dict[str, List[Any]]) -> float:
        """
        Compute the expected utility of an action by averaging over posterior samples.
        
        Args:
            action: Dictionary mapping action dimension names to values
            state_samples: Dictionary mapping state factor names to lists of values
                (represents multiple samples from the posterior)
        
        Returns:
            float: Expected utility (higher is better)
        """
        utilities = self.batch_evaluate(action, state_samples)
        
        if not utilities:
            return 0.0
            
        return float(np.mean(utilities))
    
    def evaluate_tf_batch(self, actions: List[Dict[str, Any]],
                         posterior_samples: Dict[str, tf.Tensor] = None) -> tf.Tensor:
        """
        Evaluate multiple actions using TensorFlow.
        
        Args:
            actions: List of action dictionaries
            posterior_samples: Dictionary of posterior samples
            
        Returns:
            Tensor of expected utilities for each action
        """
        # Convert actions to tensors
        action_tensors = []
        for action in actions:
            if hasattr(self, 'dimensions') and self.dimensions is not None:
                action_tensor = tf.constant([action.get(dim.name, 0.0) for dim in self.dimensions])
            else:
                budget_values = [v for k, v in action.items() if k.endswith('_budget')]
                if budget_values:
                    action_tensor = tf.constant(budget_values)
                else:
                    numeric_values = [v for k, v in action.items() 
                                     if isinstance(v, (int, float))]
                    action_tensor = tf.constant(numeric_values) if numeric_values else tf.constant([0.0])
            action_tensors.append(action_tensor)
        
        # Evaluate each action
        utilities = []
        for action_tensor in action_tensors:
            utility = self.evaluate_tf(action_tensor, posterior_samples)
            expected_utility = tf.reduce_mean(utility)
            utilities.append(expected_utility)
        
        return tf.stack(utilities)
    
    def expected_utility_tf(self, action: tf.Tensor,
                           state_samples: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Compute the expected utility of an action by averaging over posterior samples.
        
        Args:
            action: Tensor containing action values
            state_samples: Dictionary mapping state factor names to tensors
                (represents multiple samples from the posterior)
        
        Returns:
            Scalar tensor with the expected utility
        """
        utilities = self.evaluate_tf(action, state_samples)
        return tf.reduce_mean(utilities)
    
    def as_callable(self) -> Callable:
        """
        Return this utility function as a callable function.
        
        Returns:
            Function that takes (action, state_samples) and returns utilities tensor
        """
        return self.evaluate_tf
    
    def find_best_action_gradient(self, action_space, posterior_samples=None, 
                                 learning_rate=0.01, num_steps=100):
        """
        Find the best action using gradient-based optimization.
        
        Args:
            action_space: The action space to search in
            posterior_samples: Posterior samples to evaluate actions with
            learning_rate: Learning rate for optimization
            num_steps: Number of optimization steps
            
        Returns:
            Tuple of (best_action, expected_utility)
        """
        try:
            # Optimize using action_space's TF optimization method
            if hasattr(action_space, 'optimize_action_tf'):
                return action_space.optimize_action_tf(
                    self.evaluate_tf,
                    posterior_samples,
                    num_steps=num_steps,
                    learning_rate=learning_rate
                )
            else:
                print("Action space does not support gradient optimization")
                return None, 0.0
        except Exception as e:
            print(f"Error in gradient optimization: {e}")
            return None, 0.0 