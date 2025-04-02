"""
Umbrella decision utility function for BayesBrainGPT.

This module provides a specialized utility function for the umbrella decision problem,
where the cost of not taking an umbrella depends on whether it rains.
"""

import tensorflow as tf
from typing import Dict, Any
from .utility_functions import CostBenefitUtility

class UmbrellaUtility(CostBenefitUtility):
    """
    Utility function for umbrella decision making.
    For taking umbrella: fixed cost
    For not taking umbrella: rain_cost if it rains, 0 if it doesn't
    """
    
    def __init__(self, name: str, cost: float = 1.0, rain_cost: float = 5.0,
                 action_name: str = "umbrella", description: str = "", action_space=None):
        """
        Initialize umbrella utility function.
        
        Args:
            name: Name of the utility function
            cost: Cost of taking the umbrella (default: 1.0)
            rain_cost: Cost of getting wet in rain (default: 5.0)
            action_name: Name of the action dimension (default: "umbrella")
            description: Description of the utility function
            action_space: Optional action space
        """
        print(f"\nDEBUG - Initializing UmbrellaUtility:")
        print(f"Take umbrella cost: {cost}")
        print(f"Rain cost: {rain_cost}")
        print(f"Action name: {action_name}")
        
        # Set up costs for the umbrella decision
        costs = {action_name: cost}
        benefits = {action_name: 0}  # No benefits, only costs
        
        # Initialize parent class
        super().__init__(
            name=name,
            costs=costs,
            benefits=benefits,
            description=description,
            action_space=action_space
        )
        
        # Store parameters
        self.rain_cost = rain_cost
        self.action_name = action_name
        print(f"Initialized with costs={costs}, rain_cost={rain_cost}")
    
    def evaluate_tf(self, action: tf.Tensor, state_samples: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Evaluate utility with rain probability weighting.
        
        Args:
            action: Tensor of action values (0 or 1 for umbrella)
            state_samples: Dictionary with 'rain' probability
            
        Returns:
            Tensor of utility values
        """
        print(f"\nDEBUG - evaluate_tf:")
        print(f"Action tensor: {action}")
        print(f"State samples: {state_samples}")
        
        # Get rain probability from state samples
        rain_prob = state_samples.get('rain', tf.constant(0.0))
        print(f"Rain probability: {rain_prob}")
        
        # For umbrella decision:
        # - If take umbrella (action=1): fixed cost
        # - If no umbrella (action=0): rain_cost if it rains, 0 if it doesn't
        if hasattr(self, 'costs_tensor') and hasattr(self, 'benefits_tensor'):
            print(f"Using tensor-based computation")
            print(f"Costs tensor: {self.costs_tensor}")
            
            # For taking umbrella: fixed cost
            # For not taking umbrella: rain_cost * rain_prob
            result = tf.reduce_sum(action * self.costs_tensor + 
                                 (1 - action) * (-self.rain_cost * rain_prob))
            print(f"Tensor computation result: {result}")
            return result
        
        # For dictionary-based computation
        if hasattr(self, 'dimensions') and self.dimensions is not None:
            action_dict = {dim.name: float(action[i]) for i, dim in enumerate(self.dimensions)}
        else:
            action_dict = {self.action_name: float(action[0])}
        print(f"Action dict: {action_dict}")
        
        # Calculate utility based on action
        if action_dict[self.action_name] == 1:
            # Taking umbrella: fixed cost
            result = tf.constant(-self.costs[self.action_name])
            print(f"Taking umbrella - fixed cost: {result}")
        else:
            # Not taking umbrella: rain_cost if it rains, 0 if it doesn't
            result = tf.constant(-self.rain_cost * float(rain_prob[0]))
            print(f"Not taking umbrella - rain cost: {result}")
        
        return result
    
    def evaluate(self, action: Dict[str, Any], state_sample: Dict[str, Any]) -> float:
        """
        Evaluate utility for a single state sample.
        This ensures consistent evaluation between batch and single-sample cases.
        
        Args:
            action: Dictionary with action values
            state_sample: Dictionary with state values
            
        Returns:
            float: Utility value
        """
        print(f"\nDEBUG - evaluate:")
        print(f"Action: {action}")
        print(f"State sample: {state_sample}")
        
        # Convert action to tensor
        if hasattr(self, 'dimensions') and self.dimensions is not None:
            action_tensor = tf.constant([action.get(dim.name, 0.0) for dim in self.dimensions])
        else:
            action_tensor = tf.constant([action.get(self.action_name, 0.0)])
        print(f"Action tensor: {action_tensor}")
        
        # Convert state sample to tensor format
        state_tensors = {k: tf.constant([v]) for k, v in state_sample.items()}
        print(f"State tensors: {state_tensors}")
        
        # Evaluate using TensorFlow
        utility_tensor = self.evaluate_tf(action_tensor, state_tensors)
        print(f"Final utility tensor: {utility_tensor}")
        
        result = float(utility_tensor)
        print(f"Final float result: {result}")
        return result 