"""
Implementation of common utility function types for Bayesian decision making.

This module provides concrete implementations of utility functions based on common patterns:
1. Cost/Benefit analysis
2. Monetary utility (revenue - cost)
3. Multi-attribute utility with weighted components
4. Risk-adjusted utility functions
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Union, Optional
from .utility_base import UtilityFunction

class CostBenefitUtility(UtilityFunction):
    """
    Utility function based on cost/benefit analysis.
    Evaluates actions by summing their costs and benefits.
    """
    
    def __init__(self, name: str, costs: Dict[str, float], benefits: Dict[str, float], 
                 description: str = "", action_space=None):
        """
        Initialize cost/benefit utility function.
        
        Args:
            name: Name of the utility function
            costs: Dictionary mapping action names to their costs
            benefits: Dictionary mapping action names to their benefits
            description: Description of the utility function
            action_space: Optional action space
        """
        super().__init__(name, description, action_space)
        
        # Store costs and benefits
        self.costs = costs
        self.benefits = benefits
        
        # If action space is provided, ensure costs and benefits match action dimensions
        if action_space is not None and action_space.dimensions is not None:
            # Get action names from action space dimensions
            action_names = [dim.name for dim in action_space.dimensions]
            
            # Create ordered lists of costs and benefits matching action space
            self.costs_list = [costs.get(name, 0.0) for name in action_names]
            self.benefits_list = [benefits.get(name, 0.0) for name in action_names]
            
            # Convert to tensors
            self.costs_tensor = tf.convert_to_tensor(self.costs_list, dtype=tf.float32)
            self.benefits_tensor = tf.convert_to_tensor(self.benefits_list, dtype=tf.float32)
    
    def __str__(self) -> str:
        return f"Cost/Benefit Utility: {self.name}\nCosts: {self.costs}\nBenefits: {self.benefits}"
    
    def evaluate_tf(self, action: tf.Tensor, state_samples: Dict[str, tf.Tensor]) -> tf.Tensor:
        # If we have action space dimensions, use tensor operations
        if hasattr(self, 'costs_tensor') and hasattr(self, 'benefits_tensor'):
            # Compute benefit - cost using tensor operations
            return tf.reduce_sum(action * (self.benefits_tensor - self.costs_tensor))
        else:
            # Fallback to dictionary-based computation
            if hasattr(self, 'dimensions') and self.dimensions is not None:
                action_dict = {dim.name: float(action[i]) for i, dim in enumerate(self.dimensions)}
            else:
                # If no dimensions set, assume action is a single value
                action_dict = {'umbrella': float(action[0])}
            
            # Get costs and benefits for this action
            cost = sum(self.costs.get(k, 0.0) for k in action_dict.keys())
            benefit = sum(self.benefits.get(k, 0.0) for k in action_dict.keys())
            
            # Return benefit - cost (higher is better)
            return tf.constant(benefit - cost)

class MonetaryUtility(UtilityFunction):
    """
    Utility function for monetary decisions.
    Computes revenue - cost for each action.
    """
    
    def __init__(self, name: str, revenue_factors: List[str], cost_factors: List[str],
                 description: str = "", action_space=None):
        """
        Initialize monetary utility function.
        
        Args:
            name: Name of the utility function
            revenue_factors: List of state factors that contribute to revenue
            cost_factors: List of state factors that contribute to costs
            description: Description of the utility function
            action_space: Optional action space
        """
        super().__init__(name, description, action_space)
        self.revenue_factors = revenue_factors
        self.cost_factors = cost_factors
    
    def __str__(self) -> str:
        return f"Monetary Utility: {self.name}\nRevenue factors: {self.revenue_factors}\nCost factors: {self.cost_factors}"
    
    def evaluate_tf(self, action: tf.Tensor, state_samples: Dict[str, tf.Tensor]) -> tf.Tensor:
        # Compute revenue from relevant factors
        revenue = tf.zeros_like(next(iter(state_samples.values())))
        for factor in self.revenue_factors:
            if factor in state_samples:
                revenue += state_samples[factor]
        
        # Compute costs from relevant factors
        cost = tf.zeros_like(next(iter(state_samples.values())))
        for factor in self.cost_factors:
            if factor in state_samples:
                cost += state_samples[factor]
        
        # Return revenue - cost
        return revenue - cost

class MultiAttributeUtility(UtilityFunction):
    """
    Multi-attribute utility function that combines multiple objectives with weights.
    """
    
    def __init__(self, name: str, attributes: Dict[str, Dict[str, Any]],
                 description: str = "", action_space=None):
        """
        Initialize multi-attribute utility function.
        
        Args:
            name: Name of the utility function
            attributes: Dictionary mapping attribute names to their configuration:
                {
                    "attribute_name": {
                        "weight": float,  # Weight for this attribute
                        "factors": List[str],  # State factors that contribute
                        "higher_better": bool  # Whether higher values are better
                    }
                }
            description: Description of the utility function
            action_space: Optional action space
        """
        super().__init__(name, description, action_space)
        self.attributes = attributes
    
    def __str__(self) -> str:
        return f"Multi-Attribute Utility: {self.name}\nAttributes: {self.attributes}"
    
    def evaluate_tf(self, action: tf.Tensor, state_samples: Dict[str, tf.Tensor]) -> tf.Tensor:
        # Initialize total utility
        total_utility = tf.zeros_like(next(iter(state_samples.values())))
        
        # Compute weighted sum of each attribute
        for attr_name, attr_config in self.attributes.items():
            weight = attr_config["weight"]
            factors = attr_config["factors"]
            higher_better = attr_config.get("higher_better", True)
            
            # Sum relevant factors for this attribute
            attr_value = tf.zeros_like(next(iter(state_samples.values())))
            for factor in factors:
                if factor in state_samples:
                    attr_value += state_samples[factor]
            
            # Invert if lower is better
            if not higher_better:
                attr_value = -attr_value
            
            # Add weighted attribute to total
            total_utility += weight * attr_value
        
        return total_utility

class RiskAdjustedUtility(UtilityFunction):
    """
    Utility function that applies risk adjustment to monetary outcomes.
    Supports both logarithmic and exponential utility functions.
    """
    
    def __init__(self, name: str, risk_type: str = "log", alpha: float = 0.1,
                 description: str = "", action_space=None):
        """
        Initialize risk-adjusted utility function.
        
        Args:
            name: Name of the utility function
            risk_type: Type of risk adjustment ("log" or "exp")
            alpha: Risk aversion parameter for exponential utility
            description: Description of the utility function
            action_space: Optional action space
        """
        super().__init__(name, description, action_space)
        self.risk_type = risk_type
        self.alpha = alpha
    
    def __str__(self) -> str:
        if self.risk_type == "log":
            return f"Logarithmic Risk-Adjusted Utility: {self.name}"
        else:
            return f"Exponential Risk-Adjusted Utility: {self.name} (alpha={self.alpha})"
    
    def evaluate_tf(self, action: tf.Tensor, state_samples: Dict[str, tf.Tensor]) -> tf.Tensor:
        # First compute base monetary utility
        monetary_utility = tf.zeros_like(next(iter(state_samples.values())))
        for factor, values in state_samples.items():
            if factor.endswith("_revenue"):
                monetary_utility += values
            elif factor.endswith("_cost"):
                monetary_utility -= values
        
        # Apply risk adjustment
        if self.risk_type == "log":
            # Logarithmic utility: U(x) = log(x)
            # Add small constant to avoid log(0)
            return tf.math.log(monetary_utility + 1e-10)
        else:
            # Exponential utility: U(x) = 1 - e^(-αx)
            return 1.0 - tf.exp(-self.alpha * monetary_utility) 