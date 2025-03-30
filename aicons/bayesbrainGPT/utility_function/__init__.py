"""
Utility Functions for BayesBrainGPT

This package provides a collection of utility functions for Bayesian decision-making,
allowing for evaluation of actions under different objectives and contexts.
"""

from aicons.bayesbrainGPT.decision_making.action_space import ActionSpace

# Base utility functions
from .utility_base import (
    UtilityFunction
)
from . import utility_base  # Explicitly expose the module

# Marketing-specific utility functions
from .marketing_utility import (
    MarketingROIUtility,
    ConstrainedMarketingROI,
    WeatherDependentMarketingROI
)

# Multi-objective utility functions
from .multi_objective_utility import (
    WeightedSumUtility,
    ParetoUtility,
    ConstrainedMultiObjectiveUtility,
    AdaptiveWeightUtility
)

# Linear utility functions
from .linear_utility import LinearUtility

# Convenience mapping of utility types to their classes
UTILITY_FACTORIES = {
    # Marketing utilities
    "marketing_roi": MarketingROIUtility,
    "constrained_marketing_roi": ConstrainedMarketingROI,
    "weather_dependent_marketing_roi": WeatherDependentMarketingROI,
    
    # Multi-objective utilities
    "weighted_sum": WeightedSumUtility,
    "pareto": ParetoUtility,
    "constrained_multi_objective": ConstrainedMultiObjectiveUtility,
    "adaptive_weight": AdaptiveWeightUtility,
    
    # Linear utilities
    "linear": LinearUtility
}


def create_utility(utility_type: str, action_space: ActionSpace, **kwargs) -> UtilityFunction:
    """Factory for creating utility functions based on type."""
    if utility_type not in UTILITY_FACTORIES:
        raise ValueError(f"Unknown utility type: {utility_type}")
    
    # Special handling for WeightedSumUtility and LinearUtility
    if utility_type in ['weighted_sum', 'linear']:
        if 'weights' not in kwargs:
            raise ValueError(f"{utility_type} utility requires weights parameter")
        if isinstance(kwargs['weights'], dict):
            # Convert dict weights to list in order of action space dimensions
            # Strip _budget suffix when looking up weights
            kwargs['weights'] = [kwargs['weights'][dim.name.replace('_budget', '')] for dim in action_space.dimensions]
    
    # Add action space to kwargs
    kwargs['action_space'] = action_space
    
    # Create utility function using factory
    utility_class = UTILITY_FACTORIES[utility_type]
    
    # Remove name and description from kwargs for utility classes that don't accept them
    if utility_type in ['marketing_roi', 'constrained_marketing_roi', 'weather_dependent_marketing_roi']:
        kwargs.pop('name', None)
        kwargs.pop('description', None)
    
    return utility_class(**kwargs)


__all__ = [
    # Base classes
    "UtilityFunction",
    
    # Marketing utilities
    "MarketingROIUtility",
    "ConstrainedMarketingROI",
    "WeatherDependentMarketingROI",
    
    # Multi-objective utilities
    "WeightedSumUtility",
    "ParetoUtility",
    "ConstrainedMultiObjectiveUtility",
    "AdaptiveWeightUtility",
    
    # Linear utilities
    "LinearUtility",
    
    # Factory function
    "create_utility",
    "UTILITY_FACTORIES",
    
    # Modules
    "utility_base"
]
