"""
Utility Functions for BayesBrainGPT

This package provides a collection of utility functions for Bayesian decision-making,
allowing for evaluation of actions under different objectives and contexts.
"""

# Base utility functions
from aicons.bayesbrainGPT.utility_function.utility_base import (
    UtilityFunction,
    TensorFlowUtilityFunction
)

# Marketing-specific utility functions
from aicons.bayesbrainGPT.utility_function.marketing_utility import (
    MarketingROIUtility,
    ConstrainedMarketingROI,
    WeatherDependentMarketingROI
)

# Multi-objective utility functions
from aicons.bayesbrainGPT.utility_function.multi_objective_utility import (
    WeightedSumUtility,
    ParetoUtility,
    ConstrainedMultiObjectiveUtility,
    AdaptiveWeightUtility
)

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
    "adaptive_weight": AdaptiveWeightUtility
}


def create_utility(utility_type: str, action_space=None, **kwargs):
    """
    Factory function to create utility functions by type.
    
    Args:
        utility_type: String identifier for the utility type
        action_space: Optional action space to connect with the utility function
        **kwargs: Arguments to pass to the utility constructor
        
    Returns:
        An instance of the specified utility function
        
    Raises:
        ValueError: If utility_type is not recognized
    """
    if utility_type not in UTILITY_FACTORIES:
        raise ValueError(f"Unknown utility type: {utility_type}. " 
                         f"Available types are: {list(UTILITY_FACTORIES.keys())}")
    
    utility_class = UTILITY_FACTORIES[utility_type]
    
    # Add action_space to kwargs if provided
    if action_space is not None:
        kwargs['action_space'] = action_space
        
    return utility_class(**kwargs)


__all__ = [
    # Base classes
    "UtilityFunction",
    "TensorFlowUtilityFunction",
    
    # Marketing utilities
    "MarketingROIUtility",
    "ConstrainedMarketingROI",
    "WeatherDependentMarketingROI",
    
    # Multi-objective utilities
    "WeightedSumUtility",
    "ParetoUtility",
    "ConstrainedMultiObjectiveUtility",
    "AdaptiveWeightUtility",
    
    # Factory function
    "create_utility",
    "UTILITY_FACTORIES"
]
