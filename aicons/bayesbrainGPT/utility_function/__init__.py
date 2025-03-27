"""
Utility Functions for BayesBrainGPT

This package provides a collection of utility functions for Bayesian decision-making,
allowing for evaluation of actions under different objectives and contexts.
"""

# Base utility functions
from .utility_base import (
    UtilityFunction,
    TensorFlowUtilityFunction
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
    try:
        if utility_type not in UTILITY_FACTORIES:
            error_msg = f"Unknown utility type: {utility_type}. Available types are: {list(UTILITY_FACTORIES.keys())}"
            raise ValueError(error_msg)
        
        utility_class = UTILITY_FACTORIES[utility_type]
        
        # For WeightedSumUtility, we need to handle action_space differently
        if utility_type == "weighted_sum":
            # Create the utility functions first if they're provided
            if 'utility_functions' in kwargs:
                utility_fns = kwargs.pop('utility_functions')
                # Set action space for each utility function
                for fn in utility_fns:
                    if hasattr(fn, 'set_action_space'):
                        fn.set_action_space(action_space)
            else:
                # Create default utility functions based on weights
                if 'weights' in kwargs:
                    weights = kwargs.pop('weights')
                    utility_fns = []
                    for name in weights.keys():
                        # Create a MarketingROIUtility for each weight
                        fn = MarketingROIUtility(
                            name=f"default_{name}",
                            description=f"Default utility function for {name}",
                            revenue_per_sale=1.0,  # Default values
                            num_ads=1,
                            num_days=1
                        )
                        if action_space is not None:
                            fn.set_action_space(action_space)
                        utility_fns.append(fn)
                    kwargs['utility_functions'] = utility_fns
                else:
                    raise ValueError("WeightedSumUtility requires either utility_functions or weights")
        
        # For LinearUtility, provide a default name if not given
        if utility_type == "linear":
            if 'name' not in kwargs:
                kwargs['name'] = "Linear Utility"
            if 'description' not in kwargs:
                kwargs['description'] = "A linear combination of action values"
        
        # Add action_space to kwargs if provided
        if action_space is not None:
            kwargs['action_space'] = action_space
        
        try:
            utility = utility_class(**kwargs)
            return utility
        except Exception as e:
            raise
            
    except Exception as e:
        raise


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
    
    # Linear utilities
    "LinearUtility",
    
    # Factory function
    "create_utility",
    "UTILITY_FACTORIES",
    
    # Modules
    "utility_base"
]
