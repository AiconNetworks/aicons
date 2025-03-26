"""
Refactored BayesBrain Module for Bayesian Decision Making

This module provides the refactored BayesBrain class, which implements core Bayesian functionality:
1. State representation - for maintaining beliefs about the world
2. Perception - for updating beliefs based on sensor data
3. Decision making - for selecting optimal actions based on utility

The brain is designed to be a pure Bayesian processor, receiving:
- State factors from AIcon
- Sensor data from AIcon
- Action space from AIcon
- Utility function from AIcon

And providing:
- Updated beliefs with uncertainty
- Optimal actions with expected utility
- Posterior distributions
"""

from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import uuid
from abc import ABC, abstractmethod
import time

# Import core components
from .utility_function import create_utility
from .decision_making.action_space import ActionSpace
from .state_representation import BayesianState
from .perception.perception import BayesianPerception

class BayesBrain:
    """
    A Bayesian decision-making brain that implements core Bayesian functionality.
    
    The brain maintains a probabilistic belief state about the world,
    updates this belief based on sensor data, and makes decisions by selecting
    actions that maximize expected utility.
    
    Components:
    - State: Maintains beliefs about the world
    - Perception: Updates beliefs based on sensor data
    - Utility Function: Evaluates actions
    - Action Space: Defines possible actions
    
    The brain is designed to be a pure Bayesian processor, receiving all components
    from the AIcon and providing updated beliefs and decisions.
    """
    
    def __init__(self, name: str = "BayesBrain", description: str = ""):
        """
        Initialize a BayesBrain instance.
        
        Args:
            name: Name of the brain instance
            description: Description of this brain's purpose
        """
        self.name = name
        self.description = description
        self.id = str(uuid.uuid4())
        
        # Core Bayesian components
        self.state = BayesianState()
        self.perception = None
        
        # Components provided by AIcon
        self.utility_function = None
        self.action_space = None
        self.aicon = None  # Reference to the AIcon this brain belongs to
        
        # Decision making parameters
        self.decision_params = {
            "num_samples": 1000,
            "exploration_rate": 0.1,
            "optimization_method": "monte_carlo"  # or "gradient"
        }
        
        # Decision making state
        self.last_decision_time = None
        self.last_action = None
        self.last_utility = None
    
    def set_aicon(self, aicon: Any) -> None:
        """
        Set the AIcon this brain belongs to.
        
        Args:
            aicon: The AIcon instance
        """
        self.aicon = aicon
    
    def compute_posteriors(self) -> Dict[str, Any]:
        """
        Compute posterior distributions for all state factors.
        
        Returns:
            Dictionary mapping state factors to their posterior distributions
        """
        if self.perception is None:
            return self.state.get_beliefs()
            
        return self.perception.get_posterior_samples()
    
    def compute_action_utilities(self, posteriors: Dict[str, Any], num_samples: Optional[int] = None) -> Dict[Dict[str, Any], float]:
        """
        Compute expected utility for all possible actions given posteriors.
        
        Args:
            posteriors: Dictionary of posterior distributions
            num_samples: Number of samples to use for Monte Carlo methods
            
        Returns:
            Dictionary mapping actions to their expected utilities
        """
        if self.action_space is None or self.utility_function is None:
            return {}
            
        if num_samples is None:
            num_samples = self.decision_params["num_samples"]
            
        action_utilities = {}
        
        # Sample actions and compute utilities
        for _ in range(num_samples):
            action = self.action_space.sample()
            
            # Calculate expected utility
            if posteriors:
                utility = self.utility_function.expected_utility(action, posteriors)
            else:
                utility = self.utility_function.evaluate(action, self.state.get_beliefs())
                
            action_utilities[action] = utility
            
        return action_utilities
    
    def select_best_action(self, action_utilities: Dict[Dict[str, Any], float]) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Select the best action using argmax on expected utilities.
        
        Args:
            action_utilities: Dictionary mapping actions to their expected utilities
            
        Returns:
            Tuple of (best_action, best_utility)
        """
        if not action_utilities:
            return None, 0.0
            
        # Find action with maximum utility
        best_action = max(action_utilities.items(), key=lambda x: x[1])
        return best_action
    
    def take_action(self, num_samples: Optional[int] = None) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Take an action by finding the best action and making a decision.
        
        This method:
        1. Computes posteriors for all state factors
        2. Computes expected utility for all possible actions
        3. Selects the best action using argmax
        4. Calls the AIcon's make_decision method to execute the action
        
        Args:
            num_samples: Number of actions to sample (for Monte Carlo methods)
            
        Returns:
            Tuple of (action_taken, expected_utility)
        """
        # Compute posteriors
        posteriors = self.compute_posteriors()
        
        # Compute utilities for all actions
        action_utilities = self.compute_action_utilities(posteriors, num_samples)
        
        # Select best action
        best_action, expected_utility = self.select_best_action(action_utilities)
        
        if best_action and self.aicon:
            # Let the AIcon make the decision
            success = self.aicon.make_decision(best_action)
            if not success:
                print("Warning: AIcon failed to make decision")
            else:
                # Update decision state
                self.last_decision_time = time.time()
                self.last_action = best_action
                self.last_utility = expected_utility
        
        return best_action, expected_utility
    
    def update_beliefs(self, sensor_data: Dict[str, Tuple[Any, float]]) -> None:
        """
        Update beliefs based on new sensor data from AIcon.
        
        This method:
        1. Updates beliefs using the perception system
        2. Optionally triggers decision-making if sensor data indicates significant changes
        
        Args:
            sensor_data: Dictionary mapping state factors to (value, reliability) tuples
        """
        if self.perception is not None:
            self.perception.update(sensor_data)
            
            # TODO: Add control chart logic here to detect significant changes
            # For now, we'll just update beliefs without triggering decisions
            # Later, we can add logic to check if sensor data indicates
            # significant changes that should trigger a decision
    
    def set_state_factors(self, factors: Dict[str, Dict[str, Any]]) -> None:
        """
        Set state factors from AIcon.
        
        Args:
            factors: Dictionary of state factors with their properties
        """
        for name, factor in factors.items():
            if factor["type"] == "continuous":
                self.state.add_continuous_latent(
                    name=name,
                    mean=factor["value"],
                    uncertainty=factor["params"]["scale"]
                )
            elif factor["type"] == "categorical":
                self.state.add_categorical_latent(
                    name=name,
                    initial_value=factor["value"],
                    possible_values=factor["categories"],
                    probs=factor["params"]["probs"]
                )
            elif factor["type"] == "discrete":
                self.state.add_discrete_latent(
                    name=name,
                    initial_value=factor["value"],
                    min_value=factor.get("constraints", {}).get("lower", 0),
                    max_value=factor.get("constraints", {}).get("upper")
                )
    
    def set_utility_function(self, utility: Any) -> None:
        """
        Set the utility function provided by AIcon.
        
        Args:
            utility: The utility function to use for evaluating actions
        """
        self.utility_function = utility
    
    def set_action_space(self, action_space: ActionSpace) -> None:
        """
        Set the action space provided by AIcon.
        
        Args:
            action_space: The action space to use for decision making
        """
        self.action_space = action_space
    
    def initialize_perception(self) -> None:
        """Initialize the perception system for updating beliefs."""
        self.perception = BayesianPerception(self)
    
    def set_decision_params(self, params: Dict[str, Any]) -> None:
        """
        Set parameters for decision-making.
        
        Args:
            params: Dictionary of decision parameters
        """
        self.decision_params.update(params)
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state beliefs."""
        return self.state.get_beliefs()
    
    def get_action_space(self) -> Optional[ActionSpace]:
        """Get the current action space."""
        return self.action_space
    
    def get_utility_function(self) -> Optional[Any]:
        """Get the current utility function."""
        return self.utility_function
    
    def get_posterior_samples(self) -> Dict[str, Any]:
        """Get samples from the posterior distribution."""
        if self.perception is not None:
            return self.perception.get_posterior_samples()
        return {}
    
    def get_decision_history(self) -> List[Tuple[Dict[str, Any], float, float]]:
        """
        Get the history of decisions made by the brain.
        
        Returns:
            List of (action, utility, timestamp) tuples
        """
        if self.last_action is None:
            return []
        return [(self.last_action, self.last_utility, self.last_decision_time)]

# Example usage
if __name__ == "__main__":
    # Create a BayesBrain instance
    brain = BayesBrain(
        name="MarketingBrain",
        description="A Bayesian brain for marketing optimization"
    )
    
    # Set state factors from AIcon
    state_factors = {
        "base_conversion_rate": {
            "type": "continuous",
            "value": 0.05,
            "params": {"scale": 0.01}
        },
        "primary_channel": {
            "type": "categorical",
            "value": "google",
            "categories": ["google", "facebook", "twitter"],
            "params": {"probs": [0.7, 0.2, 0.1]}
        },
        "optimal_daily_ads": {
            "type": "discrete",
            "value": 8,
            "constraints": {"lower": 0, "upper": 10}
        }
    }
    brain.set_state_factors(state_factors)
    
    # Set up action space from AIcon
    from .decision_making.action_space import create_budget_allocation_space
    action_space = create_budget_allocation_space(
        total_budget=1000.0,
        num_ads=2,
        budget_step=100.0
    )
    brain.set_action_space(action_space)
    
    # Set up utility function from AIcon
    from .utility_function import create_utility
    utility_function = create_utility(
        utility_type="marketing_roi",
        action_space=action_space,
        revenue_per_sale=10.0,
        num_ads=2,
        num_days=3
    )
    brain.set_utility_function(utility_function)
    
    # Initialize perception
    brain.initialize_perception()
    
    # Example sensor data from AIcon
    sensor_data = {
        "base_conversion_rate": (0.063, 0.8),
        "primary_channel": ("google", 0.9),
        "optimal_daily_ads": (8, 0.7)
    }
    
    # Update beliefs
    brain.update_beliefs(sensor_data)
    
    # Make a decision
    best_action, expected_utility = brain.take_action()
    
    print(f"Best action: {best_action}")
    print(f"Expected utility: {expected_utility}") 