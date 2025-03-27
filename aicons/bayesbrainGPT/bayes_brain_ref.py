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

Note on State Factor Management:
- AIcon.add_state_factor() is the primary public API for adding factors
- It delegates to brain.state.add_factor() for single factor addition
- This set_state_factors() method is a bulk version for internal use
- Always use add_state_factor() from AIcon for proper relationship handling
"""

from typing import Dict, Any, Optional, Tuple, List, Union, Callable
import numpy as np
import uuid
from abc import ABC, abstractmethod
import time

# Import core components
from .utility_function import create_utility
from .decision_making.action_space import ActionSpace
from .state_representation import BayesianState
from .perception.perception import BayesianPerception
from .state_representation.latent_variables import ContinuousLatentVariable, CategoricalLatentVariable, DiscreteLatentVariable

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
        from aicons.bayesbrainGPT.perception.perception import BayesianPerception
        self.perception = BayesianPerception(self)
        
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
    
    def get_state_factors(self) -> Dict[str, Any]:
        """
        Get all state factors from the brain's state.
        
        Returns:
            Dictionary of state factors
        """
        return self.state.get_state_factors()
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state beliefs.
        
        Returns:
            Dictionary of current state beliefs
        """
        return self.state.get_state()
    
    def get_posterior_samples(self) -> Dict[str, Any]:
        """
        Get samples from the posterior distribution.
        
        Returns:
            Dictionary of posterior samples
        """
        return self.state.sample_from_posterior()
    
    def find_best_action(self, num_samples: Optional[int] = None) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Find the best action based on current beliefs.
        
        Args:
            num_samples: Number of actions to sample (for Monte Carlo methods)
            
        Returns:
            Tuple of (best_action, expected_utility)
        """
        if not self.action_space or not self.utility_function:
            raise ValueError("Action space and utility function must be set before finding best action")
            
        # Get current state
        current_state = self.get_state()
        
        # Sample actions
        if num_samples is None:
            num_samples = self.decision_params["num_samples"]
            
        # Get action utilities
        action_utilities = self.compute_action_utilities(current_state, num_samples)
        
        # Select best action
        return self.select_best_action(action_utilities)
    
    def compute_action_utilities(self, state: Dict[str, Any], num_samples: int) -> Dict[Dict[str, Any], float]:
        """
        Compute expected utilities for all possible actions.
        
        Args:
            state: Current state
            num_samples: Number of samples to use
            
        Returns:
            Dictionary mapping actions to their expected utilities
        """
        action_utilities = {}
        
        # Sample actions from the action space
        actions = self.action_space.sample_actions(num_samples)
        
        # Compute utility for each action
        for action in actions:
            # Sample from posterior given action
            posterior_samples = self.state.sample_from_posterior(num_samples)
            
            # Compute utility for each sample
            utilities = []
            for sample in posterior_samples:
                utility = self.utility_function(action, sample)
                utilities.append(utility)
                
            # Average utility across samples
            expected_utility = np.mean(utilities)
            action_utilities[action] = expected_utility
            
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
            if success:
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
    
    def set_state_factors(self, factors: Dict[str, Any]) -> None:
        """
        Set state factors from AIcon.
        
        Args:
            factors: Dictionary of factor information in the format:
                {
                    "name": {
                        "type": "continuous" | "categorical" | "discrete",
                        "value": value,
                        "params": {
                            "loc": float,
                            "scale": float,
                            "constraints": {"lower": float, "upper": float},
                            "categories": List[str],
                            "probs": List[float],
                            "rate": float
                        },
                        "relationships": {
                            "depends_on": List[str]
                        }
                    }
                }
        """
        for name, factor in factors.items():
            if name in self.state.factors:
                # Update existing factor
                self.state.factors[name].update(factor["value"])
            else:
                # Add new factor if it doesn't exist
                self.state.add_factor(
                    name=name,
                    factor_type=factor["type"],
                    value=factor["value"],
                    params=factor.get("params", {}),
                    relationships=factor.get("relationships")
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
    
    def set_decision_params(self, params: Dict[str, Any]) -> None:
        """
        Set parameters for decision-making.
        
        Args:
            params: Dictionary of decision parameters
        """
        self.decision_params.update(params)
    
    def get_action_space(self) -> Optional[ActionSpace]:
        """Get the current action space."""
        return self.action_space
    
    def get_utility_function(self) -> Optional[Any]:
        """Get the current utility function."""
        return self.utility_function
    
    def get_decision_history(self) -> List[Tuple[Dict[str, Any], float, float]]:
        """
        Get the history of decisions made by the brain.
        
        Returns:
            List of (action, utility, timestamp) tuples
        """
        if self.last_action is None:
            return []
        return [(self.last_action, self.last_utility, self.last_decision_time)]
    
    def add_sensor(self, name: str, sensor: Any = None, factor_mapping: Optional[Dict[str, str]] = None) -> None:
        """
        Add a sensor to the brain's perception system.
        
        Args:
            name: Name of the sensor
            sensor: The sensor object or function
            factor_mapping: Optional mapping between sensor outputs and state factors
        """
        # Initialize perception if not already done
        if not hasattr(self, 'perception'):
            from aicons.bayesbrainGPT.perception.perception import BayesianPerception
            self.perception = BayesianPerception(self)
        
        # Auto-create required factors if sensor has get_expected_factors method
        if hasattr(sensor, 'get_expected_factors'):
            expected_factors = sensor.get_expected_factors()
            current_state = self.state.get_state_factors() or {}
            
            # Create any missing factors with default values
            for factor_name, factor_info in expected_factors.items():
                # Map the factor name if a mapping exists
                if factor_mapping and factor_name in factor_mapping:
                    mapped_name = factor_mapping[factor_name]
                else:
                    mapped_name = factor_name
                
                # Check if factor exists and validate type
                if mapped_name in current_state:
                    existing_factor = current_state[mapped_name]
                    if existing_factor["type"] != factor_info["type"]:
                        raise ValueError(f"Type mismatch for factor {mapped_name}: sensor provides {factor_info['type']} but state has {existing_factor['type']}")
                
                # Create new factor if it doesn't exist
                if mapped_name not in current_state:
                    # Extract factor properties from factor_info
                    factor_type = factor_info.get('type', 'continuous')
                    default_value = factor_info.get('default_value', 0.0)
                    uncertainty = factor_info.get('uncertainty', 0.1)
                    lower_bound = factor_info.get('lower_bound', None)
                    upper_bound = factor_info.get('upper_bound', None)
                    categories = factor_info.get('categories', None)
                    description = factor_info.get('description', f"Factor from {name} sensor")
                    
                    print(f"Auto-creating missing factor: {mapped_name} ({factor_type})")
                    
                    # Create the appropriate type of factor
                    if factor_type == 'continuous':
                        self.state.add_factor(
                            name=mapped_name,
                            factor_type='continuous',
                            value=default_value,
                            params={
                                'loc': default_value,
                                'scale': uncertainty,
                                'constraints': {
                                    'lower': lower_bound,
                                    'upper': upper_bound
                                } if lower_bound is not None or upper_bound is not None else None
                            }
                        )
                    elif factor_type == 'categorical' and categories:
                        self.state.add_factor(
                            name=mapped_name,
                            factor_type='categorical',
                            value=default_value,
                            params={
                                'categories': categories,
                                'probs': [1.0/len(categories)] * len(categories)
                            }
                        )
                    elif factor_type == 'discrete':
                        if upper_bound is not None:
                            # Categorical distribution for bounded discrete values
                            categories = list(range(int(lower_bound or 0), int(upper_bound) + 1))
                            self.state.add_factor(
                                name=mapped_name,
                                factor_type='discrete',
                                value=default_value,
                                params={
                                    'categories': categories,
                                    'probs': [1.0/len(categories)] * len(categories)
                                }
                            )
                        else:
                            # Poisson distribution for unbounded discrete values
                            self.state.add_factor(
                                name=mapped_name,
                                factor_type='discrete',
                                value=default_value,
                                params={
                                    'rate': default_value
                                }
                            )
        
        # Register the sensor with perception
        self.perception.register_sensor(name, sensor, factor_mapping)
        return sensor
    
    def get_sensors(self) -> List[Callable]:
        """Get the current list of sensors"""
        return getattr(self, 'sensors', [])
    
    def collect_sensor_data(self, environment: Any) -> Dict[str, Any]:
        """
        Collect data from all sensors
        
        Args:
            environment: The environment to collect data from
            
        Returns:
            Dictionary of sensor data
        """
        sensor_data = {}
        for sensor in self.get_sensors():
            sensor_data.update(sensor(environment))
        return sensor_data
    
    def update_from_sensor(self, sensor_name: str, environment: Any = None) -> bool:
        """
        Update beliefs based on data from a specific sensor.
        
        Args:
            sensor_name: Name of the sensor to use
            environment: Optional environment data
            
        Returns:
            True if update was successful
        """
        return self.perception.update_from_sensor(sensor_name, environment)
    
    def update_from_all_sensors(self, environment: Any = None) -> bool:
        """
        Update beliefs based on data from all sensors.
        
        Args:
            environment: Optional environment data
            
        Returns:
            True if update was successful
        """
        return self.perception.update_all(environment)

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