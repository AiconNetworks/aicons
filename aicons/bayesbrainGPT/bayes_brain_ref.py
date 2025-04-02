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
import os
import json
import ast

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
        
        # Store posterior samples here
        self.posterior_samples = None
        self.last_posterior_update = None  # Timestamp of last posterior update
        self.update_history = []
    
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
        return self.state.get_beliefs()
    
    def get_posterior_samples(self, num_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Get samples from the posterior distribution.
        
        Args:
            num_samples: Number of samples to return. If None, return all available.
            
        Returns:
            Dictionary of posterior samples
        """
        if self.posterior_samples is None:
            print("No posterior samples available")
            return {}
        
        # If num_samples specified, randomly sample that many
        if num_samples is not None and num_samples < len(next(iter(self.posterior_samples.values()))):
            indices = np.random.choice(len(next(iter(self.posterior_samples.values()))), 
                                    size=num_samples, replace=False)
            return {
                name: samples[indices] if isinstance(samples, np.ndarray) else samples
                for name, samples in self.posterior_samples.items()
            }
        
        return self.posterior_samples
    
    def find_best_action(self, num_samples: Optional[int] = None) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Find the best action based on current beliefs.
        
        Args:
            num_samples: Number of actions to sample (for Monte Carlo methods)
            
        Returns:
            Tuple of (best_action, expected_utility)
        """
        print("\n=== Starting find_best_action ===")
        print(f"Action space type: {type(self.action_space)}")
        print(f"Utility function type: {type(self.utility_function)}")
        
        if not self.action_space or not self.utility_function:
            raise ValueError("Action space and utility function must be set before finding best action")
            
        # Get posterior samples from perception
        print("\nGetting posterior samples from perception system...")
        posterior_samples = self.get_posterior_samples()
        
        # If no posterior samples available, use prior samples
        if not posterior_samples:
            print("No posterior samples available, using prior samples...")
            posterior_samples = self.state.get_prior_samples(num_samples)
            
        print(f"\nPosterior samples shape: {len(next(iter(posterior_samples.values())))} samples")
        print(f"Posterior sample keys: {list(posterior_samples.keys())}")
        print("\nFirst posterior sample:")
        for key, value in posterior_samples.items():
            print(f"- {key}: {value[0] if isinstance(value, np.ndarray) else value}")
        
        # Initialize action utilities dictionary
        action_utilities = {}
        
        # Get all possible actions from action space dimensions
        possible_actions = self.action_space.enumerate_actions()
        print(f"\nEvaluating {len(possible_actions)} possible actions")
        print("\nAction space dimensions:")
        for dim in self.action_space.dimensions:
            print(f"- {dim.name}: {dim.dim_type}")
            if dim.dim_type == 'continuous':
                print(f"  Range: [{dim.min_value}, {dim.max_value}]")
                print(f"  Step: {dim.step}")
        
        # Evaluate each possible action
        for action in possible_actions:
            print(f"\nEvaluating action: {action}")
            print(f"Action type: {type(action)}")
            print(f"Action keys: {action.keys()}")
            print(f"Action values: {action.values()}")
            utility = 0.0
            
            # Convert action values to float32
            action_float32 = {k: float(v) for k, v in action.items()}
            print(f"\nConverted action to float32: {action_float32}")
            
            # If we have a method-based utility function
            if hasattr(self.utility_function, 'evaluate'):
                print("\nUsing method-based utility function")
                # If we have posterior samples, use expected_utility method
                if posterior_samples:
                    if hasattr(self.utility_function, 'expected_utility'):
                        print("\nUsing expected_utility method")
                        utility = self.utility_function.expected_utility(action_float32, posterior_samples)
                        print(f"Computed utility: {utility}")
                    else:
                        # Compute expected utility over posterior samples
                        all_utilities = []
                        print("\nComputing utilities for each posterior sample:")
                        for j in range(len(next(iter(posterior_samples.values())))):
                            # Extract the j-th sample for each parameter
                            sample = {k: v[j] for k, v in posterior_samples.items()}
                            print(f"\nPosterior sample {j+1}:")
                            for k, v in sample.items():
                                print(f"- {k}: {v}")
                            
                            sample_utility = self.utility_function.evaluate(action_float32, sample)
                            print(f"Utility for this sample: {sample_utility}")
                            all_utilities.append(sample_utility)
                        
                        utility = sum(all_utilities) / len(all_utilities)
                        print(f"\nAverage utility over {len(all_utilities)} samples: {utility}")
                else:
                    # Use state factors directly if no posterior samples
                    print("\nUsing state factors directly")
                    utility = self.utility_function.evaluate(action_float32, self.state_factors)
                    print(f"Computed utility: {utility}")
            
            # If we have a callable utility function 
            elif callable(self.utility_function):
                print("\nUsing callable utility function")
                if posterior_samples:
                    # Compute expected utility over posterior samples
                    all_utilities = []
                    print("\nComputing utilities for each posterior sample:")
                    for j in range(len(next(iter(posterior_samples.values())))):
                        # Extract the j-th sample for each parameter
                        sample = {k: v[j] for k, v in posterior_samples.items()}
                        print(f"\nPosterior sample {j+1}:")
                        for k, v in sample.items():
                            print(f"- {k}: {v}")
                        
                        sample_utility = self.utility_function(action_float32, sample)
                        print(f"Utility for this sample: {sample_utility}")
                        all_utilities.append(sample_utility)
                    
                    utility = sum(all_utilities) / len(all_utilities)
                    print(f"\nAverage utility over {len(all_utilities)} samples: {utility}")
                else:
                    # Use state factors directly if no posterior samples
                    print("\nUsing state factors directly")
                    utility = self.utility_function(action_float32, self.state_factors)
                    print(f"Computed utility: {utility}")
            
            # Store the utility for this action using a tuple as the key
            action_key = tuple(sorted(action.items()))
            print(f"\nStoring utility with key: {action_key}")
            action_utilities[action_key] = utility
            print(f"Stored utility {utility} for action {action}")
        
        # Use select_best_action to choose the best action
        print("\nSelecting best action from utilities:")
        for key, util in action_utilities.items():
            print(f"- Action: {dict(key)}, Utility: {util}")
        
        best_action_key = max(action_utilities.items(), key=lambda x: x[1])[0]
        best_action = dict(best_action_key)
        expected_utility = action_utilities[best_action_key]
        
        print(f"\nBest action: {best_action}")
        print(f"Expected utility: {expected_utility}")
        
        return best_action, expected_utility
    
    def compute_action_utilities(self, state: Dict[str, Any], num_samples: int) -> Dict[str, float]:
        """
        Compute expected utilities for all possible actions.
        
        Args:
            state: Current state
            num_samples: Number of samples to use
            
        Returns:
            Dictionary mapping action keys to their expected utilities
        """
        print("\n=== Debug: compute_action_utilities ===")
        print(f"Input state: {state}")
        
        action_utilities = {}
        
        # Get posterior samples once for all actions
        posterior_samples = self.get_posterior_samples(num_samples)
        print(f"\nPosterior samples: {posterior_samples}")
        
        # Get all possible actions from action space dimensions
        possible_actions = self.action_space.enumerate_actions()
        print(f"\nEvaluating {len(possible_actions)} possible actions")
        
        # Evaluate each possible action
        for action in possible_actions:
            print(f"\nProcessing action: {action}")
            
            # Create a single sample dictionary with current values
            sample = {}
            for name, samples in posterior_samples.items():
                if isinstance(samples, np.ndarray):
                    # If it's an array, use the first value
                    sample[name] = float(samples[0])
                else:
                    # If it's a scalar value, use it directly
                    sample[name] = float(samples)
            
            print(f"\nFinal sample dictionary: {sample}")
            
            # Use evaluate method
            try:
                utility = self.utility_function.evaluate(action, sample)
                print(f"Computed utility: {utility}")
                # Convert action to tuple for use as key
                action_key = tuple(sorted(action.items()))
                action_utilities[action_key] = utility
            except Exception as e:
                print(f"Error computing utility: {str(e)}")
                print(f"Action type: {type(action)}")
                print(f"Sample type: {type(sample)}")
                raise
            
        return action_utilities
    
    def select_best_action(self, action_utilities: Dict[str, float]) -> Tuple[Optional[Dict[str, Any]], float]:
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
        best_action, best_utility = max(action_utilities.items(), key=lambda x: x[1])
        
        print(f"\nSelected best action: {best_action}")
        print(f"Expected utility: {best_utility}")
        
        return best_action, best_utility
    
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
        
        At time t:
        - We have posterior_samples_t (stored in brain)
        
        At time t+1:
        - Get new sensor data
        - Use posterior_samples_t as prior
        - Generate new posterior_samples_t+1
        - Store posterior_samples_t+1 in brain
        """
        print("\n=== Starting Belief Update ===")
        print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get current posterior samples (from time t)
        current_samples = self.get_posterior_samples()
        if current_samples:
            print(f"Using current posterior as prior (from {self.last_posterior_update})")
        
        # Update with new sensor data
        success = self.perception.update_from_sensor(sensor_name, environment)
        
        if success:
            # Generate new posterior samples for time t+1
            new_sensor_data = self.perception.collect_sensor_data(environment)
            updated_samples = self.perception.sample_posterior(new_sensor_data)
            
            # Store new posterior samples (time t+1) in brain
            self.set_posterior_samples(updated_samples)
            
            # Record update in history
            self.update_history.append({
                'timestamp': self.last_posterior_update,
                'sensor': sensor_name,
                'num_samples': len(next(iter(updated_samples.values()))),
                'parameters': list(updated_samples.keys())
            })
            
            print(f"Successfully updated beliefs at {self.last_posterior_update}")
            print(f"Total updates in history: {len(self.update_history)}")
            return True
        
        return False
    
    def update_from_all_sensors(self, environment: Any = None) -> bool:
        """
        Update beliefs based on data from all sensors.
        
        Args:
            environment: Optional environment data
            
        Returns:
            True if update was successful
        """
        return self.perception.update_all(environment)
    
    def set_posterior_samples(self, samples: Dict[str, Any]):
        """
        Set the posterior samples directly.
        
        Args:
            samples: Dictionary of posterior samples with factor information
        """
        # Store posterior samples directly
        self.posterior_samples = samples
        
        self.last_posterior_update = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Updated posterior samples at {self.last_posterior_update}")
        
        # Print sample statistics
        print("\nPosterior Sample Statistics:")
        for name, samples in self.posterior_samples.items():
            if isinstance(samples, np.ndarray):
                print(f"- {name}:")
                print(f"  Mean: {np.mean(samples):.3f}")
                print(f"  Std: {np.std(samples):.3f}")
                print(f"  Min: {np.min(samples):.3f}")
                print(f"  Max: {np.max(samples):.3f}")
            else:
                print(f"- {name}: {samples}")
    
    def _load_saved_state(self):
        """Load saved state from file if available."""
        pass  # No file loading needed
    
    def _save_state(self):
        """Save current state to file."""
        pass  # No file saving needed
