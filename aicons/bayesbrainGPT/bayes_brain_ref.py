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
import tensorflow as tf
import tensorflow_probability as tfp

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
    
    def __init__(self, name: str = None, description: str = None):
        """
        Initialize the Bayesian brain.
        
        Args:
            name: Name of the brain instance
            description: Optional description of this brain
        """
        self.name = name
        self.description = description
        self.state = BayesianState()
        self.perception = BayesianPerception(self)
        self.action_space = ActionSpace([])
        self.utility_function = None
        self.aicon = None
        self.last_action = None
        self.last_utility = None
        self.sensors = []  # Initialize sensors list
        
        # HMC configuration
        self.hmc_config = {
            'num_results': 1000,
            'num_burnin_steps': 500,
            'step_size': 0.01,  # Reduced step size for better acceptance
            'num_leapfrog_steps': 5,  # Reduced number of steps
            'target_accept_prob': 0.65,  # Lower target acceptance rate
            'use_bijectors': True,  # Enable bijectors for constrained sampling
            'constraint_bijectors': {
                'lower_bound': tfp.bijectors.Exp(),  # For positive-only variables
                'upper_bound': tfp.bijectors.Sigmoid(),  # For bounded variables
                'both_bounds': tfp.bijectors.Sigmoid()  # For variables with both bounds
            }
        }
        
        # Decision parameters
        self.decision_params = {}
        
        # Single uncertainty value for the entire brain
        self.uncertainty = 0.0  # Default uncertainty
        self.uncertainty_trigger = None  # What caused the last uncertainty change
    
    def set_aicon(self, aicon: Any) -> None:
        """
        Set the AIcon this brain belongs to.
        
        Args:
            aicon: The AIcon instance
        """
        self.aicon = aicon
    
    def find_best_action(self, num_samples: Optional[int] = None) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Find the best action given the current state and beliefs.
        
        Args:
            num_samples: Optional number of samples to use for evaluation
            
        Returns:
            Tuple of (best_action, expected_utility)
        """
        start_time = time.time()
        print("\n=== Starting find_best_action ===")
        
        if not self.action_space or not self.utility_function:
            raise ValueError("Action space and utility function must be set before finding best action")
            
        # Get posterior samples from perception
        posterior_samples = self.state.get_posterior_samples(num_samples)
        if not posterior_samples:
            posterior_samples = self.state.get_prior_samples(num_samples)
        
        # Print sample statistics
        print("\nPosterior Sample Statistics:")
        for name, samples in posterior_samples.items():
            if isinstance(samples, np.ndarray):
                print(f"- {name}:")
                print(f"  Shape: {samples.shape}")
                print(f"  Mean: {np.mean(samples):.3f}")
                print(f"  Std: {np.std(samples):.3f}")
                print(f"  Min: {np.min(samples):.3f}")
                print(f"  Max: {np.max(samples):.3f}")
            else:
                print(f"- {name}: {samples}")
        
        # Convert posterior samples to tensors with proper shape
        posterior_tensors = {}
        for name, samples in posterior_samples.items():
            if isinstance(samples, np.ndarray):
                # Ensure samples are float32 and have the correct shape
                posterior_tensors[name] = tf.constant(samples, dtype=tf.float32)
            else:
                # For scalar values, create a tensor with the same shape as other samples
                sample_shape = next(iter(posterior_tensors.values())).shape if posterior_tensors else (1,)
                posterior_tensors[name] = tf.fill(sample_shape, float(samples))
        
        # Get all possible actions from action space dimensions
        possible_actions = self.action_space.enumerate_actions()
        total_actions = len(possible_actions)
        print(f"\nFound {total_actions} possible actions to evaluate")
        
        # Initialize action utilities dictionary
        action_utilities = {}
        
        # Evaluate each possible action
        best_action = None
        best_utility = float('-inf')
        last_progress = -1
        
        for i, action in enumerate(possible_actions):
            # Show progress every 1% or at least every 100 actions
            progress = int((i / total_actions) * 100)
            if progress != last_progress and (progress % 1 == 0 or i % 100 == 0):
                print(f"\rEvaluating actions: {i+1}/{total_actions} ({progress}%)", end="")
                last_progress = progress
            
            # Convert action values to float32
            action_float32 = {k: float(v) for k, v in action.items()}
            
            # Convert action to tensor format
            if hasattr(self.utility_function, 'dimensions') and self.utility_function.dimensions is not None:
                action_tensor = tf.constant([action_float32.get(dim.name, 0.0) for dim in self.utility_function.dimensions])
            else:
                # Try to identify budget values from keys
                budget_values = [v for k, v in action_float32.items() if k.endswith('_budget')]
                if budget_values:
                    action_tensor = tf.constant(budget_values)
                else:
                    # Fallback to all numeric values in the action
                    numeric_values = [v for k, v in action_float32.items() 
                                     if isinstance(v, (int, float))]
                    action_tensor = tf.constant(numeric_values) if numeric_values else tf.constant([0.0])
            
            # Calculate utility using evaluate_tf
            utility_tensor = self.utility_function.evaluate_tf(action_tensor, posterior_tensors)
            utility = float(tf.reduce_mean(utility_tensor))
            
            # Store utility
            action_key = tuple(sorted(action.items()))
            action_utilities[action_key] = utility
            
            # Check if this is the best action so far
            if utility > best_utility:
                best_utility = utility
                best_action = action
                print(f"\nNew best action found with utility {utility:.2f}")
        
        end_time = time.time()
        print(f"\n\nTotal time taken: {end_time - start_time:.2f} seconds")
        
        if best_action is None:
            print("WARNING: No valid actions found!")
            return None, float('-inf')
        
        print("\n=== Best Action Selected ===")
        print(f"Action: {best_action}")
        print(f"Expected utility: {best_utility}")
        
        return best_action, best_utility
    
    def compute_action_utilities(self, state: Dict[str, Any], num_samples: int) -> Dict[str, float]:
        """
        Compute expected utilities for all possible actions.
        
        Args:
            state: Current state
            num_samples: Number of samples to use
            
        Returns:
            Dictionary mapping action keys to their expected utilities
        """
        action_utilities = {}
        
        # Get posterior samples once for all actions
        posterior_samples = self.state.get_posterior_samples(num_samples)
        
        # Get all possible actions from action space dimensions
        possible_actions = self.action_space.enumerate_actions()
        
        # Evaluate each possible action
        for action in possible_actions:
            # Create a single sample dictionary with current values
            sample = {}
            for name, samples in posterior_samples.items():
                if isinstance(samples, np.ndarray):
                    # If it's an array, use the first value
                    sample[name] = float(samples[0])
                else:
                    # If it's a scalar value, use it directly
                    sample[name] = float(samples)
            
            # Use evaluate method
            try:
                utility = self.utility_function.evaluate(action, sample)
                # Convert action to tuple for use as key
                action_key = tuple(sorted(action.items()))
                action_utilities[action_key] = utility
            except Exception as e:
                raise ValueError(f"Error computing utility: {str(e)}")
            
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
        
        # Initialize sensors list if not already done
        if not hasattr(self, 'sensors'):
            self.sensors = []
        
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
        
        # Store the sensor in the brain's sensors list
        self.sensors.append(sensor)
        
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
        print(f"Updating from sensor: {sensor_name}")
        
        # Get current posterior samples (from time t)
        current_samples = self.state.get_posterior_samples()
        if current_samples:
            print(f"\nUsing current posterior as prior (from {self.state.last_posterior_update})")
            print("Current posterior samples:")
            for name, samples in current_samples.items():
                if isinstance(samples, np.ndarray):
                    print(f"  {name}:")
                    print(f"    Mean: {np.mean(samples):.4f}")
                    print(f"    Std: {np.std(samples):.4f}")
        else:
            print("\nNo current posterior samples available, will use prior")
        
        print("\n=== Collecting Sensor Data ===")
        # Update with new sensor data
        success = self.perception.update_from_sensor(sensor_name, environment)
        print(f"Initial sensor update success: {success}")
        
        if success:
            # Get new sensor data
            print("\n=== Getting New Sensor Data ===")
            new_sensor_data = self.perception.collect_sensor_data(environment)
            print(f"Collected sensor data: {new_sensor_data}")
            
            if not new_sensor_data:
                print("WARNING: No new sensor data available")
                return False
            
            # Generate new posterior samples for time t+1
            print("\n=== Computing Posterior ===")
            print("Input observations for posterior:")
            for factor, (value, reliability) in new_sensor_data.items():
                print(f"  {factor}: value={value:.4f}, reliability={reliability:.2f}")
            
            updated_samples = self.perception.sample_posterior(new_sensor_data)
            
            if not updated_samples:
                print("ERROR: Failed to generate posterior samples")
                return False
            
            print("\n=== Posterior Sampling Results ===")
            for name, samples in updated_samples.items():
                if isinstance(samples, np.ndarray):
                    print(f"  {name}:")
                    print(f"    Mean: {np.mean(samples):.4f}")
                    print(f"    Std: {np.std(samples):.4f}")
                    print(f"    Min: {np.min(samples):.4f}")
                    print(f"    Max: {np.max(samples):.4f}")
            
            # Store new posterior samples (time t+1) in brain
            print("\n=== Updating Brain State ===")
            self.state.set_posterior_samples(updated_samples)
            
            # Update state factors with new posterior values
            print("\n=== Updating State Factors ===")
            self.perception.update_state_from_posterior()
            
            # Record update in history
            self.state.update_history.append({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'sensor': sensor_name,
                'values': {name: float(np.mean(samples)) if isinstance(samples, np.ndarray) else samples 
                          for name, samples in updated_samples.items()}
            })
            
            print(f"\nSuccessfully updated beliefs at {self.state.last_posterior_update}")
            print(f"Total updates in history: {len(self.state.update_history)}")
            return True
        
        print("\nWARNING: Sensor update failed")
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
    
    def _load_saved_state(self):
        """Load saved state from file if available."""
        pass  # No file loading needed
    
    def _save_state(self):
        """Save current state to file."""
        pass  # No file saving needed
    
    def set_hmc_config(self, config: Dict[str, Any]) -> None:
        """
        Update the HMC configuration parameters.
        
        Args:
            config: Dictionary containing HMC parameters to update. Valid keys are:
                   - num_results: Number of samples to generate
                   - num_burnin_steps: Number of burn-in steps
                   - step_size: Step size for HMC
                   - num_leapfrog_steps: Number of leapfrog steps
                   - target_accept_prob: Target acceptance probability
        """
        # Update only the provided parameters
        for key, value in config.items():
            if key in self.hmc_config:
                self.hmc_config[key] = value
                print(f"Updated HMC parameter {key} to {value}")
            else:
                print(f"Warning: Ignoring unknown HMC parameter {key}")
    
    def update_uncertainty(self, new_uncertainty: float, trigger: str):
        """
        Update the brain's overall uncertainty and record what triggered the change.
        
        Args:
            new_uncertainty: New uncertainty value
            trigger: What caused the uncertainty change (e.g. "low_acceptance", "sampling_failed")
        """
        self.uncertainty = new_uncertainty
        self.uncertainty_trigger = trigger
        
        # Record in history
        self.state.update_history.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "type": "uncertainty_update",
            "new_uncertainty": new_uncertainty,
            "trigger": trigger
        })
