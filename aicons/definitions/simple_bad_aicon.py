"""
Simple BadAIcon Implementation

This module provides a clean, simple implementation of BadAIcon that properly uses BayesBrain.
"""

from typing import Dict, List, Any, Optional, Union, Callable
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import time
import uuid
import os
from datetime import datetime

# TFP shortcuts
tfd = tfp.distributions
tfb = tfp.bijectors

# Fix imports - add action_space and utility_function imports
from aicons.bayesbrainGPT.bayes_brain import BayesBrain
from aicons.bayesbrainGPT.decision_making.action_space import (
    ActionSpace, 
    create_budget_allocation_space,
    create_time_budget_allocation_space,
    create_multi_campaign_action_space,
    create_marketing_ads_space
)
from aicons.bayesbrainGPT.utility_function.utility_function import (TensorFlowUtilityFunction,
    create_utility_function, create_custom_marketing_utility
)

class SimpleBadAIcon:
    """
    A simple implementation of Budget Allocation Decision (BAD) AIcon
    that properly uses BayesBrain.
    """
    
    def __init__(self, name: str, capabilities: List[str] = None):
        """
        Initialize a SimpleBadAIcon with an empty BayesBrain.
        
        Args:
            name: Name of the AIcon
            capabilities: List of capabilities (optional)
        """
        # Generate a unique identifier for this AIcon
        self.id = str(uuid.uuid4())
        
        self.name = name
        self.capabilities = capabilities or []
        self.type = "bad"
        
        # Create an empty BayesBrain
        self.brain = BayesBrain()
        
        # Initialize other attributes
        self.campaigns = {}
        
        # Running state
        self.is_running = False
        self.run_stats = {
            "iterations": 0,
            "start_time": None,
            "last_update_time": None,
            "updates": []
        }
        
        # Persistence manager (initialized when needed)
        self._persistence = None
    
    
    def add_factor_continuous(self, name: str, value: float, uncertainty: float = 0.1, 
                              lower_bound: Optional[float] = None, upper_bound: Optional[float] = None,
                              description: str = ""):
        """
        Add a continuous factor with TensorFlow distribution.
        
        Args:
            name: Factor name
            value: Initial value (mean)
            uncertainty: Standard deviation
            lower_bound: Optional lower bound
            upper_bound: Optional upper bound
            description: Optional description
        """
        print(f"Creating continuous factor with TensorFlow: {name}")
        
        # Create constraints dictionary
        constraints = {}
        if lower_bound is not None:
            constraints["lower"] = lower_bound
        if upper_bound is not None:
            constraints["upper"] = upper_bound
            
        # Create TensorFlow distribution directly
        tf_dist = None
        if lower_bound is not None and upper_bound is not None:
            # Truncated normal for bounded variables
            tf_dist = tfd.TruncatedNormal(
                loc=float(value), 
                scale=float(uncertainty),
                low=float(lower_bound),
                high=float(upper_bound)
            )
        elif lower_bound is not None:
            # Transformed distribution for lower-bounded variables
            shift = float(lower_bound)
            # Fix: Use tfb.Chain instead of @ operator for composing bijectors
            bijector = tfb.Chain([tfb.Softplus(), tfb.Shift(shift=shift)])
            tf_dist = tfd.TransformedDistribution(
                distribution=tfd.Normal(loc=float(value)-shift, scale=float(uncertainty)),
                bijector=bijector
            )
        elif upper_bound is not None:
            # Transformed distribution for upper-bounded variables
            shift = float(upper_bound)
            # Fix: Use tfb.Chain instead of @ operator for composing bijectors
            bijector = tfb.Chain([tfb.Softplus(), tfb.Scale(-1.0), tfb.Shift(shift=shift)])
            tf_dist = tfd.TransformedDistribution(
                distribution=tfd.Normal(loc=shift-float(value), scale=float(uncertainty)),
                bijector=bijector
            )
        else:
            # Unconstrained normal distribution
            tf_dist = tfd.Normal(loc=float(value), scale=float(uncertainty))
        
        # Package factor with TensorFlow distribution
        factor = {
            "type": "continuous",
            "distribution": "normal",
            "params": {"loc": float(value), "scale": float(uncertainty)},
            "shape": [],
            "value": float(value),
            "constraints": constraints if constraints else None,
            "description": description or f"Continuous factor: {name}",
            "tf_distribution": tf_dist  # Store the TF distribution directly
        }
        
        # Update the brain's state with this factor
        current_state = self.brain.get_state_factors() or {}
        current_state[name] = factor
        self.brain.set_state_factors(current_state)
        
        print(f"Added continuous factor with TensorFlow distribution: {name}")
        return factor
    
    def add_factor_categorical(self, name: str, value: str, categories: List[str], 
                               probs: Optional[List[float]] = None,
                               description: str = ""):
        """
        Add a categorical factor with TensorFlow distribution.
        
        Args:
            name: Factor name
            value: Current value (must be in categories)
            categories: List of possible categories
            probs: Optional probabilities for each category (should sum to 1)
            description: Optional description
        """
        print(f"Creating categorical factor with TensorFlow: {name}")
        
        if value not in categories:
            raise ValueError(f"Value '{value}' not in provided categories: {categories}")
        
        if probs is None:
            # Equal probability for all categories
            probs = [1.0 / len(categories)] * len(categories)
        
        # Convert probs to tensor
        probs_tensor = tf.constant(probs, dtype=tf.float32)
        
        # Create TensorFlow categorical distribution
        tf_dist = tfd.Categorical(probs=probs_tensor)
        
        factor = {
            "type": "categorical",
            "distribution": "categorical",
            "params": {"probs": probs},
            "categories": categories,
            "value": value,
            "description": description or f"Categorical factor: {name}",
            "tf_distribution": tf_dist  # Store the TF distribution directly
        }
        
        # Update the brain's state with this factor
        current_state = self.brain.get_state_factors() or {}
        current_state[name] = factor
        self.brain.set_state_factors(current_state)
        
        print(f"Added categorical factor with TensorFlow distribution: {name}")
        return factor
    
    def add_factor_discrete(self, name: str, value: int, min_value: int = 0, 
                           max_value: Optional[int] = None,
                           description: str = ""):
        """
        Add a discrete integer factor with TensorFlow distribution.
        
        Args:
            name: Factor name
            value: Current integer value
            min_value: Minimum possible value
            max_value: Maximum possible value
            description: Optional description
        """
        print(f"Creating discrete factor with TensorFlow: {name}")
        
        # Create TensorFlow distribution
        tf_dist = None
        
        if max_value is None:
            # Poisson distribution for unbounded discrete values
            tf_dist = tfd.Poisson(rate=float(value))
            
            factor = {
                "type": "discrete",
                "distribution": "poisson",
                "params": {"rate": float(value)},
                "value": int(value),
                "constraints": {"lower": min_value},
                "description": description or f"Discrete factor: {name}",
                "tf_distribution": tf_dist
            }
        else:
            # Categorical distribution for bounded discrete values
            num_values = max_value - min_value + 1
            categories = list(range(min_value, max_value + 1))
            index = categories.index(value)
            
            probs = [0.0] * num_values
            probs[index] = 1.0
            
            # Convert probs to tensor
            probs_tensor = tf.constant(probs, dtype=tf.float32)
            tf_dist = tfd.Categorical(probs=probs_tensor)
            
            factor = {
                "type": "discrete",
                "distribution": "categorical",
                "params": {"probs": probs},
                "categories": categories,
                "value": int(value),
                "description": description or f"Discrete factor: {name}",
                "tf_distribution": tf_dist
            }
        
        # Update the brain's state with this factor
        current_state = self.brain.get_state_factors() or {}
        current_state[name] = factor
        self.brain.set_state_factors(current_state)
        
        print(f"Added discrete factor with TensorFlow distribution: {name}")
        return factor
    
    def configure_state_factors(self, state_factors: Dict[str, Any] = None):
        """
        Configure the state factors for the BayesBrain.
        
        Args:
            state_factors: Dictionary of state factors (optional)
        """
        if state_factors:
            # Set the state factors directly in the brain
            self.brain.set_state_factors(state_factors)
    
    def configure_utility_function(self, utility_function):
        """
        Configure the utility function for decision-making.
        
        Args:
            utility_function: A callable that takes an action and returns a utility value
        """
        self.brain.set_utility_function(utility_function)
    
    def configure_posterior_samples(self, posterior_samples: Dict[str, np.ndarray]):
        """
        Configure the posterior samples for Bayesian inference.
        
        Args:
            posterior_samples: Dictionary of posterior samples
        """
        self.brain.set_posterior_samples(posterior_samples)
    
    def add_sensor(self, name, sensor=None, factor_mapping=None):
        """
        Add a sensor to collect observations for the AIcon.
        
        Args:
            name: Name of the sensor
            sensor: Sensor object or function that returns observations
                (mapping factor names to values or (value, reliability) tuples)
            factor_mapping: Optional dictionary mapping sensor factor names to state factor names
                For example: {"base_conversion_rate": "conversion_rate"}
        
        Returns:
            The sensor for convenience
        """
        # Initialize perception if not already done
        if not hasattr(self.brain, 'perception'):
            from aicons.bayesbrainGPT.perception.perception import BayesianPerception
            self.brain.perception = BayesianPerception(self.brain)
        
        # Check if sensor is None
        if sensor is None:
            # Try to create a sensor based on name
            from aicons.bayesbrainGPT.sensors.tf_sensors import MarketingSensor, WeatherSensor
            
            if name.lower() in ["marketing", "campaign", "ad"]:
                sensor = MarketingSensor()
            elif name.lower() in ["weather", "weather_station"]:
                sensor = WeatherSensor()
            else:
                # Create a default sensor function that returns no data
                sensor = lambda env=None: {}
        
        # ENHANCEMENT: Auto-create required factors
        # Check if the sensor has a get_expected_factors method
        if hasattr(sensor, 'get_expected_factors'):
            expected_factors = sensor.get_expected_factors()
            current_state = self.brain.get_state_factors() or {}
            
            # Create any missing factors with default values
            for factor_name, factor_info in expected_factors.items():
                # Map the factor name if a mapping exists
                if factor_mapping and factor_name in factor_mapping:
                    mapped_name = factor_mapping[factor_name]
                else:
                    mapped_name = factor_name
                
                # Check if the factor already exists (with either name)
                if mapped_name not in current_state and factor_name not in current_state:
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
                        self.add_factor_continuous(
                            mapped_name, default_value, uncertainty,
                            lower_bound=lower_bound, upper_bound=upper_bound,
                            description=description
                        )
                    elif factor_type == 'categorical' and categories:
                        self.add_factor_categorical(
                            mapped_name, default_value, categories,
                            description=description
                        )
                    elif factor_type == 'discrete':
                        self.add_factor_discrete(
                            mapped_name, default_value, 
                            min_value=lower_bound, max_value=upper_bound,
                            description=description
                        )
        
        # Register the sensor with perception
        self.brain.perception.register_sensor(name, sensor, factor_mapping)
        return sensor
    
    def add_factor_mapping(self, sensor_factor_name, state_factor_name):
        """
        Add a mapping between a sensor factor name and a state factor name.
        This allows sensors and state factors to use different naming conventions.
        
        Args:
            sensor_factor_name: Factor name used by sensors (e.g., "base_conversion_rate")
            state_factor_name: Corresponding factor name in the state (e.g., "conversion_rate")
        """
        if not hasattr(self.brain, 'perception'):
            from aicons.bayesbrainGPT.perception.perception import BayesianPerception
            self.brain.perception = BayesianPerception(self.brain)
            
        self.brain.perception.add_factor_mapping(sensor_factor_name, state_factor_name)
    
    def update_from_sensor(self, sensor_name, environment=None, factor_mapping=None):
        """
        Update beliefs based on data from a specific sensor.
        
        Args:
            sensor_name: Name of the sensor to use
            environment: Optional environment data to pass to the sensor
            factor_mapping: Optional one-time mapping of sensor factor names to state factor names
            
        Returns:
            True if update was successful
        """
        if not hasattr(self.brain, 'perception'):
            from aicons.bayesbrainGPT.perception.perception import BayesianPerception
            self.brain.perception = BayesianPerception(self.brain)
            print(f"No sensors registered yet. Add sensors with add_sensor() first.")
            return False
        
        return self.brain.perception.update_from_sensor(sensor_name, environment, factor_mapping)
    
    def update_from_all_sensors(self, environment=None):
        """
        Update beliefs based on data from all sensors.
        
        Args:
            environment: Optional environment data to pass to sensors
            
        Returns:
            True if update was successful
        """
        if not hasattr(self.brain, 'perception'):
            from aicons.bayesbrainGPT.perception.perception import BayesianPerception
            self.brain.perception = BayesianPerception(self.brain)
            print(f"No sensors registered yet. Add sensors with add_sensor() first.")
            return False
        
        return self.brain.perception.update_all(environment)
    
    def get_posterior_samples(self):
        """
        Get the current posterior samples from the last update.
        
        Returns:
            Dictionary mapping factor names to posterior samples
        """
        if not hasattr(self.brain, 'perception'):
            return {}
        
        return self.brain.perception.posterior_samples
    
    def get_action_space(self):
        """
        Get the action space from the brain.
        
        Returns:
            The brain's action space
        """
        return self.brain.get_action_space()
    
    def get_utility_function(self):
        """
        Get the utility function from the brain.
        
        Returns:
            The brain's utility function
        """
        return self.brain.get_utility_function()
    
    def perceive_and_decide(self, environment):
        """
        Perceive the environment and make a decision.
        
        Args:
            environment: Environment data
            
        Returns:
            Tuple of (best_action, utility)
        """
        # Check for action space and utility function
        if self.get_action_space() is None:
            raise ValueError("Cannot perceive and decide without an action space. Call create_action_space() first.")
            
        if self.get_utility_function() is None:
            raise ValueError("Cannot perceive and decide without a utility function. Call create_utility_function() first.")
            
        return self.brain.perceive_and_decide(environment)
    
    def get_state(self, format_nicely: bool = False):
        """
        Get the current state (prior distributions) from the BayesBrain.
        
        Args:
            format_nicely: Whether to return a formatted human-readable representation
            
        Returns:
            Dictionary of state factors, or a formatted string if format_nicely is True
        """
        state = self.brain.get_state_factors()
        
        if not format_nicely:
            return state
        
        # Format nicely for human readability
        formatted = []
        formatted.append(f"AIcon State ({len(state)} factors):")
        
        for name, factor in state.items():
            factor_type = factor["type"]
            factor_value = factor["value"]
            distribution = factor["distribution"]
            
            # Basic factor info
            factor_str = f"\n{name}:"
            factor_str += f"\n  Type: {factor_type}"
            factor_str += f"\n  Distribution: {distribution}"
            factor_str += f"\n  Current value: {factor_value}"
            
            # Distribution parameters
            params = factor["params"]
            if factor_type == "continuous":
                factor_str += f"\n  Mean: {params.get('loc', 'N/A')}"
                factor_str += f"\n  Uncertainty: {params.get('scale', 'N/A')}"
                
                if "constraints" in factor and factor["constraints"]:
                    constraints = factor["constraints"]
                    bounds = []
                    if "lower" in constraints:
                        bounds.append(f">= {constraints['lower']}")
                    if "upper" in constraints:
                        bounds.append(f"<= {constraints['upper']}")
                    factor_str += f"\n  Constraints: {', '.join(bounds)}"
                    
            elif factor_type == "categorical":
                categories = factor.get("categories", [])
                probs = params.get("probs", [])
                
                # Format categories and their probabilities
                if categories and probs and len(categories) == len(probs):
                    factor_str += "\n  Categories (probability):"
                    for cat, prob in zip(categories, probs):
                        factor_str += f"\n    {cat}: {prob:.2f}"
                else:
                    factor_str += f"\n  Categories: {categories}"
                    factor_str += f"\n  Probabilities: {probs}"
                    
            elif factor_type == "discrete":
                if "categories" in factor:
                    factor_str += f"\n  Possible values: {factor['categories']}"
                elif "rate" in params:
                    factor_str += f"\n  Rate parameter: {params['rate']}"
                    
                if "constraints" in factor and factor["constraints"]:
                    constraints = factor["constraints"]
                    factor_str += f"\n  Constraints: {constraints}"
            
            # Add description if available
            if "description" in factor and factor["description"]:
                factor_str += f"\n  Description: {factor['description']}"
                
            formatted.append(factor_str)
            
        return "\n".join(formatted)
    
    def add_conditional_factor(self, name: str, parent_factor_names: List[str], 
                              conditional_dist_fn: Callable, description: str = ""):
        """
        Add a factor that depends on other factors (hierarchical relationship).
        
        This creates a true hierarchical Bayesian model where one factor's
        distribution depends on the values of other factors.
        
        Args:
            name: Name of the factor
            parent_factor_names: Names of parent factors this factor depends on
            conditional_dist_fn: Function that takes parent values and returns a distribution
            description: Description of what this factor represents
            
        Returns:
            The created factor
        """
        # Delegate to the BayesianState in BayesBrain
        if not hasattr(self.brain, 'state'):
            # Create BayesianState if not already present
            from aicons.bayesbrainGPT.state_representation.bayesian_state import BayesianState
            self.brain.state = BayesianState()
            
        # Check if the BayesianState has the add_conditional_factor method
        if not hasattr(self.brain.state, 'add_conditional_factor'):
            raise NotImplementedError("Hierarchical modeling not supported by the current state implementation")
            
        # Add the conditional factor
        return self.brain.state.add_conditional_factor(
            name=name,
            parent_factor_names=parent_factor_names,
            conditional_dist_fn=conditional_dist_fn,
            description=description
        )
    
    def create_hierarchical_model(self):
        """
        Create a hierarchical Bayesian model from all factors.
        
        This builds a joint probability distribution that respects the
        conditional dependencies between factors.
        
        Returns:
            Joint distribution representing the hierarchical model
        """
        # Delegate to the BayesianState in BayesBrain
        if not hasattr(self.brain, 'state'):
            raise ValueError("No state object found in BayesBrain")
            
        # Check if the BayesianState has the create_joint_distribution method
        if not hasattr(self.brain.state, 'create_joint_distribution'):
            raise NotImplementedError("Hierarchical modeling not supported by the current state implementation")
            
        # Create and return the joint distribution
        return self.brain.state.create_joint_distribution()
    
    def sample_from_hierarchical_prior(self, n_samples=1):
        """
        Sample from the hierarchical prior distribution.
        
        This samples from the joint distribution of all factors, respecting
        the hierarchical relationships between them.
        
        Args:
            n_samples: Number of samples to draw
            
        Returns:
            Dictionary mapping factor names to sample values
        """
        # Delegate to the BayesianState in BayesBrain
        if not hasattr(self.brain, 'state'):
            raise ValueError("No state object found in BayesBrain")
            
        # Check if the BayesianState has the sample_from_prior method
        if not hasattr(self.brain.state, 'sample_from_prior'):
            raise NotImplementedError("Hierarchical sampling not supported by the current state implementation")
            
        # Sample from the prior
        return self.brain.state.sample_from_prior(n_samples)

    def run(self, mode='once', sensor_name=None, interval=60, duration=None, environment=None):
        """
        Run the AIcon's perception system for updates.
        
        Args:
            mode: Run mode - 'once', 'continuous', or 'finite'
                - 'once': Run a single perception update and return
                - 'continuous': Run continuously until stopped
                - 'finite': Run for a specific duration or number of updates
            sensor_name: Name of specific sensor to use (if None, use all sensors)
            interval: Time between updates in seconds (for continuous/finite modes)
            duration: Duration to run in seconds or number of updates (for finite mode)
            environment: Optional environment data to pass to sensors
            
        Returns:
            Dictionary with run statistics
        """
        # First check for action space and utility function
        if self.get_action_space() is None:
            raise ValueError("Cannot run AIcon without an action space. Call create_action_space() first.")
            
        if self.get_utility_function() is None:
            raise ValueError("Cannot run AIcon without a utility function. Call create_utility_function() first.")
            
        # Check if we have perception and sensors
        if not hasattr(self.brain, 'perception'):
            from aicons.bayesbrainGPT.perception.perception import BayesianPerception
            self.brain.perception = BayesianPerception(self.brain)
            print(f"No sensors registered yet. Add sensors with add_sensor() first.")
            return self.run_stats
        
        # Check if there are any state factors defined
        state_factors = self.brain.get_state_factors()
        if not state_factors:
            print(f"No state factors (priors) defined. Add factors before running.")
            return self.run_stats
        
        # Reset run statistics
        self.run_stats = {
            "iterations": 0,
            "start_time": datetime.now(),
            "last_update_time": None,
            "updates": []
        }
        
        # Run once
        if mode == 'once':
            print(f"Running single perception update...")
            start_time = time.time()
            
            # Update from specific sensor or all sensors
            if sensor_name:
                success = self.update_from_sensor(sensor_name, environment)
            else:
                success = self.update_from_all_sensors(environment)
                
            end_time = time.time()
            
            # Record stats
            self.run_stats["iterations"] = 1
            self.run_stats["last_update_time"] = datetime.now()
            self.run_stats["updates"].append({
                "time": datetime.now(),
                "success": success,
                "sensor": sensor_name or "all",
                "duration_sec": end_time - start_time
            })
            
            print(f"Perception update completed. Success: {success}")
            
        # Run continuously or for a finite time
        elif mode in ['continuous', 'finite']:
            print(f"Starting {mode} perception run...")
            start_run_time = time.time()
            self.is_running = True
            iterations = 0
            
            try:
                while self.is_running:
                    # Run perception update
                    start_time = time.time()
                    iterations += 1
                    
                    print(f"\nUpdate #{iterations} at {datetime.now()}")
                    
                    # Update from specific sensor or all sensors
                    if sensor_name:
                        success = self.update_from_sensor(sensor_name, environment)
                    else:
                        success = self.update_from_all_sensors(environment)
                        
                    end_time = time.time()
                    update_duration = end_time - start_time
                    
                    # Record stats
                    self.run_stats["iterations"] = iterations
                    self.run_stats["last_update_time"] = datetime.now()
                    self.run_stats["updates"].append({
                        "time": datetime.now(),
                        "success": success,
                        "sensor": sensor_name or "all",
                        "duration_sec": update_duration
                    })
                    
                    print(f"Update completed in {update_duration:.2f}s. Success: {success}")
                    
                    # Check if we should stop for finite mode
                    if mode == 'finite':
                        if isinstance(duration, int) and iterations >= duration:
                            print(f"Reached {duration} iterations. Stopping.")
                            break
                        elif isinstance(duration, (float, int)) and (time.time() - start_run_time) >= duration:
                            print(f"Reached duration of {duration}s. Stopping.")
                            break
                    
                    # Sleep until next update
                    time.sleep(interval)
                    
            except KeyboardInterrupt:
                print("\nPerception run interrupted by user.")
            finally:
                self.is_running = False
                print(f"Perception run completed after {iterations} iterations.")
                
        else:
            print(f"Unknown run mode: {mode}. Use 'once', 'continuous', or 'finite'.")
            
        return self.run_stats
        
    def stop(self):
        """Stop a running continuous perception process."""
        if self.is_running:
            self.is_running = False
            print("Stopping perception run...")
            return True
        else:
            print("No perception run is currently active.")
            return False

    def validate_budget_allocation(self, total_budget, num_ads, budget_step):
        """
        Validates that budget allocation parameters are compatible.
        
        Args:
            total_budget: Total budget to allocate
            num_ads: Number of ads
            budget_step: Step size for budget allocation
            
        Returns:
            Tuple of (is_valid, message, suggested_fixes)
        """
        is_valid = True
        message = ""
        suggestions = []
        
        # Check if budget is divisible by step size
        if total_budget % budget_step != 0:
            is_valid = False
            message = f"Total budget ${total_budget} is not divisible by step size ${budget_step}"
            
            # Suggest alternative step sizes that would work
            for divisor in [5, 10, 20, 25, 50, 100]:
                if total_budget % divisor == 0:
                    suggestions.append(f"- Change step size to ${divisor}")
            
            # Or suggest rounding the total budget
            rounded_budget = round(total_budget / budget_step) * budget_step
            suggestions.append(f"- Change total budget to ${rounded_budget}")
        
        return is_valid, message, suggestions
    
    def create_action_space(self, space_type: str = 'marketing', **kwargs):
        """
        Create an action space for the AIcon.
        
        Args:
            space_type: Type of action space to create ('marketing', 'budget', 'budget_allocation', 'time_budget', 'multi_campaign')
            **kwargs: Additional arguments for the specific space type
                - For 'budget_allocation':
                  - total_budget: Total budget to allocate
                  - items: List of ad IDs to use
                  - min_allocation_percent: Minimum allocation percentage (0.0-1.0)
                  - max_allocation_percent: Maximum allocation percentage (0.0-1.0)
                  - step_size: Step size for allocation (as a percentage, e.g., 0.05 for 5%)
                  - ad_names: Optional dictionary mapping item IDs to display names
            
        Returns:
            The created ActionSpace
        """
        action_space = None
        
        if space_type == 'marketing':
            # Create a marketing ads space
            total_budget = kwargs.get('total_budget', 1000.0)
            num_ads = kwargs.get('num_ads', 2)
            budget_step = kwargs.get('budget_step', 10.0)
            min_budget = kwargs.get('min_budget', 0.0)
            ad_names = kwargs.get('ad_names', None)
            
            # Validate parameters
            is_valid, message, suggestions = self.validate_budget_allocation(total_budget, num_ads, budget_step)
            if not is_valid and not kwargs.get('ignore_validation', False):
                print(f"⚠️ Warning: {message}")
                print("Suggested fixes:")
                for suggestion in suggestions:
                    print(suggestion)
                print("To ignore this warning, add ignore_validation=True to your function call.")
            
            action_space = create_marketing_ads_space(
                total_budget=total_budget,
                num_ads=num_ads,
                budget_step=budget_step,
                min_budget=min_budget,
                ad_names=ad_names
            )
            
        elif space_type == 'budget_allocation':
            # Create a budget allocation space for specific ad IDs
            total_budget = kwargs.get('total_budget', 1000.0)
            items = kwargs.get('items', [])
            
            # Get allocation parameters
            min_allocation_percent = kwargs.get('min_allocation_percent', 0.0)
            max_allocation_percent = kwargs.get('max_allocation_percent', 1.0)
            step_size = kwargs.get('step_size', 0.05)
            
            # Convert percentages to absolute budget values
            min_budget = min_allocation_percent * total_budget
            budget_step = step_size * total_budget
            
            # Use number of items if provided, otherwise use explicit num_ads
            num_ads = kwargs.get('num_ads', len(items))
            
            # Validate parameters
            is_valid, message, suggestions = self.validate_budget_allocation(total_budget, num_ads, budget_step)
            if not is_valid and not kwargs.get('ignore_validation', False):
                print(f"⚠️ Warning: {message}")
                print("Suggested fixes:")
                for suggestion in suggestions:
                    print(suggestion)
                print("To ignore this warning, add ignore_validation=True to your function call.")
                print("Adjusting step size to ensure valid allocations...")
                
                # Auto-adjust step size to ensure it divides the total budget
                for divisor in [5, 10, 20, 25, 50, 100]:
                    if total_budget % divisor == 0:
                        budget_step = divisor
                        step_size = divisor / total_budget
                        print(f"Adjusted step size to ${budget_step} (or {step_size*100:.1f}% of total)")
                        break
            
            # Create the budget allocation space
            action_space = create_budget_allocation_space(
                total_budget=total_budget,
                num_ads=num_ads,
                budget_step=budget_step,
                min_budget=min_budget
            )
            
            # Store item IDs for later reference
            action_space.item_ids = items
            
            # Store ad names if provided
            ad_names = kwargs.get('ad_names', None)
            if ad_names:
                action_space.ad_names = ad_names
            
        elif space_type == 'budget':
            # Create a simple budget allocation space
            total_budget = kwargs.get('total_budget', 1000.0)
            num_ads = kwargs.get('num_ads', 2)
            budget_step = kwargs.get('budget_step', 100.0)
            min_budget = kwargs.get('min_budget', 0.0)
            
            # Validate parameters
            is_valid, message, suggestions = self.validate_budget_allocation(total_budget, num_ads, budget_step)
            if not is_valid and not kwargs.get('ignore_validation', False):
                print(f"⚠️ Warning: {message}")
                print("Suggested fixes:")
                for suggestion in suggestions:
                    print(suggestion)
                print("To ignore this warning, add ignore_validation=True to your function call.")
            
            action_space = create_budget_allocation_space(
                total_budget=total_budget,
                num_ads=num_ads,
                budget_step=budget_step,
                min_budget=min_budget
            )
            
        elif space_type == 'time_budget':
            # Create a time-based budget allocation space
            total_budget = kwargs.get('total_budget', 1000.0)
            num_ads = kwargs.get('num_ads', 2)
            num_days = kwargs.get('num_days', 3)
            budget_step = kwargs.get('budget_step', 100.0)
            min_budget = kwargs.get('min_budget', 0.0)
            
            # Validate parameters
            is_valid, message, suggestions = self.validate_budget_allocation(total_budget, num_ads * num_days, budget_step)
            if not is_valid and not kwargs.get('ignore_validation', False):
                print(f"⚠️ Warning: {message}")
                print("Suggested fixes:")
                for suggestion in suggestions:
                    print(suggestion)
                print("To ignore this warning, add ignore_validation=True to your function call.")
            
            action_space = create_time_budget_allocation_space(
                total_budget=total_budget,
                num_ads=num_ads,
                num_days=num_days,
                budget_step=budget_step,
                min_budget=min_budget
            )
            
        elif space_type == 'multi_campaign':
            # Create a multi-campaign action space
            campaigns = kwargs.get('campaigns', {})
            budget_step = kwargs.get('budget_step', 100.0)
            
            action_space = create_multi_campaign_action_space(
                campaigns=campaigns,
                budget_step=budget_step
            )
            
        elif space_type == 'custom':
            # Create a custom action space
            dimensions = kwargs.get('dimensions', [])
            constraints = kwargs.get('constraints', [])
            
            action_space = ActionSpace(
                dimensions=dimensions,
                constraints=constraints
            )
            
        else:
            raise ValueError(f"Unknown action space type: {space_type}")
            
        # Store the action space in the brain
        self.brain.set_action_space(action_space)
        
        # Get name if provided, otherwise use type
        name = kwargs.get('name', f"{space_type}_space")
        print(f"Created {space_type} action space '{name}' with {len(action_space.dimensions)} dimensions")
        
        return action_space
    
    def create_utility_function(self, utility_type: str = 'marketing_roi', **kwargs):
        """
        Create a utility function for the AIcon.
        
        Args:
            utility_type: Type of utility function to create
                - 'marketing_roi': Standard marketing ROI utility
                - 'constrained_marketing_roi': ROI with business constraints
                - 'weather_dependent_marketing_roi': ROI affected by weather
                - 'weighted_sum': Multi-objective weighted sum utility
                - 'pareto': Pareto-optimality based utility
            **kwargs: Parameters for the specific utility function
            
        Returns:
            The created utility function
        """
        try:
            # Get the action space if it exists
            action_space = self.get_action_space()
            
            # Add action space dimensions if they exist
            if action_space:
                kwargs['num_ads'] = len([d for d in action_space.dimensions 
                                        if d.name.endswith('_budget')])
                kwargs['ad_names'] = [d.name.replace('_budget', '') 
                                    for d in action_space.dimensions 
                                    if d.name.endswith('_budget')]
            
            # Create the utility function using the factory
            utility_function = create_utility(utility_type, action_space=action_space, **kwargs)
            
            # Store in brain
            self.brain.utility_function = utility_function
            
            print(f"Created {utility_type} utility function: {utility_function.name}")
            print(f"Description: {utility_function.description}")
            
            return utility_function
            
        except Exception as e:
            print(f"Could not create utility function: {e}")
            
            # If utility_type is 'help', log available utility types
            if utility_type == 'help':
                print(f"Available utility types: {list(UTILITY_FACTORIES.keys())}")
                return None
                
            # Fallback to a simple utility function
            if utility_type == 'marketing_roi':
                def simple_utility(action, state_sample):
                    # Simple profit calculation (20% ROI)
                    total_budget = sum(value for key, value in action.items() if key.endswith('_budget'))
                    return total_budget * 0.2
                    
                # Use LambdaUtility from the new module
                from aicons.bayesbrainGPT.utility_function.custom_utility import LambdaUtility
                utility_function = LambdaUtility(
                    evaluation_fn=simple_utility,
                    name="Simple Marketing ROI Utility",
                    description="Fallback utility function with fixed 20% ROI"
                )
                
                self.brain.utility_function = utility_function
                return utility_function
            else:
                print(f"No fallback available for utility type: {utility_type}")
                return None
    
    def find_best_action(self, num_samples: int = 100, use_gradient: bool = False):
        """
        Find the best action based on the utility function.
        
        Args:
            num_samples: Number of samples to evaluate
            use_gradient: Whether to use gradient-based optimization (for TensorFlow utility)
            
        Returns:
            Tuple of (best_allocation, utility)
        """
        action_space = self.get_action_space()
        utility_function = self.get_utility_function()
        
        if action_space is None:
            raise ValueError("Cannot find best action without an action space. Call create_action_space() first.")
            
        if utility_function is None:
            raise ValueError("Cannot find best action without a utility function. Call create_utility_function() first.")
            
        # Delegate to the brain's implementation
        return self.brain.find_best_action(num_samples=num_samples, use_gradient=use_gradient)
    
    def sample_action(self):
        """
        Sample a random action from the action space.
        
        Returns:
            A randomly sampled action, or None if no action space is configured
        """
        action_space = self.get_action_space()
        if action_space is None:
            raise ValueError("Cannot sample action without an action space. Call create_action_space() first.")
            
        return self.brain.sample_action()
    
    def create_hierarchical_model_tf(self, num_ads=2, num_days=3):
        """
        Create a TensorFlow-based hierarchical Bayesian model for ad performance.
        This model can be used for both inference and decision-making.
        
        Args:
            num_ads: Number of ads
            num_days: Number of days
            
        Returns:
            Joint distribution representing the model
        """
        # Create joint distribution
        joint_dist = tfd.JointDistributionNamed({
            # Conversion rates for each ad
            "phi": tfd.Independent(
                tfd.Normal(loc=tf.ones(num_ads)*0.05,
                           scale=tf.ones(num_ads)*0.01),
                reinterpreted_batch_ndims=1
            ),
            # Cost per click for each ad
            "c": tfd.Independent(
                tfd.Gamma(concentration=tf.ones(num_ads)*5.0,
                          rate=tf.ones(num_ads)*7.0),
                reinterpreted_batch_ndims=1
            ),
            # Day multipliers
            "delta": tfd.Independent(
                tfd.LogNormal(loc=tf.zeros(num_days), scale=tf.ones(num_days)*0.3),
                reinterpreted_batch_ndims=1
            ),
            # Observation noise
            "sigma": tfd.HalfNormal(scale=1.0)
        })
        
        # Store the model for use in perception and decision-making
        self.tf_model = joint_dist
        
        # Create a matching action space if needed
        if self.get_action_space() is None:
            self.create_action_space(
                space_type='marketing',
                total_budget=1000.0,
                num_ads=num_ads,
                budget_step=10.0
            )
            
        # Create a matching utility function if needed
        if self.get_utility_function() is None:
            self.create_utility_function(
                utility_type='marketing_roi',
                revenue_per_sale=10.0,
                num_days=num_days
            )
            
        return joint_dist 

    def get_action_dimensions(self):
        """
        Get information about the dimensions of the action space.
        
        Returns:
            A dictionary with information about the action space dimensions,
            or None if no action space has been created yet.
        """
        # Delegate to BayesBrain's method
        dimensions_info = self.brain.get_action_dimensions()
        
        if dimensions_info is None:
            print("No action space has been created yet. Call create_action_space() first.")
            
        return dimensions_info 
        
    def get_action_space_details(self):
        """
        Get detailed information about the action space configuration.
        
        Returns:
            A dictionary containing detailed information about the action space, including:
            - type: The type of action space
            - dimensions: List of dimension names
            - constraints: List of constraint descriptions
            - total_budget: The total budget if it's a budget allocation space
            - min_budget: The minimum budget allowed per dimension
            - step_size: The budget step size
            - is_discrete: Whether the action space is discrete
        """
        action_space = self.get_action_space()
        if action_space is None:
            print("No action space has been created yet. Call create_action_space() first.")
            return None
            
        # Basic info all action spaces have
        details = {
            "dimensions": [dim.name for dim in action_space.dimensions],
            "num_dimensions": len(action_space.dimensions),
            "constraints": [str(c) for c in action_space.constraints] if hasattr(action_space, 'constraints') else [],
            "is_discrete": action_space.is_discrete if hasattr(action_space, 'is_discrete') else False
        }
        
        # Get budget-specific attributes that might exist
        budget_attrs = ['total_budget', 'min_budget', 'budget_step', 'item_ids', 'ad_names']
        for attr in budget_attrs:
            if hasattr(action_space, attr):
                details[attr] = getattr(action_space, attr)
        
        # Try to infer action space type
        if all(dim.name.endswith('_budget') for dim in action_space.dimensions):
            if hasattr(action_space, 'item_ids'):
                details["type"] = "budget_allocation"
            else:
                details["type"] = "marketing"
        else:
            details["type"] = "custom"
            
        return details
        
    def call_action_space_method(self, method_name, **kwargs):
        """
        Call a method directly on the action space object.
        
        This allows direct access to action space methods without needing to access
        the action space object directly.
        
        Args:
            method_name: Name of the method to call on the action space
            **kwargs: Arguments to pass to the method
            
        Returns:
            The result of calling the method, or None if the action space doesn't exist
            or doesn't have the requested method
        """
        action_space = self.get_action_space()
        if action_space is None:
            print("No action space has been created yet. Call create_action_space() first.")
            return None
            
        if not hasattr(action_space, method_name):
            print(f"Action space does not have method '{method_name}'")
            print(f"Available methods: {[m for m in dir(action_space) if not m.startswith('_')]}")
            return None
            
        method = getattr(action_space, method_name)
        if not callable(method):
            print(f"'{method_name}' is an attribute, not a method")
            return method
            
        try:
            return method(**kwargs)
        except Exception as e:
            print(f"Error calling {method_name}: {str(e)}")
            return None

    # Persistence methods
    def connect_to_db(self, db_connection_string: str = None):
        """
        Connect to the database for persistence.
        
        Args:
            db_connection_string: Optional database connection string
            
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            from aicons.bayesbrainGPT.persistence.persistence import AIconPersistence
            self._persistence = AIconPersistence(db_connection_string)
            
            # Initialize the schema if needed
            self._persistence.initialize_schema()
            
            return True
        except Exception as e:
            import logging
            logging.error(f"Failed to connect to database: {e}")
            return False
    
    def save_state(self, name: str = None, db_connection_string: str = None, save_pickle: bool = True) -> bool:
        """
        Save the current state of the AIcon to the database.
        
        Args:
            name: Optional new name for the AIcon (uses current name if None)
            db_connection_string: Optional database connection string
            save_pickle: Whether to save a pickle representation for complex objects
            
        Returns:
            True if save is successful, False otherwise
        """
        if name:
            self.name = name
            
        try:
            # Initialize persistence if not already done
            if self._persistence is None:
                self.connect_to_db(db_connection_string)
                
            if self._persistence is None:
                raise ValueError("No database connection available")
                
            # Save the AIcon state
            aicon_id = self._persistence.save_aicon(self, save_pickle=save_pickle)
            
            if aicon_id:
                print(f"AIcon state saved successfully with ID: {aicon_id}")
                return True
            else:
                print("Failed to save AIcon state")
                return False
                
        except Exception as e:
            import logging
            logging.error(f"Error saving AIcon state: {e}")
            print(f"Error saving AIcon state: {e}")
            return False
    
    def load_state(self, aicon_id: str = None, db_connection_string: str = None) -> bool:
        """
        Load the state of the AIcon from the database.
        
        Args:
            aicon_id: ID of the AIcon to load (uses current ID if None)
            db_connection_string: Optional database connection string
            
        Returns:
            True if load is successful, False otherwise
        """
        try:
            # Use current ID if none provided
            if aicon_id is None:
                aicon_id = self.id
                
            # Initialize persistence if not already done
            if self._persistence is None:
                self.connect_to_db(db_connection_string)
                
            if self._persistence is None:
                raise ValueError("No database connection available")
                
            # Load the AIcon data
            aicon_data = self._persistence.load_aicon(aicon_id)
            
            if not aicon_data:
                print(f"No AIcon found with ID: {aicon_id}")
                return False
                
            # Update basic attributes
            self.name = aicon_data.get('name', self.name)
            self.type = aicon_data.get('config', {}).get('type', self.type)
            self.capabilities = aicon_data.get('config', {}).get('capabilities', self.capabilities)
            
            # Restore brain from pickle if available
            if 'brain_pickle' in aicon_data:
                self.brain = aicon_data['brain_pickle']
            # Otherwise restore brain state from JSON
            elif 'state' in aicon_data and 'brain' in aicon_data['state']:
                brain_data = aicon_data['state']['brain']
                
                # Restore state factors
                if 'state_factors' in brain_data:
                    self.brain.set_state_factors(brain_data['state_factors'])
                
                # Restore posterior samples
                if 'posterior_samples' in brain_data:
                    posterior_samples = {}
                    for k, v in brain_data['posterior_samples'].items():
                        if isinstance(v, list):
                            posterior_samples[k] = np.array(v)
                        else:
                            posterior_samples[k] = v
                    self.brain.set_posterior_samples(posterior_samples)
                
                # Restore decision parameters
                if 'decision_params' in brain_data:
                    self.brain.set_decision_params(brain_data['decision_params'])
            
            # Restore campaigns
            if 'state' in aicon_data and 'campaigns' in aicon_data['state']:
                self.campaigns = aicon_data['state']['campaigns']
                
            # Restore run stats
            if 'state' in aicon_data and 'run_stats' in aicon_data['state']:
                self.run_stats = aicon_data['state']['run_stats']
                
            # Restore running state
            if 'state' in aicon_data and 'is_running' in aicon_data['state']:
                self.is_running = aicon_data['state']['is_running']
                
            print(f"AIcon state loaded successfully from ID: {aicon_id}")
            return True
                
        except Exception as e:
            import logging
            logging.error(f"Error loading AIcon state: {e}")
            print(f"Error loading AIcon state: {e}")
            return False
    
    @classmethod
    def load_from_db(cls, aicon_id: str, db_connection_string: str = None):
        """
        Create a new AIcon instance and load its state from the database.
        
        Args:
            aicon_id: ID of the AIcon to load
            db_connection_string: Optional database connection string
            
        Returns:
            A new SimpleBadAIcon instance with the loaded state, or None if loading failed
        """
        try:
            # Initialize persistence
            from aicons.bayesbrainGPT.persistence.persistence import AIconPersistence
            persistence = AIconPersistence(db_connection_string)
            
            # Load the AIcon data
            aicon_data = persistence.load_aicon(aicon_id)
            
            if not aicon_data:
                print(f"No AIcon found with ID: {aicon_id}")
                return None
                
            # Create a new instance
            name = aicon_data.get('name', f"LoadedAIcon_{aicon_id[:8]}")
            aicon = cls(name)
            
            # Set the ID
            aicon.id = aicon_id
            
            # Load the state
            aicon._persistence = persistence
            aicon.load_state(aicon_id)
            
            return aicon
                
        except Exception as e:
            import logging
            logging.error(f"Error loading AIcon: {e}")
            print(f"Error loading AIcon: {e}")
            return None 