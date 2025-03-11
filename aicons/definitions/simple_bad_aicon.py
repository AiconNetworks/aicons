"""
Simple BadAIcon Implementation

This module provides a clean, simple implementation of BadAIcon that properly uses BayesBrain.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# TFP shortcuts
tfd = tfp.distributions
tfb = tfp.bijectors

# Fix imports - remove action_space imports
from aicons.bayesbrainGPT.bayes_brain import BayesBrain

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
        self.name = name
        self.capabilities = capabilities or []
        self.type = "bad"
        
        # Create an empty BayesBrain
        self.brain = BayesBrain()
        
        # Initialize other attributes
        self.campaigns = {}
    
    
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
            tf_dist = tfd.TransformedDistribution(
                distribution=tfd.Normal(loc=float(value)-shift, scale=float(uncertainty)),
                bijector=tfb.Shift(shift=shift) @ tfb.Softplus()
            )
        elif upper_bound is not None:
            # Transformed distribution for upper-bounded variables
            shift = float(upper_bound)
            tf_dist = tfd.TransformedDistribution(
                distribution=tfd.Normal(loc=shift-float(value), scale=float(uncertainty)),
                bijector=tfb.Shift(shift=shift) @ tfb.Scale(-1.0) @ tfb.Softplus()
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
    
    def add_sensor(self, name, sensor_function=None):
        """
        Add a sensor to collect observations for the AIcon.
        
        Args:
            name: Name of the sensor
            sensor_function: Function that returns observations
                (mapping factor names to values or (value, reliability) tuples)
        
        Returns:
            The sensor function for convenience
        """
        if sensor_function is None:
            # Create a default sensor function that returns no data
            sensor_function = lambda env=None: {}
            
        # Initialize perception if not already done
        if not hasattr(self.brain, 'perception'):
            from aicons.bayesbrainGPT.perception.perception import BayesianPerception
            self.brain.perception = BayesianPerception(self.brain)
        
        # Register the sensor with perception
        self.brain.perception.register_sensor(name, sensor_function)
        return sensor_function
    
    def update_from_sensor(self, sensor_name, environment=None):
        """
        Update beliefs based on data from a specific sensor.
        
        Args:
            sensor_name: Name of the sensor to use
            environment: Optional environment data to pass to the sensor
            
        Returns:
            True if update was successful
        """
        if not hasattr(self.brain, 'perception'):
            from aicons.bayesbrainGPT.perception.perception import BayesianPerception
            self.brain.perception = BayesianPerception(self.brain)
            print(f"No sensors registered yet. Add sensors with add_sensor() first.")
            return False
        
        return self.brain.perception.update_from_sensor(sensor_name, environment)
    
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
    
    def sample_allocation(self):
        """
        Sample a random allocation from the action space.
        
        Returns:
            A randomly sampled allocation, or None if no action space is configured
        """
        return self.brain.sample_action()
    
    def find_best_allocation(self, num_samples: int = 100):
        """
        Find the best allocation based on the utility function.
        
        Args:
            num_samples: Number of samples to evaluate
            
        Returns:
            Tuple of (best_allocation, utility)
        """
        return self.brain.find_best_action(num_samples=num_samples)
    
    def perceive_and_decide(self, environment):
        """
        Perceive the environment and make a decision.
        
        Args:
            environment: Environment data
            
        Returns:
            Tuple of (best_action, utility)
        """
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