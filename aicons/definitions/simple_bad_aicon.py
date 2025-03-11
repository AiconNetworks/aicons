"""
Simple BadAIcon Implementation

This module provides a clean, simple implementation of BadAIcon that properly uses BayesBrain.
"""

from typing import Dict, List, Any, Optional
import numpy as np

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
    
    def add_sensor(self, sensor_function):
        """
        Add a sensor function to the BayesBrain.
        
        Args:
            sensor_function: A callable that takes an environment and returns sensor data
        """
        self.brain.add_sensor(sensor_function)
    
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