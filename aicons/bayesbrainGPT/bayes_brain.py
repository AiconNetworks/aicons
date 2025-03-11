"""
BayesBrain Module for Bayesian Decision Making

This module provides the BayesBrain class, which integrates:
1. State representation - for maintaining beliefs about the world
2. Sensors - for gathering data from the environment
3. Perception - for updating beliefs based on sensor data
4. Action space - for defining possible actions
5. Utility functions - for evaluating actions
6. Decision making - for selecting optimal actions
"""

from typing import Dict, Any, Callable, Optional, Tuple, List
import numpy as np

class BayesBrain:
    """
    A Bayesian decision-making brain that integrates state representation,
    sensors, perception, action spaces, and utility functions.
    
    The BayesBrain maintains a probabilistic belief state about the world,
    updates this belief based on new data, and makes decisions by selecting
    actions that maximize expected utility.
    """
    def __init__(self):
        """Initialize an empty BayesBrain"""
        # Action space component
        self.action_space = None
        
        # State representation component
        self.state_factors = {}
        
        # Perception component (posterior samples)
        self.posterior_samples = {}
        
        # Utility function component
        self.utility_function = None
        
        # Sensors component
        self.sensors = []
        
        # Decision making parameters
        self.decision_params = {
            "num_samples": 1000,
            "exploration_rate": 0.1
        }
    
    # Action space methods
    def set_action_space(self, action_space):
        """
        Set the action space for decision-making
        
        Args:
            action_space: An ActionSpace object defining the possible actions
        """
        self.action_space = action_space
    
    def get_action_space(self):
        """Get the current action space"""
        return self.action_space
    
    # State representation methods
    def set_state_factors(self, state_factors: Dict[str, Any]):
        """
        Set the state factors for belief representation
        
        Args:
            state_factors: Dictionary of state factors
        """
        self.state_factors = state_factors
    
    def get_state_factors(self):
        """Get the current state factors"""
        return self.state_factors
    
    def update_state_factor(self, factor_name: str, value: Any):
        """
        Update a specific state factor
        
        Args:
            factor_name: Name of the factor to update
            value: New value for the factor
        """
        if factor_name in self.state_factors:
            self.state_factors[factor_name] = value
    
    # Perception methods
    def set_posterior_samples(self, posterior_samples: Dict[str, np.ndarray]):
        """
        Set the posterior samples for Bayesian inference
        
        Args:
            posterior_samples: Dictionary of posterior samples for different parameters
        """
        self.posterior_samples = posterior_samples
    
    def get_posterior_samples(self):
        """Get the current posterior samples"""
        return self.posterior_samples
    
    def update_posterior_samples(self, new_data: Dict[str, Any], update_function: Callable):
        """
        Update posterior samples based on new data
        
        Args:
            new_data: New data to update the posterior samples with
            update_function: Function that takes (posterior_samples, new_data) and returns updated samples
        """
        if not self.posterior_samples:
            return
        
        self.posterior_samples = update_function(self.posterior_samples, new_data)
    
    # Utility function methods
    def set_utility_function(self, utility_function: Callable):
        """
        Set the utility function for decision-making
        
        Args:
            utility_function: A function that takes an action and returns a utility value
        """
        self.utility_function = utility_function
    
    def get_utility_function(self):
        """Get the current utility function"""
        return self.utility_function
    
    # Sensor methods
    def add_sensor(self, sensor: Callable):
        """
        Add a sensor function that can gather data from the environment
        
        Args:
            sensor: A function that takes the current state and returns sensor data
        """
        self.sensors.append(sensor)
    
    def get_sensors(self):
        """Get the current list of sensors"""
        return self.sensors
    
    def collect_sensor_data(self, environment: Any) -> Dict[str, Any]:
        """
        Collect data from all sensors
        
        Args:
            environment: The environment to collect data from
            
        Returns:
            Dictionary of sensor data
        """
        sensor_data = {}
        for sensor in self.sensors:
            sensor_data.update(sensor(environment))
        return sensor_data
    
    # Decision making methods
    def sample_action(self):
        """
        Sample an action from the action space
        
        Returns:
            A randomly sampled action from the action space, or None if no action space is set
        """
        if self.action_space is None:
            return None
        return self.action_space.sample()
    
    def find_best_action(self, num_samples: Optional[int] = None) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Find the best action using the utility function
        
        Args:
            num_samples: Number of samples to try when searching for the best action
            
        Returns:
            Tuple of (best_action, best_utility), where best_action is None if no valid action is found
        """
        if self.action_space is None or self.utility_function is None:
            return None, 0.0
        
        if num_samples is None:
            num_samples = self.decision_params["num_samples"]
        
        best_action = None
        best_utility = float('-inf')
        
        # Try a reasonable number of samples to find a good action
        num_samples = min(num_samples, self.action_space.get_size() if hasattr(self.action_space, 'get_size') else num_samples)
        
        for _ in range(num_samples):
            action = self.action_space.sample()
            utility = self.utility_function(action)
            
            if utility > best_utility:
                best_utility = utility
                best_action = action
        
        return best_action, best_utility
    
    def set_decision_params(self, params: Dict[str, Any]):
        """
        Set parameters for the decision-making process
        
        Args:
            params: Dictionary of decision parameters
        """
        self.decision_params.update(params)
    
    def get_decision_params(self):
        """Get the current decision parameters"""
        return self.decision_params
    
    # Full perception-decision cycle
    def perceive_and_decide(self, environment: Any) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Run a full perception-decision cycle:
        1. Collect sensor data from the environment
        2. Update posterior samples based on sensor data
        3. Find the best action using the updated beliefs
        
        Args:
            environment: The environment to perceive
            
        Returns:
            Tuple of (best_action, best_utility)
        """
        # Collect sensor data
        sensor_data = self.collect_sensor_data(environment)
        
        # Update posterior samples
        if sensor_data and self.posterior_samples:
            def default_update(samples, data):
                # Simple default update function
                # In a real implementation, this would be replaced with proper Bayesian inference
                return samples
            
            self.update_posterior_samples(sensor_data, default_update)
        
        # Find the best action
        return self.find_best_action() 