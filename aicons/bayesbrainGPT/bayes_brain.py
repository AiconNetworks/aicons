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
import uuid
import os

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
        # Brain ID
        self.id = str(uuid.uuid4())
        
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
    
    def find_best_action(self, num_samples: Optional[int] = None, use_gradient: bool = False) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Find the best action according to the utility function
        
        Args:
            num_samples: Number of actions to sample (for Monte Carlo methods)
            use_gradient: Whether to use gradient-based optimization (for TensorFlow utility)
            
        Returns:
            Tuple of (best_action, best_utility), where best_action is None if no valid action is found
        """
        if self.action_space is None:
            return None, 0.0
            
        if self.utility_function is None:
            return None, 0.0
            
        if num_samples is None:
            num_samples = self.decision_params["num_samples"]
        
        # Get posterior samples - handle both direct access and perception component
        posterior_samples = {}
        if hasattr(self, 'posterior_samples') and self.posterior_samples:
            posterior_samples = self.posterior_samples
        elif hasattr(self, 'perception') and hasattr(self.perception, 'posterior_samples'):
            posterior_samples = self.perception.posterior_samples
            
        # Check if this is a TensorFlow utility
        is_tensorflow_utility = hasattr(self.utility_function, 'evaluate_tf')
        
        # If we have a TensorFlow utility and posterior samples
        if is_tensorflow_utility and posterior_samples:
            import tensorflow as tf
            
            # Convert posterior samples to TensorFlow format
            tf_samples = {}
            for param_name, samples in posterior_samples.items():
                if isinstance(samples, np.ndarray):
                    tf_samples[param_name] = tf.convert_to_tensor(samples, dtype=tf.float32)
                else:
                    tf_samples[param_name] = samples
                    
            # Use TensorFlow optimization if requested and action space supports it
            if use_gradient and hasattr(self.action_space, 'is_discrete') and not self.action_space.is_discrete:
                # Check if action_space has the optimize_action_tf method
                if hasattr(self.action_space, 'optimize_action_tf'):
                    best_action = self.action_space.optimize_action_tf(
                        self.utility_function.evaluate_tf, tf_samples, num_steps=100
                    )
                    # Calculate utility for the best action
                    if hasattr(self.utility_function, 'evaluate_tf'):
                        action_tensor = tf.constant([best_action[dim.name] for dim in self.action_space.dimensions])
                        utility = tf.reduce_mean(self.utility_function.evaluate_tf(action_tensor, tf_samples)).numpy()
                        return best_action, utility
            
            # Use action space's TensorFlow evaluation method if available
            if hasattr(self.action_space, 'evaluate_actions_tf'):
                return self.action_space.evaluate_actions_tf(
                    self.utility_function.evaluate_tf, tf_samples, num_actions=num_samples
                )
                
            # Otherwise use vectorized batch evaluation
            sampled_actions = [self.action_space.sample() for _ in range(num_samples)]
            try:
                # Check if evaluate_tf_batch exists before calling it
                if hasattr(self.utility_function, 'evaluate_tf_batch'):
                    # Try using the TensorFlow batch evaluation
                    utility_values = self.utility_function.evaluate_tf_batch(
                        sampled_actions, 
                        posterior_samples=tf_samples
                    )
                    best_index = tf.argmax(utility_values).numpy()
                    return sampled_actions[best_index], float(utility_values[best_index])
                else:
                    # Fall back to evaluate_tf with individual evaluation
                    utilities = []
                    for action in sampled_actions:
                        # Convert action to tensor
                        action_tensor = tf.constant([action[dim.name] for dim in self.action_space.dimensions])
                        # Calculate utility
                        utility = tf.reduce_mean(self.utility_function.evaluate_tf(action_tensor, tf_samples))
                        utilities.append(utility)
                    # Find best action
                    best_index = tf.argmax(tf.stack(utilities)).numpy()
                    return sampled_actions[best_index], float(utilities[best_index])
            except Exception as e:
                # Fallback to individual evaluation
                import logging
                logging.warning(f"Error in TF evaluation, falling back to standard: {e}")
        
        # Use action space's evaluate_actions method if available
        if hasattr(self.action_space, 'evaluate_actions'):
            return self.action_space.evaluate_actions(
                self.utility_function.evaluate, posterior_samples, num_actions=num_samples
            )
        
        # Default implementation - sample and evaluate
        best_action = None
        best_utility = float('-inf')
        
        for _ in range(num_samples):
            action = self.action_space.sample()
            utility = 0.0
            
            # If we have a method-based utility function
            if hasattr(self.utility_function, 'evaluate'):
                # If we have posterior samples, use expected_utility method
                if posterior_samples:
                    if hasattr(self.utility_function, 'expected_utility'):
                        utility = self.utility_function.expected_utility(action, posterior_samples)
                    else:
                        # Compute expected utility over posterior samples
                        all_utilities = []
                        for i in range(len(next(iter(posterior_samples.values())))):
                            # Extract the i-th sample for each parameter
                            sample = {k: v[i] for k, v in posterior_samples.items()}
                            all_utilities.append(self.utility_function.evaluate(action, sample))
                        utility = sum(all_utilities) / len(all_utilities)
                else:
                    # Use state factors directly if no posterior samples
                    utility = self.utility_function.evaluate(action, self.state_factors)
            # If we have a callable utility function 
            elif callable(self.utility_function):
                if posterior_samples:
                    # Compute expected utility over posterior samples
                    all_utilities = []
                    for i in range(len(next(iter(posterior_samples.values())))):
                        # Extract the i-th sample for each parameter
                        sample = {k: v[i] for k, v in posterior_samples.items()}
                        all_utilities.append(self.utility_function(action, sample))
                    utility = sum(all_utilities) / len(all_utilities)
                else:
                    # Use state factors directly if no posterior samples
                    utility = self.utility_function(action, self.state_factors)
            
            if utility > best_utility:
                best_utility = utility
                best_action = action
        
        return best_action, best_utility
    
    def set_decision_params(self, params: Dict[str, Any]):
        """
        Set parameters for decision-making
        
        Args:
            params: Dictionary of decision parameters
        """
        self.decision_params.update(params)
    
    def get_decision_params(self):
        """Get the current decision parameters"""
        return self.decision_params
    
    def perceive_and_decide(self, environment: Any) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Perceive the environment and decide on the best action
        
        Args:
            environment: The environment to perceive
            
        Returns:
            Tuple of (best_action, best_utility), where best_action is None if no valid action is found
        """
        # Collect sensor data
        sensor_data = self.collect_sensor_data(environment)
        
        # If we have sensor data, update posterior samples
        if sensor_data:
            # Define a default update function if not provided
            def default_update(samples, data):
                # Simple default update function
                # In a real implementation, this would be replaced with proper Bayesian inference
                return samples
            
            # Update posterior samples
            self.update_posterior_samples(sensor_data, default_update)
        
        # Find the best action
        return self.find_best_action()
    
    def get_action_dimensions(self) -> Dict[str, Any]:
        """
        Get the dimensions of the action space
        
        Returns:
            Dictionary describing the dimensions of the action space, or an empty dict if no action space is set
        """
        if self.action_space is None:
            return {}
        
        return getattr(self.action_space, 'dimensions', {})
        
    # Persistence methods
    def save(self, persistence_manager=None, db_connection_string=None):
        """
        Save the brain state using the provided persistence manager
        
        Args:
            persistence_manager: Optional AIconPersistence instance
            db_connection_string: Optional database connection string
            
        Returns:
            The brain ID if successful, None otherwise
        """
        if persistence_manager is None:
            # Import here to avoid circular imports
            from aicons.bayesbrainGPT.persistence.persistence import AIconPersistence
            persistence_manager = AIconPersistence(db_connection_string)
        
        try:
            # Create a wrapper object to hold the brain for saving
            brain_container = type('BrainContainer', (), {
                'brain': self,
                'name': f"Brain_{self.id[:8]}",
                'id': self.id
            })
            
            return persistence_manager.save_aicon(brain_container)
        except Exception as e:
            import logging
            logging.error(f"Failed to save brain: {e}")
            return None
    
    @classmethod
    def load(cls, brain_id, persistence_manager=None, db_connection_string=None):
        """
        Load a brain from the database
        
        Args:
            brain_id: ID of the brain to load
            persistence_manager: Optional AIconPersistence instance
            db_connection_string: Optional database connection string
            
        Returns:
            The loaded BayesBrain instance, or None if loading failed
        """
        if persistence_manager is None:
            # Import here to avoid circular imports
            from aicons.bayesbrainGPT.persistence.persistence import AIconPersistence
            persistence_manager = AIconPersistence(db_connection_string)
        
        try:
            # Load the brain data
            brain_container = persistence_manager.load_aicon(brain_id)
            if brain_container and 'brain_pickle' in brain_container:
                return brain_container['brain_pickle']
            
            # If there's no pickle, create a new brain and restore its state
            brain = cls()
            brain.id = brain_id
            
            if brain_container and 'state' in brain_container and 'brain' in brain_container['state']:
                # Extract and restore key components from the saved state
                brain_data = brain_container['state']['brain']
                
                # Restore state factors
                if 'state_factors' in brain_data:
                    brain.state_factors = brain_data['state_factors']
                
                # Restore posterior samples
                if 'posterior_samples' in brain_data:
                    posterior_samples = {}
                    for k, v in brain_data['posterior_samples'].items():
                        if isinstance(v, list):
                            posterior_samples[k] = np.array(v)
                        else:
                            posterior_samples[k] = v
                    brain.posterior_samples = posterior_samples
                
                # Restore decision parameters
                if 'decision_params' in brain_data:
                    brain.decision_params = brain_data['decision_params']
            
            return brain
        except Exception as e:
            import logging
            logging.error(f"Failed to load brain: {e}")
            return None 