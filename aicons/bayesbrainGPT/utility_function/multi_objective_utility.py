"""
Multi-objective utility functions for BayesBrainGPT.

This module provides utility functions for scenarios with multiple, possibly competing objectives.
It includes weighted combination approaches and Pareto optimization methods.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Optional, Callable, Tuple

from aicons.bayesbrainGPT.utility_function.utility_base import UtilityFunction, TensorFlowUtilityFunction


class WeightedSumUtility(UtilityFunction, TensorFlowUtilityFunction):
    """
    Utility function that combines multiple objectives through a weighted sum.
    
    This utility calculates a weighted sum of multiple utility functions to balance
    different possibly competing objectives.
    """
    
    def __init__(self, utility_functions: List[UtilityFunction], 
                 weights: List[float] = None,
                 names: List[str] = None):
        """
        Initialize the weighted sum utility function.
        
        Args:
            utility_functions: List of utility functions to combine
            weights: Weights for each utility function (default: equal weights)
            names: Optional names for each objective
        """
        super().__init__(name="Weighted Sum Utility", 
                         description="Calculates a weighted sum of multiple utility functions")
        
        self.utility_functions = utility_functions
        self.n_objectives = len(utility_functions)
        
        # Default to equal weights if not provided
        self.weights = weights if weights is not None else [1.0 / self.n_objectives] * self.n_objectives
        
        # Normalize weights to sum to 1
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        # Set names for each objective
        self.names = names if names is not None else [f"objective_{i+1}" for i in range(self.n_objectives)]
    
    def evaluate(self, action: Dict[str, Any], state_sample: Dict[str, Any]) -> float:
        """
        Evaluate the weighted sum utility for a specific state sample.
        
        Args:
            action: Dictionary with action parameters
            state_sample: Dictionary with state parameters
        
        Returns:
            float: Weighted sum utility value
        """
        weighted_sum = 0.0
        
        # Calculate each utility function and apply weights
        for i, utility_fn in enumerate(self.utility_functions):
            value = utility_fn.evaluate(action, state_sample)
            weighted_sum += value * self.weights[i]
        
        return weighted_sum
    
    def evaluate_tf(self, action: tf.Tensor, 
                   state_samples: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        TensorFlow implementation of the weighted sum utility.
        
        Args:
            action: Tensor with action parameters
            state_samples: Dictionary with state parameter tensors
                
        Returns:
            Tensor of weighted sum utility values
        """
        # Initialize with zeros tensor of appropriate shape
        # We'll determine shape from the first utility function
        first_util = self.utility_functions[0].evaluate_tf(action, state_samples)
        weighted_sum = tf.zeros_like(first_util)
        
        # Calculate each utility function and apply weights
        for i, utility_fn in enumerate(self.utility_functions):
            if hasattr(utility_fn, 'evaluate_tf'):
                value = utility_fn.evaluate_tf(action, state_samples)
                weighted_sum += value * self.weights[i]
            else:
                raise ValueError(f"Utility function {self.names[i]} does not have evaluate_tf method")
        
        return weighted_sum
    
    def get_component_values(self, action: Dict[str, Any], 
                           state_sample: Dict[str, Any]) -> Dict[str, float]:
        """
        Get individual component utility values for analysis.
        
        Args:
            action: Dictionary with action parameters
            state_sample: Dictionary with state parameters
            
        Returns:
            Dictionary mapping objective names to their utility values
        """
        component_values = {}
        
        for i, utility_fn in enumerate(self.utility_functions):
            name = self.names[i]
            value = utility_fn.evaluate(action, state_sample)
            component_values[name] = value
        
        return component_values
    
    def update_weights(self, new_weights: List[float]) -> None:
        """
        Update the weights for the component utility functions.
        
        Args:
            new_weights: New weights for each utility function
        """
        if len(new_weights) != self.n_objectives:
            raise ValueError(f"Expected {self.n_objectives} weights, got {len(new_weights)}")
        
        # Normalize weights to sum to 1
        total_weight = sum(new_weights)
        self.weights = [w / total_weight for w in new_weights]


class ParetoUtility(UtilityFunction):
    """
    Utility function based on Pareto dominance principles.
    
    Instead of combining objectives, this utility maintains the multi-dimensional
    nature of the objective space and identifies Pareto-optimal solutions.
    """
    
    def __init__(self, utility_functions: List[UtilityFunction],
                 aggregation_method: str = 'chebyshev',
                 reference_point: List[float] = None,
                 names: List[str] = None):
        """
        Initialize the Pareto utility function.
        
        Args:
            utility_functions: List of utility functions to combine
            aggregation_method: Method to aggregate multiple objectives:
                - 'chebyshev': Minimize maximum distance from reference point
                - 'hypervolume': Maximize hypervolume dominated by solutions
            reference_point: Reference point for Chebyshev distance (default: origin)
            names: Optional names for each objective
        """
        super().__init__(name="Pareto Utility", 
                         description="Evaluates actions based on Pareto dominance")
        
        self.utility_functions = utility_functions
        self.n_objectives = len(utility_functions)
        self.aggregation_method = aggregation_method
        
        # Set reference point
        self.reference_point = reference_point
        if reference_point is None:
            # Default to zeros
            self.reference_point = [0.0] * self.n_objectives
        
        # Set names for each objective
        self.names = names if names is not None else [f"objective_{i+1}" for i in range(self.n_objectives)]
        
        # Pre-calculated objective ranges for normalization
        self.obj_min = [float('-inf')] * self.n_objectives
        self.obj_max = [float('inf')] * self.n_objectives
        self.ranges_calibrated = False
    
    def evaluate(self, action: Dict[str, Any], state_sample: Dict[str, Any]) -> float:
        """
        Evaluate the Pareto utility for a specific state sample.
        
        Args:
            action: Dictionary with action parameters
            state_sample: Dictionary with state parameters
        
        Returns:
            float: Scalar utility based on the chosen aggregation method
        """
        # Get vector of objective values
        objective_values = self._get_objective_vector(action, state_sample)
        
        # Apply aggregation method
        if self.aggregation_method == 'chebyshev':
            return self._chebyshev_distance(objective_values)
        elif self.aggregation_method == 'hypervolume':
            return self._hypervolume_indicator(objective_values)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
    
    def _get_objective_vector(self, action: Dict[str, Any], 
                            state_sample: Dict[str, Any]) -> np.ndarray:
        """Get vector of all objective values."""
        objective_values = np.zeros(self.n_objectives)
        
        for i, utility_fn in enumerate(self.utility_functions):
            objective_values[i] = utility_fn.evaluate(action, state_sample)
        
        return objective_values
    
    def _normalize_objectives(self, objective_values: np.ndarray) -> np.ndarray:
        """Normalize objectives to [0,1] range based on observed min/max."""
        if not self.ranges_calibrated:
            # Temporary normalization (might not be ideal)
            return objective_values
        
        normalized = np.zeros_like(objective_values)
        for i in range(self.n_objectives):
            range_i = self.obj_max[i] - self.obj_min[i]
            if range_i > 0:
                normalized[i] = (objective_values[i] - self.obj_min[i]) / range_i
            else:
                normalized[i] = objective_values[i]
        
        return normalized
    
    def _chebyshev_distance(self, objective_values: np.ndarray) -> float:
        """
        Calculate negative Chebyshev distance to reference point.
        
        Negative because we want to maximize utility but minimize distance.
        """
        # Normalize objectives
        normalized = self._normalize_objectives(objective_values)
        ref_point = np.array(self.reference_point)
        
        # Calculate distance (use negative since we want to maximize utility)
        chebyshev_dist = -np.max(np.abs(normalized - ref_point))
        return float(chebyshev_dist)
    
    def _hypervolume_indicator(self, objective_values: np.ndarray) -> float:
        """
        Simplified hypervolume indicator relative to reference point.
        
        For a single solution, this is just the volume of the hypercube.
        """
        # Normalize objectives
        normalized = self._normalize_objectives(objective_values)
        ref_point = np.array(self.reference_point)
        
        # Take product of distances from reference point
        # Only consider dimensions where we're better than reference
        diffs = normalized - ref_point
        positive_diffs = np.maximum(0, diffs)
        
        # Product of positive differences (hypervolume)
        hypervolume = np.prod(positive_diffs + 1e-10)  # Small epsilon to avoid zero
        
        return float(hypervolume)
    
    def calibrate_ranges(self, action_samples: List[Dict[str, Any]], 
                        state_samples: List[Dict[str, Any]]) -> None:
        """
        Calibrate objective ranges for normalization.
        
        Args:
            action_samples: List of sample actions
            state_samples: List of sample states
        """
        # Initialize min/max values
        self.obj_min = [float('inf')] * self.n_objectives
        self.obj_max = [float('-inf')] * self.n_objectives
        
        # Evaluate each action-state pair
        for action in action_samples:
            for state in state_samples:
                obj_values = self._get_objective_vector(action, state)
                
                # Update min/max values
                for i in range(self.n_objectives):
                    self.obj_min[i] = min(self.obj_min[i], obj_values[i])
                    self.obj_max[i] = max(self.obj_max[i], obj_values[i])
        
        self.ranges_calibrated = True
    
    def is_pareto_dominated(self, action1: Dict[str, Any], action2: Dict[str, Any], 
                         state_sample: Dict[str, Any]) -> bool:
        """
        Check if action1 is Pareto-dominated by action2.
        
        Action1 is dominated if action2 is at least as good in all objectives
        and strictly better in at least one objective.
        
        Args:
            action1, action2: Two actions to compare
            state_sample: State sample for evaluation
            
        Returns:
            True if action1 is dominated by action2, False otherwise
        """
        obj1 = self._get_objective_vector(action1, state_sample)
        obj2 = self._get_objective_vector(action2, state_sample)
        
        # Check if action2 is at least as good as action1 in all objectives
        at_least_as_good = np.all(obj2 >= obj1)
        
        # Check if action2 is strictly better in at least one objective
        strictly_better = np.any(obj2 > obj1)
        
        return at_least_as_good and strictly_better
    
    def find_pareto_front(self, action_set: List[Dict[str, Any]], 
                        state_sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find the Pareto-optimal front from a set of actions.
        
        Args:
            action_set: List of candidate actions
            state_sample: State sample for evaluation
            
        Returns:
            List of non-dominated actions (Pareto front)
        """
        pareto_front = []
        
        for i, action in enumerate(action_set):
            dominated = False
            
            # Check if this action is dominated by any other action
            for other_action in action_set:
                if other_action is action:
                    continue
                    
                if self.is_pareto_dominated(action, other_action, state_sample):
                    dominated = True
                    break
            
            if not dominated:
                pareto_front.append(action)
        
        return pareto_front


class ConstrainedMultiObjectiveUtility(WeightedSumUtility):
    """
    Multi-objective utility with additional constraints.
    
    Extends the weighted sum approach with hard and soft constraints
    that can penalize or reject actions that violate constraints.
    """
    
    def __init__(self, utility_functions: List[UtilityFunction], 
                 weights: List[float] = None,
                 names: List[str] = None,
                 hard_constraints: List[Callable[[Dict[str, Any], Dict[str, Any]], bool]] = None,
                 soft_constraints: List[Callable[[Dict[str, Any], Dict[str, Any]], float]] = None,
                 soft_penalty_weight: float = 100.0):
        """
        Initialize constrained multi-objective utility.
        
        Args:
            utility_functions: List of utility functions to combine
            weights: Weights for each utility function
            names: Optional names for each objective
            hard_constraints: List of functions that return True if constraint is satisfied
            soft_constraints: List of functions that return penalty values (higher = worse)
            soft_penalty_weight: Weight for soft constraint penalties
        """
        super().__init__(utility_functions, weights, names)
        self.description = "Multi-objective utility with constraints"
        
        # Constraints
        self.hard_constraints = hard_constraints or []
        self.soft_constraints = soft_constraints or []
        self.soft_penalty_weight = soft_penalty_weight
    
    def evaluate(self, action: Dict[str, Any], state_sample: Dict[str, Any]) -> float:
        """
        Evaluate constrained multi-objective utility.
        
        Args:
            action: Dictionary with action parameters
            state_sample: Dictionary with state parameters
        
        Returns:
            float: Utility value with constraint handling
        """
        # Check hard constraints first
        for constraint in self.hard_constraints:
            if not constraint(action, state_sample):
                return float('-inf')  # Infeasible action
        
        # Get base utility
        base_utility = super().evaluate(action, state_sample)
        
        # Apply soft constraints as penalties
        penalty = 0.0
        for constraint in self.soft_constraints:
            penalty += constraint(action, state_sample)
        
        # Subtract penalty from base utility
        return base_utility - self.soft_penalty_weight * penalty


class AdaptiveWeightUtility(WeightedSumUtility):
    """
    Weighted sum utility with adaptive weight adjustment.
    
    Automatically adjusts weights based on the relative performance
    of different objectives in previous evaluations.
    """
    
    def __init__(self, utility_functions: List[UtilityFunction], 
                 initial_weights: List[float] = None,
                 names: List[str] = None,
                 adaptation_rate: float = 0.1,
                 target_ratios: List[float] = None,
                 window_size: int = 10):
        """
        Initialize adaptive weight utility.
        
        Args:
            utility_functions: List of utility functions to combine
            initial_weights: Initial weights for each utility function
            names: Optional names for each objective
            adaptation_rate: Rate at which to adjust weights (0-1)
            target_ratios: Target contribution ratios for each objective
            window_size: Number of evaluations to consider for adaptation
        """
        super().__init__(utility_functions, initial_weights, names)
        self.description = "Multi-objective utility with adaptive weights"
        
        self.adaptation_rate = adaptation_rate
        self.n_objectives = len(utility_functions)
        
        # Target ratios for objective contributions
        self.target_ratios = target_ratios
        if target_ratios is None:
            # Default to equal contributions
            self.target_ratios = [1.0 / self.n_objectives] * self.n_objectives
        
        # Normalize target ratios
        sum_targets = sum(self.target_ratios)
        self.target_ratios = [t / sum_targets for t in self.target_ratios]
        
        # History for weight adaptation
        self.window_size = window_size
        self.history = []  # Stores raw component values
    
    def evaluate(self, action: Dict[str, Any], state_sample: Dict[str, Any]) -> float:
        """
        Evaluate adaptive weight utility and update history.
        
        Args:
            action: Dictionary with action parameters
            state_sample: Dictionary with state parameters
        
        Returns:
            float: Weighted sum utility value
        """
        # Get raw component values
        component_values = []
        for utility_fn in self.utility_functions:
            component_values.append(utility_fn.evaluate(action, state_sample))
        
        # Store in history
        self.history.append(component_values)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        # Adapt weights if we have enough history
        if len(self.history) >= self.window_size:
            self._adapt_weights()
        
        # Calculate weighted sum with current weights
        weighted_sum = 0.0
        for i, value in enumerate(component_values):
            weighted_sum += value * self.weights[i]
        
        return weighted_sum
    
    def _adapt_weights(self) -> None:
        """
        Adapt weights based on historical objective values.
        
        Adjusts weights to try to achieve target contribution ratios.
        """
        # Calculate average contribution of each objective
        avg_values = np.zeros(self.n_objectives)
        for values in self.history:
            avg_values += np.abs(np.array(values))
        avg_values /= len(self.history)
        
        # Calculate current contribution ratios
        total_contribution = np.sum(avg_values)
        if total_contribution == 0:
            return  # Avoid division by zero
            
        current_ratios = avg_values / total_contribution
        
        # Calculate adjustment factor based on ratio differences
        adjustment = np.zeros(self.n_objectives)
        for i in range(self.n_objectives):
            if current_ratios[i] > 0:
                # If current ratio is higher than target, reduce weight
                # If lower, increase weight
                target = self.target_ratios[i]
                current = current_ratios[i]
                adjustment[i] = (target / current - 1.0)
        
        # Apply adjustment with adaptation rate
        new_weights = np.array(self.weights) * (1.0 + self.adaptation_rate * adjustment)
        
        # Ensure non-negative weights
        new_weights = np.maximum(0.0, new_weights)
        
        # Normalize weights to sum to 1
        if np.sum(new_weights) > 0:
            new_weights /= np.sum(new_weights)
            self.weights = new_weights.tolist()
    
    def reset_adaptation(self) -> None:
        """Reset adaptation history and revert to initial weights."""
        self.history = [] 