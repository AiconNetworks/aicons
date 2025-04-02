"""
Action Space Module for BayesBrainGPT

This module provides structures for defining action spaces in Bayesian decision making.
It allows configuration of multi-dimensional action spaces with different shapes and types.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Union, Optional, Sequence, Callable
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tabulate import tabulate
import json
import itertools


class ActionDimension:
    """
    Represents a single dimension in an action space.
    
    This can be discrete (a set of options) or continuous (a range of values).
    """
    
    def __init__(
        self, 
        name: str,
        dim_type: str,  # 'discrete' or 'continuous'
        values: Optional[Sequence] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        step: Optional[float] = None
    ):
        """
        Initialize a dimension of the action space.
        
        Args:
            name: Name of the dimension
            dim_type: Type of dimension - 'discrete' or 'continuous'
            values: For discrete dimensions, the possible values
            min_value: For continuous dimensions, the minimum value
            max_value: For continuous dimensions, the maximum value
            step: For continuous dimensions with steps, the step size
        """
        self.name = name
        self.dim_type = dim_type.lower()
        
        if self.dim_type not in ['discrete', 'continuous']:
            raise ValueError(f"Dimension type must be 'discrete' or 'continuous', got {dim_type}")
        
        # Set up discrete dimension
        if self.dim_type == 'discrete':
            if values is None:
                raise ValueError("Discrete dimensions must provide 'values'")
            self.values = list(values)
            self.size = len(self.values)
        
        # Set up continuous dimension
        else:  # continuous
            if min_value is None or max_value is None:
                raise ValueError("Continuous dimensions must provide 'min_value' and 'max_value'")
            self.min_value = min_value
            self.max_value = max_value
            self.step = step
            
            # Calculate size for stepped continuous dimensions
            if step is not None:
                self.size = int(np.ceil((max_value - min_value) / step)) + 1
            else:
                self.size = float('inf')  # Infinite size for truly continuous dimensions
    
    def sample(self) -> Any:
        """Sample a random value from this dimension."""
        if self.dim_type == 'discrete':
            return np.random.choice(self.values)
        else:  # continuous
            if self.step is not None:
                # For stepped continuous, sample one of the steps
                num_steps = self.size - 1  # -1 because we include both endpoints
                step_idx = np.random.randint(0, num_steps + 1)
                return self.min_value + step_idx * self.step
            else:
                # For truly continuous, sample uniformly
                return np.random.uniform(self.min_value, self.max_value)
    
    def contains(self, value: Any) -> bool:
        """Check if a value is within this dimension."""
        if self.dim_type == 'discrete':
            return value in self.values
        else:  # continuous
            if self.step is not None:
                # Check if it's a valid step
                if value < self.min_value or value > self.max_value:
                    return False
                
                # Check if it's aligned with a step
                steps_from_min = (value - self.min_value) / self.step
                return np.isclose(steps_from_min, round(steps_from_min))
            else:
                # Simple range check for truly continuous
                return self.min_value <= value <= self.max_value
    
    def enumerate_values(self) -> List[Any]:
        """Enumerate all possible values in this dimension."""
        if self.dim_type == 'discrete':
            return self.values
        else:  # continuous
            if self.step is not None:
                # Return all steps
                return [self.min_value + i * self.step for i in range(self.size)]
            else:
                # For truly continuous, return a reasonable discretization
                return list(np.linspace(self.min_value, self.max_value, 10))
    
    def to_tf_variable(self):
        """Convert to TensorFlow variable for optimization."""
        if self.dim_type == 'discrete':
            # For discrete, return a categorical distribution
            return tfp.distributions.Categorical(probs=tf.ones(len(self.values))/len(self.values))
        else:  # continuous
            # For continuous, return uniform distribution
            if self.step is not None:
                # For stepped continuous, use categorical over steps
                steps = self.enumerate_values()
                return tfp.distributions.Categorical(probs=tf.ones(len(steps))/len(steps))
            else:
                # For truly continuous, use uniform
                return tfp.distributions.Uniform(low=self.min_value, high=self.max_value)


class ActionSpace:
    """
    Multi-dimensional action space for Bayesian decision making.
    
    Can represent complex action spaces with multiple interacting dimensions.
    """
    
    def __init__(self, dimensions: List[ActionDimension], constraints: Optional[List[callable]] = None):
        """
        Initialize a multi-dimensional action space.
        
        Args:
            dimensions: List of ActionDimension objects defining the space
            constraints: Optional list of functions that check if an action is valid
        """
        self.dimensions = dimensions
        self.constraints = constraints or []
        
        # Create a dictionary mapping dimension names to their objects
        self.dimension_map = {dim.name: dim for dim in dimensions}
        
        # Check if the space is fully discrete (all dimensions are discrete)
        self.is_discrete = all(dim.dim_type == 'discrete' for dim in dimensions)
        
        # Compute total size for discrete spaces
        if self.is_discrete:
            self.size = 1
            for dim in dimensions:
                self.size *= dim.size
        else:
            self.size = float('inf')
    
    def sample(self):
        """
        Sample a random action from the action space.
        This ensures all constraints are satisfied.
        
        Returns:
            Dictionary mapping dimension names to values
        """
        # If discrete dimensions, sample uniformly
        if self.is_discrete:
            return {dim.name: dim.sample() for dim in self.dimensions}
        
        # For continuous spaces with constraints, use rejection sampling
        max_attempts = 100
        
        for _ in range(max_attempts):
            # Sample each dimension
            action = {dim.name: dim.sample() for dim in self.dimensions}
            
            # Check if all constraints are satisfied
            if self._check_constraints(action, tolerance=0.01):
                return action
        
        # If we couldn't find a valid sample, raise exception
        raise ValueError(f"Could not find a valid action after {max_attempts} attempts")
    
    def contains(self, action: Dict[str, Any]) -> bool:
        """
        Check if an action is valid within this space.
        
        Args:
            action: Dict mapping dimension names to values
            
        Returns:
            bool: True if the action is valid
        """
        # Check that all required dimensions are present
        if set(action.keys()) != set(self.dimension_map.keys()):
            return False
        
        # Check that each dimension contains its value
        for dim_name, value in action.items():
            if dim_name not in self.dimension_map:
                return False
            
            dimension = self.dimension_map[dim_name]
            if not dimension.contains(value):
                return False
        
        # Check constraints with a small tolerance
        return self._check_constraints(action, tolerance=0.01)
    
    def _check_constraints(self, action, tolerance=0.0):
        """
        Check if an action satisfies all constraints.
        
        Args:
            action: Dict mapping dimension names to values
            tolerance: Optional tolerance for constraints
            
        Returns:
            bool: True if all constraints are satisfied
        """
        for constraint in self.constraints:
            if not constraint(action, tolerance=tolerance):
                return False
        return True
    
    def enumerate_actions(self, max_actions: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Enumerate all possible actions in the space.
        
        Args:
            max_actions: Maximum number of actions to return. If None, return all possible actions.
            
        Returns:
            List of action dictionaries
        """
        if not self.is_discrete:
            # If space has continuous dimensions with step size, enumerate all possible combinations
            if all(hasattr(dim, 'step') and dim.step is not None for dim in self.dimensions):
                # Calculate total number of possible combinations
                total_combinations = 1
                for dim in self.dimensions:
                    num_steps = int((dim.max_value - dim.min_value) / dim.step) + 1
                    total_combinations *= num_steps
                
                # If max_actions is None or total combinations is less than max_actions,
                # enumerate all combinations
                if max_actions is None or total_combinations <= max_actions:
                    actions = []
                    # Generate all possible combinations of values
                    for values in itertools.product(*[
                        np.arange(dim.min_value, dim.max_value + dim.step, dim.step)
                        for dim in self.dimensions
                    ]):
                        action = {dim.name: value for dim, value in zip(self.dimensions, values)}
                        # Check if action satisfies all constraints
                        if self._check_constraints(action):
                            actions.append(action)
                    return actions
                else:
                    # For large spaces, sample randomly
                    return [self.sample() for _ in range(max_actions)]
            else:
                # For truly continuous spaces, sample randomly
                if max_actions is None:
                    max_actions = 1000  # Default to 1000 samples for continuous spaces
                return [self.sample() for _ in range(max_actions)]
        
        # For discrete spaces, enumerate all possible combinations
        if max_actions is None or self.size <= max_actions:
            # Generate all possible combinations
            actions = []
            for values in itertools.product(*[dim.values for dim in self.dimensions]):
                action = {dim.name: value for dim, value in zip(self.dimensions, values)}
                if self._check_constraints(action):
                    actions.append(action)
            return actions
        else:
            # For large discrete spaces, sample randomly
            return [self.sample() for _ in range(max_actions)]
    
    def evaluate_actions(self, utility_fn: Callable[[Dict[str, Any], Dict[str, Any]], float], 
                        posterior_samples: Dict[str, np.ndarray], 
                        num_actions: int = 100) -> Tuple[Dict[str, Any], float]:
        """
        Evaluate actions using a utility function and posterior samples.
        
        Args:
            utility_fn: Function that takes (action, sample) and returns utility
            posterior_samples: Dictionary of posterior samples
            num_actions: Number of actions to evaluate
            
        Returns:
            Tuple of (best_action, expected_utility)
        """
        # Enumerate or sample actions
        actions = self.enumerate_actions(max_actions=num_actions)
        
        best_action = None
        best_utility = float('-inf')
        
        # Calculate expected utility for each action
        for action in actions:
            # Calculate expected utility over all posterior samples
            utilities = []
            for i in range(len(next(iter(posterior_samples.values())))):
                # Extract the i-th sample from each posterior parameter
                sample = {param: values[i] for param, values in posterior_samples.items()}
                
                # Calculate utility for this action and sample
                utility = utility_fn(action, sample)
                utilities.append(utility)
            
            # Average utility across all samples
            expected_utility = np.mean(utilities)
            
            # Check if this is the best action so far
            if expected_utility > best_utility:
                best_utility = expected_utility
                best_action = action
        
        return best_action, best_utility
    
    def to_tf_distributions(self):
        """
        Convert the action space to TensorFlow distributions for optimization.
        
        Returns:
            Dictionary mapping dimension names to TF distributions
        """
        return {dim.name: dim.to_tf_variable() for dim in self.dimensions}
        
    def evaluate_actions_tf(self, utility_fn: Callable, posterior_samples: Dict[str, tf.Tensor], 
                           num_actions: int = 100) -> Tuple[Dict[str, Any], float]:
        """
        Evaluate actions using a TensorFlow utility function and posterior samples.
        
        Args:
            utility_fn: TensorFlow function that computes utility
            posterior_samples: Dictionary of TensorFlow posterior samples tensors
            num_actions: Number of actions to evaluate
            
        Returns:
            Tuple of (best_action, expected_utility)
        """
        # Enumerate or sample actions
        actions = self.enumerate_actions(max_actions=num_actions)
        
        best_action = None
        best_utility = float('-inf')
        
        # Get sample dimension (assuming all posterior samples have same first dimension)
        sample_tensor = next(iter(posterior_samples.values()))
        num_samples = tf.shape(sample_tensor)[0]
        
        # Calculate expected utility for each action
        for action in actions:
            # Convert action to tensor format expected by utility function
            action_tensor = tf.constant([action[dim.name] for dim in self.dimensions])
            
            # Calculate utilities across all samples (vectorized)
            utilities = utility_fn(action_tensor, posterior_samples)
            
            # Calculate expected utility (mean across samples)
            expected_utility = tf.reduce_mean(utilities).numpy()
            
            # Check if this is the best action so far
            if expected_utility > best_utility:
                best_utility = expected_utility
                best_action = action
        
        return best_action, best_utility

    def optimize_action_tf(self, utility_fn: Callable, posterior_samples: Dict[str, tf.Tensor],
                          num_steps: int = 100, learning_rate: float = 0.01) -> Dict[str, Any]:
        """
        Optimize action using TensorFlow gradient descent.
        Works for continuous action spaces.
        
        Args:
            utility_fn: TensorFlow function that computes utility
            posterior_samples: Dictionary of TensorFlow posterior samples
            num_steps: Number of optimization steps
            learning_rate: Learning rate for optimizer
            
        Returns:
            Optimal action dictionary
        """
        if self.is_discrete:
            raise ValueError("Gradient-based optimization not suitable for discrete action spaces")
        
        # Create variables for each dimension
        variables = {}
        for dim in self.dimensions:
            # Initialize in middle of range for continuous dimensions
            if dim.dim_type == 'continuous':
                init_value = (dim.min_value + dim.max_value) / 2
                variables[dim.name] = tf.Variable(init_value, dtype=tf.float32)
        
        # Create optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Optimization loop
        for step in range(num_steps):
            with tf.GradientTape() as tape:
                # Prepare action tensor from variables
                action_tensor = tf.stack([variables[dim.name] for dim in self.dimensions])
                
                # Calculate negative utility (for minimization)
                neg_utility = -tf.reduce_mean(utility_fn(action_tensor, posterior_samples))
            
            # Calculate gradients
            gradients = tape.gradient(neg_utility, list(variables.values()))
            
            # Apply gradients
            optimizer.apply_gradients(zip(gradients, variables.values()))
            
            # Apply constraints (project back to feasible space)
            self._project_to_constraints(variables)
        
        # Convert optimized variables to action dictionary
        action = {name: var.numpy() for name, var in variables.items()}
        
        # Ensure the action satisfies constraints
        if not self._check_constraints(action):
            # If constraints not satisfied, fall back to enumeration
            return self.evaluate_actions_tf(utility_fn, posterior_samples)[0]
        
        return action
    
    def _project_to_constraints(self, variables: Dict[str, tf.Variable]) -> None:
        """Project variables back to satisfy constraints."""
        # Apply bounds for each dimension
        for dim in self.dimensions:
            if dim.dim_type == 'continuous' and dim.name in variables:
                # Clip to bounds
                variables[dim.name].assign(tf.clip_by_value(
                    variables[dim.name],
                    dim.min_value,
                    dim.max_value
                ))

    def get_dimensions_info(self) -> Dict[str, Any]:
        """
        Get structured information about the dimensions in this action space.
        
        Returns:
            A dictionary with information about the dimensions:
            - num_dimensions: Number of dimensions
            - dimension_names: List of dimension names
            - dimension_types: List of dimension types
        """
        return {
            "num_dimensions": len(self.dimensions),
            "dimension_names": [dim.name for dim in self.dimensions],
            "dimension_types": [dim.dim_type for dim in self.dimensions],
        }

    def pprint(self) -> str:
        """
        Pretty print the action space in a human-readable format.
        
        Returns:
            A formatted string representation of the action space
        """
        output = []
        output.append("Action Space:")
        output.append("=" * 50)
        
        # Print dimensions
        output.append("\nDimensions:")
        output.append("-" * 30)
        
        for dim in self.dimensions:
            output.append(f"\nName: {dim.name}")
            output.append(f"Type: {dim.dim_type}")
            if dim.dim_type == 'continuous':
                output.append(f"Range: [{dim.min_value}, {dim.max_value}]")
                output.append(f"Step: {dim.step}")
            elif dim.dim_type == 'discrete':
                output.append(f"Values: {dim.values}")
        
        # Print constraints
        if self.constraints:
            output.append("\nConstraints:")
            output.append("-" * 30)
            for i, constraint in enumerate(self.constraints, 1):
                output.append(f"\nConstraint {i}:")
                if isinstance(constraint, dict):
                    output.append(f"Type: {constraint.get('type', 'unknown')}")
                else:
                    output.append(f"Type: Function")
                    output.append(f"Description: {constraint.__doc__ or 'No description available'}")
        
        return "\n".join(output)
    
    def raw_print(self) -> str:
        """
        Print the action space in raw format.
        
        Returns:
            A raw representation of the action space
        """
        output = []
        
        # Print dimensions
        dims = []
        for dim in self.dimensions:
            if dim.dim_type == 'continuous':
                dims.append(f"{dim.name}: [{dim.min_value}, {dim.max_value}] step={dim.step}")
            elif dim.dim_type == 'discrete':
                dims.append(f"{dim.name}: {dim.values}")
        
        output.append(f"dimensions: {dims}")
        
        # Print size information
        if self.is_discrete:
            output.append(f"size: {self.size} (discrete)")
        else:
            if all(hasattr(dim, 'step') and dim.step is not None for dim in self.dimensions):
                # Calculate total combinations for stepped continuous dimensions
                total_combinations = 1
                for dim in self.dimensions:
                    num_steps = int((dim.max_value - dim.min_value) / dim.step) + 1
                    total_combinations *= num_steps
                output.append(f"size: {total_combinations} (stepped continuous)")
            else:
                output.append("size: infinite (continuous)")
        
        # Print constraints
        constraints = []
        for constraint in self.constraints:
            if isinstance(constraint, dict):
                constraints.append(str(constraint))
            else:
                constraints.append(str(constraint))
        
        output.append(f"constraints: {constraints}")
        
        return "\n".join(output)
    
    def __repr__(self) -> str:
        """Default string representation"""
        return f"ActionSpace(dimensions={len(self.dimensions)}, constraints={len(self.constraints)})"


# Utility functions for creating common action spaces

def create_budget_allocation_space(total_budget: float, num_ads: int, 
                               budget_step: float = 10.0, min_budget: float = 0.0,
                               ad_names: Optional[List[str]] = None) -> ActionSpace:
    """
    Create an action space for allocating budget across multiple ads.
    
    Args:
        total_budget: Total budget to allocate
        num_ads: Number of ads to allocate budget to
        budget_step: Step size for budget allocation
        min_budget: Minimum budget per ad (default is 0.0)
        ad_names: Optional list of ad names to use for dimension names
            
    Returns:
        An ActionSpace instance with dimensions for each ad's budget
    """
    # Create dimensions for each ad's budget
    dimensions = []
    
    for i in range(num_ads):
        # Use provided ad names if available, otherwise use default names
        if ad_names and i < len(ad_names):
            name = f"{ad_names[i]}_budget"
        else:
            name = f"ad_{i+1}_budget"
            
        dimensions.append(
            ActionDimension(
                name=name,
                dim_type="continuous",
                min_value=min_budget,
                max_value=total_budget,
                step=budget_step
            )
        )
    
    # Add constraint that budgets must sum to total_budget
    def budget_sum_constraint(action, tolerance=0.0):
        return np.isclose(sum(action.values()), total_budget, rtol=tolerance)
    
    action_space = ActionSpace(dimensions, constraints=[budget_sum_constraint])
    # Store total_budget as an attribute for reference
    action_space.total_budget = total_budget
    return action_space


def create_time_budget_allocation_space(
    total_budget: float,
    num_ads: int,
    num_days: int = 3,
    budget_step: float = 100.0,
    min_budget: float = 0.0
) -> ActionSpace:
    """
    Create an action space for allocating budget across multiple ads and days.
    
    Args:
        total_budget: Total budget to allocate
        num_ads: Number of ads
        num_days: Number of days to allocate budget for
        budget_step: Step size for budget allocation
        min_budget: Minimum budget per ad per day
            
    Returns:
        ActionSpace for time-based budget allocation
    """
    # Create dimensions for each ad's budget on each day
    dimensions = []
    for i in range(num_ads):
        for j in range(num_days):
            dimensions.append(
                ActionDimension(
                    name=f"ad_{i+1}_day_{j+1}_budget",
                    dim_type="continuous",
                    min_value=min_budget,
                    max_value=total_budget,
                    step=budget_step
                )
            )
    
    # Add constraint that budgets must sum to total_budget
    def budget_sum_constraint(action, tolerance=0.0):
        return np.isclose(sum(action.values()), total_budget, rtol=tolerance)
    
    action_space = ActionSpace(dimensions, constraints=[budget_sum_constraint])
    # Store total_budget as an attribute for reference
    action_space.total_budget = total_budget
    return action_space


def create_multi_campaign_action_space(
    campaigns: Dict[str, Dict[str, Any]],
    budget_step: float = 100.0
) -> ActionSpace:
    """
    Create an action space for allocating budget across multiple campaigns and days.
    
    Args:
        campaigns: Dictionary mapping campaign_id to campaign data
            Each campaign should have:
            - 'total_budget': Total budget for this campaign
            - 'days': List of day identifiers
            - 'ads': List of ad identifiers
        budget_step: Step size for budget allocation
            
    Returns:
        ActionSpace for multi-campaign budget allocation
    """
    # Create dimensions for each campaign's ads on each day
    dimensions = []
    constraints = []
    
    total_budget = 0.0
    
    for campaign_id, campaign_data in campaigns.items():
        campaign_budget = campaign_data.get('total_budget', 0.0)
        days = campaign_data.get('days', [1])
        ads = campaign_data.get('ads', [1])
        
        # Track total budget across all campaigns
        total_budget += campaign_budget
        
        # Create dimensions for each ad on each day
        for ad_id in ads:
            for day in days:
                dimensions.append(
                    ActionDimension(
                        name=f"{campaign_id}_ad_{ad_id}_day_{day}_budget",
                        dim_type="continuous",
                        min_value=0.0,
                        max_value=campaign_budget,
                        step=budget_step
                    )
                )
        
        # Closure to capture campaign_id and days
        def make_campaign_constraint(cid, day_list, budget):
            def campaign_constraint(action, tolerance=0.0):
                campaign_keys = [f"{cid}_day_{day}_budget" for day in day_list]
                campaign_budget = sum(action[k] for k in campaign_keys if k in action)
                return np.isclose(campaign_budget, budget, rtol=tolerance)
            return campaign_constraint
        
        # Add constraint that each campaign's budget is respected
        constraints.append(make_campaign_constraint(campaign_id, days, campaign_budget))
    
    # Budget sum constraint
    def budget_sum_constraint(action, tolerance=0.0):
        return np.isclose(sum(action.values()), total_budget, rtol=tolerance)
    
    action_space = ActionSpace(dimensions, constraints=[budget_sum_constraint] + constraints)
    # Store total_budget as an attribute
    action_space.total_budget = total_budget
    return action_space


def create_marketing_ads_space(
    total_budget: float, 
    num_ads: int, 
    budget_step: float = 100.0, 
    min_budget: float = 0.0,
    ad_names: Optional[List[str]] = None
) -> ActionSpace:
    """
    Create an action space for allocating budget across marketing ads.
    
    Args:
        total_budget: Total budget to allocate
        num_ads: Number of ads to allocate budget to
        budget_step: Step size for budget allocation
        min_budget: Minimum budget per ad
        ad_names: Optional list of ad names to use for dimension names
            
    Returns:
        ActionSpace for marketing budget allocation
    """
    # Create dimensions for each ad's budget
    dimensions = []
    
    for i in range(num_ads):
        # Use provided ad names if available, otherwise use default names
        if ad_names and i < len(ad_names):
            name = f"{ad_names[i]}_budget"
        else:
            name = f"ad_{i+1}_budget"
            
        dimensions.append(
            ActionDimension(
                name=name,
                dim_type="continuous",
                min_value=min_budget,
                max_value=total_budget,
                step=budget_step
            )
        )
    
    # Add constraint that budgets must sum to total_budget
    def budget_sum_constraint(action, tolerance=0.0):
        return np.isclose(sum(action.values()), total_budget, rtol=tolerance)
    
    action_space = ActionSpace(dimensions, constraints=[budget_sum_constraint])
    # Store total_budget as an attribute for reference
    action_space.total_budget = total_budget
    return action_space
