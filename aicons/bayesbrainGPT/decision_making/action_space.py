"""
Action Space Module for BayesBrainGPT

This module provides structures for defining action spaces in Bayesian decision making.
It allows configuration of multi-dimensional action spaces with different shapes and types.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Union, Optional, Sequence
import numpy as np


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
            name: Name of the dimension (e.g., 'ad1_budget', 'day_of_week')
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


class ActionSpace:
    """
    Multi-dimensional action space for Bayesian decision making.
    
    Can represent complex action spaces with multiple interacting dimensions,
    such as budget allocations across multiple ads, time periods, etc.
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
    
    def sample(self) -> Dict[str, Any]:
        """
        Sample a random point in the action space.
        
        Returns:
            Dict mapping dimension names to their sampled values
        """
        # Try to find a valid sample that meets all constraints
        max_attempts = 100  # Prevent infinite loops
        
        for _ in range(max_attempts):
            # Sample each dimension
            action = {dim.name: dim.sample() for dim in self.dimensions}
            
            # Check constraints
            if self._check_constraints(action):
                return action
        
        # If we couldn't find a valid sample, return None or raise exception
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
        for name, value in action.items():
            if name not in self.dimension_map or not self.dimension_map[name].contains(value):
                return False
        
        # Check additional constraints
        return self._check_constraints(action)
    
    def _check_constraints(self, action: Dict[str, Any]) -> bool:
        """Check if an action satisfies all constraints."""
        return all(constraint(action) for constraint in self.constraints)
    
    def enumerate_actions(self, max_actions: int = 1000) -> List[Dict[str, Any]]:
        """
        Enumerate all possible actions in the space.
        
        For continuous dimensions, this uses discretization.
        For very large spaces, this will return a sample.
        
        Args:
            max_actions: Maximum number of actions to enumerate
            
        Returns:
            List of action dictionaries
        """
        if not self.is_discrete:
            # If space has continuous dimensions, return a reasonable sampling
            return [self.sample() for _ in range(min(max_actions, 1000))]
        
        # For small discrete spaces, enumerate everything
        if self.size <= max_actions:
            return self._enumerate_discrete_actions()
        
        # For large discrete spaces, return a sample
        return [self.sample() for _ in range(max_actions)]
    
    def _enumerate_discrete_actions(self) -> List[Dict[str, Any]]:
        """Enumerate all actions in a discrete space."""
        # Get all values for each dimension
        all_values = [dim.enumerate_values() for dim in self.dimensions]
        
        # Generate all combinations
        actions = []
        self._enumerate_recursive(all_values, 0, {}, actions)
        
        # Filter by constraints
        return [action for action in actions if self._check_constraints(action)]
    
    def _enumerate_recursive(
        self, all_values: List[List[Any]], 
        dim_idx: int, 
        current_action: Dict[str, Any],
        actions: List[Dict[str, Any]]
    ) -> None:
        """
        Recursively enumerate all combinations of discrete values.
        
        Args:
            all_values: List of value lists for each dimension
            dim_idx: Current dimension index
            current_action: Partially built action
            actions: List to store complete actions
        """
        # Base case: we've assigned values to all dimensions
        if dim_idx >= len(self.dimensions):
            actions.append(current_action.copy())
            return
        
        # Recursive case: try each value for the current dimension
        dim = self.dimensions[dim_idx]
        for value in all_values[dim_idx]:
            current_action[dim.name] = value
            self._enumerate_recursive(all_values, dim_idx + 1, current_action, actions)


# Utility functions for creating common action spaces

def create_budget_allocation_space(
    total_budget: float,
    num_ads: int,
    budget_step: float = 100.0,
    min_budget: float = 0.0
) -> ActionSpace:
    """
    Create an action space for allocating budget across multiple ads.
    
    Args:
        total_budget: Total budget to allocate
        num_ads: Number of ads
        budget_step: Step size for budget allocation
        min_budget: Minimum budget per ad
        
    Returns:
        ActionSpace for budget allocation
    """
    # Create dimensions for each ad's budget
    dimensions = []
    for i in range(num_ads):
        dimensions.append(
            ActionDimension(
                name=f"ad_{i+1}_budget",
                dim_type="continuous",
                min_value=min_budget,
                max_value=total_budget,
                step=budget_step
            )
        )
    
    # Add constraint that budgets must sum to total_budget
    def budget_sum_constraint(action):
        return np.isclose(sum(action.values()), total_budget)
    
    return ActionSpace(dimensions, constraints=[budget_sum_constraint])


def create_time_budget_allocation_space(
    total_budget: float,
    num_ads: int,
    num_days: int,
    budget_step: float = 100.0,
    min_budget: float = 0.0
) -> ActionSpace:
    """
    Create an action space for allocating budget across multiple ads and days.
    
    Args:
        total_budget: Total budget to allocate
        num_ads: Number of ads
        num_days: Number of days
        budget_step: Step size for budget allocation
        min_budget: Minimum budget per ad per day
        
    Returns:
        ActionSpace for time-based budget allocation
    """
    # Create dimensions for each ad's budget on each day
    dimensions = []
    for d in range(num_days):
        for i in range(num_ads):
            dimensions.append(
                ActionDimension(
                    name=f"day_{d+1}_ad_{i+1}_budget",
                    dim_type="continuous",
                    min_value=min_budget,
                    max_value=total_budget,
                    step=budget_step
                )
            )
    
    # Add constraint that budgets must sum to total_budget
    def budget_sum_constraint(action):
        return np.isclose(sum(action.values()), total_budget)
    
    return ActionSpace(dimensions, constraints=[budget_sum_constraint])


def create_multi_campaign_action_space(
    campaigns: Dict[str, Dict[str, Any]],
    budget_step: float = 100.0
) -> ActionSpace:
    """
    Create a customizable action space for multiple campaigns with different parameters.
    
    Args:
        campaigns: Dictionary mapping campaign IDs to their parameters
            Each campaign dict should have:
                - total_budget: Total budget for this campaign
                - min_budget: Minimum budget per day (optional)
                - max_budget: Maximum budget per day (optional)
                - days: List of day names or identifiers (optional)
        budget_step: Step size for budget allocation
        
    Returns:
        ActionSpace for the multi-campaign scenario
    """
    dimensions = []
    
    for campaign_id, params in campaigns.items():
        total_budget = params.get('total_budget', 0)
        min_budget = params.get('min_budget', 0)
        max_budget = params.get('max_budget', total_budget)
        days = params.get('days', [1])  # Default to a single day
        
        # Add dimensions for each day in this campaign
        if isinstance(days, int):
            # If days is an integer, create that many numbered days
            days = list(range(1, days + 1))
            
        for day in days:
            dimensions.append(
                ActionDimension(
                    name=f"{campaign_id}_day_{day}_budget",
                    dim_type="continuous",
                    min_value=min_budget,
                    max_value=max_budget,
                    step=budget_step
                )
            )
    
    # Add constraint that budgets for each campaign must sum to its total_budget
    constraints = []
    for campaign_id, params in campaigns.items():
        total_budget = params.get('total_budget', 0)
        days = params.get('days', [1])
        
        if isinstance(days, int):
            days = list(range(1, days + 1))
        
        # Closure to capture campaign_id and days
        def make_campaign_constraint(cid, day_list, budget):
            def campaign_constraint(action):
                campaign_keys = [f"{cid}_day_{day}_budget" for day in day_list]
                campaign_budget = sum(action[k] for k in campaign_keys if k in action)
                return np.isclose(campaign_budget, budget)
            return campaign_constraint
        
        constraints.append(make_campaign_constraint(campaign_id, days, total_budget))
    
    return ActionSpace(dimensions, constraints=constraints)


# Example usage
if __name__ == "__main__":
    # Example 1: Simple budget allocation across 3 ads
    budget_space = create_budget_allocation_space(
        total_budget=500.0,
        num_ads=3,
        budget_step=100.0
    )
    
    # Sample an action and print it
    action = budget_space.sample()
    print("Sample budget allocation:")
    print(action)
    
    # Example 2: Budget allocation across 2 ads and 3 days
    time_budget_space = create_time_budget_allocation_space(
        total_budget=600.0,
        num_ads=2,
        num_days=3,
        budget_step=50.0
    )
    
    # Sample an action and print it
    action = time_budget_space.sample()
    print("\nSample time-based budget allocation:")
    print(action)
    
    # Example 3: Multi-campaign with different parameters
    campaigns = {
        "summer_sale": {
            "total_budget": 1000.0,
            "min_budget": 0.0,
            "max_budget": 500.0,
            "days": ["mon", "tue", "wed"]
        },
        "product_launch": {
            "total_budget": 1500.0,
            "min_budget": 100.0,
            "max_budget": 1000.0,
            "days": ["thu", "fri", "sat", "sun"]
        }
    }
    
    multi_campaign_space = create_multi_campaign_action_space(
        campaigns=campaigns,
        budget_step=100.0
    )
    
    # Sample an action and print it
    action = multi_campaign_space.sample()
    print("\nSample multi-campaign budget allocation:")
    print(action)
