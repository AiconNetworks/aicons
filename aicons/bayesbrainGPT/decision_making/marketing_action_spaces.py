"""
Marketing Action Spaces Module for BayesBrainGPT

This module provides specialized action spaces for marketing optimization,
building on top of the core action space functionality.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from .action_space import ActionSpace, ActionDimension


def create_budget_allocation_space(
    total_budget: float,
    num_ads: int,
    budget_step: float = 10.0,
    min_budget: float = 0.0,
    ad_names: Optional[List[str]] = None
) -> ActionSpace:
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
    
    # Pre-compute valid budget combinations
    def find_valid_combinations():
        # Convert to units of budget_step
        total_units = int(total_budget / budget_step)
        min_units = int(min_budget / budget_step)
        
        # Initialize list to store valid combinations
        valid_combinations = []
        
        # Helper function to generate combinations recursively
        def generate_combinations(current, remaining_units, remaining_ads):
            if remaining_ads == 1:
                # Last ad gets all remaining units
                if min_units <= remaining_units <= total_units:
                    valid_combinations.append(current + [remaining_units])
            else:
                # Try all possible allocations for current ad
                for units in range(min_units, min(remaining_units + 1, total_units + 1)):
                    generate_combinations(
                        current + [units],
                        remaining_units - units,
                        remaining_ads - 1
                    )
        
        # Start generating combinations
        generate_combinations([], total_units, num_ads)
        
        # Convert back to actual budget values
        return [
            {dim.name: units * budget_step for dim, units in zip(dimensions, combo)}
            for combo in valid_combinations
        ]
    
    # Get all valid combinations
    valid_actions = find_valid_combinations()
    
    # Create a custom enumerate_actions method that returns pre-computed combinations
    def custom_enumerate_actions(max_actions=None):
        if max_actions is None or len(valid_actions) <= max_actions:
            return valid_actions
        else:
            # If we need fewer actions, sample from valid combinations
            return np.random.choice(valid_actions, size=max_actions, replace=False).tolist()
    
    # Create action space with custom enumerate_actions
    action_space = ActionSpace(dimensions, constraints=[])
    action_space.enumerate_actions = custom_enumerate_actions
    
    # Store total_budget as an attribute for reference
    action_space.total_budget = total_budget
    action_space.valid_actions = valid_actions
    
    print(f"Created budget allocation space with {len(valid_actions)} valid combinations")
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