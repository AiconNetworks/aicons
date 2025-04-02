"""
Decision Making Module for BayesBrainGPT

This module provides tools for decision-making in BayesBrainGPT,
including action spaces and Bayesian decision-making.
"""

from .action_space import (
    ActionDimension,
    ActionSpace
)

from .marketing_action_spaces import (
    create_budget_allocation_space,
    create_time_budget_allocation_space,
    create_multi_campaign_action_space,
    create_marketing_ads_space
)
