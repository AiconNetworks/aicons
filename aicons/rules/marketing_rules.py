from typing import Dict, Any
from enum import Enum

class MarketingActions(Enum):
    INCREASE_BUDGET = "increase_budget"
    DECREASE_BUDGET = "decrease_budget"
    CHANGE_CHANNEL = "change_channel"
    ADJUST_TARGETING = "adjust_targeting"

def evaluate_campaign_performance(state: Dict[str, Any]) -> MarketingActions:
    """Evaluate current state and determine marketing actions"""
    roi = state.get('campaign_roi', 0)
    engagement = state.get('engagement_rate', 0)
    
    if roi < 0.5 and engagement < 0.3:
        return MarketingActions.DECREASE_BUDGET
    elif roi > 2.0 and engagement > 0.6:
        return MarketingActions.INCREASE_BUDGET
    
    return MarketingActions.ADJUST_TARGETING

def get_budget_adjustment(action: MarketingActions, current_budget: float) -> float:
    """Calculate budget adjustments based on action"""
    adjustments = {
        MarketingActions.INCREASE_BUDGET: 1.2,
        MarketingActions.DECREASE_BUDGET: 0.8,
        MarketingActions.ADJUST_TARGETING: 1.0
    }
    return current_budget * adjustments.get(action, 1.0) 