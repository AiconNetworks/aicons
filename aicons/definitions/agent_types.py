from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class AgentDefinition:
    """Defines an AI agent's characteristics and capabilities"""
    name: str
    type: str
    capabilities: List[str]
    state_factors: Dict[str, Any]
    decision_rules: List[str]

# Example agent definitions
MARKETING_AGENT = AgentDefinition(
    name="marketing_specialist",
    type="marketing",
    capabilities=[
        "campaign_optimization",
        "budget_management",
        "audience_targeting"
    ],
    state_factors={
        "campaign_roi": {
            "type": "continuous",
            "initial_value": 1.0
        },
        "engagement_rate": {
            "type": "continuous",
            "initial_value": 0.5
        }
    },
    decision_rules=[
        "evaluate_campaign_performance",
        "get_budget_adjustment"
    ]
)

# Add more agent definitions as needed
AGENT_TYPES = {
    "marketing": MARKETING_AGENT,
    # Add other agent types
} 