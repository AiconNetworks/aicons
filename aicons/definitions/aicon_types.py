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

ROOM_ENVIRONMENT_AGENT = AgentDefinition(
    name="room_environment_monitor",
    type="room_environment",
    capabilities=[
        "temperature_monitoring",
        "occupancy_tracking",
        "comfort_assessment",
        "air_quality_monitoring"
    ],
    state_factors={
        "room_temperature": {
            "type": "continuous",
            "initial_value": 20.0,
            "description": "Room temperature in Celsius"
        },
        "comfort_level": {
            "type": "categorical",
            "initial_value": "comfortable",
            "possible_values": ["cold", "comfortable", "hot"],
            "description": "Comfort level based on temperature"
        },
        "room_occupancy": {
            "type": "continuous",
            "initial_value": 2.0,
            "description": "Number of people in the room (0-10)",
            "relationships": {
                "depends_on": ["comfort_level"],
                "model": {
                    "comfort_level": {
                        "type": "categorical_effect",
                        "effects": {
                            "cold": 0.0,
                            "comfortable": 2.0,
                            "hot": 1.0
                        }
                    }
                }
            }
        },
        "co2_level": {
            "type": "continuous",
            "initial_value": 400.0,
            "description": "CO2 concentration in ppm",
            "relationships": {
                "depends_on": ["room_occupancy", "comfort_level"],
                "model": {
                    "room_occupancy": {
                        "type": "linear",
                        "base": 400.0,
                        "coefficient": 50.0
                    },
                    "comfort_level": {
                        "type": "categorical_effect",
                        "effects": {
                            "cold": 0.0,
                            "comfortable": 0.0,
                            "hot": 50.0
                        }
                    }
                }
            }
        }
    },
    decision_rules=[
        "evaluate_comfort_conditions",
        "suggest_ventilation_actions",
        "monitor_air_quality"
    ]
)

AIQUON_AGENT = AgentDefinition(
    name="aiquon",
    type="aiquon",
    capabilities=[
        "bayesian_inference",
        "factor_management",
        "state_tracking",
        "relationship_learning"
    ],
    state_factors={},  # Empty by default since factors will be defined dynamically
    decision_rules=[
        "update_state",
        "infer_relationships",
        "predict_changes"
    ]
)

# Add more agent definitions as needed
AGENT_TYPES = {
    "marketing": MARKETING_AGENT,
    "room_environment": ROOM_ENVIRONMENT_AGENT,
    "aiquon": AIQUON_AGENT
} 