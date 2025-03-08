import torch
from bayesbrainGPT.state_representation import EnvironmentState
from bayesbrainGPT.perception import BayesianPerception
from bayesbrainGPT.sensors.base import Sensor
from typing import Dict, Any

class MarketingSensor(Sensor):
    """Marketing sensor that pulls data from marketing platforms"""
    def __init__(self, state: EnvironmentState, reliability: float = 0.85):
        super().__init__("marketing_sensor", state, reliability)
        
    def _setup_observable_factors(self):
        """Define which marketing metrics this sensor can observe"""
        self.observable_factors = [
            name for name, factor in self.state.factors.items()
            if name in ["conversion_rate", "click_through_rate", "average_order_value"]
        ]

    def fetch_data(self) -> Dict[str, torch.Tensor]:
        """Actively fetch marketing data"""
        # In real world, this would fetch data from your marketing platforms
        return {
            "conversion_rate": torch.tensor(0.05),
            "click_through_rate": torch.tensor(0.02),
            "average_order_value": torch.tensor(100.0)
        }

    def receive_data(self, data: Dict[str, Any]):
        """Handle webhook updates from marketing platforms"""
        # This could handle real-time updates from marketing APIs
        raise NotImplementedError("Push updates not implemented for marketing sensor")

# Create marketing-specific state configuration
MARKETING_STATE_CONFIG = {
    "conversion_rate": {
        "type": "continuous",
        "value": 0.03,  # Prior mean
        "description": "Conversion rate from ad click to purchase"
    },
    "click_through_rate": {
        "type": "continuous",
        "value": 0.01,  # Prior mean
        "description": "Click-through rate for ads"
    },
    "campaign_effectiveness": {
        "type": "bayesian_linear",
        "explanatory_vars": {
            "budget": 1000.0,
            "seasonality": 0.5
        },
        "theta_prior": {
            "mean": [0.0, 0.0],
            "variance": [1.0, 1.0]
        },
        "variance": 0.1,
        "description": "Campaign effectiveness model"
    }
}

def main():
    # Initialize state with marketing priors
    state = EnvironmentState(MARKETING_STATE_CONFIG)
    print("Initial state:")
    print(state)
    
    # Create perception system
    perception = BayesianPerception(state)
    
    # Register marketing sensor with state and reliability score
    marketing_sensor = MarketingSensor(state, reliability=0.85)
    perception.register_sensor(marketing_sensor)
    
    # Update perception
    perception.update_from_sensor("marketing_sensor")
    
    print("\nUpdated state after sensor data:")
    print(state)

if __name__ == "__main__":
    main() 