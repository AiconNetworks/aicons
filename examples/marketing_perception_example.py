import torch
from bayesbrainGPT.state_representation import EnvironmentState
from bayesbrainGPT.perception import BayesianPerception
from bayesbrainGPT.perception.sensor import Sensor
from typing import Dict, Any

class MarketingSensor(Sensor):
    def __init__(self):
        super().__init__("marketing_sensor")
        
    def get_data(self) -> Dict[str, torch.Tensor]:
        # In real world, this would fetch data from your marketing platforms
        return {
            "conversion_rate": torch.tensor(0.05),
            "click_through_rate": torch.tensor(0.02),
            "average_order_value": torch.tensor(100.0)
        }

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
    
    # Register marketing sensor
    marketing_sensor = MarketingSensor()
    perception.register_sensor(marketing_sensor)
    
    # Update perception
    perception.update_from_sensor("marketing_sensor")
    
    print("\nUpdated state after sensor data:")
    print(state)

if __name__ == "__main__":
    main() 