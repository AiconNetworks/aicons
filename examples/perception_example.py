import torch
from bayesbrainGPT.state_representation import EnvironmentState
from bayesbrainGPT.perception import BayesianPerception
from bayesbrainGPT.sensors.traffic import StaticTrafficSensor
from bayesbrainGPT.config import DEFAULT_STATE_CONFIG

def create_test_config():
    return {
        "temperature": {
            "type": "continuous",
            "value": 20.0,
            "description": "Temperature in Celsius"
        },
        "weather": {
            "type": "categorical",
            "value": "sunny",
            "description": "Weather condition",
            "possible_values": ["sunny", "rainy", "cloudy"]
        },
        "traffic_density": {
            "type": "continuous",
            "value": 2.0,
            "description": "Traffic density (0-10)",
            "relationships": {
                "depends_on": ["weather"],
                "model": {
                    "weather": {
                        "type": "categorical_effect",
                        "effects": {
                            "sunny": 0.0,
                            "rainy": 2.0,
                            "cloudy": 1.0
                        }
                    }
                }
            }
        },
        "average_speed": {
            "type": "continuous",
            "value": 40.0,
            "description": "Average vehicle speed",
            "relationships": {
                "depends_on": ["traffic_density", "weather"],
                "model": {
                    "traffic_density": {
                        "type": "linear",
                        "base": 60.0,  # Base speed
                        "coefficient": -5.0  # Speed reduction per unit of traffic
                    },
                    "weather": {
                        "type": "categorical_effect",
                        "effects": {
                            "sunny": 0.0,
                            "rainy": -10.0,
                            "cloudy": -5.0
                        }
                    }
                }
            }
        }
    }

def main():
    # Create state with our test config
    state = EnvironmentState(factors_config=create_test_config())
    perception = BayesianPerception(state)
    
    print("\nINITIAL STATE:")
    print(f"Traffic Density: {state.factors['traffic_density'].value}")
    print(f"Weather: {state.factors['weather'].value}")
    print(f"Average Speed: {state.factors['average_speed'].value}")
    
    # Create and register traffic sensor
    traffic_sensor = StaticTrafficSensor(state, "downtown", reliability=0.9)
    perception.register_sensor(traffic_sensor)
    
    # Run full Bayesian update
    perception.update_all()

if __name__ == "__main__":
    main() 