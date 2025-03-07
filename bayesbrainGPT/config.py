# Default configuration for the environment state factors.
DEFAULT_STATE_CONFIG = {
    "temperature": {
        "type": "continuous",
        "value": 20.0,
        "description": "Ambient temperature in Celsius"
    },
    "weather": {
        "type": "categorical",
        "value": "sunny",
        "possible_values": ["sunny", "rainy", "cloudy"],
        "description": "Weather condition",
        "relationships": {
            "affects": ["average_speed", "road_quality"],
            "model": {
                "average_speed": {
                    "type": "categorical_effect",
                    "effects": {"sunny": 0.0, "rainy": -15.0, "cloudy": -5.0}
                }
            }
        }
    },
    "traffic_density": {
        "type": "continuous",
        "value": 2.0,
        "description": "Traffic density observation",
        "relationships": {
            "affects": ["average_speed", "vehicle_count"],
            "model": {
                "average_speed": {"type": "linear", "coefficient": -10.0, "base": 60.0},
                "vehicle_count": {"type": "exponential", "base": 50.0, "scale": 1.2}
            }
        }
    },
    "rain_prediction": {
        "type": "bayesian_linear",
        "explanatory_vars": {"humidity": 0.7, "pressure": 1013},
        "theta_prior": {"mean": [0.0, 0.0], "variance": [1.0, 1.0]},
        "variance": 1.0,
        "description": "Rain prediction model based on humidity and pressure"
    },
    "average_speed": {
        "type": "continuous",
        "value": 40.0,
        "description": "Average vehicle speed",
        "relationships": {
            "depends_on": ["traffic_density", "weather"],
            "noise_model": {"type": "gaussian", "base_std": 5.0}
        }
    }
}
