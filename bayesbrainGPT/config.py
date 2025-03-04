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
        "description": "Current weather condition"
    },
    "traffic_density": {
        "type": "discrete",
        "value": 2,
        "description": "Traffic density level (1-5)"
    },
    "rain_prediction": {
        "type": "bayesian_linear",
        "explanatory_vars": {"humidity": 0.7, "pressure": 1013},
        "theta_prior": {"mean": [0.0, 0.0], "variance": [1.0, 1.0]},
        "variance": 1.0,
        "description": "Rain prediction model based on humidity and pressure"
    }
}
