# Default configuration for the environment state factors.
DEFAULT_STATE_CONFIG = {
    "rain": {
        "type": "bayesian_linear",  # We want to use our Bayesian linear model for rain.
        "value": None,              # Initially, the value is not set.
        "description": "Rain amount in mm predicted by a Bayesian linear model",
        "explanatory_vars": {
            "humidity": 0.75,       # Default humidity value.
            "pressure": 1012        # Default pressure in hPa.
        },
        "theta_prior": {
            "theta0": 0.0,
            "theta_humidity": 1.0,
            "theta_pressure": 0.001
        },
        "variance": 4.0
    },
    "temperature": {
        "type": "continuous",
        "value": 20.0,
        "description": "Temperature in °C"
    },
    "traffic": {
        "type": "categorical",
        "value": "Light",
        "description": "Traffic condition"
    },
    "weather": {
        "type": "categorical",
        "value": "Clear",
        "description": "Overall weather condition"
    }
}
