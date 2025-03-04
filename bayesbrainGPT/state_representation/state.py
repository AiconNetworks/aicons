# state.py
from typing import Dict, Any
from .factors import ContinuousFactor, CategoricalFactor, DiscreteFactor

class EnvironmentState:
    """
    Represents the overall state of the environment.
    Holds a dictionary of factors and provides methods for updating and retrieving the state.
    """
    def __init__(self, factors=None):
        self.factors = factors or {}

    def update_state(self, new_data):
        for key, value in new_data.items():
            if key in self.factors:
                if isinstance(value, dict) and key == "rain_prediction":
                    self.factors[key].update_explanatory_vars(value["explanatory_vars"])
                else:
                    self.factors[key].value = value

    def __str__(self):
        return "\n".join(f"{k}: {v}" for k, v in self.factors.items())

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the current state as a simple dictionary.
        """
        return {key: factor.value for key, factor in self.factors.items()}

    def __repr__(self) -> str:
        state_repr = "\n".join([repr(factor) for factor in self.factors.values()])
        return f"EnvironmentState:\n{state_repr}"


# Example usage:
if __name__ == "__main__":
    # Build the initial state.
    state = EnvironmentState()
    print("Initial State:")
    print(state)

    # Suppose new sensor data arrives:
    sensor_data = {
        "rain": 15.0,           # in mm
        "temperature": 25.0,      # in °C
        "traffic": "Heavy",       # traffic condition
        "weather": "Stormy"       # weather condition
    }
    
    # Update the state with sensor data.
    state.update_state(sensor_data)
    print("\nUpdated State:")
    print(state)
