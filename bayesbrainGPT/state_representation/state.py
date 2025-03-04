# state.py
from typing import Dict, Any
from factors import Factor, ContinuousFactor, CategoricalFactor, DiscreteFactor

class EnvironmentState:
    """
    Represents the overall state of the environment.
    Holds a dictionary of factors and provides methods for updating and retrieving the state.
    """
    def __init__(self, factors: Dict[str, Factor] = None):
        if factors is None:
            self.factors = self.build_default_state()
        else:
            self.factors = factors

    @staticmethod
    def build_default_state() -> Dict[str, Factor]:
        """
        Build the default state with predefined factors and default values.
        """
        return {
            "rain": ContinuousFactor(name="rain", value=0.0, description="Rain amount in mm"),
            "temperature": ContinuousFactor(name="temperature", value=20.0, description="Temperature in °C"),
            "traffic": CategoricalFactor(name="traffic", value="Light", description="Traffic condition"),
            "weather": CategoricalFactor(name="weather", value="Clear", description="Overall weather condition"),
        }

    def update_state(self, sensor_data: Dict[str, Any]) -> None:
        """
        Update the state with new sensor data.
        """
        for key, new_value in sensor_data.items():
            if key in self.factors:
                self.factors[key].update(new_value)

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
