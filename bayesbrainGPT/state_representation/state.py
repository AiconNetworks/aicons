# state.py
import json
from typing import Dict, Any
from pathlib import Path
from .factors import ContinuousFactor, CategoricalFactor, DiscreteFactor, BayesianLinearFactor

class EnvironmentState:
    """
    Represents the overall state of the environment.
    Can be initialized from either:
    1. LLM-derived state (from real LLM or mock data)
    2. Configuration-based state (for testing/default behavior)
    """
    def __init__(self, factors_config=None, use_llm=False, mock_llm=True):
        self.factors = {}
        
        if use_llm:
            if mock_llm:
                # Use mock LLM data from file
                mock_file = Path(__file__).parent / "llm_state_mkt.txt"
                with open(mock_file, 'r') as f:
                    llm_factors = json.load(f)
                self._initialize_from_llm_data(llm_factors)
            else:
                # Use real LLM integration
                from ..llm_integration import fetch_state_context_from_llm
                llm_factors = fetch_state_context_from_llm("Get current marketing state factors")
                self._initialize_from_llm_data(llm_factors)
        else:
            # Use provided configuration
            self._initialize_from_config(factors_config or {})

    def _initialize_from_llm_data(self, llm_factors):
        """Initialize state from LLM-derived data"""
        for factor in llm_factors:
            name = factor.get("description", "").lower().replace(" ", "_")
            if factor["type"] == "continuous":
                self.factors[name] = ContinuousFactor(
                    name=name,
                    initial_value=float(factor["value"]),
                    description=factor["description"]
                )
            elif factor["type"] == "categorical":
                self.factors[name] = CategoricalFactor(
                    name=name,
                    initial_value=factor["value"],
                    description=factor["description"]
                )
            elif factor["type"] == "discrete":
                self.factors[name] = DiscreteFactor(
                    name=name,
                    initial_value=int(float(factor["value"])),
                    description=factor["description"]
                )

    def _initialize_from_config(self, config):
        """Initialize state from configuration dictionary"""
        for name, factor_config in config.items():
            if factor_config["type"] == "continuous":
                self.factors[name] = ContinuousFactor(
                    name=name,
                    initial_value=factor_config["value"],
                    description=factor_config["description"]
                )
            elif factor_config["type"] == "categorical":
                self.factors[name] = CategoricalFactor(
                    name=name,
                    initial_value=factor_config["value"],
                    description=factor_config["description"]
                )
            elif factor_config["type"] == "discrete":
                self.factors[name] = DiscreteFactor(
                    name=name,
                    initial_value=factor_config["value"],
                    description=factor_config["description"]
                )
            elif factor_config["type"] == "bayesian_linear":
                self.factors[name] = BayesianLinearFactor(
                    name=name,
                    explanatory_vars=factor_config["explanatory_vars"],
                    theta_prior=factor_config["theta_prior"],
                    variance=factor_config["variance"],
                    description=factor_config["description"]
                )

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

    def reset(self):
        """Reset all factors to their initial values"""
        for factor in self.factors.values():
            factor.reset()


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
