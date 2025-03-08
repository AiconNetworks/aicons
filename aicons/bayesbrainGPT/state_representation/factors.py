# bayesian_factors.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np
import torch

class BaseFactor:
    def __init__(self, name: str, initial_value: Any = None, description: str = "", relationships: Dict = None):
        self.name = name
        self.initial_value = initial_value
        self.value = initial_value  # Current value starts as initial value
        self.description = description
        self.relationships = relationships or {}  # Store relationships

    def __str__(self):
        return f"{self.value}"

    def reset(self):
        """Reset to initial value"""
        self.value = self.initial_value

class ContinuousFactor(BaseFactor):
    """
    A factor that holds a continuous (real-valued) variable.
    """
    def __init__(self, name: str, initial_value: float, description: str = "", relationships: Dict = None):
        super().__init__(name, initial_value, description, relationships)
        self._uncertainty = 1.0  # Default uncertainty
        
    @property
    def uncertainty(self):
        """Get the current uncertainty (standard deviation) of the belief"""
        return self._uncertainty
        
    def update_uncertainty(self, new_uncertainty: float):
        """Update the uncertainty of our belief"""
        self._uncertainty = max(0.0, new_uncertainty)  # Ensure non-negative

    def update(self, new_value: float) -> None:
        if not isinstance(new_value, (int, float)):
            raise ValueError("Expected a numeric value for ContinuousFactor.")
        self.value = new_value

class CategoricalFactor(BaseFactor):
    """
    A factor that holds a categorical value (as a string).
    """
    def __init__(self, name: str, initial_value: str, description: str = "", 
                 relationships: Dict = None, possible_values: List[str] = None):
        super().__init__(name, initial_value, description, relationships)
        self.possible_values = possible_values or [initial_value]  # If no values provided, use initial value
        if initial_value not in self.possible_values:
            self.possible_values.append(initial_value)

    def update(self, new_value: str) -> None:
        if not isinstance(new_value, str):
            raise ValueError("Expected a string value for CategoricalFactor.")
        if new_value not in self.possible_values:
            raise ValueError(f"Value {new_value} not in possible values: {self.possible_values}")
        self.value = new_value

class DiscreteFactor(BaseFactor):
    """
    A factor that holds a discrete (integer) value.
    """
    def update(self, new_value: int) -> None:
        if not isinstance(new_value, int):
            raise ValueError("Expected an integer value for DiscreteFactor.")
        self.value = new_value

class BayesianLinearFactor(BaseFactor):
    """
    A continuous factor modeled by a Bayesian linear regression.
    
    The factor's value is predicted using:
        r = theta0 + theta1*x1 + ... + theta_n*xn + epsilon,
    where theta_i are parameters with associated priors.
    """
    def __init__(self, name, explanatory_vars=None, theta_prior=None, variance=1.0, description=""):
        super().__init__(name, None, description)
        self.explanatory_vars = explanatory_vars or {}
        self.theta_prior = theta_prior or {}
        self.variance = variance

    def update_explanatory_vars(self, new_vars):
        self.explanatory_vars.update(new_vars)

    def __str__(self):
        return f"vars={self.explanatory_vars}, prior={self.theta_prior}, var={self.variance}"

    def get_prior_params(self) -> Dict[str, Any]:
        """Get prior parameters in a format suitable for Pyro"""
        return {
            "mean": torch.tensor([v["mean"] for v in self.theta_prior.values()]),
            "variance": torch.tensor([v["variance"] for v in self.theta_prior.values()])
        }

# Example usage:
if __name__ == "__main__":
    # Initialize Bayesian linear factor for "rain"
    explanatory_vars = {"humidity": 0.8, "pressure": 1010}
    theta_prior = {"theta0": 0.0, "theta_humidity": 1.0, "theta_pressure": 0.001}
    rain_factor = BayesianLinearFactor(
        name="rain", 
        explanatory_vars=explanatory_vars,
        theta_prior=theta_prior,
        variance=4.0,
        description="Rain amount predicted by Bayesian linear regression"
    )
    
    # Predict current rain value
    print("Initial prediction for rain:", rain_factor.predict())
    
    # Suppose sensor data is observed: 12 mm of rain.
    observed_rain = 12.0
    rain_factor.update(observed_rain)
    
    # After update, see the new state.
    print("After update:", rain_factor)
    
    # Demonstrate usage of Categorical and Discrete factors:
    traffic = CategoricalFactor(name="traffic", value="Light", description="Traffic condition")
    print("Traffic factor:", traffic)
    
    day_of_week = DiscreteFactor(name="day_of_week", value=1, description="Day of the week (1=Monday)")
    print("Day of week factor:", day_of_week)
