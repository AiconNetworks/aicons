# bayesian_factors.py
from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np
import torch

class BaseFactor:
    def __init__(self, name, value=None, description=""):
        self.name = name
        self.value = value
        self.description = description

    def __str__(self):
        return f"{self.value}"

class ContinuousFactor(BaseFactor):
    """
    A factor that holds a continuous (real-valued) variable.
    """
    def update(self, new_value: float) -> None:
        if not isinstance(new_value, (int, float)):
            raise ValueError("Expected a numeric value for ContinuousFactor.")
        self.value = new_value

class CategoricalFactor(BaseFactor):
    """
    A factor that holds a categorical value (as a string).
    """
    def update(self, new_value: str) -> None:
        if not isinstance(new_value, str):
            raise ValueError("Expected a string value for CategoricalFactor.")
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
