# bayesian_factors.py
from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np

class Factor(ABC):
    """
    A generic abstract factor.
    Each factor has a name, a value, and a description.
    """
    def __init__(self, name: str, value: Any = None, description: str = ""):
        self.name = name
        self.value = value
        self.description = description

    @abstractmethod
    def update(self, new_value: Any) -> None:
        """
        Update the factor's value with new data.
        """
        pass

    def __repr__(self):
        return f"{self.name}: {self.value}  ({self.description})"

class ContinuousFactor(Factor):
    """
    A factor that holds a continuous (real-valued) variable.
    """
    def update(self, new_value: float) -> None:
        if not isinstance(new_value, (int, float)):
            raise ValueError("Expected a numeric value for ContinuousFactor.")
        self.value = new_value

class CategoricalFactor(Factor):
    """
    A factor that holds a categorical value (as a string).
    """
    def update(self, new_value: str) -> None:
        if not isinstance(new_value, str):
            raise ValueError("Expected a string value for CategoricalFactor.")
        self.value = new_value

class DiscreteFactor(Factor):
    """
    A factor that holds a discrete (integer) value.
    """
    def update(self, new_value: int) -> None:
        if not isinstance(new_value, int):
            raise ValueError("Expected an integer value for DiscreteFactor.")
        self.value = new_value

class BayesianLinearFactor(ContinuousFactor):
    """
    A continuous factor modeled by a Bayesian linear regression.
    
    The factor's value is predicted using:
        r = theta0 + theta1*x1 + ... + theta_n*xn + epsilon,
    where theta_i are parameters with associated priors.
    """
    def __init__(self, name: str, explanatory_vars: Dict[str, float],
                 theta_prior: Dict[str, float], variance: float, description: str = ""):
        """
        :param explanatory_vars: A dict of explanatory variable names and their current values.
        :param theta_prior: A dict containing the current estimates for the parameters,
                            e.g., {"theta0": 0.0, "theta_humidity": 1.0, ...}.
        :param variance: The noise variance sigma^2.
        """
        super().__init__(name, value=None, description=description)
        self.explanatory_vars = explanatory_vars  # e.g., {"humidity": 0.8, "pressure": 1010}
        self.theta = theta_prior  # e.g., {"theta0": 0.0, "theta_humidity": 1.0, "theta_pressure": 0.001}
        self.variance = variance  # noise variance

    def predict(self) -> float:
        """
        Predict the value using the linear model.
        """
        prediction = self.theta.get("theta0", 0.0)
        for var, val in self.explanatory_vars.items():
            prediction += self.theta.get(f"theta_{var}", 0.0) * val
        return prediction

    def update(self, observed_value: float) -> None:
        """
        Update the parameters using the new observed value.
        This is a very simple (not fully Bayesian) update for illustration.
        In a full implementation, you would compute the posterior distribution of theta.
        """
        # Set the current observed value
        self.value = observed_value
        
        # Compute the error between the observed value and current prediction.
        prediction = self.predict()
        error = observed_value - prediction
        learning_rate = 0.01  # small constant for illustration
        
        # Update theta0
        self.theta["theta0"] = self.theta.get("theta0", 0.0) + learning_rate * error
        
        # Update coefficients for each explanatory variable
        for var, val in self.explanatory_vars.items():
            key = f"theta_{var}"
            self.theta[key] = self.theta.get(key, 0.0) + learning_rate * error * val

    def __repr__(self):
        return (f"{self.name}: predicted={self.predict():.2f}, observed={self.value}, "
                f"theta={self.theta}  ({self.description})")

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
