"""
Bayesian Brain-inspired Latent Variables

This module provides representations of latent variables in a Bayesian brain model.
These latent variables represent unobserved causes that the brain infers from sensory data.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np

class LatentVariable(ABC):
    """
    Base class for latent variables in a Bayesian brain model.
    
    In the Bayesian brain hypothesis, latent variables represent
    hidden causes that the brain infers from sensory observations.
    """
    def __init__(self, name: str, initial_value: Any = None, description: str = "", relationships: Dict = None):
        """
        Initialize a latent variable.
        
        Args:
            name: Name of the latent variable
            initial_value: Initial value (prior mean)
            description: Description of what this variable represents
            relationships: Dictionary of relationships with other variables
        """
        self.name = name
        self.initial_value = initial_value
        self.value = initial_value  # Current value (posterior mean)
        self.description = description
        self.relationships = relationships or {}  # Store hierarchical relationships

    def __str__(self):
        return f"{self.value}"
        
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', value={self.value}, description='{self.description}')"

    def reset(self):
        """Reset to initial value (prior mean)"""
        self.value = self.initial_value


class ContinuousLatentVariable(LatentVariable):
    """
    A latent variable that can take continuous (real-valued) values.
    
    In the Bayesian brain hypothesis, continuous latent variables might represent
    quantities like temperature, brightness, or other real-valued hidden causes.
    """
    def __init__(self, name: str, initial_value: float, uncertainty: float = 1.0, 
                 description: str = "", relationships: Dict = None):
        """
        Initialize a continuous latent variable.
        
        Args:
            name: Name of the latent variable
            initial_value: Initial value (prior mean)
            uncertainty: Initial uncertainty (prior standard deviation)
            description: Description of what this variable represents
            relationships: Dictionary of relationships with other variables
        """
        super().__init__(name, initial_value, description, relationships)
        self._uncertainty = uncertainty  # Prior/posterior uncertainty (std dev)
        
    @property
    def uncertainty(self):
        """Get the current uncertainty (standard deviation) of the belief"""
        return self._uncertainty
        
    def update_uncertainty(self, new_uncertainty: float):
        """Update the uncertainty of our belief"""
        self._uncertainty = max(0.0, new_uncertainty)  # Ensure non-negative

    def update(self, new_value: float, new_uncertainty: Optional[float] = None) -> None:
        """
        Update the value and optionally the uncertainty.
        
        In Bayesian terms, this updates the posterior mean and optionally
        the posterior standard deviation.
        
        Args:
            new_value: New value (posterior mean)
            new_uncertainty: New uncertainty (posterior std dev), if None keeps current
        """
        if not isinstance(new_value, (int, float)):
            raise ValueError("Expected a numeric value for ContinuousLatentVariable.")
        self.value = new_value
        if new_uncertainty is not None:
            self.update_uncertainty(new_uncertainty)
            
    def sample_prior(self, n_samples: int = 1) -> Union[float, np.ndarray]:
        """
        Sample from the prior distribution.
        
        Args:
            n_samples: Number of samples to draw
            
        Returns:
            Single value or array of samples from prior distribution
        """
        # Assume Gaussian prior for continuous variables
        samples = np.random.normal(self.initial_value, self._uncertainty, size=n_samples)
        return samples[0] if n_samples == 1 else samples
        
    def sample_posterior(self, n_samples: int = 1) -> Union[float, np.ndarray]:
        """
        Sample from the posterior distribution.
        
        Args:
            n_samples: Number of samples to draw
            
        Returns:
            Single value or array of samples from posterior distribution
        """
        # Assume Gaussian posterior for continuous variables
        samples = np.random.normal(self.value, self._uncertainty, size=n_samples)
        return samples[0] if n_samples == 1 else samples
        
    def log_prior(self, value: float) -> float:
        """
        Compute log prior probability of a value.
        
        Args:
            value: Value to compute probability for
            
        Returns:
            Log probability of the value under the prior
        """
        # Assume Gaussian prior
        return -0.5 * np.log(2 * np.pi * self._uncertainty**2) - \
               0.5 * ((value - self.initial_value) / self._uncertainty)**2


class CategoricalLatentVariable(LatentVariable):
    """
    A latent variable that can take one of several discrete categorical values.
    
    In the Bayesian brain hypothesis, categorical latent variables might represent
    discrete concepts like "sunny" vs "rainy" weather, or object categories.
    """
    def __init__(self, name: str, initial_value: str, possible_values: List[str], 
                 prior_probs: Optional[List[float]] = None,
                 description: str = "", relationships: Dict = None):
        """
        Initialize a categorical latent variable.
        
        Args:
            name: Name of the latent variable
            initial_value: Initial most likely value
            possible_values: List of all possible values
            prior_probs: List of prior probabilities for each value (default uniform)
            description: Description of what this variable represents
            relationships: Dictionary of relationships with other variables
        """
        super().__init__(name, initial_value, description, relationships)
        self.possible_values = possible_values
        
        # Ensure initial value is in possible values
        if initial_value not in self.possible_values:
            self.possible_values.append(initial_value)
            
        # Set up prior probabilities (default to uniform if not provided)
        if prior_probs is None:
            self.prior_probs = {val: 1.0/len(self.possible_values) for val in self.possible_values}
        else:
            if len(prior_probs) != len(possible_values):
                raise ValueError("prior_probs must have same length as possible_values")
            self.prior_probs = {val: prob for val, prob in zip(possible_values, prior_probs)}
            
        # Current posterior probabilities (start with prior)
        self.posterior_probs = dict(self.prior_probs)

    def update(self, new_value: str, posterior_probs: Optional[Dict[str, float]] = None) -> None:
        """
        Update the value and optionally the posterior probabilities.
        
        Args:
            new_value: New most likely value
            posterior_probs: New posterior probabilities, if None keeps current
        """
        if not isinstance(new_value, str):
            raise ValueError("Expected a string value for CategoricalLatentVariable.")
        if new_value not in self.possible_values:
            raise ValueError(f"Value {new_value} not in possible values: {self.possible_values}")
        self.value = new_value
        
        if posterior_probs is not None:
            # Validate and update posterior probabilities
            if not all(val in posterior_probs for val in self.possible_values):
                raise ValueError("posterior_probs must contain all possible values")
            
            # Normalize probabilities
            total = sum(posterior_probs.values())
            self.posterior_probs = {k: v/total for k, v in posterior_probs.items()}
            
    def sample_prior(self, n_samples: int = 1) -> Union[str, List[str]]:
        """
        Sample from the prior distribution.
        
        Args:
            n_samples: Number of samples to draw
            
        Returns:
            Single value or list of samples from prior distribution
        """
        values = list(self.prior_probs.keys())
        probs = list(self.prior_probs.values())
        
        samples = np.random.choice(values, size=n_samples, p=probs)
        return samples[0] if n_samples == 1 else samples.tolist()
        
    def sample_posterior(self, n_samples: int = 1) -> Union[str, List[str]]:
        """
        Sample from the posterior distribution.
        
        Args:
            n_samples: Number of samples to draw
            
        Returns:
            Single value or list of samples from posterior distribution
        """
        values = list(self.posterior_probs.keys())
        probs = list(self.posterior_probs.values())
        
        samples = np.random.choice(values, size=n_samples, p=probs)
        return samples[0] if n_samples == 1 else samples.tolist()
        
    def log_prior(self, value: str) -> float:
        """
        Compute log prior probability of a value.
        
        Args:
            value: Value to compute probability for
            
        Returns:
            Log probability of the value under the prior
        """
        if value not in self.prior_probs:
            return -float('inf')
        return np.log(self.prior_probs[value])


class DiscreteLatentVariable(LatentVariable):
    """
    A latent variable that can take discrete integer values.
    
    In the Bayesian brain hypothesis, discrete latent variables might represent
    counts, indices, or other integer-valued hidden causes.
    """
    def __init__(self, name: str, initial_value: int, min_value: Optional[int] = None, 
                 max_value: Optional[int] = None, description: str = "", relationships: Dict = None):
        """
        Initialize a discrete latent variable.
        
        Args:
            name: Name of the latent variable
            initial_value: Initial value (prior mean or mode)
            min_value: Minimum possible value (default: None = no minimum)
            max_value: Maximum possible value (default: None = no maximum)
            description: Description of what this variable represents
            relationships: Dictionary of relationships with other variables
        """
        super().__init__(name, initial_value, description, relationships)
        self.min_value = min_value
        self.max_value = max_value
        self._uncertainty = 1.0  # Default uncertainty for Poisson/Negative Binomial
        
    @property
    def uncertainty(self):
        """Get the current uncertainty parameter"""
        return self._uncertainty
        
    def update_uncertainty(self, new_uncertainty: float):
        """Update the uncertainty parameter"""
        self._uncertainty = max(0.0, new_uncertainty)
        
    def update(self, new_value: int, new_uncertainty: Optional[float] = None) -> None:
        """
        Update the value and optionally the uncertainty.
        
        Args:
            new_value: New value (posterior mean or mode)
            new_uncertainty: New uncertainty parameter, if None keeps current
        """
        if not isinstance(new_value, int):
            raise ValueError("Expected an integer value for DiscreteLatentVariable.")
            
        # Check boundaries if specified
        if self.min_value is not None and new_value < self.min_value:
            raise ValueError(f"Value {new_value} below minimum {self.min_value}")
        if self.max_value is not None and new_value > self.max_value:
            raise ValueError(f"Value {new_value} above maximum {self.max_value}")
            
        self.value = new_value
        
        if new_uncertainty is not None:
            self.update_uncertainty(new_uncertainty)
            
    def sample_prior(self, n_samples: int = 1) -> Union[int, np.ndarray]:
        """
        Sample from the prior distribution.
        
        Args:
            n_samples: Number of samples to draw
            
        Returns:
            Single value or array of samples from prior distribution
        """
        # Assume Poisson distribution for unbounded discrete variables
        # or Binomial for bounded ones
        if self.min_value is None and self.max_value is None:
            # Poisson distribution centered at initial value
            samples = np.random.poisson(self.initial_value, size=n_samples)
        else:
            # Bounded case: use Binomial with appropriate parameters
            if self.min_value is None:
                n = self.max_value
                p = self.initial_value / n
            elif self.max_value is None:
                # Hard to model with standard distributions, default to Poisson
                samples = np.random.poisson(self.initial_value - self.min_value, size=n_samples)
                samples = samples + self.min_value
                return samples[0] if n_samples == 1 else samples
            else:
                n = self.max_value - self.min_value
                p = (self.initial_value - self.min_value) / n
            
            samples = np.random.binomial(n, p, size=n_samples)
            if self.min_value is not None:
                samples = samples + self.min_value
                
        return samples[0] if n_samples == 1 else samples


class HierarchicalLatentVariable(ContinuousLatentVariable):
    """
    A latent variable with hierarchical dependencies on other latent variables.
    
    In the Bayesian brain hypothesis, hierarchical latent variables represent hidden
    causes that depend on other hidden causes at different levels of abstraction.
    """
    def __init__(self, name: str, parents: Dict[str, Any], relation_type: str = "linear",
                 parameters: Dict[str, Any] = None, initial_value: float = 0.0,
                 uncertainty: float = 1.0, description: str = ""):
        """
        Initialize a hierarchical latent variable.
        
        Args:
            name: Name of the latent variable
            parents: Dictionary mapping parent variable names to their values
            relation_type: Type of relation ("linear", "exponential", etc.)
            parameters: Parameters of the relation
            initial_value: Initial value (prior mean when parents are at reference values)
            uncertainty: Uncertainty (standard deviation of noise term)
            description: Description of what this variable represents
        """
        super().__init__(name, initial_value, uncertainty, description)
        self.parents = parents
        self.relation_type = relation_type
        self.parameters = parameters or {}
        
    def predict_from_parents(self, parent_values: Dict[str, Any]) -> float:
        """
        Predict the value of this variable from parent values.
        
        Args:
            parent_values: Dictionary mapping parent names to their values
            
        Returns:
            Predicted value based on the hierarchical relation
        """
        if self.relation_type == "linear":
            # Linear relation: y = b0 + b1*x1 + b2*x2 + ... + noise
            intercept = self.parameters.get("intercept", 0.0)
            coeffs = {k: v for k, v in self.parameters.items() if k != "intercept"}
            
            # Check all needed parents are provided
            for parent in coeffs:
                if parent not in parent_values:
                    raise ValueError(f"Missing parent value for '{parent}'")
            
            # Compute prediction
            predicted = intercept
            for parent, coef in coeffs.items():
                predicted += coef * parent_values[parent]
                
            return predicted
            
        elif self.relation_type == "exponential":
            # Exponential relation: y = a * exp(b1*x1 + b2*x2 + ...) + noise
            scale = self.parameters.get("scale", 1.0)
            coeffs = {k: v for k, v in self.parameters.items() if k != "scale"}
            
            # Check all needed parents are provided
            for parent in coeffs:
                if parent not in parent_values:
                    raise ValueError(f"Missing parent value for '{parent}'")
            
            # Compute prediction
            exponent = 0.0
            for parent, coef in coeffs.items():
                exponent += coef * parent_values[parent]
                
            predicted = scale * np.exp(exponent)
            return predicted
            
        else:
            raise ValueError(f"Unsupported relation type: {self.relation_type}")
    
    def update_from_parents(self, parent_values: Dict[str, Any], noise: float = 0.0):
        """
        Update the value based on parent values.
        
        Args:
            parent_values: Dictionary mapping parent names to their values
            noise: Optional noise term to add (default: 0)
        """
        predicted = self.predict_from_parents(parent_values)
        self.value = predicted + noise


# For backward compatibility
BaseFactor = LatentVariable
ContinuousFactor = ContinuousLatentVariable
CategoricalFactor = CategoricalLatentVariable
DiscreteFactor = DiscreteLatentVariable
BayesianLinearFactor = HierarchicalLatentVariable

# Example usage:
if __name__ == "__main__":
    # Create a continuous latent variable for temperature
    temperature = ContinuousLatentVariable(
        name="temperature",
        initial_value=25.0,  # 25°C
        uncertainty=3.0,     # ±3°C uncertainty
        description="Ambient temperature"
    )
    
    # Create a categorical latent variable for weather
    weather = CategoricalLatentVariable(
        name="weather",
        initial_value="Sunny",
        possible_values=["Sunny", "Cloudy", "Rainy", "Stormy"],
        prior_probs=[0.5, 0.3, 0.15, 0.05],  # Probabilities for each category
        description="Weather condition"
    )
    
    # Create a discrete latent variable for number of visitors
    visitors = DiscreteLatentVariable(
        name="visitors",
        initial_value=50,
        min_value=0,
        description="Number of visitors per day"
    )
    
    # Create a hierarchical latent variable for sales that depends on weather and temperature
    sales = HierarchicalLatentVariable(
        name="sales",
        parents={"temperature": 25.0, "weather": "Sunny"},
        relation_type="linear",
        parameters={
            "intercept": 1000,  # Base sales
            "temperature": 10,   # Each degree adds $10 in sales
            "Sunny": 200,        # Sunny weather adds $200 in sales
            "Rainy": -100        # Rainy weather reduces sales by $100
        },
        uncertainty=50.0,  # $50 uncertainty due to random factors
        description="Daily sales in dollars"
    )
    
    # Update a latent variable with new evidence
    temperature.update(28.0, 1.5)  # More certain about temperature now
    
    # Sample from distributions
    temp_samples = temperature.sample_posterior(5)
    weather_samples = weather.sample_posterior(5)
    
    print(f"Temperature samples: {temp_samples}")
    print(f"Weather samples: {weather_samples}")
    
    # Predict sales based on updated beliefs
    parent_values = {"temperature": temperature.value, "weather": weather.value}
    predicted_sales = sales.predict_from_parents(parent_values)
    print(f"Predicted sales: ${predicted_sales:.2f}") 