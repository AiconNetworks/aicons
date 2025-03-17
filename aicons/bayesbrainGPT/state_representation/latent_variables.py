"""
Bayesian Brain-inspired Latent Variables

This module provides representations of latent variables in a Bayesian brain model.
These latent variables represent unobserved causes that the brain infers from sensory data.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# TFP shortcuts
tfd = tfp.distributions
tfb = tfp.bijectors

class LatentVariable(ABC):
    """
    Base class for latent variables in a Bayesian brain model.
    
    In the Bayesian brain hypothesis, latent variables represent
    hidden causes that the brain infers from sensory observations.
    """
    def __init__(self, name: str, initial_value: Any = None, description: str = "", 
                 relationships: Dict = None, tf_distribution = None):
        """
        Initialize a latent variable.
        
        Args:
            name: Name of the latent variable
            initial_value: Initial value (prior mean)
            description: Description of what this variable represents
            relationships: Dictionary of relationships with other variables
            tf_distribution: TensorFlow probability distribution
        """
        self.name = name
        self.initial_value = initial_value
        self.value = initial_value  # Current value (posterior mean)
        self.description = description
        self.relationships = relationships or {}  # Store hierarchical relationships
        self.tf_distribution = tf_distribution  # Store TensorFlow distribution

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
                 description: str = "", relationships: Dict = None, 
                 tf_distribution = None, constraints = None):
        """
        Initialize a continuous latent variable.
        
        Args:
            name: Name of the latent variable
            initial_value: Initial value (prior mean)
            uncertainty: Initial uncertainty (prior standard deviation)
            description: Description of what this variable represents
            relationships: Dictionary of relationships with other variables
            tf_distribution: TensorFlow probability distribution
            constraints: Constraints on the variable (e.g., bounds)
        """
        super().__init__(name, initial_value, description, relationships, tf_distribution)
        self._uncertainty = uncertainty  # Prior/posterior uncertainty (std dev)
        self.constraints = constraints or {}
        
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
        if self.tf_distribution:
            # Use TensorFlow distribution if available
            samples = self.tf_distribution.sample(n_samples).numpy()
            return samples[0] if n_samples == 1 else samples
            
        # Fallback to numpy for Gaussian prior
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
                 description: str = "", relationships: Dict = None,
                 tf_distribution = None, probabilities = None):
        """
        Initialize a categorical latent variable.
        
        Args:
            name: Name of the latent variable
            initial_value: Initial most likely value
            possible_values: List of all possible values
            prior_probs: List of prior probabilities for each value (default uniform)
            description: Description of what this variable represents
            relationships: Dictionary of relationships with other variables
            tf_distribution: TensorFlow probability distribution
            probabilities: Probabilities for each category (alternative to prior_probs)
        """
        super().__init__(name, initial_value, description, relationships, tf_distribution)
        self.possible_values = possible_values
        
        # Ensure initial value is in possible values
        if initial_value not in self.possible_values:
            self.possible_values.append(initial_value)
        
        # Use probabilities if provided (from TF distribution creation)
        if probabilities is not None:
            if len(probabilities) != len(possible_values):
                raise ValueError("probabilities must have same length as possible_values")
            self.prior_probs = {val: prob for val, prob in zip(possible_values, probabilities)}
        # Set up prior probabilities (default to uniform if not provided)
        elif prior_probs is None:
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
        if self.tf_distribution:
            # Use TensorFlow distribution if available
            samples = self.tf_distribution.sample(n_samples).numpy()
            
            # Convert indices to category names
            if n_samples == 1:
                return self.possible_values[int(samples)]
            else:
                return [self.possible_values[int(idx)] for idx in samples]
            
        # Fallback to numpy
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
    counts or indices of hidden causes, like number of objects.
    """
    def __init__(self, name: str, initial_value: int, description: str = "", 
                 relationships: Dict = None, tf_distribution = None,
                 distribution_params = None, constraints = None):
        """
        Initialize a discrete latent variable.
        
        Args:
            name: Name of the latent variable
            initial_value: Initial value (prior mean or mode)
            description: Description of what this variable represents
            relationships: Dictionary of relationships with other variables
            tf_distribution: TensorFlow probability distribution
            distribution_params: Parameters for the distribution (e.g., rate for Poisson)
            constraints: Constraints on the variable (e.g., min/max values)
        """
        super().__init__(name, initial_value, description, relationships, tf_distribution)
        self.distribution_params = distribution_params or {}
        self.constraints = constraints or {}
        
        # For Poisson distribution, parameter is the rate (mean)
        if 'rate' not in self.distribution_params and not tf_distribution:
            self.distribution_params['rate'] = float(initial_value)

    def update(self, new_value: int, new_params: Optional[Dict] = None) -> None:
        """
        Update the value and optionally the distribution parameters.
        
        Args:
            new_value: New value (posterior mean or mode)
            new_params: New distribution parameters, if None keeps current
        """
        if not isinstance(new_value, (int, np.integer)):
            raise ValueError("Expected an integer value for DiscreteLatentVariable.")
        self.value = new_value
        
        if new_params is not None:
            self.distribution_params.update(new_params)
            
    def sample_prior(self, n_samples: int = 1) -> Union[int, np.ndarray]:
        """
        Sample from the prior distribution.
        
        Args:
            n_samples: Number of samples to draw
            
        Returns:
            Single value or array of samples from prior distribution
        """
        if self.tf_distribution:
            # Use TensorFlow distribution if available
            samples = self.tf_distribution.sample(n_samples).numpy()
            
            if hasattr(samples, '__len__') and n_samples == 1:
                return int(samples[0])
            elif n_samples == 1:
                return int(samples)
                
            # Convert to integers if needed
            return samples.astype(int)
            
        # Fallback to numpy for Poisson/categorical
        if 'categories' in self.distribution_params:
            # For categorical-like discrete 
            categories = self.distribution_params['categories']
            probs = self.distribution_params.get('probs', [1.0/len(categories)] * len(categories))
            samples = np.random.choice(categories, size=n_samples, p=probs)
        else:
            # For Poisson
            rate = self.distribution_params.get('rate', float(self.initial_value))
            samples = np.random.poisson(rate, size=n_samples)
            
        return int(samples[0]) if n_samples == 1 else samples


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