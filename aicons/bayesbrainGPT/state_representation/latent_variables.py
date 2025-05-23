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
    def __init__(self, name: str, value: Any = None, params: Dict = None, 
                 relationships: Dict = None, description: str = ""):
        """
        Initialize a latent variable.
        
        Args:
            name: Name of the latent variable
            value: Initial value (prior mean)
            params: Dictionary of parameters for the factor
            relationships: Dictionary of relationships with other variables
            description: Description of what this variable represents
        """
        self.name = name
        self.value = value
        self.initial_value = value  # Store initial value for reset
        self.description = description
        self.relationships = relationships or {}
        self.params = params or {}
        self.tf_distribution = None  # Will be set by subclasses

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
    def __init__(self, name: str, value: float, params: Dict = None, 
                 relationships: Dict = None, description: str = ""):
        """
        Initialize a continuous latent variable.
        
        Args:
            name: Name of the latent variable
            value: Initial value (prior mean)
            params: Dictionary of parameters including:
                   - loc: Location parameter (mean)
                   - scale: Scale parameter (std dev)
                   - constraints: Optional constraints dict with 'lower' and 'upper'
            relationships: Dictionary of relationships with other variables
            description: Description of what this variable represents
        """
        super().__init__(name, value, params, relationships, description)
        self._uncertainty = params.get('scale', 1.0) if params else 1.0
        self.constraints = params.get('constraints', {}) if params else {}
        
        # Create TensorFlow distribution
        self.tf_distribution = tfd.Normal(
            loc=float(value),
            scale=float(self._uncertainty)
        )

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


class CategoricalLatentVariable(LatentVariable):
    """
    A latent variable that can take one of several discrete categorical values.
    
    In the Bayesian brain hypothesis, categorical latent variables might represent
    discrete concepts like "sunny" vs "rainy" weather, or object categories.
    """
    def __init__(self, name: str, value: str, params: Dict = None, 
                 relationships: Dict = None, description: str = ""):
        """
        Initialize a categorical latent variable.
        
        Args:
            name: Name of the latent variable
            value: Initial most likely value
            params: Dictionary of parameters including:
                   - categories: List of all possible values
                   - probs: List of probabilities for each value
            relationships: Dictionary of relationships with other variables
            description: Description of what this variable represents
        """
        super().__init__(name, value, params, relationships, description)
        
        # Extract parameters
        self.possible_values = params.get('categories', [value])
        probabilities = params.get('probs', [1.0/len(self.possible_values)] * len(self.possible_values))
        
        # Ensure initial value is in possible values
        if value not in self.possible_values:
            self.possible_values.append(value)
            probabilities.append(1.0/len(self.possible_values))
        
        # Set up probabilities
        self.prior_probs = {val: prob for val, prob in zip(self.possible_values, probabilities)}
        self.posterior_probs = dict(self.prior_probs)
        
        # Create TensorFlow distribution
        self.tf_distribution = tfd.Categorical(probs=probabilities)

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


class DiscreteLatentVariable(LatentVariable):
    """
    Discrete latent variable with either categorical or Poisson distribution.
    
    For categorical-like discrete variables:
        - distribution_params should contain 'categories' and 'probs'
    
    For Poisson-distributed variables:
        - distribution_params should contain 'rate'
    """
    def __init__(self, name: str, value: int, params: Dict[str, Any], relationships: Dict[str, List[str]] = None, description: str = ""):
        super().__init__(name, value, params, relationships, description)
        
        # Store distribution parameters
        self.distribution_params = params
        
        # Create appropriate distribution
        if 'categories' in params:
            # Categorical-like discrete
            self.distribution = tfd.Categorical(
                probs=params['probs']
            )
        else:
            # Poisson distribution
            self.distribution = tfd.Poisson(
                rate=float(params.get('rate', value))
            )
            
    def log_prob(self, value: int) -> float:
        """Compute log probability of a value"""
        return float(self.distribution.log_prob(value))
        
    def __str__(self) -> str:
        """String representation"""
        if 'categories' in self.distribution_params:
            return f"{self.name}: {self.value} (categorical: {self.distribution_params['categories']})"
        else:
            return f"{self.name}: {self.value} (Poisson rate: {self.distribution_params.get('rate', self.value)})"


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
