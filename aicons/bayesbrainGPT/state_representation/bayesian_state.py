"""
Bayesian Brain-inspired State Representation

This module provides a state representation based on the Bayesian brain hypothesis,
where the brain's internal model consists of latent variables that explain sensory observations.
"""

import json
from typing import Dict, Any, List, Optional, Union
import numpy as np
from pathlib import Path
import tensorflow as tf
import tensorflow_probability as tfp

# TFP shortcuts
tfd = tfp.distributions
tfb = tfp.bijectors

# Import directly from latent_variables
from .latent_variables import ContinuousLatentVariable, CategoricalLatentVariable, DiscreteLatentVariable, HierarchicalLatentVariable

class BayesianState:
    """
    Represents a collection of latent variables in a Bayesian brain model.
    
    In the Bayesian brain hypothesis, the brain maintains a generative model of the world,
    consisting of latent (hidden) variables that explain sensory observations.
    This class manages these latent variables and their prior distributions.
    
    Can be initialized from:
    1. Configuration-based priors
    2. LLM-derived latent variables
    3. Manual latent variable addition
    """
    def __init__(self, latent_config=None, use_llm=False, mock_llm=True):
        """
        Initialize a Bayesian state with latent variables.
        
        Args:
            latent_config: Configuration for latent variables
            use_llm: Whether to use LLM for initialization
            mock_llm: Whether to use mock LLM data
        """
        # For compatibility with existing code, we keep using "factors" in the internal dict
        self.factors = {}
        
        # Initialize latent variables from config or LLM
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
        elif latent_config:
            # Use provided configuration
            self._initialize_from_config(latent_config)
        
        # For hierarchical generative models
        self.prior_distributions = {}  # TFP/Pyro distributions for each latent variable
        self.hierarchical_relations = {}  # Conditional dependencies between variables
    
    def _initialize_from_llm_data(self, llm_factors):
        """
        Initialize latent variables from LLM-derived data.
        
        Args:
            llm_factors: List of factor data from LLM
        """
        for factor in llm_factors:
            name = factor.get("description", "").lower().replace(" ", "_")
            if factor["type"] == "continuous":
                self.factors[name] = ContinuousLatentVariable(
                    name=name,
                    initial_value=float(factor["value"]),
                    description=factor["description"]
                )
            elif factor["type"] == "categorical":
                self.factors[name] = CategoricalLatentVariable(
                    name=name,
                    initial_value=factor["value"],
                    description=factor["description"]
                )
            elif factor["type"] == "discrete":
                self.factors[name] = DiscreteLatentVariable(
                    name=name,
                    initial_value=int(float(factor["value"])),
                    description=factor["description"]
                )

    def _initialize_from_config(self, config):
        """
        Initialize latent variables from configuration dictionary.
        
        Args:
            config: Dictionary of latent variable configurations
        """
        for name, factor_config in config.items():
            if factor_config["type"] == "continuous":
                self.factors[name] = ContinuousLatentVariable(
                    name=name,
                    initial_value=factor_config["value"],
                    description=factor_config["description"],
                    relationships=factor_config.get("relationships", {})
                )
            elif factor_config["type"] == "categorical":
                self.factors[name] = CategoricalLatentVariable(
                    name=name,
                    initial_value=factor_config["value"],
                    description=factor_config["description"],
                    relationships=factor_config.get("relationships", {}),
                    possible_values=factor_config.get("possible_values", None)
                )
            elif factor_config["type"] == "discrete":
                self.factors[name] = DiscreteLatentVariable(
                    name=name,
                    initial_value=factor_config["value"],
                    description=factor_config["description"]
                )
            elif factor_config["type"] == "bayesian_linear":
                self.factors[name] = HierarchicalLatentVariable(
                    name=name,
                    parents=factor_config["explanatory_vars"],
                    parameters=factor_config["theta_prior"],
                    uncertainty=factor_config["variance"],
                    description=factor_config["description"]
                )

    def __str__(self):
        """String representation of the Bayesian state"""
        return "\n".join(f"{k}: {v}" for k, v in self.factors.items())

    def get_beliefs(self) -> Dict[str, Any]:
        """
        Retrieve current beliefs about latent variables as a simple dictionary.
        
        In the Bayesian brain hypothesis, these represent the brain's current
        best estimates of the latent variables' values.
        
        Returns:
            Dictionary mapping latent variable names to their current values
        """
        return {key: factor.value for key, factor in self.factors.items()}

    def __repr__(self) -> str:
        """Detailed representation of the Bayesian state"""
        state_repr = "\n".join([repr(factor) for factor in self.factors.values()])
        return f"BayesianState:\n{state_repr}"

    def reset(self):
        """Reset all latent variables to their initial values (prior means)"""
        for factor in self.factors.values():
            factor.reset()

    # Methods for adding different types of latent variables
    
    def add_continuous_latent(self, name: str, mean: float, uncertainty: float = 1.0, 
                            description: str = "", relationships: Dict = None,
                            lower_bound: Optional[float] = None, upper_bound: Optional[float] = None) -> ContinuousLatentVariable:
        """
        Add a continuous latent variable to the state with TensorFlow distribution.
        
        In the Bayesian brain hypothesis, continuous latent variables represent
        hidden causes that can take any real value, like temperature or intensity.
        
        Args:
            name: Name of the latent variable
            mean: Prior mean (initial value)
            uncertainty: Prior uncertainty (standard deviation)
            description: Description of what this latent variable represents
            relationships: Dictionary of relationships with other variables
            lower_bound: Optional lower bound constraint
            upper_bound: Optional upper bound constraint
            
        Returns:
            The created continuous latent variable
        """
        # Create TensorFlow distribution directly
        tf_dist = None
        if lower_bound is not None and upper_bound is not None:
            # Truncated normal for bounded variables
            tf_dist = tfd.TruncatedNormal(
                loc=float(mean), 
                scale=float(uncertainty),
                low=float(lower_bound),
                high=float(upper_bound)
            )
        elif lower_bound is not None:
            # Transformed distribution for lower-bounded variables
            shift = float(lower_bound)
            tf_dist = tfd.TransformedDistribution(
                distribution=tfd.Normal(loc=float(mean)-shift, scale=float(uncertainty)),
                bijector=tfb.Shift(shift=shift) @ tfb.Softplus()
            )
        elif upper_bound is not None:
            # Transformed distribution for upper-bounded variables
            shift = float(upper_bound)
            tf_dist = tfd.TransformedDistribution(
                distribution=tfd.Normal(loc=shift-float(mean), scale=float(uncertainty)),
                bijector=tfb.Shift(shift=shift) @ tfb.Scale(-1.0) @ tfb.Softplus()
            )
        else:
            # Unconstrained normal distribution
            tf_dist = tfd.Normal(loc=float(mean), scale=float(uncertainty))
            
        # Create latent variable
        latent = ContinuousLatentVariable(
            name=name,
            initial_value=mean,
            uncertainty=uncertainty,
            description=description,
            relationships=relationships,
            tf_distribution=tf_dist  # Attach TF distribution
        )
        
        # Store constraints if provided
        if lower_bound is not None or upper_bound is not None:
            constraints = {}
            if lower_bound is not None:
                constraints["lower"] = lower_bound
            if upper_bound is not None:
                constraints["upper"] = upper_bound
            latent.constraints = constraints
            
        self.factors[name] = latent
        return latent

    def add_categorical_latent(self, name: str, initial_value: str, possible_values: List[str], 
                             description: str = "", relationships: Dict = None,
                             probs: Optional[List[float]] = None) -> CategoricalLatentVariable:
        """
        Add a categorical latent variable to the state with TensorFlow distribution.
        
        In the Bayesian brain hypothesis, categorical latent variables represent
        hidden causes that can take one of several discrete values, like "sunny" or "rainy".
        
        Args:
            name: Name of the latent variable
            initial_value: Initial (most probable) value
            possible_values: List of all possible values
            description: Description of what this latent variable represents
            relationships: Dictionary of relationships with other variables
            probs: Optional probability for each category (should sum to 1)
            
        Returns:
            The created categorical latent variable
        """
        # Create probability distribution
        if probs is None:
            # Equal probability for all categories
            probs = [1.0 / len(possible_values)] * len(possible_values)
            
        # Convert probs to tensor and create TF distribution
        probs_tensor = tf.constant(probs, dtype=tf.float32)
        tf_dist = tfd.Categorical(probs=probs_tensor)
        
        latent = CategoricalLatentVariable(
            name=name,
            initial_value=initial_value,
            description=description,
            relationships=relationships,
            possible_values=possible_values,
            tf_distribution=tf_dist,  # Attach TF distribution
            probabilities=probs
        )
        self.factors[name] = latent
        return latent

    def add_discrete_latent(self, name: str, initial_value: int, description: str = "", 
                          relationships: Dict = None, min_value: int = 0,
                          max_value: Optional[int] = None,
                          rate: Optional[float] = None) -> DiscreteLatentVariable:
        """
        Add a discrete latent variable to the state with TensorFlow distribution.
        
        In the Bayesian brain hypothesis, discrete latent variables represent
        hidden causes that can take integer values, like counts or indices.
        
        Args:
            name: Name of the latent variable
            initial_value: Initial (most probable) value
            description: Description of what this latent variable represents
            relationships: Dictionary of relationships with other variables
            min_value: Minimum possible value
            max_value: Maximum possible value (if None, unbounded above)
            rate: Rate parameter for Poisson distribution (if unbounded)
            
        Returns:
            The created discrete latent variable
        """
        # Create TensorFlow distribution
        tf_dist = None
        
        if max_value is None:
            # Poisson distribution for unbounded discrete values
            if rate is None:
                rate = float(initial_value)
            tf_dist = tfd.Poisson(rate=rate)
            
            constraints = {"lower": min_value}
            distribution_params = {"rate": rate}
        else:
            # Categorical distribution for bounded discrete values
            num_values = max_value - min_value + 1
            categories = list(range(min_value, max_value + 1))
            
            # Create probabilities centered on initial value
            probs = [0.0] * num_values
            index = categories.index(initial_value)
            probs[index] = 1.0
            
            # Convert to tensor
            probs_tensor = tf.constant(probs, dtype=tf.float32)
            tf_dist = tfd.Categorical(probs=probs_tensor)
            
            distribution_params = {"probs": probs}
            constraints = {"lower": min_value, "upper": max_value}
            
        latent = DiscreteLatentVariable(
            name=name,
            initial_value=initial_value,
            description=description,
            relationships=relationships,
            tf_distribution=tf_dist,  # Attach TF distribution
            distribution_params=distribution_params,
            constraints=constraints
        )
        self.factors[name] = latent
        return latent

    def add_hierarchical_latent(self, name: str, explanatory_vars: Dict = None, theta_prior: Dict = None,
                               variance: float = 1.0, description: str = "") -> HierarchicalLatentVariable:
        """
        Add a hierarchical latent variable to the state.
        
        In the Bayesian brain hypothesis, hierarchical latent variables represent
        hidden causes that depend on other hidden causes at a higher level of abstraction.
        
        Args:
            name: Name of the latent variable
            explanatory_vars: Dictionary of higher-level variables that explain this one
            theta_prior: Prior distribution for the relationship strengths
            variance: Variance of the residual noise
            description: Description of what this latent variable represents
            
        Returns:
            The created hierarchical latent variable
        """
        latent = HierarchicalLatentVariable(
            name=name,
            parents=explanatory_vars or {},
            parameters=theta_prior or {},
            uncertainty=variance,
            description=description
        )
        self.factors[name] = latent
        return latent
    
    def set_hierarchical_relation(self, child: str, parents: List[str], relation_type: str = "linear", 
                                parameters: Dict = None):
        """
        Set a hierarchical relation between latent variables.
        
        In the Bayesian brain hypothesis, hierarchical relations define how higher-level
        hidden causes generate or explain lower-level hidden causes.
        
        Args:
            child: Name of the child latent variable
            parents: List of parent latent variable names
            relation_type: Type of relation ("linear", "exponential", etc.)
            parameters: Parameters of the relation
        """
        if child not in self.factors:
            raise ValueError(f"Child latent variable {child} not found")
        
        for parent in parents:
            if parent not in self.factors:
                raise ValueError(f"Parent latent variable {parent} not found")
        
        # Store the hierarchical relation
        self.hierarchical_relations[child] = {
            "parents": parents,
            "type": relation_type,
            "parameters": parameters or {}
        }
        
        # For compatibility with the existing system
        if hasattr(self.factors[child], "relationships"):
            if not self.factors[child].relationships:
                self.factors[child].relationships = {}
            
            if "depends_on" not in self.factors[child].relationships:
                self.factors[child].relationships["depends_on"] = []
            
            # Add parents to dependencies
            for parent in parents:
                if parent not in self.factors[child].relationships["depends_on"]:
                    self.factors[child].relationships["depends_on"].append(parent)

    # Additional methods for hierarchical generative models can be added here


# For backward compatibility with any code that directly imports EnvironmentState
EnvironmentState = BayesianState
# Ensure exports include both classes
__all__ = ['BayesianState', 'EnvironmentState'] 