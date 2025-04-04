"""
Bayesian Brain-inspired State Representation

This module provides a state representation based on the Bayesian brain hypothesis,
where the brain's internal model consists of latent variables that explain sensory observations.
"""

import json
from typing import Dict, Any, List, Optional, Union, Callable, Set
import numpy as np
from pathlib import Path
import tensorflow as tf
import tensorflow_probability as tfp
import time

# TFP shortcuts
tfd = tfp.distributions
tfb = tfp.bijectors

# Import directly from latent_variables
from .latent_variables import (
    ContinuousLatentVariable, 
    CategoricalLatentVariable, 
    DiscreteLatentVariable, 
    HierarchicalLatentVariable
)

class BayesianState:
    """
    Represents a collection of latent variables in a Bayesian brain model.
    
    In the Bayesian brain hypothesis, the brain maintains a generative model of the world,
    consisting of latent (hidden) variables that explain sensory observations.
    This class manages these latent variables and their prior distributions.
    
    Can be initialized from:
    1. Configuration-based priors
    2. Manual latent variable addition
    """
    def __init__(self):
        """
        Initialize a Bayesian state with latent variables.
        
        Args:
            latent_config: Configuration for latent variables
        """
        self.factors = {}
        self.hierarchical_relations = {}
        self.conditional_distributions = {}
        self.topological_order = []
        self.prior_distributions = {}  # TFP distributions for each latent variable
        self.conditional_distributions = {}  # Functions that return conditional distributions
        self.topological_order = []  # Ordering of factors for joint distribution
        
        # State management
        self.posterior_samples = None  # Current posterior samples
        self.last_posterior_update = None  # Timestamp of last posterior update
        self.update_history = []  # History of state updates
    


    def __str__(self):
        """String representation of the state"""
        output = "State Representation:\n"
        for factor_name in self.topological_order:
            factor = self.factors[factor_name]
            if isinstance(factor, ContinuousLatentVariable):
                output += f"{factor_name}: {factor.value:.2f}"
                if hasattr(factor, 'constraints') and factor.constraints:
                    output += f" (constraints: {factor.constraints})"
                if hasattr(factor, '_uncertainty'):
                    output += f" (uncertainty: {factor._uncertainty:.2f})"
            elif isinstance(factor, CategoricalLatentVariable):
                output += f"{factor_name}: {factor.value}"
                if hasattr(factor, 'prior_probs'):
                    output += f" (probs: {factor.prior_probs})"
            else:  # DiscreteLatentVariable
                output += f"{factor_name}: {factor.value}"
                if hasattr(factor, 'distribution_params'):
                    if 'categories' in factor.distribution_params:
                        output += f" (categories: {factor.distribution_params['categories']})"
                    elif 'rate' in factor.distribution_params:
                        output += f" (Poisson rate: {factor.distribution_params['rate']})"
            output += "\n"
            
        output += "\nFactor Relationships:\n"
        for factor_name in self.topological_order:
            factor = self.factors[factor_name]
            depends_on = factor.relationships.get('depends_on', [])
            output += f"\n{factor_name}:\n"
            if depends_on:
                output += f"  Depends on: {depends_on}\n"
            else:
                output += "  Root factor (no dependencies)\n"
                
        return output

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

    def add_factor(self, name: str, factor_type: str, value: Any, params: Dict[str, Any], relationships: Optional[Dict[str, List[str]]] = None) -> Union[ContinuousLatentVariable, CategoricalLatentVariable, DiscreteLatentVariable]:
        """
        Add a new factor to the state with proper validation and hierarchical relationships.
        
        This method enforces several validation rules:
        1. Factor Creation Order: Parent factors must exist before dependent factors
        2. Relationship Format: Relationships must be a dict with 'depends_on' as a list
        3. Circular Dependencies: No cycles allowed in factor relationships
        4. Factor Type Validation: Each type has specific parameter requirements
        
        Args:
            name: Name of the factor
            factor_type: Type of factor ('continuous', 'categorical', or 'discrete')
            value: Initial value of the factor
            params: Dictionary of parameters for the factor
            relationships: Optional dictionary specifying hierarchical relationships
                         Format: {'depends_on': [parent_factor_names]}
        
        Returns:
            The created latent variable (ContinuousLatentVariable, CategoricalLatentVariable, or DiscreteLatentVariable)
            
        Raises:
            ValueError: If validation fails for any of the following:
                - Factor name already exists
                - Invalid factor type
                - Missing required parameters
                - Parent factors don't exist
                - Circular dependencies detected
                - Invalid relationship format
        """
        # Validate factor name
        if name in self.factors:
            raise ValueError(f"Factor '{name}' already exists")
            
        # Validate relationships format
        if relationships is None:
            relationships = {}
        if not isinstance(relationships, dict):
            raise ValueError("Relationships must be a dictionary")
        if "depends_on" in relationships and not isinstance(relationships["depends_on"], list):
            raise ValueError("'depends_on' must be a list of factor names")
            
        # Check if this is a root factor (no dependencies)
        is_root = not relationships or not relationships.get("depends_on")
        
        # Validate parent factors exist
        if not is_root:
            parent_names = relationships["depends_on"]
            missing_parents = [p for p in parent_names if p not in self.factors]
            if missing_parents:
                raise ValueError(f"Parent factors do not exist: {missing_parents}")
                
        # Check for circular dependencies using DFS
        def has_cycle(current: str, visited: Set[str], path: Set[str]) -> bool:
            visited.add(current)
            path.add(current)
            
            if current in self.hierarchical_relations:
                for parent in self.hierarchical_relations[current]["parents"]:
                    if parent not in visited:
                        if has_cycle(parent, visited, path):
                            return True
                    elif parent in path:
                        return True
                        
            path.remove(current)
            return False
            
        # Test for cycles if this is not a root factor
        if not is_root:
            visited = set()
            path = set()
            if has_cycle(name, visited, path):
                raise ValueError(f"Circular dependency detected involving factor '{name}'")
                
        # Validate factor type and parameters
        if factor_type == "continuous":
            required_params = {"loc", "scale"}
            missing_params = required_params - set(params.keys())
            if missing_params:
                raise ValueError(f"Missing required parameters for continuous factor: {missing_params}")
                
            # Validate constraints if present
            if "constraints" in params:
                constraints = params["constraints"]
                if not isinstance(constraints, dict):
                    raise ValueError("Constraints must be a dictionary")
                if "lower" in constraints and "upper" in constraints:
                    if constraints["lower"] is not None and constraints["upper"] is not None:
                        if constraints["lower"] >= constraints["upper"]:
                            raise ValueError("Lower bound must be less than upper bound")
                        
        elif factor_type == "categorical":
            required_params = {"categories", "probs"}
            missing_params = required_params - set(params.keys())
            if missing_params:
                raise ValueError(f"Missing required parameters for categorical factor: {missing_params}")
                
            if not isinstance(params["categories"], list):
                raise ValueError("Categories must be a list")
            if not isinstance(params["probs"], list):
                raise ValueError("Probabilities must be a list")
            if len(params["categories"]) != len(params["probs"]):
                raise ValueError("Number of categories must match number of probabilities")
            if not np.isclose(sum(params["probs"]), 1.0):
                raise ValueError("Probabilities must sum to 1.0")
                
        elif factor_type == "discrete":
            if "categories" in params:
                # Finite discrete values
                required_params = {"categories", "probs"}
                missing_params = required_params - set(params.keys())
                if missing_params:
                    raise ValueError(f"Missing required parameters for discrete factor: {missing_params}")
                    
                if not isinstance(params["categories"], list):
                    raise ValueError("Categories must be a list")
                if not isinstance(params["probs"], list):
                    raise ValueError("Probabilities must be a list")
                if len(params["categories"]) != len(params["probs"]):
                    raise ValueError("Number of categories must match number of probabilities")
                if not np.isclose(sum(params["probs"]), 1.0):
                    raise ValueError("Probabilities must sum to 1.0")
            else:
                # Poisson distribution
                if "rate" not in params:
                    raise ValueError("Missing required parameter 'rate' for Poisson distribution")
                if params["rate"] <= 0:
                    raise ValueError("Rate must be positive")
                    
        else:
            raise ValueError(f"Invalid factor type: {factor_type}")
            
        # Create the factor based on type
        if factor_type == "continuous":
            factor = ContinuousLatentVariable(
                name=name,
                value=value,
                params=params,
                relationships=relationships
            )
        elif factor_type == "categorical":
            factor = CategoricalLatentVariable(
                name=name,
                value=value,
                params=params,
                relationships=relationships
            )
        else:  # discrete
            factor = DiscreteLatentVariable(
                name=name,
                value=value,
                params=params,
                relationships=relationships
            )
            
        # Add factor to state
        self.factors[name] = factor
        
        # Update hierarchical relations if not a root factor
        if not is_root:
            self.hierarchical_relations[name] = {
                "parents": relationships["depends_on"],
                "children": []
            }
            for parent in relationships["depends_on"]:
                if parent in self.hierarchical_relations:
                    self.hierarchical_relations[parent]["children"].append(name)
                    
        # Update topological order
        self._update_topological_order()
        
        return factor

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

    def add_conditional_factor(self, name: str, parent_factor_names: List[str], 
                                conditional_dist_fn: Callable, description: str = ""):
        """
        Add a factor that depends on other factors (hierarchical relationship).
        
        In the Bayesian brain hypothesis, conditional relationships form the
        hierarchical generative model that explains how the world works.
        
        Args:
            name: Name of the factor
            parent_factor_names: Names of parent factors this factor depends on
            conditional_dist_fn: Function that takes parent values and returns a distribution
            description: Description of what this factor represents
            
        Returns:
            The created factor
        """
        # Verify parent factors exist
        for parent_name in parent_factor_names:
            if parent_name not in self.factors:
                raise ValueError(f"Parent factor '{parent_name}' does not exist")
        
        # Store the conditional distribution function
        self.conditional_distributions[name] = conditional_dist_fn
        
        # Create a hierarchical latent variable
        factor = HierarchicalLatentVariable(
            name=name,
            parents={parent: 1.0 for parent in parent_factor_names},  # Placeholder weights
            parameters={},  # Will be determined by conditional_dist_fn
            uncertainty=1.0,  # Placeholder, will be determined by conditional_dist_fn
            description=description
        )
        
        # Add to factors
        self.factors[name] = factor
        
        # Update hierarchical relations
        self.hierarchical_relations[name] = {
            "parents": parent_factor_names,
            "type": "conditional"
        }
        
        # Update topological order (a simple implementation)
        self._update_topological_order()
        
        return factor
        
    def _update_topological_order(self):
        """
        Update the topological ordering of factors based on dependencies.
        This ensures factors are evaluated in the correct order in the joint distribution.
        """
        # Start with factors that have no parents
        self.topological_order = []
        visited = set()
        
        # Find root factors (no parents)
        root_factors = []
        for name in self.factors:
            if name not in self.hierarchical_relations:
                root_factors.append(name)
                
        # Do topological sort
        def visit(factor_name):
            if factor_name in visited:
                return
            visited.add(factor_name)
            
            # Visit parents first
            if factor_name in self.hierarchical_relations:
                for parent in self.hierarchical_relations[factor_name]["parents"]:
                    visit(parent)
                    
            self.topological_order.append(factor_name)
            
        # Visit all factors
        for name in self.factors:
            visit(name)
    
    def create_joint_distribution(self):
        """
        Create a TensorFlow Probability joint distribution from all factors.
        Handles hierarchical relationships between factors.
        
        Returns:
            A joint distribution object that can be used for sampling
        """
        # Create distributions in topological order
        distributions = []
        for factor_name in self.topological_order:
            factor = self.factors[factor_name]
            
            if isinstance(factor, ContinuousLatentVariable):
                # For continuous variables, use Normal distribution with constraints
                if hasattr(factor, 'constraints') and factor.constraints:
                    # Apply constraints using bijectors
                    bijectors = []
                    if factor.constraints.get('lower') is not None:
                        bijectors.append(tfb.Shift(factor.constraints['lower']))
                    if factor.constraints.get('upper') is not None:
                        bijectors.append(tfb.Exp())  # Ensure positive
                    if bijectors:
                        dist = tfd.TransformedDistribution(
                            distribution=tfd.Normal(
                                loc=factor.value,
                                scale=factor._uncertainty
                            ),
                            bijector=tfb.Chain(bijectors)
                        )
                    else:
                        dist = tfd.Normal(
                            loc=factor.value,
                            scale=factor._uncertainty
                        )
                else:
                    dist = tfd.Normal(
                        loc=factor.value,
                        scale=factor._uncertainty
                    )
                    
            elif isinstance(factor, CategoricalLatentVariable):
                # For categorical variables, use Categorical distribution
                categories = list(factor.prior_probs.keys())
                probs = list(factor.prior_probs.values())
                dist = tfd.Categorical(probs=probs)
                
            else:  # DiscreteLatentVariable
                if hasattr(factor, 'distribution_params') and 'categories' in factor.distribution_params:
                    # For categorical-like discrete
                    dist = tfd.Categorical(
                        probs=factor.distribution_params['probs']
                    )
                else:
                    # For Poisson
                    dist = tfd.Poisson(
                        rate=factor.distribution_params.get('rate', float(factor.value))
                    )
            
            distributions.append(dist)
            
        # Create joint distribution that respects dependencies
        return tfd.JointDistributionSequential(distributions)
    
    def _get_conditional_dist(self, factor_name, parent_values):
        """
        Get conditional distribution for a factor given parent values.
        
        Args:
            factor_name: Name of the factor
            parent_values: Values of parent factors
            
        Returns:
            TensorFlow Probability distribution conditioned on parent values
        """
        # Get parent names
        parent_names = self.hierarchical_relations[factor_name]["parents"]
        
        # Create dictionary mapping parent names to values
        parent_dict = {name: value for name, value in zip(parent_names, parent_values)}
        
        # Call conditional distribution function
        return self.conditional_distributions[factor_name](**parent_dict)
    
    def sample_from_prior(self, n_samples=1):
        """
        Sample from the joint prior distribution of all factors.
        
        Args:
            n_samples: Number of samples to draw
            
        Returns:
            List of dictionaries mapping factor names to sample values
        """
        # Create joint distribution
        joint_dist = self.create_joint_distribution()
        
        # Sample from joint distribution
        samples = joint_dist.sample(n_samples)
        
        # Convert to list of dictionaries mapping factor names to values
        if n_samples == 1:
            # Single sample case
            return {name: value.numpy() for name, value in zip(self.topological_order, samples)}
        else:
            # Multiple samples case
            return [{name: value.numpy() for name, value in zip(self.topological_order, sample)} 
                   for sample in samples]

    def get_state_factors(self) -> Dict[str, Any]:
        """
        Get all state factors in a format suitable for the perception system's TFP distributions.
        
        Returns:
            Dictionary containing state factors with their properties:
            - type: Factor type ('continuous', 'categorical', 'discrete')
            - value: Current value
            - distribution: TFP distribution type ('normal', 'truncated_normal', 'categorical', 'poisson')
            - params: Distribution parameters (loc, scale, probs, etc.)
            - constraints: Any constraints on the factor
            - relationships: Hierarchical relationships with other factors
            - uncertainty: Uncertainty/variance parameter
        """
        state_factors = {}
        
        for name, factor in self.factors.items():
            factor_info = {
                "type": "continuous" if isinstance(factor, ContinuousLatentVariable) else
                       "categorical" if isinstance(factor, CategoricalLatentVariable) else
                       "discrete",
                "value": factor.value,
                "distribution": None,
                "params": {},
                "constraints": {},
                "relationships": factor.relationships,
                "uncertainty": None
            }
            
            if isinstance(factor, ContinuousLatentVariable):
                factor_info["distribution"] = "normal"
                factor_info["params"] = {
                    "loc": factor.value,
                    "scale": factor._uncertainty
                }
                if hasattr(factor, 'constraints') and factor.constraints:
                    factor_info["constraints"] = factor.constraints
                    if "lower" in factor.constraints or "upper" in factor.constraints:
                        factor_info["distribution"] = "truncated_normal"
                factor_info["uncertainty"] = factor._uncertainty
                
            elif isinstance(factor, CategoricalLatentVariable):
                factor_info["distribution"] = "categorical"
                factor_info["params"] = {
                    "probs": list(factor.prior_probs.values()),
                    "categories": list(factor.prior_probs.keys())
                }
                
            else:  # DiscreteLatentVariable
                if hasattr(factor, 'distribution_params') and 'categories' in factor.distribution_params:
                    factor_info["distribution"] = "categorical"
                    factor_info["params"] = {
                        "probs": factor.distribution_params['probs'],
                        "categories": factor.distribution_params['categories']
                    }
                else:
                    factor_info["distribution"] = "poisson"
                    factor_info["params"] = {
                        "rate": factor.distribution_params.get('rate', float(factor.value))
                    }
            
            state_factors[name] = factor_info
            
        return state_factors

    def get_prior_samples(self, num_samples: int = 100) -> Dict[str, np.ndarray]:
        """
        Generate samples from the prior distributions for each factor.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Dictionary mapping factor names to arrays of samples
        """
        samples = {}
        for name, factor in self.factors.items():
            if isinstance(factor, ContinuousLatentVariable):
                # For continuous factors, sample from their distribution
                if isinstance(factor.tf_distribution, tfp.distributions.Normal):
                    samples[name] = np.random.normal(
                        loc=factor.params['loc'],
                        scale=factor.params['scale'],
                        size=num_samples
                    )
                elif isinstance(factor.tf_distribution, tfp.distributions.TruncatedNormal):
                    # For truncated normal, we'll use rejection sampling
                    raw_samples = np.random.normal(
                        loc=factor.params['loc'],
                        scale=factor.params['scale'],
                        size=num_samples * 2  # Generate extra samples for rejection
                    )
                    # Apply constraints
                    mask = (raw_samples >= factor.constraints['lower']) & (raw_samples <= factor.constraints['upper'])
                    samples[name] = raw_samples[mask][:num_samples]
                else:
                    raise ValueError(f"Unsupported distribution type: {type(factor.tf_distribution)}")
            elif isinstance(factor, DiscreteLatentVariable):
                # For discrete factors, sample from their probability distribution
                values = factor.params['values']
                probs = factor.params.get('probs', np.ones(len(values)) / len(values))
                samples[name] = np.random.choice(values, size=num_samples, p=probs)
            elif isinstance(factor, CategoricalLatentVariable):
                # For categorical factors, sample from their probability distribution
                categories = factor.params['categories']
                probs = factor.params.get('probs', np.ones(len(categories)) / len(categories))
                samples[name] = np.random.choice(categories, size=num_samples, p=probs)
            else:
                raise ValueError(f"Unsupported factor type: {type(factor)}")
        
        return samples

    def get_posterior_samples(self, num_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Get samples from the posterior distribution.
        
        Args:
            num_samples: Number of samples to return. If None, return all available.
            
        Returns:
            Dictionary of posterior samples
        """
        if self.posterior_samples is None:
            return {}
        
        # If num_samples specified, randomly sample that many
        if num_samples is not None and num_samples < len(next(iter(self.posterior_samples.values()))):
            indices = np.random.choice(len(next(iter(self.posterior_samples.values()))), 
                                    size=num_samples, replace=False)
            return {
                name: samples[indices] if isinstance(samples, np.ndarray) else samples
                for name, samples in self.posterior_samples.items()
            }
        
        return self.posterior_samples

    def set_posterior_samples(self, samples: Dict[str, Any]):
        """
        Set the posterior samples directly.
        
        Args:
            samples: Dictionary of posterior samples with factor information
        """
        # Store posterior samples directly
        self.posterior_samples = samples
        
        self.last_posterior_update = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Updated posterior samples at {self.last_posterior_update}")
        
        # Print sample statistics
        print("\nPosterior Sample Statistics:")
        for name, samples in self.posterior_samples.items():
            if isinstance(samples, np.ndarray):
                print(f"- {name}:")
                print(f"  Mean: {np.mean(samples):.3f}")
                print(f"  Std: {np.std(samples):.3f}")
                print(f"  Min: {np.min(samples):.3f}")
                print(f"  Max: {np.max(samples):.3f}")
            else:
                print(f"- {name}: {samples}")
            
        # Add to update history
        self.update_history.append({
            "timestamp": self.last_posterior_update,
            "samples": samples
        })
