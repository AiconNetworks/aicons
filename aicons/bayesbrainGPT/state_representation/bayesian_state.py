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
        self.prior_distributions = {}  # TFP distributions for each latent variable
        self.hierarchical_relations = {}  # Conditional dependencies between variables
        self.conditional_distributions = {}  # Functions that return conditional distributions
        self.topological_order = []  # Ordering of factors for joint distribution
    
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
                bijector=tfb.Chain([tfb.Shift(shift=shift), tfb.Softplus()])
            )
        elif upper_bound is not None:
            # Transformed distribution for upper-bounded variables
            shift = float(upper_bound)
            tf_dist = tfd.TransformedDistribution(
                distribution=tfd.Normal(loc=shift-float(mean), scale=float(uncertainty)),
                bijector=tfb.Chain([tfb.Shift(shift=shift), tfb.Scale(-1.0), tfb.Softplus()])
            )
        else:
            # Standard normal for unconstrained variables
            tf_dist = tfd.Normal(loc=float(mean), scale=float(uncertainty))
            
        # Create the latent variable
        latent = ContinuousLatentVariable(
            name=name,
            initial_value=float(mean),
            uncertainty=float(uncertainty),
            description=description,
            relationships=relationships or {},
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
        
        This implements the hierarchical generative model from the Bayesian brain
        hypothesis, with explicit conditional dependencies between factors.
        
        Returns:
            Joint distribution representing the entire generative model
        """
        # Ensure topological order is up to date
        self._update_topological_order()
        
        # Build dictionary for JointDistributionNamed
        dist_dict = {}
        
        # Add factors in topological order
        for factor_name in self.topological_order:
            factor = self.factors[factor_name]
            
            # Check if this is a conditional factor
            if factor_name in self.hierarchical_relations:
                # This is a conditional factor
                parent_names = self.hierarchical_relations[factor_name]["parents"]
                
                # Create lambda function that depends on parent values
                dist_dict[factor_name] = lambda *args, fn=factor_name: self._get_conditional_dist(fn, args)
            else:
                # This is a root factor (no parents)
                if hasattr(factor, "tf_distribution") and factor.tf_distribution is not None:
                    dist_dict[factor_name] = factor.tf_distribution
                elif factor.type == "continuous":
                    # Create continuous distribution
                    dist_dict[factor_name] = tfd.Normal(
                        loc=float(factor.value),
                        scale=float(factor.uncertainty)
                    )
                elif factor.type == "categorical":
                    # Create categorical distribution
                    probs = factor.prior_probs.values() if hasattr(factor, "prior_probs") else None
                    if probs is None:
                        probs = [1.0/len(factor.possible_values)] * len(factor.possible_values)
                    dist_dict[factor_name] = tfd.Categorical(probs=probs)
                elif factor.type == "discrete":
                    # Create discrete distribution
                    if hasattr(factor, "distribution_params") and "rate" in factor.distribution_params:
                        dist_dict[factor_name] = tfd.Poisson(rate=factor.distribution_params["rate"])
                    else:
                        dist_dict[factor_name] = tfd.Poisson(rate=float(factor.value))
        
        # Create and return joint distribution
        return tfd.JointDistributionNamed(dist_dict)
    
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
            Dictionary mapping factor names to sample values
        """
        # Create joint distribution
        joint_dist = self.create_joint_distribution()
        
        # Sample from joint distribution
        samples = joint_dist.sample(n_samples)
        
        return samples

    def add_factor(self, name: str, factor_type: str, value: Any,
                   params: Dict[str, Any], relationships: Dict[str, Any]) -> LatentVariable:
        """
        Add a factor to the state with proper hierarchical relationships.
        
        This is the main method for adding factors, ensuring proper hierarchical structure.
        It delegates to the appropriate type-specific method based on factor_type.
        
        Args:
            name: Name of the factor
            factor_type: Type of factor ('continuous', 'categorical', 'discrete')
            value: Initial value
            params: Type-specific parameters
            relationships: Hierarchical relationships with other factors
            
        Returns:
            The created latent variable
        """
        # Validate relationships format
        if relationships is None:
            relationships = {}
        elif not isinstance(relationships, dict):
            raise ValueError("Relationships must be a dictionary")
        
        # Validate depends_on is a list if present
        if 'depends_on' in relationships:
            if not isinstance(relationships['depends_on'], list):
                raise ValueError("depends_on must be a list of factor names")
            if not all(isinstance(x, str) for x in relationships['depends_on']):
                raise ValueError("All elements in depends_on must be strings")
            
        # Check if this is a root factor (no dependencies)
        is_root = not relationships.get('depends_on')
        
        # Validate parent factors exist
        if not is_root:
            missing_parents = [parent for parent in relationships['depends_on'] 
                             if parent not in self.factors]
            if missing_parents:
                raise ValueError(f"Cannot create factor '{name}' because parent factors do not exist: {missing_parents}")
            
        # Check for circular dependencies
        if not is_root:
            # Create a temporary graph with the new factor
            temp_graph = self.hierarchical_relations.copy()
            temp_graph[name] = relationships
            
            # Check for cycles using DFS
            def has_cycle(graph: Dict[str, Dict[str, Any]], 
                         node: str, 
                         visited: Set[str], 
                         path: Set[str]) -> bool:
                visited.add(node)
                path.add(node)
                
                if node in graph and 'depends_on' in graph[node]:
                    for parent in graph[node]['depends_on']:
                        if parent not in visited:
                            if has_cycle(graph, parent, visited, path):
                                return True
                        elif parent in path:
                            return True
                        
                path.remove(node)
                return False
                
            # Check for cycles starting from the new factor
            if has_cycle(temp_graph, name, set(), set()):
                raise ValueError(f"Adding factor '{name}' would create a circular dependency")
        
        # Create factor based on type
        if factor_type == "continuous":
            factor = self.add_continuous_latent(
                name=name,
                mean=float(value),
                uncertainty=float(params.get('scale', 0.1)),
                description=f"Continuous factor {name}",
                lower_bound=params.get('lower_bound'),
                upper_bound=params.get('upper_bound'),
                relationships=relationships
            )
        elif factor_type == "categorical":
            if not params.get('categories') or not params.get('probs'):
                raise ValueError("Categories and probabilities required for categorical factors")
            
            # Handle hierarchical relationships for categorical variables
            if not is_root:
                # If this categorical variable depends on other factors,
                # we need to specify conditional probabilities for each combination
                conditional_probs = relationships.get('conditional_probs', {})
                if not conditional_probs:
                    # If no conditional probabilities provided, use uniform distribution
                    # This should be updated based on actual data or expert knowledge
                    conditional_probs = {
                        cat: {dep: 1.0/len(params['categories']) for dep in params['categories']}
                        for cat in params['categories']
                    }
            else:
                conditional_probs = None
            
            factor = self.add_categorical_latent(
                name=name,
                initial_value=str(value),
                possible_values=params['categories'],
                probs=params['probs'],
                description=f"Categorical factor {name}",
                relationships=relationships,
                conditional_probs=conditional_probs
            )
        elif factor_type == "discrete":
            if params.get('categories') and params.get('probs'):
                # Finite discrete values
                factor = self.add_discrete_latent(
                    name=name,
                    initial_value=int(value),
                    min_value=min(params['categories']),
                    max_value=max(params['categories']),
                    description=f"Discrete factor {name}",
                    relationships=relationships
                )
            else:
                # Poisson distribution for counts
                factor = self.add_discrete_latent(
                    name=name,
                    initial_value=int(value),
                    rate=float(value),
                    description=f"Count factor {name}",
                    relationships=relationships
                )
        else:
            raise ValueError(f"Unknown factor type: {factor_type}")
        
        # Add to state factors dictionary
        self.factors[name] = {
            "type": factor_type,
            "value": value,
            "params": params,
            "relationships": relationships
        }
        
        # Update hierarchical relations if not a root factor
        if not is_root:
            self.hierarchical_relations[name] = relationships
        
        # Update topological order
        self._update_topological_order()
        
        return factor


# For backward compatibility with any code that directly imports EnvironmentState
EnvironmentState = BayesianState
# Ensure exports include both classes
__all__ = ['BayesianState', 'EnvironmentState'] 