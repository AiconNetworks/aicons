# bayesbrainGPT/state_representation/__init__.py

from .bayesian_state import BayesianState, EnvironmentState
from .latent_variables import (
    LatentVariable,
    ContinuousLatentVariable,
    CategoricalLatentVariable,
    DiscreteLatentVariable,
    HierarchicalLatentVariable
)

# For backward compatibility
from .latent_variables import (
    LatentVariable as BaseFactor,
    ContinuousLatentVariable as ContinuousFactor,
    CategoricalLatentVariable as CategoricalFactor,
    DiscreteLatentVariable as DiscreteFactor,
    HierarchicalLatentVariable as BayesianLinearFactor
)

__all__ = [
    # New Bayesian brain classes
    'BayesianState',
    'EnvironmentState',
    'LatentVariable',
    'ContinuousLatentVariable',
    'CategoricalLatentVariable',
    'DiscreteLatentVariable',
    'HierarchicalLatentVariable',
    # Backward compatibility names
    'BaseFactor',
    'ContinuousFactor',
    'CategoricalFactor',
    'DiscreteFactor',
    'BayesianLinearFactor'
]

# Empty file to mark directory as Python package
