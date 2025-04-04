# bayesbrainGPT/state_representation/__init__.py

from .bayesian_state import BayesianState
from .latent_variables import (
    LatentVariable,
    ContinuousLatentVariable,
    CategoricalLatentVariable,
    DiscreteLatentVariable,
    HierarchicalLatentVariable
)

__all__ = [
    # New Bayesian brain classes
    'BayesianState',
    'LatentVariable',
    'ContinuousLatentVariable',
    'CategoricalLatentVariable',
    'DiscreteLatentVariable',
    'HierarchicalLatentVariable'
]

# Empty file to mark directory as Python package
