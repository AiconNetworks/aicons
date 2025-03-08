# bayesbrainGPT/state_representation/__init__.py

from .state import EnvironmentState
from .factors import (
    BaseFactor,
    ContinuousFactor,
    CategoricalFactor,
    DiscreteFactor,
    BayesianLinearFactor
)

__all__ = [
    'EnvironmentState',
    'BaseFactor',
    'ContinuousFactor',
    'CategoricalFactor',
    'DiscreteFactor',
    'BayesianLinearFactor'
]

# Empty file to mark directory as Python package
