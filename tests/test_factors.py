import pytest
import torch
from bayesbrainGPT.state_representation.factors import (
    ContinuousFactor,
    CategoricalFactor,
    DiscreteFactor,
    BayesianLinearFactor
)

def test_continuous_factor():
    """Test continuous factor behavior"""
    factor = ContinuousFactor("temp", 20.0, "Temperature in Celsius")
    assert factor.value == 20.0
    factor.update(25.0)
    assert factor.value == 25.0
    with pytest.raises(ValueError):
        factor.update("invalid")

def test_categorical_factor():
    """Test categorical factor behavior"""
    factor = CategoricalFactor("weather", "sunny", "Weather condition")
    assert factor.value == "sunny"
    factor.update("rainy")
    assert factor.value == "rainy"
    with pytest.raises(ValueError):
        factor.update(42)

def test_discrete_factor():
    """Test discrete factor behavior"""
    factor = DiscreteFactor("day", 1, "Day of week")
    assert factor.value == 1
    factor.update(2)
    assert factor.value == 2
    with pytest.raises(ValueError):
        factor.update(3.14)

def test_bayesian_linear_factor():
    """Test Bayesian linear factor behavior"""
    explanatory_vars = {"x1": 1.0, "x2": 2.0}
    theta_prior = {
        "theta0": {"mean": 0.0, "variance": 1.0},
        "theta1": {"mean": 0.0, "variance": 1.0}
    }
    factor = BayesianLinearFactor(
        "y",
        explanatory_vars=explanatory_vars,
        theta_prior=theta_prior,
        variance=1.0,
        description="Test factor"
    )
    
    # Test prior parameter extraction
    prior_params = factor.get_prior_params()
    assert isinstance(prior_params["mean"], torch.Tensor)
    assert isinstance(prior_params["variance"], torch.Tensor)
    assert len(prior_params["mean"]) == len(theta_prior) 