import pytest
from aicons.bayesbrainGPT.state_representation import EnvironmentState
from aicons.bayesbrainGPT.perception import BayesianPerception
from aicons.bayesbrainGPT.config import DEFAULT_STATE_CONFIG
import torch

def test_state_initialization_from_config():
    """Test state initialization from config"""
    state = EnvironmentState(factors_config=DEFAULT_STATE_CONFIG)
    assert len(state.factors) > 0
    assert "temperature" in state.factors
    assert "weather" in state.factors

def test_state_initialization_from_llm():
    """Test state initialization from mock LLM data"""
    state = EnvironmentState(use_llm=True, mock_llm=True)
    assert len(state.factors) > 0
    # Check for some expected marketing factors
    assert any("conversion" in name for name in state.factors.keys())
    assert any("campaign" in name for name in state.factors.keys())

def test_state_initialization_fails_gracefully():
    """Test state initialization with no config and no LLM"""
    state = EnvironmentState()  # Should use empty config
    assert len(state.factors) == 0

def test_perception_update():
    """Test state update through perception with new data"""
    state = EnvironmentState(factors_config=DEFAULT_STATE_CONFIG)
    perception = BayesianPerception(state)
    
    initial_temp = state.factors["temperature"].value
    initial_weather = state.factors["weather"].value
    
    new_data = {
        "temperature": (torch.tensor(25.0), 0.95),
        "weather": (torch.tensor([0.0, 1.0, 0.0]), 0.9)  # One-hot encoded: [sunny, rainy, cloudy]
    }
    
    perception.update_all(observations=new_data)
    
    assert state.factors["temperature"].value != initial_temp
    assert state.factors["weather"].value != initial_weather

def test_state_get_state():
    """Test getting state as dictionary"""
    state = EnvironmentState(factors_config=DEFAULT_STATE_CONFIG)
    state_dict = state.get_state()
    assert isinstance(state_dict, dict)
    assert "temperature" in state_dict
    assert "weather" in state_dict 