import pytest
from bayesbrainGPT.state_representation import EnvironmentState
from bayesbrainGPT.config import DEFAULT_STATE_CONFIG

def test_state_initialization():
    """Test state initialization from config"""
    state = EnvironmentState(DEFAULT_STATE_CONFIG)
    assert len(state.factors) > 0
    assert "temperature" in state.factors
    assert "weather" in state.factors

def test_state_update():
    """Test state update with new data"""
    state = EnvironmentState(DEFAULT_STATE_CONFIG)
    new_data = {
        "temperature": 25.0,
        "weather": "rainy"
    }
    state.update_state(new_data)
    assert state.factors["temperature"].value == 25.0
    assert state.factors["weather"].value == "rainy"

def test_state_get_state():
    """Test getting state as dictionary"""
    state = EnvironmentState(DEFAULT_STATE_CONFIG)
    state_dict = state.get_state()
    assert isinstance(state_dict, dict)
    assert "temperature" in state_dict
    assert "weather" in state_dict 