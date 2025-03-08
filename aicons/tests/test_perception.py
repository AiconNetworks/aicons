import pytest
import torch
import pyro
from bayesbrainGPT.state_representation import EnvironmentState
from bayesbrainGPT.perception import BayesianPerception, WeatherSensor, TrafficSensor
from bayesbrainGPT.config import DEFAULT_STATE_CONFIG

@pytest.fixture
def state():
    """Create a test state with default config"""
    return EnvironmentState(factors_config=DEFAULT_STATE_CONFIG)

@pytest.fixture
def perception(state):
    """Create a test perception system"""
    return BayesianPerception(state)

@pytest.fixture
def weather_sensor():
    """Create a test weather sensor"""
    return WeatherSensor()

@pytest.fixture
def traffic_sensor():
    """Create a test traffic sensor"""
    return TrafficSensor()

def test_perception_initialization(perception):
    """Test that perception system initializes correctly"""
    assert perception.state is not None
    assert isinstance(perception.sensors, dict)
    assert len(perception.sensors) == 0

def test_sensor_registration(perception, weather_sensor):
    """Test that sensors can be registered"""
    perception.register_sensor(weather_sensor)
    assert "weather_sensor" in perception.sensors
    assert perception.sensors["weather_sensor"] == weather_sensor

def test_weather_sensor_data(weather_sensor):
    """Test that weather sensor returns correct data format"""
    data = weather_sensor.get_data()
    assert "temperature" in data
    assert "rain" in data
    assert isinstance(data["temperature"], torch.Tensor)
    assert isinstance(data["rain"], torch.Tensor)

def test_traffic_sensor_data(traffic_sensor):
    """Test that traffic sensor returns correct data format"""
    data = traffic_sensor.get_data()
    assert "traffic_density" in data
    assert isinstance(data["traffic_density"], torch.Tensor)

def test_update_from_sensor(perception, weather_sensor):
    """Test updating perception from a single sensor"""
    perception.register_sensor(weather_sensor)
    perception.update_from_sensor("weather_sensor")
    # Check that state was updated
    assert perception.state.factors["temperature"].value is not None

def test_update_all_sensors(perception, weather_sensor, traffic_sensor):
    """Test updating perception from all sensors"""
    perception.register_sensor(weather_sensor)
    perception.register_sensor(traffic_sensor)
    perception.update_all()
    # Check that state was updated for both sensor domains
    assert perception.state.factors["temperature"].value is not None
    assert perception.state.factors["traffic_density"].value is not None

def test_invalid_sensor_update(perception):
    """Test that updating from non-existent sensor raises error"""
    with pytest.raises(ValueError):
        perception.update_from_sensor("nonexistent_sensor")

def test_bayesian_model_sampling(perception):
    """Test that the Bayesian model can sample from priors"""
    # Run the model once to check it works
    model = perception.model
    trace = pyro.poutine.trace(model).get_trace()
    assert len(trace.nodes) > 0  # Check that some sampling occurred 