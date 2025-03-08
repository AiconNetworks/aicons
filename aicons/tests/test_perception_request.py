import pytest
import torch
from bayesbrainGPT.state_representation import EnvironmentState
from bayesbrainGPT.perception import BayesianPerception
from bayesbrainGPT.sensors.traffic import StaticTrafficSensor
from bayesbrainGPT.config import DEFAULT_STATE_CONFIG

@pytest.fixture
def state():
    """Create a test state with default config"""
    return EnvironmentState(factors_config=DEFAULT_STATE_CONFIG)

@pytest.fixture
def perception(state):
    """Create perception system with state"""
    return BayesianPerception(state)

def test_brain_requesting_sensor_data(perception, state):
    """Test how brain updates beliefs when requesting sensor data"""
    # Set up sensors
    downtown_sensor = StaticTrafficSensor(state, location="downtown", reliability=0.85)
    highway_sensor = StaticTrafficSensor(state, location="highway", reliability=0.90)
    
    # Get and print prior beliefs in detail
    prior_traffic = state.factors["traffic_density"]
    print("\nPRIOR BELIEFS:")
    print(f"Traffic density value: {prior_traffic.value}")
    print(f"Traffic density type: {type(prior_traffic)}")
    print(f"Traffic density distribution: Normal(μ={prior_traffic.value}, σ=1.0)")  # Default prior
    
    # Register sensors and update perception
    perception.register_sensor(downtown_sensor)
    perception.register_sensor(highway_sensor)
    
    # Brain requests and processes sensor data from downtown
    print("\nUPDATE 1 - Downtown Sensor:")
    print(f"Sensor reliability: {downtown_sensor.reliability}")
    print(f"Sensor reading: {downtown_sensor.get_data()['traffic_density']}")
    perception.update_from_sensor(downtown_sensor.name)
    
    # Check and print first posterior in detail
    posterior_1 = state.factors["traffic_density"]
    print("\nPOSTERIOR 1:")
    print(f"Traffic density value: {posterior_1.value}")
    # TODO: Show full posterior distribution once we implement Pyro
    
    # Update with highway sensor
    print("\nUPDATE 2 - Highway Sensor:")
    print(f"Sensor reliability: {highway_sensor.reliability}")
    print(f"Sensor reading: {highway_sensor.get_data()['traffic_density']}")
    perception.update_from_sensor(highway_sensor.name)
    
    # Check and print final posterior
    posterior_2 = state.factors["traffic_density"]
    print("\nPOSTERIOR 2:")
    print(f"Traffic density value: {posterior_2.value}")
    # TODO: Show full posterior distribution once we implement Pyro

def test_brain_handling_missing_data(perception, state):
    """Test how brain handles sensors with missing or unreliable data"""
    # Get and print prior
    prior_traffic = state.factors["traffic_density"].value
    print(f"\nPrior traffic density: {prior_traffic}")
    
    # Create sensor with very low reliability
    unreliable_sensor = StaticTrafficSensor(state, location="downtown", reliability=0.2)
    perception.register_sensor(unreliable_sensor)
    
    # Get data and verify reliability score
    print("\nUpdating with unreliable downtown sensor (reliability: 0.2)...")
    perception.update_from_sensor(unreliable_sensor.name)
    
    # Check and print posterior
    posterior_traffic = state.factors["traffic_density"].value
    print(f"Posterior traffic density after unreliable sensor: {posterior_traffic}")
    
    # For now, verify the update happened (later we'll implement reliability weighting)
    assert posterior_traffic != prior_traffic 