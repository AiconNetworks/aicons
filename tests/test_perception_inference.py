import pytest
import torch
from bayesbrainGPT.state_representation import EnvironmentState
from bayesbrainGPT.perception import BayesianPerception
from bayesbrainGPT.sensors.traffic import StaticTrafficSensor

@pytest.fixture
def complex_state():
    """Create a test state with multiple factors"""
    config = {
        "traffic_density": {
            "type": "continuous",
            "value": 2.0,
            "description": "Traffic density observation"
        },
        "weather": {
            "type": "categorical",
            "value": "sunny",
            "possible_values": ["sunny", "rainy", "cloudy"],
            "description": "Weather condition"
        },
        "road_quality": {
            "type": "continuous",
            "value": 0.8,
            "description": "Road condition (not observable by traffic sensors)"
        }
    }
    return EnvironmentState(factors_config=config)

def test_unobserved_factors_unchanged(complex_state):
    """Test that factors without sensor data remain unchanged"""
    perception = BayesianPerception(complex_state)
    traffic_sensor = StaticTrafficSensor(complex_state, "downtown")
    
    # Record prior for unobserved factor
    prior_road_quality = complex_state.factors["road_quality"].value
    
    # Update with traffic sensor
    perception.register_sensor(traffic_sensor)
    perception.update_from_sensor(traffic_sensor.name)
    
    # Road quality should be unchanged
    assert complex_state.factors["road_quality"].value == prior_road_quality

def test_posterior_uncertainty(complex_state):
    """Test that sensor reliability affects posterior uncertainty"""
    perception = BayesianPerception(complex_state)
    
    # Print initial state with more detail
    print("\nPRIOR BELIEFS:")
    print("Traffic Density:")
    print(f"  Value: {complex_state.factors['traffic_density'].value}")
    print(f"  Uncertainty: {complex_state.factors['traffic_density'].uncertainty}")
    print(f"  Distribution: Normal(μ={complex_state.factors['traffic_density'].value}, σ={complex_state.factors['traffic_density'].uncertainty})")
    
    # Create two sensors with different reliabilities
    reliable_sensor = StaticTrafficSensor(complex_state, "downtown", reliability=0.9)
    unreliable_sensor = StaticTrafficSensor(complex_state, "suburb", reliability=0.3)
    
    # Update with reliable sensor
    perception.register_sensor(reliable_sensor)
    sensor_data = reliable_sensor.get_data()
    print("\nRELIABLE SENSOR DATA:")
    print(f"  Reading: {sensor_data['traffic_density'][0].item()}")
    print(f"  Reliability: {sensor_data['traffic_density'][1]}")
    print(f"  Observation noise: {1.0/sensor_data['traffic_density'][1]}")
    
    perception.update_from_sensor(reliable_sensor.name)
    reliable_posterior_std = complex_state.factors["traffic_density"].uncertainty
    
    print("\nPOSTERIOR AFTER RELIABLE SENSOR:")
    print(f"  Value: {complex_state.factors['traffic_density'].value}")
    print(f"  Uncertainty: {reliable_posterior_std}")
    print(f"  Distribution: Normal(μ={complex_state.factors['traffic_density'].value}, σ={reliable_posterior_std})")
    
    # Reset and update with unreliable sensor
    complex_state.reset()
    perception.register_sensor(unreliable_sensor)
    sensor_data = unreliable_sensor.get_data()
    
    print("\nUNRELIABLE SENSOR DATA:")
    print(f"  Reading: {sensor_data['traffic_density'][0].item()}")
    print(f"  Reliability: {sensor_data['traffic_density'][1]}")
    print(f"  Observation noise: {1.0/sensor_data['traffic_density'][1]}")
    
    perception.update_from_sensor(unreliable_sensor.name)
    unreliable_posterior_std = complex_state.factors["traffic_density"].uncertainty
    
    print("\nPOSTERIOR AFTER UNRELIABLE SENSOR:")
    print(f"  Value: {complex_state.factors['traffic_density'].value}")
    print(f"  Uncertainty: {unreliable_posterior_std}")
    print(f"  Distribution: Normal(μ={complex_state.factors['traffic_density'].value}, σ={unreliable_posterior_std})")
    
    # Compare uncertainties
    assert unreliable_posterior_std > reliable_posterior_std

def test_multiple_updates_convergence(complex_state):
    """Test that multiple updates from same sensor type converge"""
    perception = BayesianPerception(complex_state)
    sensor = StaticTrafficSensor(complex_state, "downtown")
    perception.register_sensor(sensor)
    
    previous_values = []
    for _ in range(5):
        perception.update_from_sensor(sensor.name)
        current_value = complex_state.factors["traffic_density"].value
        previous_values.append(current_value)
    
    # Values should converge (differences should get smaller)
    differences = [abs(previous_values[i] - previous_values[i-1]) 
                  for i in range(1, len(previous_values))]
    assert all(differences[i] >= differences[i+1] for i in range(len(differences)-1))

def test_conflicting_sensor_resolution(complex_state):
    """Test how system handles conflicting sensor readings"""
    perception = BayesianPerception(complex_state)
    
    # Two sensors with different readings but same reliability
    sensor1 = StaticTrafficSensor(complex_state, "downtown")  # Reports high traffic
    sensor2 = StaticTrafficSensor(complex_state, "suburb")    # Reports low traffic
    
    perception.register_sensor(sensor1)
    perception.register_sensor(sensor2)
    
    # Update with both sensors
    perception.update_from_sensor(sensor1.name)
    first_update = complex_state.factors["traffic_density"].value
    perception.update_from_sensor(sensor2.name)
    second_update = complex_state.factors["traffic_density"].value
    
    # Final value should be between the two sensor readings
    assert min(4.0, 1.0) <= second_update <= max(4.0, 1.0) 