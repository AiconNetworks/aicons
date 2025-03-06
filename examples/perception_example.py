from bayesbrainGPT.state_representation import EnvironmentState
from bayesbrainGPT.perception import BayesianPerception, WeatherSensor, TrafficSensor
from bayesbrainGPT.config import DEFAULT_STATE_CONFIG

# Initialize state with priors
state = EnvironmentState(DEFAULT_STATE_CONFIG)

# Create perception system
perception = BayesianPerception(state)

# Register sensors
weather_sensor = WeatherSensor()
traffic_sensor = TrafficSensor()
perception.register_sensor(weather_sensor)
perception.register_sensor(traffic_sensor)

# Update from specific sensor
perception.update_from_sensor("weather_sensor")

# Or update from all sensors
perception.update_all()

# Check updated state
print("Updated state:")
print(state) 