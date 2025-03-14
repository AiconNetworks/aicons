import importlib
import aicons.definitions.simple_bad_aicon
import aicons.bayesbrainGPT.sensors.tf_sensors
import aicons.bayesbrainGPT.perception.perception

importlib.reload(aicons.definitions.simple_bad_aicon)
importlib.reload(aicons.bayesbrainGPT.sensors.tf_sensors)
importlib.reload(aicons.bayesbrainGPT.perception.perception)
print("Modules reloaded successfully")

# Setup
from aicons.definitions.simple_bad_aicon import SimpleBadAIcon
from aicons.bayesbrainGPT.sensors.tf_sensors import MarketingSensor
from aicons.bayesbrainGPT.perception.perception import BayesianPerception

# Create AIcon
aicon = SimpleBadAIcon(name="Marketing Budget Allocator")

# Add sensors
marketing_sensor = MarketingSensor()
aicon.add_sensor("marketing", marketing_sensor)

# Create simulated data - just raw values
simulated_data = {
    "base_conversion_rate": 0.05,  # Using raw value (not a tuple)
    "primary_channel": "social",
    "optimal_daily_ads": 12
}

print("\nDEBUG TEST: Directly accessing perception system")
print(f"Type of simulated_data: {type(simulated_data)}")
print(f"Keys in simulated_data: {list(simulated_data.keys())}")

# Directly access the perception system through brain
perception = aicon.brain.perception

# Try to collect data from sensors
print("\nCollecting sensor data...")
try:
    observations = perception.collect_sensor_data(simulated_data)
    print(f"Observations collected: {observations}")
except Exception as e:
    print(f"Error during collect_sensor_data: {e}")
    import traceback
    traceback.print_exc()

# Try direct access to sensor methods
print("\nTesting sensor access directly...")
try:
    print("Calling get_data on marketing sensor...")
    data = marketing_sensor.get_data(simulated_data)
    print(f"Sensor data: {data}")
except Exception as e:
    print(f"Error during sensor.get_data: {e}")
    import traceback
    traceback.print_exc()

# Try direct access to fetch_data
print("\nTesting fetch_data directly...")
try:
    print("Calling fetch_data on marketing sensor...")
    data = marketing_sensor.fetch_data(simulated_data)
    print(f"Fetched data: {data}")
except Exception as e:
    print(f"Error during sensor.fetch_data: {e}")
    import traceback
    traceback.print_exc() 