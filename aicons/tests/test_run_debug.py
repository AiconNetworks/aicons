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
import traceback

# Create AIcon
aicon = SimpleBadAIcon(name="Marketing Budget Allocator")

# Add sensors
marketing_sensor = MarketingSensor()
aicon.add_sensor("marketing", marketing_sensor)

# Create simulated data with valid values matching the categories defined in the sensor
simulated_data = {
    "base_conversion_rate": 0.05,
    "primary_channel": "facebook",  # Changed from 'social' to 'facebook', which is in the defined categories
    "optimal_daily_ads": 12
}

# Print run method source for reference
import inspect
print("\nMonkey patching run method with debug version...")

# Store the original run method
original_run = aicon.run

# Define a debug version of the run method
def debug_run(self, mode='once', environment=None, **kwargs):
    print(f"DEBUG - Starting run with mode={mode}")
    print(f"DEBUG - Environment: {environment}")
    
    # Process environment (deepcopy it so we don't modify the original)
    try:
        import copy
        local_env = copy.deepcopy(environment) if environment else None
        print(f"DEBUG - Local environment: {local_env}")
        
        # Try perception update directly
        print("\nDEBUG - Trying perception update directly...")
        try:
            success = self.brain.perception.update_all(local_env)
            print(f"DEBUG - Direct perception update success: {success}")
        except Exception as e:
            print(f"DEBUG - Error in direct perception update: {e}")
            traceback.print_exc()
    except Exception as e:
        print(f"DEBUG - Error in initial environment processing: {e}")
        traceback.print_exc()
    
    # Call the original method and return its result
    try:
        print("\nDEBUG - Calling original run method...")
        return original_run(mode=mode, environment=environment, **kwargs)
    except Exception as e:
        print(f"DEBUG - Error in original run method: {e}")
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}

# Monkey patch the run method
aicon.run = lambda *args, **kwargs: debug_run(aicon, *args, **kwargs)

# Run with debug tracing active
print("\nRunning perception update...")
result = aicon.run(mode='once', environment=simulated_data)
print(f"\nRun result: {result}") 