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

# Create AIcon
aicon = SimpleBadAIcon(name="Marketing Budget Allocator")

# Add sensors
marketing_sensor = MarketingSensor()
aicon.add_sensor("marketing", marketing_sensor)

# Create simulated data with valid values matching the categories defined in the sensor
simulated_data = {
    "base_conversion_rate": 0.05,
    "primary_channel": "facebook",
    "optimal_daily_ads": 12
}

print("\nSETUP BYPASS TEST - Direct Update Without MCMC")
print("=============================================")

# Monkey patch the perception's sample_posterior method to skip MCMC
# and provide sample values directly based on the observations
original_sample_posterior = aicon.brain.perception.sample_posterior

def bypass_sample_posterior(observations):
    print(f"BYPASS: Skipping MCMC sampling, using observations directly: {observations}")
    
    # Create direct samples from observations
    samples = {}
    for factor_name, (value, reliability) in observations.items():
        # For each factor, create samples directly from the observation value
        if factor_name == "primary_channel":
            # For categorical variable, use the category directly
            samples[factor_name] = [value] * 100  # 100 identical samples
        elif factor_name == "optimal_daily_ads":
            # For discrete variable, use the value directly
            samples[factor_name] = [int(value)] * 100  # 100 identical samples
        else:
            # For continuous variables, use the value with small noise
            import numpy as np
            samples[factor_name] = np.random.normal(float(value), 0.001, 100)
    
    # Store and return the samples
    aicon.brain.perception.posterior_samples = samples
    return samples

# Apply the monkey patch
aicon.brain.perception.sample_posterior = bypass_sample_posterior

# Try updating directly
try:
    print("\nCollecting sensor data and updating state...")
    observations = aicon.brain.perception.collect_sensor_data(simulated_data)
    print(f"Collected observations: {observations}")
    
    # Try updating state from these observations
    success = aicon.brain.perception.update_state_from_posterior()
    print(f"Update state success: {success}")
    
    # Check updated state
    print("\nUpdated state factors:")
    for name, factor in aicon.brain.get_state_factors().items():
        print(f"  {name}: {factor['value']}")
    
    # Reset the original method
    aicon.brain.perception.sample_posterior = original_sample_posterior
    print("\nTest completed successfully")
    
except Exception as e:
    # Reset the original method
    aicon.brain.perception.sample_posterior = original_sample_posterior
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 