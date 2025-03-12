#!/usr/bin/env python3
"""
Test script for factor mapping in sensors and perception.

This script demonstrates how to use factor mapping to allow sensors and AIcon
state to use different naming conventions for the same factors.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# TFP shortcuts
tfd = tfp.distributions
tfb = tfp.bijectors

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Import our AIcon class and the TensorFlow sensors
from aicons.definitions.simple_bad_aicon import SimpleBadAIcon
from aicons.bayesbrainGPT.sensors.tf_sensors import MarketingSensor

def main():
    print("Testing sensor factor mapping functionality")
    print("===========================================\n")
    
    # Create an AIcon with our own factor names
    aicon = SimpleBadAIcon(name="Marketing Campaign")

    # Add factors with our own naming convention
    print("Adding continuous factor...")
    aicon.add_factor_continuous(
        name="conversion_rate", 
        value=0.05, 
        uncertainty=0.01,
        lower_bound=0.0,
        upper_bound=1.0,
        description="Conversion rate for ads"
    )

    print("Adding categorical factor...")
    aicon.add_factor_categorical(
        name="best_channel",
        value="facebook",
        categories=["facebook", "google", "tiktok", "instagram"],
        probs=[0.4, 0.3, 0.2, 0.1],
        description="Best performing ad channel"
    )

    print("Adding discrete factor...")
    aicon.add_factor_discrete(
        name="ad_count",
        value=3,
        min_value=1,
        max_value=10,
        description="Number of ads per campaign"
    )

    # Print the initial state
    print("\nInitial state:")
    print(aicon.get_state(format_nicely=True))

    # Create a MarketingSensor
    print("\nCreating MarketingSensor...")
    try:
        marketing_sensor = MarketingSensor(
            name="campaign_data",
            reliability=0.8
        )
        print("✅ Successfully created MarketingSensor")
    except Exception as e:
        print(f"❌ Error creating MarketingSensor: {e}")
        return

    # Add the sensor with factor mapping via the add_sensor method
    print("\nAdding sensor with factor mapping...")
    try:
        aicon.add_sensor(
            name="campaign_data", 
            sensor=marketing_sensor,
            factor_mapping={
                "base_conversion_rate": "conversion_rate",
                "primary_channel": "best_channel",
                "optimal_daily_ads": "ad_count"
            }
        )
        print("✅ Successfully added sensor with factor mapping")
    except Exception as e:
        print(f"❌ Error adding sensor with factor mapping: {e}")
        return

    # Define "true" values for testing - using sensor's original factor names
    true_values = {
        "base_conversion_rate": 0.08,  # Sensor expects this name
        "primary_channel": "google",   # Sensor expects this name
        "optimal_daily_ads": 5         # Sensor expects this name
    }
    print(f"\nTrue values for testing: {true_values}")

    # Print sensor's observable factors and reliabilities
    print("\nSensor observable factors:")
    for factor in marketing_sensor.observable_factors:
        reliability = marketing_sensor.factor_reliabilities.get(factor, marketing_sensor.default_reliability)
        print(f"  {factor}: reliability={reliability:.2f}")

    # Get sensor data directly to check mapping
    print("\nTesting sensor data retrieval:")
    try:
        sensor_data = marketing_sensor.get_data(environment=true_values)
        print("Sensor data (with mapping applied):")
        for state_factor, (value, reliability) in sensor_data.items():
            print(f"  {state_factor}: {value} (reliability: {reliability:.2f})")
        print("✅ Successfully retrieved sensor data with mapping")
    except Exception as e:
        print(f"❌ Error retrieving sensor data: {e}")
        return

    # Update beliefs with sensor data
    print("\nUpdating beliefs with sensor data...")
    try:
        update_result = aicon.update_from_sensor("campaign_data", environment=true_values)
        if update_result:
            print("✅ Successfully updated beliefs")
        else:
            print("❌ Failed to update beliefs")
            return
    except Exception as e:
        print(f"❌ Error updating beliefs: {e}")
        return

    # Get posterior samples
    posterior_samples = aicon.get_posterior_samples()
    print("\nPosterior samples keys:", list(posterior_samples.keys()))
    print("Number of samples:", {k: len(v) if hasattr(v, '__len__') else 'N/A' for k, v in posterior_samples.items()})

    # Print sample statistics for each factor
    for name, samples in posterior_samples.items():
        if isinstance(samples, np.ndarray):
            if samples.dtype.kind in ['U', 'S']:  # String type (categorical)
                # Get value counts for categorical variables
                values, counts = np.unique(samples, return_counts=True)
                probs = counts / counts.sum()
                print(f"\n{name} posterior:")
                for v, p in zip(values, probs):
                    print(f"  {v}: {p:.3f}")
            else:  # Numeric type
                print(f"\n{name} posterior:")
                print(f"  Mean: {np.mean(samples):.4f}")
                print(f"  Std: {np.std(samples):.4f}")
                print(f"  Min: {np.min(samples):.4f}")
                print(f"  Max: {np.max(samples):.4f}")

    # Print the updated state
    print("\nUpdated state:")
    print(aicon.get_state(format_nicely=True))
    
    print("\n✅ Test completed successfully")

if __name__ == "__main__":
    main() 