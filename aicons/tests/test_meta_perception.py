#!/usr/bin/env python
"""
Test script for the Meta Ads Sales Sensor with Perception integration.

This script demonstrates how the MetaAdsSalesSensor integrates with the
BayesianPerception system in BayesBrain.
"""

import sys
import json
import pandas as pd
from pathlib import Path
from pprint import pprint

# Add the project root to the path if necessary
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the sensor and perception classes
from aicons.bayesbrainGPT.sensors.meta_s.meta_ads_sales_sensor import MetaAdsSalesSensor
from aicons.bayesbrainGPT.perception.perception import BayesianPerception
from aicons.bayesbrainGPT.brain import BayesBrain

def test_perception_integration():
    """Test integration of MetaAdsSalesSensor with BayesianPerception."""
    print("\n" + "="*80)
    print("META ADS SALES SENSOR PERCEPTION INTEGRATION TEST")
    print("="*80 + "\n")
    
    # Step 1: Create a BayesBrain instance
    print("Creating BayesBrain instance...")
    brain = BayesBrain(name="meta_test_brain")
    
    # Step 2: Create a Meta Ads sensor
    print("\nCreating MetaAdsSalesSensor instance...")
    sensor = MetaAdsSalesSensor(
        name="meta_ads",
        reliability=0.85
    )
    
    # Step 3: Register the sensor with the perception system
    print("\nRegistering sensor with perception system...")
    perception = brain.perception
    perception.register_sensor("meta_ads", sensor)
    
    # Step 4: Get mock data from the sensor
    print("\nGetting data from sensor...")
    mock_data = sensor.mock_run(num_adsets=2, num_ads_per_adset=3)
    
    # Display campaign-level metrics
    print("\n--- Campaign Summary ---")
    print(f"Purchases: {mock_data['purchases']}")
    print(f"Add to Carts: {mock_data['add_to_carts']}")
    print(f"Initiated Checkouts: {mock_data['initiated_checkouts']}")
    
    # Step 5: Extract factors in perception-friendly format
    print("\n--- Extracted Factors for Perception ---")
    extracted_factors = sensor.extract_ad_factors(mock_data)
    
    # Show a sample of the factors (just first 5 for clarity)
    factors_sample = list(extracted_factors.items())[:5]
    print(f"Sample factors (first 5 of {len(extracted_factors)}):")
    for name, (value, reliability) in factors_sample:
        print(f"  {name}: {value} (reliability: {reliability})")
    
    # Step 6: Set up state factors in BayesBrain
    print("\n--- Setting up state factors in BayesBrain ---")
    
    # Get expected factors from sensor
    expected_factors = sensor.get_expected_factors()
    
    # Create state factors for each expected factor
    for factor_name, factor_info in expected_factors.items():
        # Create appropriate factor type
        if factor_info["type"] == "continuous":
            brain.create_continuous_factor(
                name=factor_name,
                default_value=factor_info.get("default_value", 0.0),
                uncertainty=factor_info.get("uncertainty", 1.0),
                lower_bound=factor_info.get("lower_bound", None),
                description=factor_info.get("description", "")
            )
        elif factor_info["type"] == "categorical":
            brain.create_categorical_factor(
                name=factor_name,
                categories=factor_info.get("categories", []),
                default_value=factor_info.get("default_value", ""),
                description=factor_info.get("description", "")
            )
        elif factor_info["type"] == "discrete":
            brain.create_discrete_factor(
                name=factor_name,
                default_value=factor_info.get("default_value", 0),
                lower_bound=factor_info.get("lower_bound", 0),
                upper_bound=factor_info.get("upper_bound", 1000),
                description=factor_info.get("description", "")
            )
        elif factor_info["type"] == "json":
            # For JSON types, use a special factor
            brain.create_json_factor(
                name=factor_name,
                default_value=factor_info.get("default_value", "{}"),
                description=factor_info.get("description", "")
            )
    
    print(f"Created {len(expected_factors)} state factors")
    
    # Step 7: Show how to collect sensor data and update state
    print("\n--- Collecting Sensor Data and Updating State ---")
    print("Calling perception.update_from_sensor('meta_ads')...")
    
    # In a real scenario, we would do this:
    # perception.update_from_sensor('meta_ads')
    
    # But for demonstration, we'll show the flow manually:
    # 1. Collect data from sensor
    sensor_data = sensor.get_data()
    
    # 2. Show what data would be sent to perception
    print(f"\nSensor returns {len(sensor_data)} factors with (value, reliability) tuples")
    print("Sample data (first 3 factors):")
    sample_data = list(sensor_data.items())[:3]
    for factor_name, (value, reliability) in sample_data:
        print(f"  {factor_name}: {value} (reliability: {reliability})")
    
    print("\n--- Perception Flow ---")
    print("1. Sensor provides data as (value, reliability) tuples")
    print("2. Perception uses these as observations for Bayesian inference")
    print("3. Posterior distributions are sampled based on prior + observations")
    print("4. State factors are updated with posterior estimates")
    
    # Step 8: Show how this data flows into the Bayesian update
    print("\n--- Example State Update Simulation ---")
    print("Let's simulate updating a specific ad's purchases state factor:")
    
    # Take the first ad's purchases as an example
    ad_id = list(mock_data["ad_performances"].keys())[0]
    ad_data = mock_data["ad_performances"][ad_id]
    ad_purchases = ad_data["purchases"]
    factor_name = f"ad_{ad_id}_purchases"
    
    print(f"For ad {ad_id} ({ad_data['ad_name']}):")
    print(f"  Prior value: 0.0 (default state factor value)")
    print(f"  Observation: {ad_purchases} (from sensor data)")
    print(f"  Reliability: {sensor.default_reliability}")
    print(f"  Updated value: would be closer to {ad_purchases} depending on relative uncertainties")
    
    print("\n" + "="*80)
    print("END OF META ADS PERCEPTION INTEGRATION TEST")
    print("="*80)

# Helper for creating JSON factors in BayesBrain (might not exist in all versions)
def add_json_factor_to_brain(BayesBrain):
    """Add create_json_factor method to BayesBrain if it doesn't exist."""
    if not hasattr(BayesBrain, 'create_json_factor'):
        def create_json_factor(self, name, default_value="{}", description=""):
            self.create_factor(
                name=name,
                factor_type="json",
                default_value=default_value,
                description=description
            )
        BayesBrain.create_json_factor = create_json_factor

# Add the create_json_factor method if needed
add_json_factor_to_brain(BayesBrain)

if __name__ == "__main__":
    test_perception_integration() 