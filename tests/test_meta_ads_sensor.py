#!/usr/bin/env python3
"""
Test script for MetaAdsSalesSensor changes.
Verifies that the sensor creates concrete factors with actual ad IDs.
"""

import sys
import os
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import the sensor
from aicons.bayesbrainGPT.sensors.meta_s.meta_ads_sales_sensor import MetaAdsSalesSensor
from aicons.definitions.simple_bad_aicon import SimpleBadAIcon

def test_meta_ads_sensor():
    print("\n==== Testing MetaAdsSalesSensor Factor Creation ====\n")
    
    # Create a sensor with mock data
    sensor = MetaAdsSalesSensor(
        name="test_meta_ads",
        reliability=0.9,
        # No real credentials - will use mock data
    )
    
    # Generate some mock data with known structure
    print("Generating mock data...")
    mock_data = sensor.mock_run(num_adsets=2, num_ads_per_adset=2)
    
    # Extract the ad IDs from the mock data
    ad_ids = list(mock_data["ad_performances"].keys())
    print(f"Mock data contains {len(ad_ids)} ads with IDs: {ad_ids}")
    
    # Get expected factors from the sensor
    print("\nGetting expected factors...")
    expected_factors = sensor.get_expected_factors()
    
    # Verify we don't have any wildcard patterns in the factor names
    print("\nVerifying factor names...")
    has_patterns = False
    for factor_name in expected_factors.keys():
        if "*" in factor_name:
            print(f"ERROR: Found pattern in factor name: {factor_name}")
            has_patterns = True
    
    if not has_patterns:
        print("SUCCESS: No wildcard patterns found in factor names!")
    
    # Check if we have concrete factors for each ad ID
    print("\nVerifying ad-specific factors...")
    missing_factors = []
    for ad_id in ad_ids:
        for metric in ["purchases", "clicks", "spend"]:
            expected_factor = f"ad_{ad_id}_{metric}"
            if expected_factor not in expected_factors:
                missing_factors.append(expected_factor)
    
    if missing_factors:
        print(f"ERROR: Missing expected factors: {missing_factors}")
    else:
        print("SUCCESS: All expected ad-specific factors found!")
    
    # Print some sample factor definitions
    print("\nSample factor definitions:")
    for i, (factor_name, factor_def) in enumerate(expected_factors.items()):
        if i >= 5:  # Print just a few examples
            break
        print(f"- {factor_name}: {factor_def}")
    
    # Try with a SimpleBadAIcon to ensure factors are registered correctly
    print("\nTesting with SimpleBadAIcon...")
    aicon = SimpleBadAIcon("test_aicon")
    aicon.add_sensor("meta_ads", sensor)
    
    # List all factors in the brain
    brain_factors = aicon.brain.get_state_factors()
    print(f"AIcon brain has {len(brain_factors)} factors")
    
    # Check for ad-specific factors
    for ad_id in ad_ids:
        sample_factor = f"ad_{ad_id}_purchases"
        if sample_factor in brain_factors:
            print(f"SUCCESS: Found concrete factor '{sample_factor}' in brain")
        else:
            print(f"ERROR: Missing factor '{sample_factor}' in brain")
    
    print("\n==== Test Complete ====\n")

if __name__ == "__main__":
    test_meta_ads_sensor() 