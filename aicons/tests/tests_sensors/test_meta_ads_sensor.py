#!/usr/bin/env python
"""
Test script that checks if the MetaAdsSalesSensor connects properly to an AIcon
and if hierarchical relationships between ad-level metrics and campaign metrics work.
"""

import sys
import os
import json
from pathlib import Path

# Fix the import path issue
# Get the project root directory (Babel)
current_file = os.path.abspath(__file__)
tests_dir = os.path.dirname(current_file)
aicons_dir = os.path.dirname(tests_dir)
project_root = os.path.dirname(aicons_dir)  # This is the Babel directory

# Add the project root to sys.path
sys.path.insert(0, project_root)
print(f"Added {project_root} to sys.path")

# Now import with absolute paths
from aicons.bayesbrainGPT.sensors.meta_s.meta_ads_sales_sensor import MetaAdsSalesSensor
from aicons.definitions.simple_bad_aicon import SimpleBadAIcon

def hierarchical_test():
    """
    Test to check if the sensor connects to the AIcon and if
    ad-level metrics properly roll up to campaign metrics.
    """
    print("\n=== TESTING META ADS SENSOR HIERARCHICAL FUNCTIONALITY ===")
    
    # Initialize sensor with mock data generator
    print("\n1. Creating Meta Ads Sensor...")
    sensor = MetaAdsSalesSensor(name="meta_ads", reliability=0.9)
    
    # Get COMPLETE mock data with ALL factors
    print("\n2. Generating complete mock data with all metrics...")
    mock_data = sensor.mock_run(num_adsets=2, num_ads_per_adset=2)
    
    # Print the complete mock data structure
    print(f"\nMock data contains the following keys: {list(mock_data.keys())}")
    print(f"Number of ad sets: {len(mock_data.get('adset_performances', {}))}")
    print(f"Number of ads: {len(mock_data.get('ad_performances', {}))}")
    
    # Override the run method to return our mock data
    original_run = sensor.run
    sensor.run = lambda: mock_data
    
    # Initialize AIcon
    print("\n3. Creating AIcon and adding sensor...")
    aicon = SimpleBadAIcon("test_hierarchical")
    
    # Add the sensor to the AIcon
    aicon.add_sensor("meta_ads", sensor)
    
    # Get the expected factors from the sensor
    print("\n4. Getting expected factors from sensor...")
    expected_factors = sensor.get_expected_factors()
    print(f"Sensor defines {len(expected_factors)} factors:")
    for factor_name, factor_info in expected_factors.items():
        print(f"  - {factor_name}: {factor_info['type']}")
        if 'hierarchy' in factor_info:
            print(f"    Hierarchical: {factor_info['hierarchy']}")
            if 'child_pattern' in factor_info.get('hierarchy', {}):
                print(f"    Child pattern: {factor_info['hierarchy']['child_pattern']}")
    
    # First update - campaign level
    print("\n5. Checking if sensor can update AIcon's state...")
    # Create a simpler environment with no sampling to avoid errors
    simple_env = {"use_individual_factors": True, "avoid_sampling": True}
    aicon.update_from_sensor("meta_ads", environment=simple_env)
    
    # Check which factors were created in the state
    state = aicon.brain.get_state_factors()
    
    # Verify campaign-level metrics exist
    print("\n6. Checking campaign-level metrics in state:")
    found_campaign_metrics = False
    for key in ["purchases", "add_to_carts", "initiated_checkouts"]:
        if key in state:
            found_campaign_metrics = True
            print(f"  - {key}: {state[key].get('value')}")
    
    if not found_campaign_metrics:
        print("  No campaign-level metrics found in state")
    
    # Check for ad-level metrics 
    print("\n7. Checking ad-level metrics in state:")
    ad_metrics = [k for k in state.keys() if k.startswith('ad_') and (
        k.endswith('_purchases') or 
        k.endswith('_add_to_carts') or 
        k.endswith('_initiated_checkouts')
    )]
    
    if ad_metrics:
        for metric in sorted(ad_metrics):
            print(f"  - {metric}: {state[metric].get('value')}")
    else:
        print("  No ad-level metrics found in state")
    
    # Check if hierarchical factors are properly defined
    print("\n8. Checking hierarchical relationships in expected factors:")
    hierarchical_factors = {name: info for name, info in expected_factors.items() 
                          if 'hierarchy' in info}
    
    if hierarchical_factors:
        print(f"Found {len(hierarchical_factors)} hierarchical factor definitions:")
        for name, info in hierarchical_factors.items():
            hierarchy = info.get('hierarchy', {})
            if 'role' in hierarchy:
                if hierarchy['role'] == 'aggregate':
                    print(f"  - {name}: AGGREGATE factor with child pattern: {hierarchy.get('child_pattern')}")
                elif hierarchy['role'] == 'child':
                    print(f"  - {name}: CHILD factor contributing to: {hierarchy.get('parent')}")
    else:
        print("No hierarchical factors defined")
    
    # Restore original run method
    sensor.run = original_run
    
    print("\n=== TEST COMPLETE ===")

if __name__ == "__main__":
    hierarchical_test() 