"""
Test script for the decision-making capabilities of the BaseAIcon class.

This script demonstrates how to use BaseAIcon for decision-making with manual space definitions.
"""

import os
import sys
import json
from datetime import datetime
import numpy as np

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aicons.definitions.base_aicon import BaseAIcon

def test_simple_decision():
    """Test basic decision-making with BaseAIcon."""
    print("\n=== Testing BaseAIcon Decision Making ===")
    
    # Create a BaseAIcon instance
    aicon = BaseAIcon("DecisionAIcon", "decision_test")
    print(f"Created AIcon: {aicon.name} (ID: {aicon.id})")
    
    # Add some factors related to a simple advertising scenario
    aicon.add_factor_continuous(
        name="conversion_rate_ad1", 
        value=0.05,  # 5% conversion rate
        uncertainty=0.01,
        lower_bound=0.0,
        upper_bound=1.0,
        description="Conversion rate for Ad 1"
    )
    
    aicon.add_factor_continuous(
        name="conversion_rate_ad2", 
        value=0.03,  # 3% conversion rate
        uncertainty=0.01,
        lower_bound=0.0,
        upper_bound=1.0,
        description="Conversion rate for Ad 2"
    )
    
    aicon.add_factor_continuous(
        name="cost_per_click_ad1", 
        value=0.50,  # $0.50 per click
        uncertainty=0.10,
        lower_bound=0.01,
        description="Cost per click for Ad 1"
    )
    
    aicon.add_factor_continuous(
        name="cost_per_click_ad2", 
        value=0.30,  # $0.30 per click
        uncertainty=0.05,
        lower_bound=0.01,
        description="Cost per click for Ad 2"
    )
    
    # Print the state
    print("\nAIcon State (prior beliefs):")
    print(aicon.get_state(format_nicely=True))
    
    # Create a manually defined budget allocation action space
    try:
        # Define dimensions for a budget allocation action space
        dimensions = {
            "ad1": {
                "type": "continuous",
                "min": 0.0,
                "max": 1000.0,
                "default": 500.0
            },
            "ad2": {
                "type": "continuous", 
                "min": 0.0,
                "max": 1000.0,
                "default": 500.0
            }
        }
        
        # Define constraints (total budget = 1000)
        constraints = [
            {
                "type": "sum",
                "dimensions": ["ad1", "ad2"],
                "value": 1000.0
            }
        ]
        
        action_space = aicon.create_action_space(
            dimensions=dimensions,
            constraints=constraints
        )
        
        if action_space:
            print("\nSuccessfully created manual budget allocation action space")
            
            # Define a utility function for marketing ROI
            def marketing_roi_utility(action, state):
                """
                Calculate marketing ROI utility.
                
                Args:
                    action: Dictionary of actions (ad budgets)
                    state: Dictionary of state factors
                
                Returns:
                    Expected utility (profit)
                """
                # Initialize expected profit
                expected_profit = 0.0
                
                # Revenue per conversion
                revenue_per_sale = 20.0
                
                # For each ad in the action
                for ad_name, budget in action.items():
                    # Extract the ad number (e.g., 'ad1' -> '1')
                    ad_num = ad_name[2:]
                    
                    # Factor names for this ad
                    conv_rate_name = f"conversion_rate_ad{ad_num}"
                    cost_name = f"cost_per_click_ad{ad_num}"
                    
                    # Get current beliefs
                    if conv_rate_name in state and cost_name in state:
                        conv_rate = state[conv_rate_name]["value"]
                        cost_per_click = state[cost_name]["value"]
                        
                        # Calculate number of clicks
                        if cost_per_click > 0:
                            clicks = budget / cost_per_click
                        else:
                            clicks = 0
                        
                        # Calculate conversions and revenue
                        conversions = clicks * conv_rate
                        revenue = conversions * revenue_per_sale
                        
                        # Add to expected profit
                        ad_profit = revenue - budget
                        expected_profit += ad_profit
                
                return expected_profit
            
            # Create the utility function directly
            utility_function = aicon.create_utility_function(
                function=marketing_roi_utility,
                name="manual_marketing_roi"
            )
            
            if utility_function:
                print("Successfully created manual marketing ROI utility function")
                
                # Find the best action
                print("\nFinding the best action...")
                best_action, expected_utility = aicon.find_best_action(num_samples=100)
                
                if best_action:
                    print("\nBest Action:")
                    for ad, budget in best_action.items():
                        print(f"- {ad}: ${budget:.2f}")
                    print(f"Expected utility: ${expected_utility:.2f}")
                    
                    # Try sampling a random action
                    sampled_action = aicon.sample_action()
                    if sampled_action:
                        print("\nRandom Sampled Action:")
                        for ad, budget in sampled_action.items():
                            print(f"- {ad}: ${budget:.2f}")
                else:
                    print("Failed to find best action")
            else:
                print("Failed to create utility function")
        else:
            print("Failed to create action space")
    except Exception as e:
        print(f"WARNING: Decision-making test could not complete: {e}")
        print("This is expected if BayesBrain or TensorFlow is not available")
    
    print("\n=== Decision Making Test Completed ===")
    return aicon

def test_sensor_decision():
    """Test decision-making with sensor updates."""
    print("\n=== Testing BaseAIcon Sensor-Based Decision Making ===")
    
    # Create a BaseAIcon instance
    aicon = BaseAIcon("SensorDecisionAIcon", "sensor_decision_test")
    
    # Create a simple mock ad performance sensor function
    def mock_ad_sensor(environment=None):
        """Mock sensor that returns ad performance metrics."""
        # Default values if no environment is provided
        if environment is None:
            return {
                "conversion_rate_ad1": 0.04,
                "conversion_rate_ad2": 0.06,  # Higher than initial prior
                "cost_per_click_ad1": 0.45,
                "cost_per_click_ad2": 0.35
            }
        
        # Return values from environment if provided
        return environment
    
    try:
        # Add initial factors
        aicon.add_factor_continuous("conversion_rate_ad1", 0.05, 0.01, lower_bound=0.0, upper_bound=1.0)
        aicon.add_factor_continuous("conversion_rate_ad2", 0.03, 0.01, lower_bound=0.0, upper_bound=1.0)
        aicon.add_factor_continuous("cost_per_click_ad1", 0.50, 0.10, lower_bound=0.01)
        aicon.add_factor_continuous("cost_per_click_ad2", 0.30, 0.05, lower_bound=0.01)
        
        # Add the sensor
        sensor = aicon.add_sensor("ad_performance", mock_ad_sensor)
        
        if sensor:
            print("Successfully added mock ad performance sensor")
            
            # Create manually defined action space
            dimensions = {
                "ad1": {
                    "type": "continuous",
                    "min": 0.0,
                    "max": 1000.0,
                    "default": 500.0
                },
                "ad2": {
                    "type": "continuous", 
                    "min": 0.0,
                    "max": 1000.0,
                    "default": 500.0
                }
            }
            
            # Define constraints (total budget = 1000)
            constraints = [
                {
                    "type": "sum",
                    "dimensions": ["ad1", "ad2"],
                    "value": 1000.0
                }
            ]
            
            # Create action space
            aicon.create_action_space(
                dimensions=dimensions,
                constraints=constraints
            )
            
            # Create a utility function
            def marketing_roi_utility(action, state):
                # Revenue per conversion
                revenue_per_sale = 20.0
                expected_profit = 0.0
                
                for ad_name, budget in action.items():
                    ad_num = ad_name[2:]
                    conv_rate_name = f"conversion_rate_ad{ad_num}"
                    cost_name = f"cost_per_click_ad{ad_num}"
                    
                    if conv_rate_name in state and cost_name in state:
                        conv_rate = state[conv_rate_name]["value"]
                        cost_per_click = state[cost_name]["value"]
                        
                        if cost_per_click > 0:
                            clicks = budget / cost_per_click
                        else:
                            clicks = 0
                        
                        conversions = clicks * conv_rate
                        revenue = conversions * revenue_per_sale
                        ad_profit = revenue - budget
                        expected_profit += ad_profit
                
                return expected_profit
            
            # Create utility function
            aicon.create_utility_function(
                function=marketing_roi_utility,
                name="manual_marketing_roi"
            )
            
            # Find the best action with prior beliefs
            print("\nFinding best action with prior beliefs...")
            prior_action, prior_utility = aicon.find_best_action(num_samples=100)
            
            if prior_action:
                print("\nBest Action (Prior Beliefs):")
                for ad, budget in prior_action.items():
                    print(f"- {ad}: ${budget:.2f}")
                print(f"Expected utility: ${prior_utility:.2f}")
                
                # Now update beliefs with sensor data
                print("\nUpdating beliefs with sensor data...")
                success = aicon.update_from_sensor("ad_performance")
                
                if success:
                    print("Successfully updated beliefs")
                    print("\nUpdated State:")
                    print(aicon.get_state(format_nicely=True))
                    
                    # Find the best action with updated beliefs
                    print("\nFinding best action with updated beliefs...")
                    updated_action, updated_utility = aicon.find_best_action(num_samples=100)
                    
                    if updated_action:
                        print("\nBest Action (Updated Beliefs):")
                        for ad, budget in updated_action.items():
                            print(f"- {ad}: ${budget:.2f}")
                        print(f"Expected utility: ${updated_utility:.2f}")
                        
                        # Check if the decision changed
                        if abs(prior_action.get('ad1', 0) - updated_action.get('ad1', 0)) > 50:
                            print("\nThe budget allocation changed significantly after sensor update!")
                        else:
                            print("\nThe budget allocation remained similar after sensor update.")
                    else:
                        print("Failed to find best action with updated beliefs")
                else:
                    print("Failed to update beliefs")
            else:
                print("Failed to find best action with prior beliefs")
        else:
            print("Failed to add sensor")
    except Exception as e:
        print(f"WARNING: Sensor-based decision test could not complete: {e}")
        print("This is expected if BayesBrain or TensorFlow is not available")
    
    print("\n=== Sensor-Based Decision Making Test Completed ===")
    return aicon

def perceive_and_decide(aicon, environment=None):
    """
    Custom perceive and decide function since it was removed from BaseAIcon.
    
    Args:
        aicon: The AIcon instance
        environment: Environment data
    
    Returns:
        Tuple of (best_action, expected_utility)
    """
    # First update from all sensors if possible
    if hasattr(aicon, 'update_from_all_sensors'):
        aicon.update_from_all_sensors(environment)
    
    # Then find the best action
    return aicon.find_best_action()

def test_perceive_and_decide():
    """Test the perceive_and_decide convenience method."""
    print("\n=== Testing BaseAIcon Perceive and Decide ===")
    
    # Create a BaseAIcon instance
    aicon = BaseAIcon("PerceiveDecideAIcon", "perceive_decide_test")
    
    # Mock environment data
    environment = {
        "conversion_rate_ad1": 0.03,
        "conversion_rate_ad2": 0.07,
        "cost_per_click_ad1": 0.55,
        "cost_per_click_ad2": 0.25
    }
    
    try:
        # Add initial factors
        aicon.add_factor_continuous("conversion_rate_ad1", 0.05, 0.01, lower_bound=0.0, upper_bound=1.0)
        aicon.add_factor_continuous("conversion_rate_ad2", 0.03, 0.01, lower_bound=0.0, upper_bound=1.0)
        aicon.add_factor_continuous("cost_per_click_ad1", 0.50, 0.10, lower_bound=0.01)
        aicon.add_factor_continuous("cost_per_click_ad2", 0.30, 0.05, lower_bound=0.01)
        
        # Add a mock sensor function
        def mock_sensor(env=None):
            return env or {}
        
        # Add the sensor
        aicon.add_sensor("ad_performance", mock_sensor)
        
        # Create a fully manual action space
        dimensions = {
            "ad1": {"type": "continuous", "min": 0.0, "max": 1000.0, "default": 500.0},
            "ad2": {"type": "continuous", "min": 0.0, "max": 1000.0, "default": 500.0}
        }
        
        constraints = [
            {"type": "sum", "dimensions": ["ad1", "ad2"], "value": 1000.0}
        ]
        
        aicon.create_action_space(dimensions=dimensions, constraints=constraints)
        
        # Define a manual utility function
        def marketing_roi_utility(action, state):
            revenue_per_sale = 20.0
            profit = 0.0
            
            for ad_name, budget in action.items():
                ad_num = ad_name[2:]
                conv_rate = state.get(f"conversion_rate_ad{ad_num}", {}).get("value", 0)
                cost_per_click = state.get(f"cost_per_click_ad{ad_num}", {}).get("value", 1.0)
                
                if cost_per_click > 0:
                    clicks = budget / cost_per_click
                    conversions = clicks * conv_rate
                    revenue = conversions * revenue_per_sale
                    profit += revenue - budget
            
            return profit
        
        aicon.create_utility_function(
            function=marketing_roi_utility,
            name="manual_marketing_roi"
        )
        
        # Use perceive_and_decide to both update beliefs and find best action
        print("\nCalling perceive_and_decide with environment data...")
        action, utility = perceive_and_decide(aicon, environment)
        
        if action:
            print("\nBest Action (One-Step Perceive and Decide):")
            for ad, budget in action.items():
                print(f"- {ad}: ${budget:.2f}")
            print(f"Expected utility: ${utility:.2f}")
        else:
            print("Failed to perceive and decide")
    except Exception as e:
        print(f"WARNING: Perceive and decide test could not complete: {e}")
        print("This is expected if BayesBrain or TensorFlow is not available")
    
    print("\n=== Perceive and Decide Test Completed ===")
    return aicon

if __name__ == "__main__":
    print("=== Starting BaseAIcon Decision Making Tests ===\n")
    
    # Run the tests
    test_aicon = test_simple_decision()
    test_sensor_decision()
    test_perceive_and_decide()
    
    print("\n=== All Decision Making Tests Completed ===") 