# Test BayesBrain Integration with SimpleBadAIcon
import sys
import os
import numpy as np
import importlib

# Fix import path
project_root = "/Users/infa/Documents/Babel"
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to sys.path")

# Import and reload modules to ensure latest changes
import aicons.definitions.simple_bad_aicon
importlib.reload(aicons.definitions.simple_bad_aicon)
from aicons.definitions.simple_bad_aicon import SimpleBadAIcon

# Import the MetaAdsSalesSensor for testing
from aicons.bayesbrainGPT.sensors.meta_s.meta_ads_sales_sensor import MetaAdsSalesSensor

# Create a customized Meta Ads Sales Sensor for testing
class TestMetaAdsSensor(MetaAdsSalesSensor):
    """A testing version of the Meta Ads Sales Sensor that uses mock data."""
    
    def __init__(self, name="meta_ads_test", reliability=0.9):
        # Initialize without API credentials to force mock mode
        super().__init__(name=name, reliability=reliability)
        # Override use_real_data to ensure we use mock data
        self.use_real_data = False
    
    def __call__(self, environment=None):
        """Make the sensor callable for the perception system."""
        # Generate mock data
        mock_data = self.mock_run(num_adsets=2, num_ads_per_adset=1)
        # Extract the relevant conversion metrics for our test
        observations = {
            "ad1_conversion_rate": 0.025,  # Slightly higher than prior
            "ad2_conversion_rate": 0.02,   # Lower than prior
            "ad1_cost_per_click": 0.45,    # Lower than prior
            "ad2_cost_per_click": 0.8      # Higher than prior
        }
        return observations
    
    def get_expected_factors(self):
        """Define the specific factors we need for this test."""
        return {
            "ad1_conversion_rate": {
                "type": "continuous",
                "default_value": 0.02,
                "uncertainty": 0.005,
                "lower_bound": 0.0,
                "description": "Conversion rate for ad 1"
            },
            "ad2_conversion_rate": {
                "type": "continuous",
                "default_value": 0.03,
                "uncertainty": 0.008,
                "lower_bound": 0.0,
                "description": "Conversion rate for ad 2"
            },
            "ad1_cost_per_click": {
                "type": "continuous",
                "default_value": 0.5,
                "uncertainty": 0.1,
                "lower_bound": 0.1,
                "description": "Cost per click for ad 1"
            },
            "ad2_cost_per_click": {
                "type": "continuous",
                "default_value": 0.7,
                "uncertainty": 0.15,
                "lower_bound": 0.1,
                "description": "Cost per click for ad 2"
            }
        }

print("=== TESTING BAYESBRAIN INTEGRATION WITH SIMPLEBADARICON ===")

# Step 1: Create the AIcon
print("\n=== Step 1: Create AIcon ===")
aicon = SimpleBadAIcon("test_integration_aicon")
print(f"AIcon created: {aicon.name}")

# Step 2: Set up prior beliefs
print("\n=== Step 2: Set Prior Beliefs ===")
aicon.add_factor_continuous("ad1_conversion_rate", 0.02, 0.005, lower_bound=0.0)
aicon.add_factor_continuous("ad2_conversion_rate", 0.03, 0.008, lower_bound=0.0)
aicon.add_factor_continuous("ad1_cost_per_click", 0.5, 0.1, lower_bound=0.1)
aicon.add_factor_continuous("ad2_cost_per_click", 0.7, 0.15, lower_bound=0.1)

# Verify factors are in the brain
state_factors = aicon.brain.get_state_factors()
print(f"Number of factors in brain: {len(state_factors)}")
print("Factor names:", ", ".join(state_factors.keys()))

# Step 3: Create action space and verify it's in the brain
print("\n=== Step 3: Create Action Space ===")
action_space = aicon.create_action_space(
    space_type='marketing',
    total_budget=1000.0,
    num_ads=2,
    budget_step=100.0,
    min_budget=0.0,
    ad_names=["ad1", "ad2"]
)

# Verify action space is in the brain
brain_action_space = aicon.brain.get_action_space()
print(f"Action space in brain: {brain_action_space is not None}")
print(f"Same action space object: {action_space is brain_action_space}")

# Step 4: Create utility function and verify it's in the brain
print("\n=== Step 4: Create Utility Function ===")
utility_function = aicon.create_utility_function(
    utility_type='marketing_roi',
    revenue_per_sale=50.0,
    num_ads=2,
    num_days=1
)

# Verify utility function is in the brain
brain_utility_function = aicon.brain.get_utility_function()
print(f"Utility function in brain: {brain_utility_function is not None}")
print(f"Same utility function object: {utility_function is brain_utility_function}")

# Step 5: Add sensor and update beliefs
print("\n=== Step 5: Add Sensor and Update Beliefs ===")
sensor = TestMetaAdsSensor(reliability=0.8)
aicon.add_sensor("meta_ads", sensor)
aicon.update_from_sensor("meta_ads")

# Get posterior samples
posterior_samples = aicon.get_posterior_samples()
print(f"Number of posterior factors: {len(posterior_samples)}")
print("Posterior factor names:", ", ".join(posterior_samples.keys()))

# Step 6: Find best action directly using the brain
print("\n=== Step 6: Find Best Action Using Brain ===")
best_action, utility = aicon.brain.find_best_action(num_samples=100)
print(f"Best action found with utility: ${utility:.2f}")
print("Optimal budget allocation:")
for name, budget in best_action.items():
    percentage = (budget / 1000.0) * 100
    print(f"- {name}: ${budget:.2f} ({percentage:.1f}%)")

# Step 7: Find best action via AIcon convenience method
print("\n=== Step 7: Find Best Action Via AIcon Method ===")
best_action2, utility2 = aicon.find_best_action(num_samples=100)
print(f"Best action found with utility: ${utility2:.2f}")
print("Optimal budget allocation:")
for name, budget in best_action2.items():
    percentage = (budget / 1000.0) * 100
    print(f"- {name}: ${budget:.2f} ({percentage:.1f}%)")

# Step 8: Modify action space directly in brain
print("\n=== Step 8: Modify Action Space Directly in Brain ===")
from aicons.bayesbrainGPT.decision_making.action_space import create_marketing_ads_space

# Create new action space with finer granularity
new_action_space = create_marketing_ads_space(
    total_budget=1000.0,
    num_ads=2,
    budget_step=50.0,  # Now in $50 increments instead of $100
    min_budget=0.0,
    ad_names=["ad1", "ad2"]
)

# Set directly on brain
aicon.brain.set_action_space(new_action_space)

# Verify AIcon uses the updated action space
current_action_space = aicon.get_action_space()
print(f"New action space step size: ${current_action_space.dimensions[0].step}")
print(f"Same as newly created action space: {current_action_space is new_action_space}")

# Find best action with new action space
best_action3, utility3 = aicon.find_best_action(num_samples=100)
print(f"Best action with new action space and utility: ${utility3:.2f}")
print("Optimal budget allocation with finer granularity:")
for name, budget in best_action3.items():
    percentage = (budget / 1000.0) * 100
    print(f"- {name}: ${budget:.2f} ({percentage:.1f}%)")

print("\n=== TEST COMPLETED SUCCESSFULLY ===") 