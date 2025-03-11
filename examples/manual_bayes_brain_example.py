"""
Manual BayesBrain Configuration Example

This example demonstrates how to create a BadAIcon with an empty BayesBrain
and then manually configure each component step by step.
"""

import numpy as np
from typing import Dict, Any, List
from aicons.definitions.aicon_types import BadAIcon, Campaign
from aicons.bayesbrainGPT.decision_making.action_space import (
    ActionDimension, ActionSpace, create_budget_allocation_space
)

# Create a BAD AIcon instance with an empty BayesBrain
bad_aicon = BadAIcon(
    name="ManualBayesBrainDemo",
    capabilities=["budget_allocation", "campaign_management"]
)

print("1. Created BadAIcon with empty BayesBrain")
print(f"   - Name: {bad_aicon.name}")
print(f"   - Type: {bad_aicon.aicon_type}")
print(f"   - Capabilities: {bad_aicon.capabilities}")

# At this point, the BayesBrain exists but is empty (all components are None or empty)
print("\n2. Initial BayesBrain state:")
print(f"   - Action space: {bad_aicon.brain.get_action_space()}")
print(f"   - State factors: {bad_aicon.brain.get_state_factors()}")
print(f"   - Utility function: {bad_aicon.brain.get_utility_function()}")
print(f"   - Sensors: {len(bad_aicon.brain.get_sensors())}")
print(f"   - Posterior samples: {bad_aicon.brain.get_posterior_samples()}")

# Step 3: Manually add campaigns
print("\n3. Adding campaigns manually")
campaigns = [
    Campaign(
        id="campaign_001",
        name="Facebook Awareness Campaign",
        platform="facebook",
        total_budget=5000.0,
        daily_budget=300.0,
        performance_metrics={
            "impressions": 10000,
            "clicks": 200,
            "conversions": 10,
            "cpc": 0.65
        }
    ),
    Campaign(
        id="campaign_002",
        name="Google Search Campaign",
        platform="google",
        total_budget=7000.0,
        daily_budget=400.0,
        performance_metrics={
            "impressions": 8000,
            "clicks": 300,
            "conversions": 15,
            "cpc": 0.80
        }
    )
]

for campaign in campaigns:
    bad_aicon.add_campaign(campaign)
    print(f"   - Added campaign: {campaign.name}")

# Step 4: Manually create and set the action space
print("\n4. Manually creating and setting action space")
num_ads = len(bad_aicon.campaigns)
total_budget = 1000.0
budget_step = 100.0

# Create a budget allocation action space
action_space = create_budget_allocation_space(
    total_budget=total_budget,
    num_ads=num_ads,
    budget_step=budget_step,
    min_budget=0.0
)

# Set the action space in the BayesBrain
bad_aicon.brain.set_action_space(action_space)

print(f"   - Created action space with {num_ads} dimensions")
print(f"   - Total budget: ${total_budget:.2f}, increment: ${budget_step:.2f}")
for dim in action_space.dimensions:
    print(f"   - {dim.name}: {dim.dim_type}, range: {dim.min_value} to {dim.max_value}, step: {dim.step}")

# Step 5: Manually set state factors
print("\n5. Manually setting state factors")
state_factors = {
    "roi_target": 1.5,
    "ctr_baseline": 0.025,
    "conversion_rate_baseline": 0.05,
    "risk_tolerance": 0.7,
    "exploration_rate": 0.2
}

bad_aicon.brain.set_state_factors(state_factors)
print(f"   - Set state factors: {bad_aicon.brain.get_state_factors()}")

# Step 6: Manually create and set posterior samples
print("\n6. Manually creating and setting posterior samples")
num_samples = 1000

# Conversion rate samples (phi) - around 5% conversion rate with small variance
phi_samples = np.random.normal(0.05, 0.01, size=(num_samples, num_ads))

# Cost per click samples (c) - around $0.70 per click with gamma distribution
c_samples = np.random.gamma(7, 0.1, size=(num_samples, num_ads))

# Click-through rate samples (ctr) - around 2.5% with beta distribution
ctr_samples = np.random.beta(2, 80, size=(num_samples, num_ads))

# Day-of-week effect samples (dow) - multiplicative effect of day of week
dow_samples = np.random.normal(1.0, 0.2, size=(num_samples, 7, num_ads))

posterior_samples = {
    "conversion_rate": phi_samples,
    "cost_per_click": c_samples,
    "ctr": ctr_samples,
    "day_of_week_effect": dow_samples
}

bad_aicon.brain.set_posterior_samples(posterior_samples)
print(f"   - Set posterior samples with shapes:")
for key, samples in bad_aicon.brain.get_posterior_samples().items():
    print(f"     - {key}: {samples.shape}")

# Step 7: Manually create and set utility function
print("\n7. Manually creating and setting utility function")

def custom_utility_function(action: Dict[str, float]) -> float:
    """
    Calculate expected ROI based on the action (budget allocation)
    and current posterior samples
    """
    # Get the current posterior samples
    posterior_samples = bad_aicon.brain.get_posterior_samples()
    state_factors = bad_aicon.brain.get_state_factors()
    
    # Extract relevant samples
    conversion_rates = posterior_samples.get("conversion_rate", np.zeros((1, num_ads)))
    cpcs = posterior_samples.get("cost_per_click", np.zeros((1, num_ads)))
    ctrs = posterior_samples.get("ctr", np.zeros((1, num_ads)))
    
    # Get risk tolerance from state factors
    risk_tolerance = state_factors.get("risk_tolerance", 0.5)
    
    # Convert action to numpy array in the same order as our samples
    campaign_ids = list(bad_aicon.campaigns.keys())
    budgets = np.array([action.get(campaign_id, 0.0) for campaign_id in campaign_ids])
    
    # Calculate expected clicks for each campaign
    expected_clicks = budgets / cpcs
    
    # Calculate expected conversions
    expected_conversions = expected_clicks * conversion_rates
    
    # Calculate expected revenue (assuming $10 per conversion)
    revenue_per_conversion = 10.0
    expected_revenue = expected_conversions * revenue_per_conversion
    
    # Calculate ROI
    total_budget = sum(budgets)
    if total_budget == 0:
        return 0.0
    
    roi = np.sum(expected_revenue, axis=1) / total_budget - 1.0
    
    # Apply risk adjustment based on risk tolerance
    # Higher risk tolerance means we care more about the mean
    # Lower risk tolerance means we care more about the lower percentile
    risk_percentile = 100 * (1 - risk_tolerance)
    safe_roi = np.percentile(roi, risk_percentile)
    mean_roi = np.mean(roi)
    
    # Weighted average based on risk tolerance
    adjusted_roi = risk_tolerance * mean_roi + (1 - risk_tolerance) * safe_roi
    
    return float(adjusted_roi)

bad_aicon.brain.set_utility_function(custom_utility_function)
print("   - Set custom utility function for ROI optimization")

# Step 8: Manually add sensors
print("\n8. Manually adding sensors")

def campaign_performance_sensor(environment):
    """Sensor that extracts campaign performance data from the environment"""
    if not environment or not isinstance(environment, dict):
        return {}
    
    # Extract campaign performance data if available
    campaign_performance = environment.get("campaign_performance", {})
    return {
        "observed_sales": campaign_performance.get("observed_sales", []),
        "observed_cpc": campaign_performance.get("observed_cpc", []),
        "budgets": campaign_performance.get("budgets", [])
    }

bad_aicon.brain.add_sensor(campaign_performance_sensor)
print(f"   - Added campaign performance sensor")
print(f"   - Total sensors: {len(bad_aicon.brain.get_sensors())}")

# Step 9: Sample allocations from the manually configured BayesBrain
print("\n9. Sampling allocations from the manually configured BayesBrain")
for i in range(3):
    allocation = bad_aicon.sample_allocation()
    print(f"\n   Allocation {i+1}:")
    for key, value in allocation.items():
        print(f"     {key}: ${value:.2f}")
    print(f"     Total: ${sum(allocation.values()):.2f}")

# Step 10: Find the best allocation using the utility function
print("\n10. Finding the best allocation using the utility function")
best_action, expected_utility = bad_aicon.brain.find_best_action(num_samples=100)
print(f"   Best allocation found with expected utility: {expected_utility:.4f}")
print("   Budget allocation:")
for key, value in best_action.items():
    print(f"     {key}: ${value:.2f}")
print(f"     Total: ${sum(best_action.values()):.2f}")

print("\nManual BayesBrain configuration complete!") 