"""
Empty BayesBrain Example

This example demonstrates how to create a BadAIcon with an empty BayesBrain
and use it with minimal configuration.
"""

from aicons.definitions.aicon_types import BadAIcon, Campaign
from aicons.bayesbrainGPT.decision_making.action_space import create_budget_allocation_space

# Step 1: Create a BAD AIcon instance with an empty BayesBrain
print("Step 1: Creating BadAIcon with empty BayesBrain")
bad_aicon = BadAIcon(
    name="EmptyBrainDemo",
    capabilities=["budget_allocation"]
)

# Step 2: Check the initial state of the BayesBrain
print("\nStep 2: Initial BayesBrain state")
print(f"Action space: {bad_aicon.brain.get_action_space()}")
print(f"State factors: {bad_aicon.brain.get_state_factors()}")
print(f"Utility function: {bad_aicon.brain.get_utility_function()}")
print(f"Sensors: {len(bad_aicon.brain.get_sensors())}")

# Step 3: Add campaigns
print("\nStep 3: Adding campaigns")
campaigns = [
    Campaign(
        id="campaign_001",
        name="Facebook Campaign",
        platform="facebook",
        total_budget=5000.0,
        daily_budget=300.0,
        performance_metrics={"cpc": 0.65}
    ),
    Campaign(
        id="campaign_002",
        name="Google Campaign",
        platform="google",
        total_budget=7000.0,
        daily_budget=400.0,
        performance_metrics={"cpc": 0.80}
    )
]

for campaign in campaigns:
    bad_aicon.add_campaign(campaign)
    print(f"Added campaign: {campaign.name}")

# Step 4: Set up a minimal action space
print("\nStep 4: Setting up minimal action space")
action_space = create_budget_allocation_space(
    total_budget=1000.0,
    num_ads=len(bad_aicon.campaigns),
    budget_step=100.0,
    min_budget=0.0
)
bad_aicon.brain.set_action_space(action_space)
print(f"Action space set with {len(action_space.dimensions)} dimensions")

# Step 5: Try to sample an allocation
print("\nStep 5: Sampling allocation from minimal BayesBrain")
try:
    allocation = bad_aicon.sample_allocation()
    print("Successfully sampled allocation:")
    for key, value in allocation.items():
        print(f"  {key}: ${value:.2f}")
    print(f"  Total: ${sum(allocation.values()):.2f}")
except Exception as e:
    print(f"Error sampling allocation: {e}")
    print("This is expected if the BayesBrain is not fully configured")

# Step 6: Set a minimal utility function
print("\nStep 6: Setting minimal utility function")

def simple_utility(action):
    """A simple utility function that prefers balanced allocations"""
    values = list(action.values())
    if not values:
        return 0.0
    
    # Simple utility: higher for more balanced allocations
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    
    # Lower variance means more balanced allocation
    return -variance

bad_aicon.brain.set_utility_function(simple_utility)
print("Utility function set")

# Step 7: Try to find the best allocation
print("\nStep 7: Finding best allocation with minimal BayesBrain")
try:
    best_action, utility = bad_aicon.brain.find_best_action(num_samples=50)
    print(f"Best allocation found with utility: {utility:.4f}")
    print("Budget allocation:")
    for key, value in best_action.items():
        print(f"  {key}: ${value:.2f}")
    print(f"  Total: ${sum(best_action.values()):.2f}")
except Exception as e:
    print(f"Error finding best allocation: {e}")
    print("This is expected if the BayesBrain is not fully configured")

print("\nEmpty BayesBrain example complete!") 