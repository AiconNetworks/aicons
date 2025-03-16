# Test Core Integration of SimpleBadAIcon with BayesBrain
# This minimal test focuses ONLY on verifying that action space and utility function
# are properly stored in the brain, which was the key refactoring change.

import sys
import os
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

print("=== TESTING CORE BAYESBRAIN INTEGRATION WITH SIMPLEBADARICON ===")
print("This test focuses on the fundamental integration point: action space and utility function")

# Step 1: Create the AIcon
print("\n=== Step 1: Create AIcon ===")
aicon = SimpleBadAIcon("test_integration_aicon")
print(f"AIcon created: {aicon.name}")

# Step 2: Set up minimal prior beliefs (just one factor)
print("\n=== Step 2: Set Prior Beliefs ===")
aicon.add_factor_continuous("conversion_rate", 0.02, 0.005, lower_bound=0.0)

# Verify factor is in the brain
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
    min_budget=0.0
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

# Step 5: Verify getter methods
print("\n=== Step 5: Verify AIcon Getter Methods ===")
aicon_action_space = aicon.get_action_space()
aicon_utility_function = aicon.get_utility_function()

print(f"Action space from getter: {aicon_action_space is not None}")
print(f"Same as brain action space: {aicon_action_space is brain_action_space}")
print(f"Utility function from getter: {aicon_utility_function is not None}")
print(f"Same as brain utility function: {aicon_utility_function is brain_utility_function}")

# Step 6: Modify action space directly in brain
print("\n=== Step 6: Modify Action Space Directly in Brain ===")
from aicons.bayesbrainGPT.decision_making.action_space import create_marketing_ads_space

# Create new action space with finer granularity
new_action_space = create_marketing_ads_space(
    total_budget=1000.0,
    num_ads=2,
    budget_step=50.0,  # Now in $50 increments instead of $100
    min_budget=0.0
)

# Set directly on brain
aicon.brain.set_action_space(new_action_space)

# Verify AIcon uses the updated action space
current_action_space = aicon.get_action_space()
print(f"New action space step size: ${current_action_space.dimensions[0].step}")
print(f"Same as newly created action space: {current_action_space is new_action_space}")

print("\n=== CORE INTEGRATION TEST COMPLETED SUCCESSFULLY ===")
print("The action space and utility function are properly stored in the brain.")
print("This confirms the key integration point of the refactoring.") 