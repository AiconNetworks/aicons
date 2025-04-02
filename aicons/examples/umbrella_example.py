"""
Example of using BayesBrainGPT for umbrella decision making.
This demonstrates a simple Bayesian decision-making scenario where the AIcon
decides whether to take an umbrella based on the probability of rain.
"""

# Standard library imports
import sys
import os
from pathlib import Path

# Third-party imports
import numpy as np
import tensorflow as tf

# Add parent directory to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

# Local imports
from aicons.definitions.aicon import AIcon
from aicons.bayesbrainGPT.utility_function.umbrella_utility import UmbrellaUtility

def main():
    # Create AIcon
    aicon = AIcon("umbrella_aicon")
    print(f"AIcon created: {aicon.name}")

    # Add state factor for rain probability using AIcon's interface
    aicon.add_state_factor(
        name="rain",
        factor_type="continuous",
        value=0.3,  # Prior probability of rain
        params={
            "loc": 0.3,
            "scale": 0.1,  # Small uncertainty in our prior
            "constraints": {"lower": 0, "upper": 1}  # Probability must be between 0 and 1
        },
        relationships={
            "depends_on": []  # Empty list for root factor
        }
    )
    print("Added rain state factor")

    # Define action space (take umbrella or not)
    aicon.define_action_space(
        space_type='custom',
        dimensions_specs=[
            {
                "name": "umbrella",
                "type": "discrete",
                "values": [0, 1]  # 0 = no umbrella, 1 = take umbrella
            }
        ]
    )
    print("Defined action space")

    # Define utility function using UmbrellaUtility
    utility = UmbrellaUtility(
        name="umbrella_utility",
        cost=1.0,  # Cost of taking umbrella
        rain_cost=5.0,  # Cost of getting wet in rain
        description="Utility function for umbrella decision"
    )
    
    # Set the utility function in the brain
    aicon.brain.set_utility_function(utility)
    print("Set utility function")

    # Debug: Print state factors
    print("\nState Factors:")
    state_factors = aicon.get_state_factors()
    for name, factor in state_factors.items():
        print(f"{name}: {factor}")

    # Debug: Print action space
    print("\nAction Space Details:")
    print(aicon.brain.action_space.raw_print())

    # Update beliefs using AIcon's interface
    print("\nUpdating beliefs...")
    aicon.update_from_sensor()  # No sensor specified, will use priors
    
    # Debug: Print posterior samples
    print("\nPosterior Samples:")
    posterior_samples = aicon.get_posterior_samples()
    for key, value in posterior_samples.items():
        print(f"{key}: shape={value.shape}, first value={value[0]}")

    # Find best action
    print("\nFinding best action...")
    best_action, expected_utility = aicon.find_best_action(num_samples=100)
    print(f"\nBest action: {'Take umbrella' if best_action['umbrella'] == 1 else 'No umbrella'}")
    print(f"Expected utility: {expected_utility}")

    # Print expected utilities for both actions
    print("\nExpected utilities for each action:")
    for action in [{"umbrella": 0}, {"umbrella": 1}]:
        utility = aicon.brain.utility_function.evaluate(action, {"rain": 0.3})
        print(f"{'Take umbrella' if action['umbrella'] == 1 else 'No umbrella'}: {utility}")

if __name__ == "__main__":
    main() 