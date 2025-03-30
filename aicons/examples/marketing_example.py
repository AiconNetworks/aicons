"""
Example of using BayesBrainGPT for marketing optimization.
This replicates the setup from the notebook and includes debugging code.
"""

import sys
import os
from pathlib import Path
import numpy as np
import tensorflow as tf

# Add parent directory to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

# Import the required classes
from aicons.definitions.aicon import AIcon
from aicons.bayesbrainGPT.sensors.meta_s.meta_ads_sales_sensor import MetaAdsSalesSensor

def main():
    # Create AIcon
    aicon = AIcon("marketing_aicon")
    print(f"AIcon created: {aicon.name}")

    # Add state factors
    aicon.add_state_factor(
        name="purchases",
        factor_type="continuous",
        value=0.0,
        params={
            "loc": 0.0,
            "scale": 1.0,
            "constraints": {"lower": 0}
        },
        relationships={
            "depends_on": []  # Empty list for root factor
        }
    )

    aicon.add_state_factor(
        name="add_to_carts",
        factor_type="continuous",
        value=0.0,
        params={
            "loc": 0.0,
            "scale": 5.0,
            "constraints": {"lower": 0}
        },
        relationships={
            "depends_on": []  # Empty list for root factor
        }
    )

    aicon.add_state_factor(
        name="initiated_checkouts",
        factor_type="continuous",
        value=0.0,
        params={
            "loc": 0.0,
            "scale": 2.0,
            "constraints": {"lower": 0}
        },
        relationships={
            "depends_on": []  # Empty list for root factor
        }
    )

    # Setup Meta Ads sensor
    access_token = "EAAZAn8wmq1IEBOZCz8oyDZBBgiazAgnQKIoAr4mFTbkV7jxi6t3APzOSxFybXNIkBgwQACdagbs5lFE8tpnNOBOOpWtS3KjZAdf9MNAlySpwEaDrX32oQwUTNmOZAaSXjT5Os5Q8YqRo57tXOUukB7QtcO8nQ8JuqrnnshCr7A0giynZBnJKfuPakrZBWoZD"
    ad_account_id = "act_252267674525035"
    campaign_id = "120218631288730217"

    sensor = MetaAdsSalesSensor(
        name="meta_ads",
        reliability=0.9,
        access_token=access_token,
        ad_account_id=ad_account_id,
        campaign_id=campaign_id,
        api_version="v18.0",
        time_granularity="hour"
    )

    aicon.add_sensor("meta_ads", sensor)
    print("Meta Ads sensor added")

    # Define action space
    aicon.define_action_space(
        space_type='marketing',
        total_budget=1000.0,
        num_ads=2,
        budget_step=100.0,
        ad_names=['google', 'facebook']
    )

    # Define utility function
    aicon.define_utility_function(
        utility_type='marketing_roi',
        name="Marketing ROI Utility",
        revenue_per_sale=50.0,  # $50 revenue per conversion
        num_days=1  # Since we're using hourly data
    )

    # Update from Meta Ads sensor
    print("Updating beliefs from Meta Ads sensor...")
    aicon.update_from_sensor("meta_ads")

    # Debug: Print action space
    print("\nAction Space Details:")
    print(aicon.brain.action_space.raw_print())

    # Debug: Print posterior samples
    print("\nPosterior Samples:")
    posterior_samples = aicon.get_posterior_samples()
    for key, value in posterior_samples.items():
        print(f"{key}: shape={value.shape}, first value={value[0]}")

    # Find best action
    print("\nFinding best action...")
    best_action, expected_utility = aicon.find_best_action(num_samples=100)
    print(f"\nBest action: {best_action}")
    print(f"Expected utility: {expected_utility}")

if __name__ == "__main__":
    main() 