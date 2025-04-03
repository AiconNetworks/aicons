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
from aicons.bayesbrainGPT.decision_making.marketing_action_spaces import create_budget_allocation_space

def print_section(title):
    """Helper function to print section headers"""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")

def print_metric(name, value, unit=""):
    """Helper function to print metrics in a consistent format"""
    print(f"{name:<30}: {value:>10} {unit}")

def main():
    print_section("Initializing Marketing AIcon")
    aicon = AIcon("marketing_aicon")
    print_metric("AIcon Name", aicon.name)

    print_section("Setting up Meta Ads Sensor")
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

    # Add sensor first
    aicon.add_sensor("meta_ads", sensor)
    print_metric("Sensor Added", "meta_ads")

    print_section("Fetching Active Ads")
    active_ads = sensor.get_active_ads()
    print_metric("Number of Active Ads", len(active_ads))
    
    # Extract ad IDs for action space
    ad_ids = [ad['ad_id'] for ad in active_ads]
    print("\nActive Ad IDs:")
    for ad_id in ad_ids:
        print(f"  - {ad_id}")

    print_section("Creating Action Space")
    action_space = create_budget_allocation_space(
        total_budget=1000.0,
        num_ads=len(active_ads),
        budget_step=100.0,
        ad_names=ad_ids
    )
    aicon.brain.action_space = action_space
    print_metric("Total Budget", "$1,000.00")
    print_metric("Budget Step Size", "$100.00")
    print("\nAction Space Summary:")
    print(f"  - Number of dimensions: {len(action_space.dimensions)}")
    print(f"  - Total possible actions: {action_space.size if action_space.size != float('inf') else 'infinite'}")
    print(f"  - Constraints: {len(action_space.constraints)}")

    print_section("Setting up Utility Function")
    aicon.define_utility_function(
        utility_type='marketing_roi',
        name="Marketing ROI Utility",
        revenue_per_sale=50.0,
        num_days=1,
        ad_names=ad_ids
    )
    print_metric("Revenue per Sale", "$50.00")
    print_metric("Time Horizon", "1 day")

    print_section("Updating Beliefs from Sensor Data")
    print("Fetching and processing Meta Ads data...")
    aicon.update_from_sensor("meta_ads")

    print_section("Analyzing Posterior Samples")
    posterior_samples = aicon.get_posterior_samples()
    print("\nKey Metrics Summary:")
    for key, value in posterior_samples.items():
        if "purchases" in key or "clicks" in key or "spend" in key:
            mean = np.mean(value)
            std = np.std(value)
            print(f"\n{key}:")
            print_metric("  Mean", f"{mean:.2f}")
            print_metric("  Std Dev", f"{std:.2f}")

    print_section("Finding Optimal Budget Allocation")
    print("Evaluating different budget allocations...")
    best_action, expected_utility = aicon.find_best_action(num_samples=100)
    
    print("\nOptimal Budget Allocation:")
    total_allocated = 0
    for ad_id, budget in best_action.items():
        print_metric(f"  {ad_id}", f"${budget:.2f}")
        total_allocated += budget
    print_metric("Total Allocated", f"${total_allocated:.2f}")
    print_metric("Expected Utility", f"${expected_utility:.2f}")

if __name__ == "__main__":
    main() 