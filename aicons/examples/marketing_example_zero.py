"""
Example of using ZeroAIcon for marketing optimization.
This replicates the setup from the original marketing example but without best actions selection.
"""

import sys
import os
from pathlib import Path
import json
import logging
import numpy as np

# Add parent directory to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

# Import ZeroAIcon
from aicons.definitions.zero import ZeroAIcon
from aicons.bayesbrainGPT.sensors.meta_s.meta_ads_sales_sensor import MetaAdsSalesSensor
from aicons.bayesbrainGPT.decision_making.marketing_action_spaces import create_budget_allocation_space

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Create ZeroAIcon
    aicon = ZeroAIcon(
        name="marketing_aicon",
        description="AIcon for marketing optimization",
        model_name="deepseek-r1:7b"
    )
    logger.info(f"Created ZeroAIcon: {aicon.name}")

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

    # Add sensor
    logger.info("Adding sensor...")
    aicon.add_sensor("meta_ads", sensor)
    logger.info("Meta Ads sensor added")

    # Get active ads from sensor
    active_ads = sensor.get_active_ads()
    logger.info(f"Found {len(active_ads)} active ads")
    
    # Extract ad IDs for action space
    ad_ids = [ad['ad_id'] for ad in active_ads]
    logger.info(f"Active ad IDs: {ad_ids}")

    # Define action space using create_budget_allocation_space
    logger.info("Setting action space...")
    aicon.define_action_space(
        space_type="budget_allocation",
        total_budget=1000.0,
        items=ad_ids,
        budget_step=100.0,
        min_budget=0.0
    )
    logger.info("Action space defined")

    # Define utility function (marketing ROI)
    logger.info("Setting utility function...")
    aicon.define_utility_function(
        utility_type="marketing_roi",
        revenue_per_sale=50.0,  # $50 revenue per conversion
        num_days=1,  # Since we're using hourly data
        ad_names=ad_ids
    )
    logger.info("Utility function defined")

    # Update from Meta Ads sensor
    logger.info("Updating beliefs from Meta Ads sensor...")
    aicon.update_from_sensor("meta_ads")

    # Find best action
    logger.info("Finding best action...")
    best_action, expected_utility = aicon.find_best_action(num_samples=100)
    logger.info(f"Best action found: {best_action}")
    logger.info(f"Expected utility: {expected_utility}")

    # Print token usage report
    logger.info("\nToken Usage Report:")
    print(json.dumps(aicon.get_token_usage_report(), indent=2))

if __name__ == "__main__":
    main() 