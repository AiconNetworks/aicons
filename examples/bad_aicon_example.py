"""
Example script demonstrating the Budget Allocation Decision (BAD) AIcon
with integrated action space for ad campaign management.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root))

from aicons.definitions.aicon_types import BadAIcon, Campaign

def main():
    """Run the BAD AIcon example"""
    print("=== Budget Allocation Decision (BAD) AIcon Example ===")
    
    # Create a BAD AIcon instance
    bad_aicon = BadAIcon(
        name="AdBudgetOptimizer",
        capabilities=["budget_allocation", "campaign_management", "bayesian_optimization"]
    )
    
    # Initialize the AIcon
    print("\n1. Initializing BAD AIcon...")
    bad_aicon.initialize()
    
    # Add some example campaigns
    print("\n2. Adding campaigns...")
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
        ),
        Campaign(
            id="campaign_003",
            name="Instagram Promotion",
            platform="instagram",
            total_budget=3000.0,
            daily_budget=200.0,
            performance_metrics={
                "impressions": 5000,
                "clicks": 150,
                "conversions": 8,
                "cpc": 0.75
            }
        )
    ]
    
    for campaign in campaigns:
        bad_aicon.add_campaign(campaign)
    
    # Process input data and get budget allocations
    print("\n3. Processing campaign performance data...")
    
    # Simulate campaign performance data
    input_data = {
        "campaign_performance": {
            "observed_sales": [
                [15, 20, 10]  # Sales for each campaign on day 1
            ],
            "observed_cpc": [0.65, 0.80, 0.75],  # CPC for each campaign
            "budgets": [
                [300.0, 400.0, 200.0]  # Current budgets for each campaign
            ]
        }
    }
    
    # Process the data and get budget allocations
    result = bad_aicon.process(input_data)
    
    # Display results
    print("\n4. Budget allocation results:")
    print(f"Meta campaign status: {result['meta_campaign_status']}")
    print(f"Expected ROI: {result['expected_roi']:.2f}")
    print("\nBudget allocations:")
    
    for campaign_id, budget in result["budget_allocations"].items():
        campaign_name = bad_aicon.campaigns[campaign_id].name
        print(f"  {campaign_name}: ${budget:.2f}")
    
    # Demonstrate action space sampling
    print("\n5. Sampling from action space:")
    print("Random valid budget allocations:")
    
    for i in range(3):
        allocation = bad_aicon.action_space.sample()
        print(f"\nSample {i+1}:")
        for key, value in allocation.items():
            print(f"  {key}: ${value:.2f}")
        print(f"  Total: ${sum(allocation.values()):.2f}")
    
    print("\nExample complete!")

if __name__ == "__main__":
    main() 