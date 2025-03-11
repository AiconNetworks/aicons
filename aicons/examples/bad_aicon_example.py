#!/usr/bin/env python3
# bad_aicon_example.py
# Example demonstrating how to use the Budget Allocation Decision (BAD) AIcon

import sys
from pathlib import Path
import json
from typing import Dict, Any
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from definitions.aicon_types import BAD_AICON, Campaign

def main():
    """
    Example of using the Budget Allocation Decision (BAD) AIcon
    for ad campaign budget management with BayesBrainGPT integration
    """
    print("Initializing Budget Allocation Decision (BAD) AIcon...")
    
    # Initialize the BAD AIcon
    BAD_AICON.initialize()
    
    # Create some example campaigns
    campaigns = [
        Campaign(
            id="campaign_001",
            name="Summer Sale Facebook Ads",
            platform="facebook",
            total_budget=2000.0,
            daily_budget=100.0,
            performance_metrics={
                "roi": 1.5,
                "impressions": 10000,
                "clicks": 200,
                "conversions": 10
            }
        ),
        Campaign(
            id="campaign_002",
            name="Product Launch Google Ads",
            platform="google",
            total_budget=3000.0,
            daily_budget=150.0,
            performance_metrics={
                "roi": 1.2,
                "impressions": 15000,
                "clicks": 300,
                "conversions": 12
            }
        ),
        Campaign(
            id="campaign_003",
            name="Retargeting Instagram Ads",
            platform="instagram",
            total_budget=1500.0,
            daily_budget=75.0,
            performance_metrics={
                "roi": 2.0,
                "impressions": 5000,
                "clicks": 150,
                "conversions": 15
            }
        )
    ]
    
    # Add campaigns to the BAD AIcon
    for campaign in campaigns:
        BAD_AICON.add_campaign(campaign)
        print(f"Added campaign: {campaign.name}")
    
    # Set budget increment for allocation (in dollars)
    BAD_AICON.set_budget_increment(50.0)
    
    print("\nSimulating campaign performance data...")
    
    # Simulate 3 days of performance data for all campaigns
    campaign_performance = {
        "observed_sales": [
            # Day 1: sales for each campaign
            [30, 15, 45],
            # Day 2: sales for each campaign
            [25, 20, 40],
            # Day 3: sales for each campaign
            [35, 10, 50]
        ],
        "observed_cpc": [0.75, 0.65, 0.85],  # CPC for each campaign
        "budgets": [
            # Day 1: budget spent for each campaign
            [100.0, 150.0, 75.0],
            # Day 2: budget spent for each campaign
            [100.0, 150.0, 75.0],
            # Day 3: budget spent for each campaign
            [100.0, 150.0, 75.0]
        ]
    }
    
    # Basic performance metrics for state update
    performance_data = {
        "roi": 1.8,
        "ctr": 0.025,
        "cpc": 0.8,
        "conversion_rate": 0.06,
        "campaign_metrics": {
            "impressions": 30000,
            "clicks": 650,
            "conversions": 37
        },
        "campaign_performance": campaign_performance
    }
    
    # Process data and get budget allocations
    print("\nProcessing performance data and allocating budget...")
    result = BAD_AICON.process(performance_data)
    
    # Print the results
    print("\nBudget Allocation Results:")
    print(json.dumps(result["budget_allocations"], indent=2))
    print(f"\nExpected ROI: ${result['expected_roi']:.2f}")
    
    # Demonstrate accessing the BayesBrainGPT state
    if BAD_AICON.state:
        print("\nCurrent BayesBrainGPT State:")
        for factor_name, factor in BAD_AICON.state.factors.items():
            print(f"- {factor_name}: {factor.value} ({factor.description})")
    
    print("\nMeta Campaign Status:")
    if BAD_AICON.meta_campaign:
        meta = BAD_AICON.meta_campaign
        print(f"- ID: {meta.id}")
        print(f"- Name: {meta.name}")
        print(f"- Total Budget: ${meta.total_budget:.2f}")
        print(f"- Daily Budget: ${meta.daily_budget:.2f}")
        print(f"- Status: {meta.status}")
        print("- Performance Metrics:")
        for metric, value in meta.performance_metrics.items():
            print(f"  - {metric}: {value}")
    
    # Run a second allocation after performance changes to demonstrate Bayesian updates
    print("\n" + "="*50)
    print("Simulating improved performance for campaign_003...")
    
    # Simulate improved performance for the third campaign
    campaign_performance_improved = {
        "observed_sales": [
            # Day 1: sales for each campaign (3rd campaign improved)
            [30, 15, 60],
            # Day 2: sales for each campaign (3rd campaign improved)
            [25, 20, 55],
            # Day 3: sales for each campaign (3rd campaign improved)
            [35, 10, 65]
        ],
        "observed_cpc": [0.75, 0.65, 0.80],  # CPC for each campaign (3rd slightly lower)
        "budgets": [
            # Day 1: budget spent for each campaign
            [100.0, 150.0, 75.0],
            # Day 2: budget spent for each campaign
            [100.0, 150.0, 75.0],
            # Day 3: budget spent for each campaign
            [100.0, 150.0, 75.0]
        ]
    }
    
    # Updated performance metrics
    performance_data_improved = {
        "roi": 2.1,  # Improved ROI
        "ctr": 0.028,
        "cpc": 0.78,
        "conversion_rate": 0.07,
        "campaign_metrics": {
            "impressions": 32000,
            "clicks": 700,
            "conversions": 45
        },
        "campaign_performance": campaign_performance_improved
    }
    
    # Process data again with improved performance
    print("\nProcessing updated performance data and reallocating budget...")
    result_improved = BAD_AICON.process(performance_data_improved)
    
    # Print the updated results
    print("\nUpdated Budget Allocation Results:")
    print(json.dumps(result_improved["budget_allocations"], indent=2))
    print(f"\nExpected ROI: ${result_improved['expected_roi']:.2f}")
    
    # Show budget shifts from first to second allocation
    print("\nBudget Shifts After Performance Update:")
    for campaign_id in result["budget_allocations"]:
        original = result["budget_allocations"][campaign_id]
        updated = result_improved["budget_allocations"][campaign_id]
        change = updated - original
        change_pct = (change / original) * 100 if original > 0 else 0
        
        direction = "➚" if change > 0 else "➘" if change < 0 else "="
        print(f"- {campaign_id}: ${original:.2f} → ${updated:.2f} {direction} (${change:.2f}, {change_pct:.1f}%)")

if __name__ == "__main__":
    main() 