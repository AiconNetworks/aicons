#!/usr/bin/env python3
"""
Test script for the Meta Ads Sales Sensor.

This script demonstrates how to:
1. Create and configure a Meta Ads Sales Sensor
2. Register it with a BayesBrain instance
3. Update brain state with sensor data 
4. Make decisions based on the updated state
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to allow imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir.parent))

# Import our classes
from aicons.definitions.simple_bad_aicon import SimpleBadAIcon
from aicons.bayesbrainGPT.sensors.meta_s.meta_ads_sales_sensor import MetaAdsSalesSensor

def main():
    print("Testing Meta Ads Sales Sensor")
    print("============================\n")
    
    # Create an AIcon with state factors relevant to Meta ads
    aicon = SimpleBadAIcon(name="Meta Advertising Manager")
    
    # Add factor for conversion rate
    print("Adding continuous factor for conversion rate...")
    aicon.add_factor_continuous(
        name="meta_conversion_rate", 
        value=0.02, 
        uncertainty=0.005,
        lower_bound=0.0,
        upper_bound=1.0,
        description="Conversion rate for Meta ads"
    )
    
    # Add factor for ROAS (Return on Ad Spend)
    print("Adding continuous factor for ROAS...")
    aicon.add_factor_continuous(
        name="meta_roas", 
        value=1.5, 
        uncertainty=0.3,
        lower_bound=0.0,
        description="Return on Ad Spend for Meta ads"
    )
    
    # Add factor for best performing campaign
    print("Adding categorical factor for best campaign...")
    aicon.add_factor_categorical(
        name="best_campaign",
        value="New Customer Acquisition",
        categories=[
            "Summer Sale 2023", 
            "New Customer Acquisition", 
            "Retargeting Shoppers", 
            "Lookalike Audience", 
            "Holiday Special"
        ],
        probs=[0.2, 0.3, 0.2, 0.2, 0.1],
        description="Best performing Meta ad campaign"
    )
    
    # Add factor for best performing creative
    print("Adding categorical factor for best creative...")
    aicon.add_factor_categorical(
        name="best_creative",
        value="Creative 1 for Retargeting Shoppers",
        categories=[
            "Creative 1 for Retargeting Shoppers",
            "Creative 2 for Retargeting Shoppers",
            "Creative 1 for New Customer Acquisition",
            "Creative 2 for New Customer Acquisition",
            "Creative 3 for New Customer Acquisition"
        ],
        probs=[0.3, 0.25, 0.2, 0.15, 0.1],
        description="Best performing Meta ad creative"
    )
    
    # Add factor for optimal daily budget
    print("Adding continuous factor for optimal budget...")
    aicon.add_factor_continuous(
        name="optimal_daily_budget", 
        value=500.0, 
        uncertainty=100.0,
        lower_bound=0.0,
        description="Optimal daily budget for Meta ads"
    )
    
    # Add factor for sales trend
    print("Adding categorical factor for sales trend...")
    aicon.add_factor_categorical(
        name="sales_trend",
        value="stable",
        categories=["up", "stable", "down"],
        probs=[0.25, 0.5, 0.25],
        description="Sales trend direction"
    )
    
    # Add factor for audience engagement
    print("Adding continuous factor for audience engagement...")
    aicon.add_factor_continuous(
        name="audience_engagement", 
        value=3.0, 
        uncertainty=1.0,
        lower_bound=0.0,
        upper_bound=10.0,
        description="Audience engagement score for Meta ads"
    )
    
    # Print the initial state
    print("\nInitial state:")
    print(aicon.get_state(format_nicely=True))
    
    # Create the Meta Ads Sales Sensor
    print("\nCreating Meta Ads Sales Sensor...")
    try:
        meta_sensor = MetaAdsSalesSensor(
            name="meta_ads_sales",
            reliability=0.85,
            factor_mapping={
                "conversion_rate": "meta_conversion_rate",
                "roas": "meta_roas",
                "best_campaign": "best_campaign",
                "best_creative": "best_creative",
                "optimal_daily_budget": "optimal_daily_budget",
                "sales_trend": "sales_trend",
                "audience_engagement": "audience_engagement"
            }
        )
        print("✅ Successfully created Meta Ads Sales Sensor")
    except Exception as e:
        print(f"❌ Error creating Meta Ads Sales Sensor: {e}")
        return
    
    # Add the sensor to the AIcon
    print("\nAdding sensor to AIcon...")
    try:
        aicon.add_sensor(
            name="meta_ads_sales", 
            sensor=meta_sensor
        )
        print("✅ Successfully added sensor to AIcon")
    except Exception as e:
        print(f"❌ Error adding sensor: {e}")
        return
    
    # Print sensor's observable factors and reliabilities
    print("\nSensor observable factors:")
    for factor in meta_sensor.observable_factors:
        reliability = meta_sensor.factor_reliabilities.get(factor, meta_sensor.default_reliability)
        print(f"  {factor}: reliability={reliability:.2f}")
    
    # Get sensor data directly to check mapping
    print("\nTesting sensor data retrieval:")
    try:
        sensor_data = meta_sensor.get_data()
        print("Sensor data (with mapping applied):")
        for state_factor, (value, reliability) in sensor_data.items():
            print(f"  {state_factor}: {value} (reliability: {reliability:.2f})")
        print("✅ Successfully retrieved sensor data with mapping")
    except Exception as e:
        print(f"❌ Error retrieving sensor data: {e}")
        return
    
    # Update beliefs with sensor data
    print("\nUpdating beliefs with sensor data...")
    try:
        update_result = aicon.update_from_sensor("meta_ads_sales")
        if update_result:
            print("✅ Successfully updated beliefs")
        else:
            print("❌ Failed to update beliefs")
            return
    except Exception as e:
        print(f"❌ Error updating beliefs: {e}")
        return
    
    # Print the updated state
    print("\nUpdated state:")
    print(aicon.get_state(format_nicely=True))
    
    # Demonstrate using the updated state for decision making
    print("\nUsing updated state for decision making:")
    state = aicon.get_state()
    roas = state["meta_roas"]["value"]
    budget = state["optimal_daily_budget"]["value"]
    trend = state["sales_trend"]["value"]
    best_campaign = state["best_campaign"]["value"]
    
    # Display a summary of key facts
    print("\nKey facts derived from Meta ads data:")
    print(f"• Current ROAS: {roas:.2f}x")
    print(f"• Best performing campaign: {best_campaign}")
    print(f"• Sales trend: {trend}")
    print(f"• Recommended daily budget: ${budget:.2f}")
    
    # Simple decision logic based on state
    if roas > 2.0 and trend == "up":
        recommendation = f"Increase daily budget to ${budget * 1.2:.2f}"
        recommendation_details = "Strong performance with upward trend justifies increased investment"
    elif roas > 1.5:
        recommendation = f"Maintain current budget at ${budget:.2f}"
        recommendation_details = "Good performance suggests maintaining current investment level"
    else:
        recommendation = f"Reduce daily budget to ${budget * 0.8:.2f}"
        recommendation_details = "Below-target performance suggests reducing investment until campaigns improve"
        
    print(f"\nRecommendation: {recommendation}")
    print(f"Reasoning: {recommendation_details}")
    
    print("\n✅ Test completed successfully")

if __name__ == "__main__":
    main() 