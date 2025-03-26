#!/usr/bin/env python
"""
Persistence Example for AIcon

This example demonstrates how to use the persistence features to save and load AIcon state.
"""

import os
import sys
import time
import argparse
from datetime import datetime

# Add parent directory to path to make imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import AIcon
from aicons.definitions.simple_bad_aicon import SimpleBadAIcon

def create_example_aicon():
    """Create an example AIcon with some state"""
    print("Creating example AIcon...")
    aicon = SimpleBadAIcon("PersistenceExample")
    
    # Add some factors
    aicon.add_factor_continuous("conversion_rate", 0.05, uncertainty=0.01)
    aicon.add_factor_continuous("click_through_rate", 0.02, uncertainty=0.005)
    aicon.add_factor_categorical("user_segment", "high_value", 
                                categories=["high_value", "medium_value", "low_value"])
    
    # Create action space and utility function
    aicon.create_action_space("marketing", num_campaigns=2, budget_step=10, total_budget=100)
    aicon.create_utility_function("marketing_roi", target_metric="conversion_rate")
    
    # Add some run stats
    aicon.run_stats["iterations"] = 5
    aicon.run_stats["start_time"] = datetime.now().isoformat()
    aicon.run_stats["last_update_time"] = datetime.now().isoformat()
    aicon.run_stats["updates"].append({
        "time": datetime.now().isoformat(),
        "action": "created"
    })
    
    return aicon

def save_aicon(aicon, db_connection):
    """Save the AIcon to the database"""
    print(f"\nSaving AIcon '{aicon.name}' with ID: {aicon.id}")
    success = aicon.save_state(db_connection_string=db_connection)
    
    if success:
        print(f"AIcon saved successfully. ID: {aicon.id}")
        return aicon.id
    else:
        print("Failed to save AIcon")
        return None

def load_aicon(aicon_id, db_connection):
    """Load the AIcon from the database"""
    print(f"\nLoading AIcon with ID: {aicon_id}")
    aicon = SimpleBadAIcon.load_from_db(aicon_id, db_connection_string=db_connection)
    
    if aicon:
        print(f"AIcon '{aicon.name}' loaded successfully")
        return aicon
    else:
        print(f"Failed to load AIcon with ID: {aicon_id}")
        return None

def update_aicon(aicon):
    """Make some changes to the AIcon state"""
    print(f"\nUpdating AIcon '{aicon.name}'")
    
    # Update a factor
    aicon.add_factor_continuous("conversion_rate", 0.06, uncertainty=0.01)
    
    # Update run stats
    aicon.run_stats["iterations"] += 1
    aicon.run_stats["last_update_time"] = datetime.now().isoformat()
    aicon.run_stats["updates"].append({
        "time": datetime.now().isoformat(),
        "action": "updated"
    })
    
    print("AIcon updated")
    return aicon

def display_aicon_state(aicon):
    """Display key information about the AIcon"""
    print(f"\nAIcon State:")
    print(f"  Name: {aicon.name}")
    print(f"  ID: {aicon.id}")
    print(f"  Type: {aicon.type}")
    print(f"  Iterations: {aicon.run_stats['iterations']}")
    
    # Show factors
    print("\n  Factors:")
    for name, factor in aicon.brain.state_factors.items():
        if factor.get('type') == 'continuous':
            value = factor.get('value', 'unknown')
            print(f"    {name}: {value}")
        elif factor.get('type') == 'categorical':
            value = factor.get('value', 'unknown')
            categories = factor.get('categories', [])
            print(f"    {name}: {value} (categories: {categories})")
    
    # Show action space
    if aicon.brain.action_space:
        print("\n  Action Space:")
        dimensions = aicon.get_action_dimensions()
        print(f"    Type: {aicon.brain.action_space.__class__.__name__}")
        print(f"    Dimensions: {len(dimensions)}")
    
    # Show run stats
    print("\n  Run Stats:")
    print(f"    Iterations: {aicon.run_stats['iterations']}")
    print(f"    Updates: {len(aicon.run_stats['updates'])}")
    
    print("\n")

def main():
    parser = argparse.ArgumentParser(description='AIcon Persistence Example')
    parser.add_argument('--db', type=str, help='Database connection string')
    args = parser.parse_args()
    
    # Get database connection string
    db_connection = args.db or os.getenv("AICON_DB_URL")
    if not db_connection:
        print("No database connection provided. Using SQLite in-memory for demo.")
        db_connection = "sqlite:///:memory:"
    
    # 1. Create a new AIcon
    aicon = create_example_aicon()
    display_aicon_state(aicon)
    
    # 2. Save the AIcon
    aicon_id = save_aicon(aicon, db_connection)
    if not aicon_id:
        return
    
    # 3. Update the AIcon
    aicon = update_aicon(aicon)
    display_aicon_state(aicon)
    
    # 4. Save the updated AIcon
    save_aicon(aicon, db_connection)
    
    # 5. Load the AIcon from the database
    loaded_aicon = load_aicon(aicon_id, db_connection)
    if not loaded_aicon:
        return
    
    # 6. Display the loaded AIcon state
    display_aicon_state(loaded_aicon)
    
    print("Persistence example completed successfully")

if __name__ == "__main__":
    main() 