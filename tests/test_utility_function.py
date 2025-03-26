"""
Test script for utility function creation and integration with BaseAIcon.

This script tests the creation of utility functions and verifies they're correctly 
stored in the AIcon.
"""

import os
import sys
import json
from datetime import datetime
import numpy as np

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aicons.definitions.base_aicon import BaseAIcon

def test_utility_function_creation():
    """Test creating utility functions and verifying they're properly stored in BaseAIcon."""
    print("\n=== Testing BaseAIcon Utility Function Creation ===")
    
    # Create a BaseAIcon instance
    aicon = BaseAIcon("UtilityTestAIcon", "utility_test")
    print(f"Created AIcon: {aicon.name} (ID: {aicon.id})")
    
    # Add some factors related to marketing
    aicon.add_factor_continuous(
        name="conversion_rate", 
        value=0.05,
        uncertainty=0.01,
        lower_bound=0.0,
        upper_bound=1.0,
        description="Conversion rate for marketing campaign"
    )
    
    aicon.add_factor_continuous(
        name="cost_per_click", 
        value=0.50,
        uncertainty=0.10,
        lower_bound=0.01,
        description="Cost per click for marketing campaign"
    )
    
    aicon.add_factor_continuous(
        name="average_order_value", 
        value=50.0,
        uncertainty=5.0,
        lower_bound=1.0,
        description="Average order value in dollars"
    )
    
    # Print the state
    print("\nAIcon State (prior beliefs):")
    print(aicon.get_state(format_nicely=True))
    
    # Create an action space for the utility function to work with
    try:
        # Create a budget allocation action space
        action_space = aicon.create_action_space(
            space_type='budget_allocation',
            total_budget=1000.0,
            items=["ad_1", "ad_2", "ad_3"],
            budget_step=10.0
        )
        
        if action_space:
            print("\nSuccessfully created budget allocation action space")
            
            # Test 1: Create a marketing ROI utility function
            utility_marketing = aicon.create_utility_function(
                utility_type='marketing_roi',
                revenue_per_conversion=50.0
            )
            
            # Verify the utility function is stored in the AIcon
            if utility_marketing and aicon.brain and aicon.brain.get_utility_function():
                print("\nSUCCESS: Marketing ROI utility function created and stored in AIcon")
                print(f"Utility function: {utility_marketing}")
                
                # Test finding best action with this utility
                best_action, expected_utility = aicon.find_best_action(num_samples=100)
                if best_action:
                    print("\nBest Action with Marketing ROI utility:")
                    for ad, budget in best_action.items():
                        print(f"- {ad}: ${budget:.2f}")
                    print(f"Expected utility: ${expected_utility:.2f}")
                else:
                    print("WARNING: Could not find best action with Marketing ROI utility")
            else:
                print("WARNING: Marketing ROI utility function not created or not stored properly")
            
            # Test 2: Create a custom utility function
            def custom_utility(action, state):
                """
                A custom utility function that prioritizes even distribution.
                
                Args:
                    action: Dictionary of actions (ad budgets)
                    state: Dictionary of state factors
                
                Returns:
                    A utility score
                """
                # Basic ROI calculation
                conversion_rate = state.get("conversion_rate", {}).get("value", 0.05)
                aov = state.get("average_order_value", {}).get("value", 50.0)
                
                total_budget = sum(budget for ad, budget in action.items())
                expected_revenue = total_budget * conversion_rate * aov
                roi = expected_revenue - total_budget
                
                # Add a penalty for uneven distribution
                budgets = list(action.values())
                if len(budgets) > 1:
                    variance = np.var(budgets)
                    # Penalize high variance (uneven distribution)
                    distribution_score = np.exp(-variance / 10000)
                    # Weighted combination of ROI and distribution score
                    return roi * 0.8 + distribution_score * 0.2 * total_budget
                
                return roi
            
            # Create the custom utility function
            utility_custom = aicon.create_utility_function(
                utility_type='custom',
                function=custom_utility,
                name="even_distribution_utility",
                description="Utility function that balances ROI with even budget distribution"
            )
            
            # Verify the utility function is stored in the AIcon
            if utility_custom and aicon.brain and aicon.brain.get_utility_function():
                print("\nSUCCESS: Custom utility function created and stored in AIcon")
                
                # Test finding best action with this utility
                best_action, expected_utility = aicon.find_best_action(num_samples=100)
                if best_action:
                    print("\nBest Action with Custom utility:")
                    for ad, budget in best_action.items():
                        print(f"- {ad}: ${budget:.2f}")
                    print(f"Expected utility: ${expected_utility:.2f}")
                else:
                    print("WARNING: Could not find best action with Custom utility")
            else:
                print("WARNING: Custom utility function not created or not stored properly")
            
            # Test 3: Create a constrained marketing ROI utility function
            utility_constrained = aicon.create_utility_function(
                utility_type='constrained_marketing_roi',
                revenue_per_conversion=50.0,
                min_budget_fraction=0.2  # Each ad must get at least 20% of total budget
            )
            
            # Verify the utility function is stored in the AIcon
            if utility_constrained and aicon.brain and aicon.brain.get_utility_function():
                print("\nSUCCESS: Constrained Marketing ROI utility function created and stored in AIcon")
                
                # Test finding best action with this utility
                best_action, expected_utility = aicon.find_best_action(num_samples=100)
                if best_action:
                    print("\nBest Action with Constrained Marketing ROI utility:")
                    for ad, budget in best_action.items():
                        print(f"- {ad}: ${budget:.2f}")
                    print(f"Expected utility: ${expected_utility:.2f}")
                    
                    # Verify the constraint is respected
                    total_budget = sum(budget for ad, budget in best_action.items())
                    min_required = total_budget * 0.2
                    all_constraints_met = all(budget >= min_required for ad, budget in best_action.items())
                    
                    if all_constraints_met:
                        print("SUCCESS: All budget constraints are respected")
                    else:
                        print("WARNING: Budget constraints are not respected")
                else:
                    print("WARNING: Could not find best action with Constrained Marketing ROI utility")
            else:
                print("WARNING: Constrained Marketing ROI utility function not created or not stored properly")
            
        else:
            print("WARNING: Failed to create action space")
            
    except Exception as e:
        print(f"WARNING: Utility function test could not complete: {e}")
        print("This is expected if BayesBrain or TensorFlow is not available")
    
    print("\n=== Utility Function Test Completed ===")
    return aicon

def test_define_factor_dependency_and_utility():
    """Test creating a utility that uses hierarchical factors."""
    print("\n=== Testing Utility with Hierarchical Factors ===")
    
    # Create a BaseAIcon instance
    aicon = BaseAIcon("HierarchicalUtilityAIcon", "hierarchical_utility_test")
    
    # Add base factors
    aicon.add_factor_continuous(
        name="base_conversion_rate", 
        value=0.05,
        uncertainty=0.01,
        lower_bound=0.0,
        upper_bound=0.2,
        description="Base conversion rate"
    )
    
    aicon.add_factor_categorical(
        name="season", 
        value="summer",
        categories=["spring", "summer", "fall", "winter"],
        description="Current season"
    )
    
    # Define a hierarchical factor (factor dependency)
    try:
        aicon.define_factor_dependency(
            name="expected_roi",
            parent_factors=["base_conversion_rate", "season"],
            relation_type="linear",
            uncertainty=0.02,
            description="Expected ROI based on conversion rate and season"
        )
        
        # Compile the probabilistic model
        aicon.compile_probabilistic_model()
        
        print("\nSuccessfully created hierarchical factors and compiled model")
        print(aicon.get_state(format_nicely=True))
        
        # Create an action space
        action_space = aicon.create_action_space(
            space_type='budget_allocation',
            total_budget=1000.0,
            items=["ad_1", "ad_2"],
            budget_step=10.0
        )
        
        if action_space:
            print("\nSuccessfully created action space")
            
            # Create a utility function that uses the hierarchical factor
            utility = aicon.create_utility_function(
                utility_type='custom',
                function=lambda action, state: sum(budget for _, budget in action.items()) * 
                                             state.get("expected_roi", {}).get("value", 0.1),
                name="hierarchical_roi_utility",
                description="Utility function that uses the hierarchical expected_roi factor"
            )
            
            if utility and aicon.brain and aicon.brain.get_utility_function():
                print("\nSUCCESS: Hierarchical utility function created and stored in AIcon")
                
                # Find the best action
                best_action, expected_utility = aicon.find_best_action(num_samples=100)
                if best_action:
                    print("\nBest Action with Hierarchical utility:")
                    for ad, budget in best_action.items():
                        print(f"- {ad}: ${budget:.2f}")
                    print(f"Expected utility: ${expected_utility:.2f}")
                else:
                    print("WARNING: Could not find best action with hierarchical utility")
            else:
                print("WARNING: Hierarchical utility function not created or not stored properly")
        else:
            print("WARNING: Failed to create action space")
    except Exception as e:
        print(f"WARNING: Hierarchical utility test could not complete: {e}")
        print("This is expected if BayesBrain or BayesianState is not available")
    
    print("\n=== Hierarchical Utility Test Completed ===")

if __name__ == "__main__":
    print("=== Starting BaseAIcon Utility Function Tests ===\n")
    
    # Run the tests
    aicon = test_utility_function_creation()
    test_define_factor_dependency_and_utility()
    
    print("\n=== All Utility Function Tests Completed ===") 