# Test safety checks in SimpleBadAIcon
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


def test_action_space_required():
    """Test that an action space is required for key methods."""
    print("\n=== Testing Action Space Requirements ===")
    
    # Create AIcon with no action space
    aicon = SimpleBadAIcon("test_safety_checks")
    
    # Add some factors
    aicon.add_factor_continuous("test_factor", 0.5, 0.1)
    
    # Add utility function but NO action space
    aicon.create_utility_function(
        utility_type='marketing_roi',
        revenue_per_sale=50.0,
        num_ads=2,
        num_days=1
    )
    
    # Test sample_action
    try:
        aicon.sample_action()
        print("✗ FAIL: sample_action() did not raise an error without action space")
    except ValueError as e:
        print(f"✓ PASS: sample_action() correctly raised an error: {e}")
    
    # Test find_best_action
    try:
        aicon.find_best_action()
        print("✗ FAIL: find_best_action() did not raise an error without action space")
    except ValueError as e:
        print(f"✓ PASS: find_best_action() correctly raised an error: {e}")
    
    # Test run
    try:
        aicon.run()
        print("✗ FAIL: run() did not raise an error without action space")
    except ValueError as e:
        print(f"✓ PASS: run() correctly raised an error: {e}")
    
    # Test perceive_and_decide
    try:
        aicon.perceive_and_decide({})
        print("✗ FAIL: perceive_and_decide() did not raise an error without action space")
    except ValueError as e:
        print(f"✓ PASS: perceive_and_decide() correctly raised an error: {e}")


def test_utility_function_required():
    """Test that a utility function is required for key methods."""
    print("\n=== Testing Utility Function Requirements ===")
    
    # Create AIcon with no utility function
    aicon = SimpleBadAIcon("test_safety_checks")
    
    # Add some factors
    aicon.add_factor_continuous("test_factor", 0.5, 0.1)
    
    # Add action space but NO utility function
    aicon.create_action_space(
        space_type='marketing',
        total_budget=1000.0,
        num_ads=2,
        budget_step=100.0
    )
    
    # Test find_best_action
    try:
        aicon.find_best_action()
        print("✗ FAIL: find_best_action() did not raise an error without utility function")
    except ValueError as e:
        print(f"✓ PASS: find_best_action() correctly raised an error: {e}")
    
    # Test run
    try:
        aicon.run()
        print("✗ FAIL: run() did not raise an error without utility function")
    except ValueError as e:
        print(f"✓ PASS: run() correctly raised an error: {e}")
    
    # Test perceive_and_decide
    try:
        aicon.perceive_and_decide({})
        print("✗ FAIL: perceive_and_decide() did not raise an error without utility function")
    except ValueError as e:
        print(f"✓ PASS: perceive_and_decide() correctly raised an error: {e}")


def test_both_components_work():
    """Test that providing both components allows operation."""
    print("\n=== Testing With Both Components ===")
    
    # Create AIcon with both action space and utility function
    aicon = SimpleBadAIcon("test_safety_checks")
    
    # Add some factors
    aicon.add_factor_continuous("ad1_conversion_rate", 0.02, 0.005)
    aicon.add_factor_continuous("ad2_conversion_rate", 0.03, 0.008)
    aicon.add_factor_continuous("ad1_cost_per_click", 0.5, 0.1)
    aicon.add_factor_continuous("ad2_cost_per_click", 0.7, 0.15)
    
    # Add both action space and utility function
    aicon.create_action_space(
        space_type='marketing',
        total_budget=1000.0,
        num_ads=2,
        budget_step=100.0
    )
    
    aicon.create_utility_function(
        utility_type='marketing_roi',
        revenue_per_sale=50.0,
        num_ads=2,
        num_days=1
    )
    
    # Test sample_action
    try:
        action = aicon.sample_action()
        print(f"✓ PASS: sample_action() works with both components: {action}")
    except ValueError as e:
        print(f"✗ FAIL: sample_action() raised an error despite having both components: {e}")
    
    # We can't fully test find_best_action without posterior samples, but we can check if it raises
    # the correct error (about posterior samples, not action space or utility function)
    try:
        aicon.find_best_action()
        print("✗ UNEXPECTED: find_best_action() should raise an error about posterior samples")
    except ValueError as e:
        if "posterior samples" in str(e).lower():
            print(f"✓ PASS: find_best_action() correctly needs posterior samples: {e}")
        else:
            print(f"✗ FAIL: find_best_action() raised wrong error: {e}")


if __name__ == "__main__":
    print("=== TESTING AICON SAFETY CHECKS ===")
    print("This script verifies that SimpleBadAIcon raises appropriate errors")
    print("when run without an action space or utility function.")
    
    # Run the tests
    test_action_space_required()
    test_utility_function_required()
    test_both_components_work()
    
    print("\n=== ALL TESTS COMPLETED ===") 