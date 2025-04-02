"""
Test cases for utility function implementations.
"""

import numpy as np
import tensorflow as tf
from .utility_functions import (
    CostBenefitUtility,
    MonetaryUtility,
    MultiAttributeUtility,
    RiskAdjustedUtility
)

def test_cost_benefit_utility():
    """Test the CostBenefitUtility implementation."""
    # Create a simple cost/benefit utility
    costs = {"action1": 10.0, "action2": 20.0}
    benefits = {"action1": 15.0, "action2": 25.0}
    
    utility = CostBenefitUtility(
        name="test_cost_benefit",
        costs=costs,
        benefits=benefits
    )
    
    # Test single action evaluation
    action = {"action1": 1.0}
    state = {"dummy": 1.0}  # Dummy state since this utility doesn't use state
    
    result = utility.evaluate(action, state)
    print(f"Cost/Benefit Utility Test: {result}")
    assert result == 5.0  # 15.0 - 10.0

def test_monetary_utility():
    """Test the MonetaryUtility implementation."""
    # Create a monetary utility
    utility = MonetaryUtility(
        name="test_monetary",
        revenue_factors=["sales_revenue", "subscription_revenue"],
        cost_factors=["operating_cost", "marketing_cost"]
    )
    
    # Test with sample state
    action = {"budget": 1000.0}
    state = {
        "sales_revenue": 5000.0,
        "subscription_revenue": 2000.0,
        "operating_cost": 1000.0,
        "marketing_cost": 500.0
    }
    
    result = utility.evaluate(action, state)
    print(f"Monetary Utility Test: {result}")
    assert result == 5500.0  # (5000 + 2000) - (1000 + 500)

def test_multi_attribute_utility():
    """Test the MultiAttributeUtility implementation."""
    # Create a multi-attribute utility
    attributes = {
        "profit": {
            "weight": 0.6,
            "factors": ["revenue", "cost"],
            "higher_better": True
        },
        "customer_satisfaction": {
            "weight": 0.4,
            "factors": ["satisfaction_score"],
            "higher_better": True
        }
    }
    
    utility = MultiAttributeUtility(
        name="test_multi_attribute",
        attributes=attributes
    )
    
    # Test with sample state
    action = {"budget": 1000.0}
    state = {
        "revenue": 5000.0,
        "cost": 2000.0,
        "satisfaction_score": 0.8
    }
    
    result = utility.evaluate(action, state)
    print(f"Multi-Attribute Utility Test: {result}")
    # Expected: 0.6 * (5000 - 2000) + 0.4 * 0.8 = 1800 + 0.32 = 1800.32
    assert abs(result - 1800.32) < 1e-6

def test_risk_adjusted_utility():
    """Test the RiskAdjustedUtility implementation."""
    # Test logarithmic utility
    log_utility = RiskAdjustedUtility(
        name="test_log_utility",
        risk_type="log"
    )
    
    # Test exponential utility
    exp_utility = RiskAdjustedUtility(
        name="test_exp_utility",
        risk_type="exp",
        alpha=0.1
    )
    
    # Test with sample state
    action = {"budget": 1000.0}
    state = {
        "sales_revenue": 5000.0,
        "operating_cost": 2000.0
    }
    
    # Test logarithmic utility
    log_result = log_utility.evaluate(action, state)
    print(f"Logarithmic Risk-Adjusted Utility Test: {log_result}")
    assert log_result > 0  # Should be positive for positive monetary value
    
    # Test exponential utility
    exp_result = exp_utility.evaluate(action, state)
    print(f"Exponential Risk-Adjusted Utility Test: {exp_result}")
    assert 0 < exp_result < 1  # Should be between 0 and 1

if __name__ == "__main__":
    # Run all tests
    test_cost_benefit_utility()
    test_monetary_utility()
    test_multi_attribute_utility()
    test_risk_adjusted_utility()
    print("\nAll utility function tests passed!") 