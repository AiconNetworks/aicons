"""
Utility Function Module for BayesBrainGPT

This module provides classes and functions for defining utility functions
that evaluate actions in Bayesian decision making.

Utility functions can be customized to embed business rules and preferences,
allowing the AIcon to make decisions that align with specific goals.
"""

import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Optional, Callable


class UtilityFunction(ABC):
    """
    Abstract base class for utility functions used in Bayesian decision making.
    
    Utility functions evaluate the "goodness" of an action given a specific state.
    They are used to compute expected utility by averaging over posterior samples.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the utility function.
        
        Args:
            name: Name of the utility function
            description: Description of what this utility function measures
        """
        self.name = name
        self.description = description
        
    @abstractmethod
    def evaluate(self, action: Dict[str, Any], state_sample: Dict[str, Any]) -> float:
        """
        Evaluate the utility of an action for a specific state sample.
        
        Args:
            action: Dictionary mapping action dimension names to values
            state_sample: Dictionary mapping state factor names to values
                (represents a single sample from the posterior)
        
        Returns:
            float: Utility value (higher is better)
        """
        pass
    
    def batch_evaluate(self, action: Dict[str, Any], 
                      state_samples: Dict[str, List[Any]]) -> List[float]:
        """
        Evaluate the utility of an action for multiple state samples.
        
        Args:
            action: Dictionary mapping action dimension names to values
            state_samples: Dictionary mapping state factor names to lists of values
                (represents multiple samples from the posterior)
        
        Returns:
            List of utility values, one for each sample
        """
        # Get the number of samples
        if not state_samples:
            return []
        
        first_factor = next(iter(state_samples.values()))
        n_samples = len(first_factor)
        
        # Evaluate for each sample
        utilities = []
        for i in range(n_samples):
            # Extract the i-th sample
            sample = {k: v[i] for k, v in state_samples.items()}
            
            # Evaluate utility
            utility = self.evaluate(action, sample)
            utilities.append(utility)
            
        return utilities
    
    def expected_utility(self, action: Dict[str, Any], 
                        state_samples: Dict[str, List[Any]]) -> float:
        """
        Compute the expected utility of an action by averaging over posterior samples.
        
        Args:
            action: Dictionary mapping action dimension names to values
            state_samples: Dictionary mapping state factor names to lists of values
                (represents multiple samples from the posterior)
        
        Returns:
            float: Expected utility (higher is better)
        """
        utilities = self.batch_evaluate(action, state_samples)
        
        if not utilities:
            return 0.0
            
        return np.mean(utilities)


class TensorFlowUtilityFunction(ABC):
    """
    Abstract base class for TensorFlow-based utility functions.
    
    These utility functions work with TensorFlow tensors and can be optimized
    using gradient-based methods.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the TensorFlow utility function.
        
        Args:
            name: Name of the utility function
            description: Description of what this utility function measures
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def evaluate_tf(self, action: tf.Tensor, 
                   state_samples: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Evaluate the utility of an action for all state samples.
        
        Args:
            action: Tensor containing action values
            state_samples: Dictionary mapping state factor names to tensors
                (represents multiple samples from the posterior)
        
        Returns:
            Tensor of utility values, one for each sample
        """
        pass
    
    def expected_utility_tf(self, action: tf.Tensor,
                           state_samples: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Compute the expected utility of an action by averaging over posterior samples.
        
        Args:
            action: Tensor containing action values
            state_samples: Dictionary mapping state factor names to tensors
                (represents multiple samples from the posterior)
        
        Returns:
            Scalar tensor with the expected utility
        """
        utilities = self.evaluate_tf(action, state_samples)
        return tf.reduce_mean(utilities)
    
    def as_callable(self) -> Callable:
        """
        Return this utility function as a callable function.
        
        Returns:
            Function that takes (action, state_samples) and returns utilities tensor
        """
        return self.evaluate_tf


# ---------------------------------------------------------------
# Concrete Utility Function Implementations
# ---------------------------------------------------------------

class MarketingROIUtility(UtilityFunction, TensorFlowUtilityFunction):
    """
    Utility function for marketing ROI (Return on Investment).
    
    This utility function calculates the expected profit from ad spend
    by modeling sales as a function of budget, conversion rates, and
    day-specific factors.
    """
    
    def __init__(self, revenue_per_sale: float = 10.0, num_ads: int = 2,
                 num_days: int = 3, ad_names: List[str] = None,
                 weights: Dict[str, float] = None):
        """
        Initialize the marketing ROI utility function.
        
        Args:
            revenue_per_sale: Revenue generated per sale/conversion
            num_ads: Number of ads in the model
            num_days: Number of days to consider
            ad_names: Optional names for the ads
            weights: Optional weights for different components of the utility
                (e.g., {'revenue': 1.0, 'cost': 1.0, 'risk': 0.5})
        """
        super().__init__(name="Marketing ROI Utility", 
                        description="Calculates expected profit from ad spend")
        
        self.revenue_per_sale = revenue_per_sale
        self.num_ads = num_ads
        self.num_days = num_days
        self.ad_names = ad_names or [f"ad_{i+1}" for i in range(num_ads)]
        
        # Default weights (can be adjusted to reflect business priorities)
        self.weights = weights or {
            'revenue': 1.0,    # Weight for revenue component
            'cost': 1.0,       # Weight for cost component
            'risk': 0.0        # Weight for risk/variance penalty
        }
    
    def evaluate(self, action: Dict[str, Any], state_sample: Dict[str, Any]) -> float:
        """
        Evaluate the marketing ROI utility for a specific state sample.
        
        Args:
            action: Dictionary with budget allocations for each ad
            state_sample: Dictionary with conversion rates, CPCs, and day factors
                - Expected keys: 'phi' (conversion rates), 'c' (CPCs), 'delta' (day factors)
        
        Returns:
            float: Utility value representing expected profit
        """
        # Extract state parameters
        phi = state_sample.get('phi', np.zeros(self.num_ads) + 0.05)  # Conversion rates
        c = state_sample.get('c', np.zeros(self.num_ads) + 0.7)       # Cost per click
        delta = state_sample.get('delta', np.ones(self.num_days))     # Day factors
        
        # Initialize total revenue and cost
        total_revenue = 0.0
        total_cost = 0.0
        
        # For each ad
        for ad_idx in range(self.num_ads):
            ad_name = self.ad_names[ad_idx]
            
            # Get budget for this ad
            budget_key = f"{ad_name}_budget"
            ad_budget = action.get(budget_key, 0.0)
            
            # Daily budget (equal allocation across days)
            daily_budget = ad_budget / self.num_days
            
            # Calculate sales and cost for this ad
            ad_sales = 0.0
            for day_idx in range(self.num_days):
                # Sales = budget * conversion_rate * day_factor
                day_sales = daily_budget * phi[ad_idx] * delta[day_idx]
                ad_sales += day_sales
            
            # Revenue = sales * revenue_per_sale
            ad_revenue = ad_sales * self.revenue_per_sale
            
            # Cost = budget * cost_per_click
            ad_cost = ad_budget * c[ad_idx]
            
            # Add to totals
            total_revenue += ad_revenue
            total_cost += ad_cost
        
        # Calculate utility components
        weighted_revenue = total_revenue * self.weights['revenue']
        weighted_cost = total_cost * self.weights['cost']
        
        # Risk component (could be based on variance or other risk metrics)
        risk_penalty = 0.0  # In this simple version, no risk penalty
        
        # Total utility = revenue - cost - risk_penalty
        utility = weighted_revenue - weighted_cost - risk_penalty
        
        return float(utility)
    
    def evaluate_tf(self, action: tf.Tensor, 
                   state_samples: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        TensorFlow implementation of the marketing ROI utility.
        
        Args:
            action: Tensor with budget allocations for each ad
            state_samples: Dictionary with tensors for conversion rates, CPCs, and day factors
                - Expected keys: 'phi' (conversion rates), 'c' (CPCs), 'delta' (day factors)
                
        Returns:
            Tensor of utility values, one for each posterior sample
        """
        # Extract state parameters
        phi = state_samples.get('phi')      # Shape [n_samples, num_ads]
        c = state_samples.get('c')          # Shape [n_samples, num_ads]
        delta = state_samples.get('delta')  # Shape [n_samples, num_days]
        
        if phi is None or c is None or delta is None:
            # Use default values if not provided
            n_samples = next(iter(state_samples.values())).shape[0]
            if phi is None:
                phi = tf.ones([n_samples, self.num_ads]) * 0.05
            if c is None:
                c = tf.ones([n_samples, self.num_ads]) * 0.7
            if delta is None:
                delta = tf.ones([n_samples, self.num_days])
        
        # Initialize total revenue and cost
        total_revenue = 0.0
        total_cost = 0.0
        
        # For each ad
        for ad_idx in range(self.num_ads):
            # Get budget for this ad
            ad_budget = action[ad_idx]
            
            # Daily budget (equal allocation across days)
            daily_budget = ad_budget / self.num_days
            
            # Calculate sales for this ad
            ad_sales = 0.0
            for day_idx in range(self.num_days):
                # Sales = budget * conversion_rate * day_factor
                day_sales = daily_budget * phi[:, ad_idx] * delta[:, day_idx]
                ad_sales += day_sales
            
            # Revenue = sales * revenue_per_sale
            ad_revenue = ad_sales * self.revenue_per_sale
            
            # Cost = budget * cost_per_click
            ad_cost = ad_budget * c[:, ad_idx]
            
            # Add to totals
            total_revenue += ad_revenue
            total_cost += ad_cost
        
        # Calculate utility components
        weighted_revenue = total_revenue * self.weights['revenue']
        weighted_cost = total_cost * self.weights['cost']
        
        # Risk component (could use variance or other risk metrics)
        risk_penalty = 0.0
        if self.weights['risk'] > 0:
            # Example: Penalize based on standard deviation of profit
            profit = weighted_revenue - weighted_cost
            # Add small epsilon to avoid NaN gradients
            risk_penalty = self.weights['risk'] * tf.math.reduce_std(profit + 1e-8)
        
        # Total utility = revenue - cost - risk_penalty
        utility = weighted_revenue - weighted_cost - risk_penalty
        
        return utility


class ConstrainedMarketingROI(MarketingROIUtility):
    """
    Marketing ROI utility function with additional business constraints.
    
    This extends the basic ROI utility with business-specific constraints
    like minimum budget per ad, maximum spend per day, brand exposure goals, etc.
    """
    
    def __init__(self, revenue_per_sale: float = 10.0, num_ads: int = 2,
                num_days: int = 3, ad_names: List[str] = None,
                weights: Dict[str, float] = None,
                constraints: Dict[str, Any] = None):
        """
        Initialize the constrained marketing ROI utility.
        
        Args:
            revenue_per_sale: Revenue generated per sale/conversion
            num_ads: Number of ads in the model
            num_days: Number of days to consider
            ad_names: Optional names for the ads
            weights: Optional weights for different components of the utility
            constraints: Dictionary of business constraints to apply, such as:
                - min_budget_per_ad: Minimum budget for each ad
                - max_daily_spend: Maximum spend per day
                - min_brand_exposure: Minimum budget for brand-focused ads
                - balanced_spend: Boolean requiring somewhat balanced spending
        """
        super().__init__(revenue_per_sale, num_ads, num_days, ad_names, weights)
        self.description = "Calculates expected profit with business constraints"
        
        # Default constraints
        self.constraints = constraints or {
            'min_budget_per_ad': 0.0,     # Minimum budget per ad
            'max_daily_spend': float('inf'),  # Maximum spend per day
            'min_brand_exposure': 0.0,    # Minimum for brand-focused ads
            'balanced_spend': False,      # Whether to require balanced spending
            'brand_focused_ads': []       # Indices of brand-focused ads
        }
    
    def evaluate(self, action: Dict[str, Any], state_sample: Dict[str, Any]) -> float:
        """
        Evaluate utility with business constraints.
        
        Args:
            action: Dictionary with budget allocations for each ad
            state_sample: Dictionary with conversion rates, CPCs, and day factors
        
        Returns:
            float: Utility value with constraint penalties
        """
        # Get base utility (profit)
        base_utility = super().evaluate(action, state_sample)
        
        # Apply constraint penalties
        penalty = self._calculate_constraint_penalties(action)
        
        return base_utility - penalty
    
    def evaluate_tf(self, action: tf.Tensor, 
                    state_samples: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        TensorFlow implementation with business constraints.
        
        Args:
            action: Tensor with budget allocations for each ad
            state_samples: Dictionary with tensors for model parameters
                
        Returns:
            Tensor of utility values with constraint penalties
        """
        # Get base utility (profit)
        base_utility = super().evaluate_tf(action, state_samples)
        
        # Apply constraint penalties
        penalty = self._calculate_constraint_penalties_tf(action)
        
        return base_utility - penalty
    
    def _calculate_constraint_penalties(self, action: Dict[str, Any]) -> float:
        """Calculate penalties for constraint violations."""
        penalty = 0.0
        
        # Extract budget values
        budgets = []
        for ad_idx in range(self.num_ads):
            ad_name = self.ad_names[ad_idx]
            budget_key = f"{ad_name}_budget"
            budget = action.get(budget_key, 0.0)
            budgets.append(budget)
        
        # Minimum budget per ad constraint
        min_budget = self.constraints['min_budget_per_ad']
        for budget in budgets:
            if budget < min_budget:
                penalty += 100.0 * (min_budget - budget)**2
        
        # Maximum daily spend constraint
        total_budget = sum(budgets)
        daily_spend = total_budget / self.num_days
        max_daily = self.constraints['max_daily_spend']
        if daily_spend > max_daily:
            penalty += 100.0 * (daily_spend - max_daily)**2
        
        # Brand exposure constraint
        brand_idx = self.constraints.get('brand_focused_ads', [])
        if brand_idx:
            brand_budget = sum(budgets[i] for i in brand_idx)
            min_brand = self.constraints['min_brand_exposure']
            if brand_budget < min_brand:
                penalty += 100.0 * (min_brand - brand_budget)**2
        
        # Balanced spend constraint
        if self.constraints.get('balanced_spend', False) and len(budgets) > 1:
            avg_budget = sum(budgets) / len(budgets)
            variance = sum((b - avg_budget)**2 for b in budgets) / len(budgets)
            # Penalize high variance in budget allocation
            penalty += 10.0 * variance
        
        return penalty
    
    def _calculate_constraint_penalties_tf(self, action: tf.Tensor) -> tf.Tensor:
        """Calculate penalties for constraint violations using TensorFlow."""
        penalty = tf.constant(0.0, dtype=action.dtype)
        
        # Minimum budget per ad constraint
        min_budget = self.constraints['min_budget_per_ad']
        for ad_idx in range(self.num_ads):
            budget = action[ad_idx]
            penalty += 100.0 * tf.maximum(0.0, min_budget - budget)**2
        
        # Maximum daily spend constraint
        total_budget = tf.reduce_sum(action)
        daily_spend = total_budget / self.num_days
        max_daily = self.constraints['max_daily_spend']
        penalty += 100.0 * tf.maximum(0.0, daily_spend - max_daily)**2
        
        # Brand exposure constraint
        brand_idx = self.constraints.get('brand_focused_ads', [])
        if brand_idx:
            brand_budgets = tf.gather(action, brand_idx)
            brand_budget = tf.reduce_sum(brand_budgets)
            min_brand = self.constraints['min_brand_exposure']
            penalty += 100.0 * tf.maximum(0.0, min_brand - brand_budget)**2
        
        # Balanced spend constraint
        if self.constraints.get('balanced_spend', False) and self.num_ads > 1:
            avg_budget = tf.reduce_mean(action)
            variance = tf.reduce_mean((action - avg_budget)**2)
            # Penalize high variance in budget allocation
            penalty += 10.0 * variance
        
        return penalty


class WeatherDependentMarketingROI(MarketingROIUtility):
    """
    Marketing ROI utility that adjusts based on weather conditions.
    
    This utility function models how different weather conditions might
    affect the performance of different ad types.
    """
    
    def __init__(self, revenue_per_sale: float = 10.0, num_ads: int = 2,
                num_days: int = 3, ad_names: List[str] = None,
                weights: Dict[str, float] = None,
                weather_effects: Dict[str, Dict[str, float]] = None):
        """
        Initialize weather-dependent marketing ROI utility.
        
        Args:
            revenue_per_sale: Revenue generated per sale/conversion
            num_ads: Number of ads in the model
            num_days: Number of days to consider
            ad_names: Optional names for the ads
            weights: Optional weights for different components of the utility
            weather_effects: Dictionary mapping weather conditions to effect multipliers
                Example: {
                    'rainy': {'ad_1': 0.8, 'ad_2': 1.2},  # Rain hurts ad_1, helps ad_2
                    'sunny': {'ad_1': 1.2, 'ad_2': 0.9}   # Sun helps ad_1, slightly hurts ad_2
                }
        """
        super().__init__(revenue_per_sale, num_ads, num_days, ad_names, weights)
        self.description = "Calculates ROI with weather condition adjustments"
        
        # Default weather effects (no effect)
        self.weather_effects = weather_effects or {
            'rainy': {ad: 1.0 for ad in self.ad_names},
            'sunny': {ad: 1.0 for ad in self.ad_names},
            'cloudy': {ad: 1.0 for ad in self.ad_names},
            'snowy': {ad: 1.0 for ad in self.ad_names}
        }
    
    def evaluate(self, action: Dict[str, Any], state_sample: Dict[str, Any]) -> float:
        """
        Evaluate utility with weather condition adjustments.
        
        Args:
            action: Dictionary with budget allocations for each ad
            state_sample: Dictionary with model parameters and weather conditions
                - Should include 'weather' key with current weather condition
        
        Returns:
            float: Adjusted utility value
        """
        # Get base utility calculation
        base_utility = super().evaluate(action, state_sample)
        
        # Apply weather condition adjustments
        weather = state_sample.get('weather', 'sunny')  # Default to sunny if not specified
        
        # If we don't have effects for this weather, use default (no adjustment)
        if weather not in self.weather_effects:
            return base_utility
        
        # Calculate weighted adjustment based on ad budgets
        total_budget = 0.0
        weighted_effect = 0.0
        
        for ad_idx, ad_name in enumerate(self.ad_names):
            budget_key = f"{ad_name}_budget"
            budget = action.get(budget_key, 0.0)
            total_budget += budget
            
            # Get effect multiplier for this ad under current weather
            effect = self.weather_effects[weather].get(ad_name, 1.0)
            weighted_effect += budget * effect
        
        # Avoid division by zero
        if total_budget == 0:
            return base_utility
            
        # Calculate overall effect as weighted average
        overall_effect = weighted_effect / total_budget
        
        # Apply effect to utility
        adjusted_utility = base_utility * overall_effect
        
        return adjusted_utility


# Utility function factory

def create_utility_function(utility_type: str, **kwargs) -> Union[UtilityFunction, TensorFlowUtilityFunction]:
    """
    Create a utility function of the specified type.
    
    Args:
        utility_type: Type of utility function to create
        **kwargs: Parameters for the specific utility function
        
    Returns:
        Utility function instance
    """
    if utility_type == 'marketing_roi':
        return MarketingROIUtility(**kwargs)
    elif utility_type == 'constrained_marketing_roi':
        return ConstrainedMarketingROI(**kwargs)
    elif utility_type == 'weather_dependent_marketing_roi':
        return WeatherDependentMarketingROI(**kwargs)
    else:
        raise ValueError(f"Unknown utility function type: {utility_type}")


# Example of creating a custom utility function from components

def create_custom_marketing_utility(
    revenue_weight: float = 1.0,
    cost_weight: float = 1.0,
    risk_weight: float = 0.0,
    min_budget_per_ad: float = 0.0,
    max_daily_spend: float = float('inf'),
    brand_focused_ads: List[int] = None,
    min_brand_exposure: float = 0.0,
    balanced_spend: bool = False,
    weather_effects: Dict[str, Dict[str, float]] = None,
    **kwargs
) -> UtilityFunction:
    """
    Create a custom marketing utility function with specified components.
    
    Args:
        revenue_weight: Weight for revenue component
        cost_weight: Weight for cost component
        risk_weight: Weight for risk/variance component
        min_budget_per_ad: Minimum budget per ad
        max_daily_spend: Maximum spend per day
        brand_focused_ads: Indices of brand-focused ads
        min_brand_exposure: Minimum budget for brand ads
        balanced_spend: Whether to require balanced spending
        weather_effects: Weather condition adjustments
        **kwargs: Additional parameters for the utility function
        
    Returns:
        Custom utility function
    """
    # Set up weights
    weights = {
        'revenue': revenue_weight,
        'cost': cost_weight,
        'risk': risk_weight
    }
    
    # Set up constraints
    constraints = {
        'min_budget_per_ad': min_budget_per_ad,
        'max_daily_spend': max_daily_spend,
        'min_brand_exposure': min_brand_exposure,
        'balanced_spend': balanced_spend,
        'brand_focused_ads': brand_focused_ads or []
    }
    
    # Choose the appropriate utility class based on parameters
    if weather_effects:
        return WeatherDependentMarketingROI(
            weights=weights,
            weather_effects=weather_effects,
            **kwargs
        )
    elif any(v != 0 for v in constraints.values()) or constraints['brand_focused_ads']:
        return ConstrainedMarketingROI(
            weights=weights,
            constraints=constraints,
            **kwargs
        )
    else:
        return MarketingROIUtility(
            weights=weights,
            **kwargs
        ) 