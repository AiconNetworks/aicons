"""
Marketing utility functions for BayesBrainGPT.

This module provides marketing-specific utility functions that calculate expected profit,
ROI, and other marketing performance metrics based on budget allocations.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Optional

from aicons.bayesbrainGPT.utility_function.utility_base import UtilityFunction


class MarketingROIUtility(UtilityFunction):
    """
    Utility function for marketing ROI (Return on Investment).
    
    This utility function calculates the expected profit from ad spend
    by modeling sales as a function of budget, conversion rates, and
    day-specific factors.
    """
    
    def __init__(self, revenue_per_sale: float = 10.0, num_ads: int = 2,
                 num_days: int = 3, ad_names: List[str] = None,
                 weights: Dict[str, float] = None, action_space=None):
        """
        Initialize the marketing ROI utility function.
        
        Args:
            revenue_per_sale: Revenue generated per sale/conversion
            num_ads: Number of ads in the model
            num_days: Number of days to consider
            ad_names: Optional names for the ads
            weights: Optional weights for different components of the utility
                (e.g., {'revenue': 1.0, 'cost': 1.0, 'risk': 0.5})
            action_space: Optional action space to connect with this utility
        """
        # Initialize parent class
        super().__init__(
            name="Marketing ROI Utility", 
            description="Calculates expected profit from ad spend",
            action_space=action_space
        )
        
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
                - Or individual factors like 'conversion_rate_ad1', 'cost_per_click_ad1', etc.
        
        Returns:
            float: Utility value representing expected profit
        """
        # Check if we have vector parameters or individual factors
        if 'phi' in state_sample:
            # Vector parameters
            phi = state_sample.get('phi', np.zeros(self.num_ads) + 0.05)  # Conversion rates
            c = state_sample.get('c', np.zeros(self.num_ads) + 0.7)       # Cost per click
            delta = state_sample.get('delta', np.ones(self.num_days))     # Day factors
        else:
            # Individual factors
            phi = np.zeros(self.num_ads)
            c = np.zeros(self.num_ads)
            delta = np.ones(self.num_days)
            
            # Extract conversion rates and costs from individual factors
            for i, ad_name in enumerate(self.ad_names):
                # Try different naming patterns
                phi[i] = state_sample.get(f"conversion_rate_{ad_name}", 
                                          state_sample.get(f"conv_rate_{ad_name}",
                                                         state_sample.get(f"cr_{ad_name}", 0.05)))
                
                c[i] = state_sample.get(f"cost_per_click_{ad_name}",
                                        state_sample.get(f"cpc_{ad_name}", 0.7))
        
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
                - If dimensions were set via set_action_space, action is a tensor of budget values
                - Otherwise, action should directly contain the budget values in order
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
            # Get budget for this ad from the action tensor
            # If we have dimensions set, the action may be organized differently
            if hasattr(self, 'dimensions') and self.dimensions is not None:
                # Find the dimension for this ad's budget
                for i, dim in enumerate(self.dimensions):
                    if dim.name == f"{self.ad_names[ad_idx]}_budget":
                        ad_budget = action[i]
                        break
                else:
                    # If we don't find a matching dimension, use the position in the tensor
                    ad_budget = action[ad_idx] if ad_idx < tf.shape(action)[0] else tf.constant(0.0)
            else:
                # Otherwise, assume the action tensor has budgets in order
                ad_budget = action[ad_idx] if ad_idx < tf.shape(action)[0] else tf.constant(0.0)
            
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
                constraints: Dict[str, Any] = None,
                action_space=None):
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
            action_space: Optional action space to connect with this utility
        """
        super().__init__(
            revenue_per_sale=revenue_per_sale,
            num_ads=num_ads,
            num_days=num_days,
            ad_names=ad_names,
            weights=weights,
            action_space=action_space
        )
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
                weather_effects: Dict[str, Dict[str, float]] = None,
                action_space=None):
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
            action_space: Optional action space to connect with this utility
        """
        super().__init__(
            revenue_per_sale=revenue_per_sale,
            num_ads=num_ads,
            num_days=num_days,
            ad_names=ad_names,
            weights=weights,
            action_space=action_space
        )
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