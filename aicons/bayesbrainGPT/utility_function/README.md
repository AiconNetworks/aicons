# BayesBrainGPT Utility Functions

This module provides a comprehensive set of utility function implementations for Bayesian decision-making. Utility functions evaluate the "goodness" of an action given a state and are essential components in decision-theoretic AI systems.

## Overview

Utility functions in BayesBrainGPT serve as the evaluation component that determines which actions are preferable over others. They transform state-action pairs into scalar values that can be maximized to find optimal decisions.

## Core Utility Function Types

### Base Classes

- **UtilityFunction**: Abstract base class for all utility functions
- **TensorFlowUtilityFunction**: Base class for utility functions that can be optimized with TensorFlow

### Marketing Utilities

- **MarketingROIUtility**: Calculates expected profit from marketing spend based on conversion rates and costs
- **ConstrainedMarketingROI**: Extends the basic ROI utility with business constraints like minimum budgets or maximum spend
- **WeatherDependentMarketingROI**: Adjusts marketing ROI based on weather conditions that affect ad performance

### Multi-Objective Utilities

- **WeightedSumUtility**: Combines multiple utility functions through a weighted sum
- **ParetoUtility**: Evaluates actions based on Pareto dominance principles
- **ConstrainedMultiObjectiveUtility**: Adds constraints to multi-objective optimization
- **AdaptiveWeightUtility**: Automatically adjusts weights to achieve target contribution ratios

### Custom Utilities

- **LambdaUtility**: Create a utility function from a custom lambda function
- **CustomCompositeUtility**: Build complex utility functions from components with custom combination logic
- **TensorFlowLambdaUtility**: Define TensorFlow-compatible utility functions for gradient-based optimization
- **RuleBasedUtility**: Evaluate actions based on business rules and heuristics
- **ParameterizedUtility**: Define utility functions with tunable parameters
- **ContextAwareUtility**: Switch between different utility functions based on context

## Usage Examples

### Basic Usage

```python
from aicons.bayesbrainGPT.utility_function import MarketingROIUtility

# Create a marketing ROI utility for 3 ads over 7 days
utility = MarketingROIUtility(
    revenue_per_sale=15.0,
    num_ads=3,
    num_days=7,
    ad_names=["search_ad", "display_ad", "social_ad"]
)

# Define an action (budget allocation)
action = {
    "search_ad_budget": 100.0,
    "display_ad_budget": 150.0,
    "social_ad_budget": 200.0
}

# Define a state sample with conversion rates and costs
state_sample = {
    "conversion_rate_search_ad": 0.08,
    "conversion_rate_display_ad": 0.04,
    "conversion_rate_social_ad": 0.06,
    "cost_per_click_search_ad": 0.9,
    "cost_per_click_display_ad": 0.5,
    "cost_per_click_social_ad": 0.7
}

# Evaluate the utility of this action given the state
value = utility.evaluate(action, state_sample)
print(f"Expected profit: ${value:.2f}")
```

### Using the Factory Function

```python
from aicons.bayesbrainGPT.utility_function import create_utility

# Create a utility using the factory function
utility = create_utility(
    utility_type="constrained_marketing_roi",
    revenue_per_sale=12.0,
    num_ads=2,
    constraints={
        'min_budget_per_ad': 50.0,
        'max_daily_spend': 200.0,
        'balanced_spend': True
    }
)
```

### Creating a Custom Utility

```python
from aicons.bayesbrainGPT.utility_function import LambdaUtility

# Define a custom utility function
def my_custom_utility(action, state):
    # Custom logic here
    return action.get("investment", 0) * state.get("expected_return", 0.05)

# Create the utility
utility = LambdaUtility(
    evaluation_fn=my_custom_utility,
    name="Investment Return Utility",
    description="Evaluates expected returns from investments"
)
```

### Combining Multiple Utilities

```python
from aicons.bayesbrainGPT.utility_function import WeightedSumUtility, MarketingROIUtility, create_utility

# Create component utilities
profit_utility = MarketingROIUtility(revenue_per_sale=10.0, num_ads=2)

# Create a utility for brand exposure (higher spending on brand ads is better)
brand_utility = create_utility(
    utility_type="lambda",
    evaluation_fn=lambda action, state: action.get("brand_ad_budget", 0) * 0.01,
    name="Brand Exposure"
)

# Combine them with different weights
combined_utility = WeightedSumUtility(
    utility_functions=[profit_utility, brand_utility],
    weights=[0.8, 0.2],  # 80% profit, 20% brand exposure
    names=["profit", "brand_exposure"]
)
```

### Rule-Based Utility

```python
from aicons.bayesbrainGPT.utility_function import RuleBasedUtility

# Create a rule-based utility
rule_utility = RuleBasedUtility()

# Add rules
rule_utility.add_rule(
    name="Minimum Social Budget",
    check_fn=lambda action, state: action.get("social_ad_budget", 0) >= 100,
    score=10.0
)

rule_utility.add_rule(
    name="Weather-Appropriate Allocation",
    check_fn=lambda action, state: (
        state.get("weather") == "rainy" and action.get("indoor_activity_budget", 0) >
        action.get("outdoor_activity_budget", 0)
    ),
    score=15.0
)
```

## TensorFlow Integration

Many utility functions implement the `evaluate_tf` method, which allows for gradient-based optimization:

```python
import tensorflow as tf
from aicons.bayesbrainGPT.utility_function import MarketingROIUtility

# Create the utility
utility = MarketingROIUtility(revenue_per_sale=12.0, num_ads=2)

# Initial budget allocation
initial_action = tf.Variable([100.0, 100.0])

# State samples (multiple samples for posterior)
state_samples = {
    'phi': tf.constant([[0.05, 0.04], [0.06, 0.03], [0.04, 0.05]]),  # 3 samples, 2 ads
    'c': tf.constant([[0.8, 0.6], [0.7, 0.5], [0.9, 0.7]])
}

# Define optimization
optimizer = tf.optimizers.Adam(learning_rate=0.1)

# Optimization loop
for step in range(100):
    with tf.GradientTape() as tape:
        # Negate utility for minimization
        loss = -tf.reduce_mean(utility.evaluate_tf(initial_action, state_samples))

    # Apply gradients
    gradients = tape.gradient(loss, [initial_action])
    optimizer.apply_gradients(zip(gradients, [initial_action]))

    # Apply constraints (make sure budgets are non-negative)
    initial_action.assign(tf.maximum(0.0, initial_action))
```

## Extending the Framework

To add a new utility function type:

1. Create a new class that inherits from `UtilityFunction`
2. Implement the `evaluate` method (and optionally `evaluate_tf` if using TensorFlow)
3. Add your class to the `UTILITY_FACTORIES` dictionary in `__init__.py`

```python
from aicons.bayesbrainGPT.utility_function import UtilityFunction

class MyCustomUtility(UtilityFunction):
    def __init__(self, param1, param2=0.5):
        super().__init__(name="My Custom Utility",
                         description="A custom utility function")
        self.param1 = param1
        self.param2 = param2

    def evaluate(self, action, state_sample):
        # Your evaluation logic here
        return result
```

## Best Practices

1. Use specialized utility functions for specific domains when available
2. Consider multi-objective utilities for balancing competing objectives
3. Use TensorFlow-enabled utilities when gradient-based optimization is needed
4. Structure complex utilities as compositions of simpler ones
5. Add constraints to enforce business rules and limitations
