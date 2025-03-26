# Utility Function Implementation Plan

## Current Status

Currently, our utility function implementation has the following components:

1. **In `bayesbrainGPT/utility_function/utility_function.py`**:

   - Abstract base classes:
     - `UtilityFunction` - Standard callable utility function
     - `TensorFlowUtilityFunction` - TensorFlow-based utility function for gradient optimization
   - Concrete implementations:
     - `MarketingROIUtility` - For marketing ROI calculations
     - `ConstrainedMarketingROI` - Marketing ROI with business constraints
     - `WeatherDependentMarketingROI` - Marketing ROI that adjusts for weather conditions
   - Factory functions:
     - `create_utility_function` - Creates a utility function of the specified type
     - `create_custom_marketing_utility` - Creates customized marketing utility functions

2. **In `BayesBrain` class**:

   - Methods for setting and getting utility functions
   - Methods for evaluating actions using utility functions

3. **In `BaseAIcon` class**:
   - Basic utility function creation that tries to directly call brain methods
   - Missing proper integration with the utility function creators

## Integration Issues

The main issues with our current implementation:

1. **Inconsistent API**: `BaseAIcon` and `SimpleBadAIcon` have different approaches to creating utility functions
2. **Improper Integration**: `BaseAIcon` doesn't properly utilize the utility function creators from the module
3. **Missing Typed Utility Functions**: `BaseAIcon` requires manual function creation rather than supporting predefined types

## Implementation Plan

### 1. Refactor `BaseAIcon.create_utility_function()`

Reimplement the method to properly leverage the utility function factories:

```python
def create_utility_function(self, utility_type: str = 'marketing_roi', **kwargs):
    """
    Create a utility function for decision-making.

    Args:
        utility_type: Type of utility function to create. Options include:
            - 'marketing_roi': For marketing ROI optimization
            - 'constrained_marketing_roi': Marketing ROI with business constraints
            - 'weather_dependent_marketing_roi': Weather-sensitive marketing ROI
            - 'custom': Custom utility function
        **kwargs: Additional parameters specific to the utility function type

    Returns:
        The created utility function
    """
    if not self.brain:
        print("BayesBrain not available, cannot create utility function")
        return None

    try:
        # Import utility function factory
        from aicons.bayesbrainGPT.utility_function.utility_function import (
            create_utility_function,
            create_custom_marketing_utility
        )

        # If a function is directly provided, use it as custom
        if 'function' in kwargs:
            function = kwargs.pop('function')
            name = kwargs.pop('name', 'custom_utility')

            utility_function = create_utility_function(
                function=function,
                name=name
            )

        # Otherwise use the utility_type to create predefined types
        elif utility_type == 'custom_marketing':
            utility_function = create_custom_marketing_utility(**kwargs)
        else:
            # Use action space dimensions if available
            action_space = self.brain.get_action_space()
            if action_space:
                # Add information about dimensions to kwargs
                kwargs['num_ads'] = len([d for d in action_space.dimensions
                                       if hasattr(d, 'name') and d.name.endswith('_budget')])
                kwargs['ad_names'] = [d.name.replace('_budget', '')
                                     for d in action_space.dimensions
                                     if hasattr(d, 'name') and d.name.endswith('_budget')]

            # Create the utility function using the factory
            utility_function = create_utility_function(utility_type, **kwargs)

        # Set the utility function in the brain
        if utility_function and hasattr(self.brain, 'set_utility_function'):
            self.brain.set_utility_function(utility_function)
            print(f"Created {utility_type} utility function")

        return utility_function

    except Exception as e:
        print(f"Could not create utility function: {e}")
        return None
```

### 2. Refactor `BaseAIcon.create_action_space()`

Reimplement the method to properly leverage the action space factories:

```python
def create_action_space(self, space_type: str = 'budget_allocation', **kwargs):
    """
    Create an action space for decision-making.

    Args:
        space_type: Type of action space to create. Options include:
            - 'budget_allocation': For allocating budget across items
            - 'marketing': For marketing campaign optimization
            - 'time_budget': For time-based budget allocation
            - 'multi_campaign': For multi-campaign budget allocation
        **kwargs: Additional parameters specific to the action space type

    Returns:
        The created action space
    """
    if not self.brain:
        print("BayesBrain not available, cannot create action space")
        return None

    try:
        # Import action space components
        from aicons.bayesbrainGPT.decision_making.action_space import (
            ActionSpace,
            create_budget_allocation_space,
            create_time_budget_allocation_space,
            create_multi_campaign_action_space,
            create_marketing_ads_space
        )

        # For a fully manual definition, directly use the provided dimensions and constraints
        if 'dimensions' in kwargs:
            dimensions = kwargs.get('dimensions', {})
            constraints = kwargs.get('constraints', [])

            action_space = ActionSpace(
                dimensions=dimensions,
                constraints=constraints
            )

        # Otherwise use predefined action space creators
        elif space_type == 'budget_allocation':
            # Budget allocation across items
            total_budget = kwargs.get('total_budget', 1000.0)
            items = kwargs.get('items', [])
            step_size = kwargs.get('step_size', 0.01)
            min_budget = kwargs.get('min_budget', 0.0)

            # Convert step_size from percentage to absolute if needed
            if 0 < step_size < 1:
                budget_step = step_size * total_budget
            else:
                budget_step = step_size

            # Create the action space
            action_space = create_budget_allocation_space(
                total_budget=total_budget,
                num_ads=len(items),
                budget_step=budget_step,
                min_budget=min_budget,
                ad_names=items
            )

        elif space_type == 'marketing':
            # Marketing optimization space
            total_budget = kwargs.get('total_budget', 1000.0)
            num_ads = kwargs.get('num_ads', 3)
            budget_step = kwargs.get('budget_step', 10.0)
            min_budget = kwargs.get('min_budget', 0.0)
            ad_names = kwargs.get('ad_names', None)

            action_space = create_marketing_ads_space(
                total_budget=total_budget,
                num_ads=num_ads,
                budget_step=budget_step,
                min_budget=min_budget,
                ad_names=ad_names
            )

        elif space_type == 'time_budget':
            # Time and budget allocation
            total_budget = kwargs.get('total_budget', 1000.0)
            num_days = kwargs.get('num_days', 7)
            num_ads = kwargs.get('num_ads', 2)
            budget_step = kwargs.get('budget_step', 10.0)
            min_budget = kwargs.get('min_budget', 0.0)

            action_space = create_time_budget_allocation_space(
                total_budget=total_budget,
                num_ads=num_ads,
                num_days=num_days,
                budget_step=budget_step,
                min_budget=min_budget
            )

        elif space_type == 'multi_campaign':
            # Multi-campaign allocation
            campaigns = kwargs.get('campaigns', {})
            budget_step = kwargs.get('budget_step', 10.0)

            action_space = create_multi_campaign_action_space(
                campaigns=campaigns,
                budget_step=budget_step
            )

        else:
            print(f"Unknown action space type: {space_type}")
            return None

        # Set the action space in the brain
        if action_space and hasattr(self.brain, 'set_action_space'):
            self.brain.set_action_space(action_space)
            print(f"Created {space_type} action space")

        return action_space

    except Exception as e:
        print(f"Could not create action space: {e}")
        return None
```

### 3. Ensure Proper `find_best_action()` Integration

Make sure the method properly utilizes the BayesBrain's functionality:

```python
def find_best_action(self, num_samples: int = 100, use_gradient: bool = False):
    """
    Find the best action based on the current state and utility function.

    Args:
        num_samples: Number of samples to use for the search
        use_gradient: Whether to use gradient-based optimization (requires TensorFlow)

    Returns:
        Tuple of (best_action, expected_utility)
    """
    if not self.brain:
        print("BayesBrain not available, cannot find best action")
        return None, 0.0

    try:
        # Make sure we have an action space and utility function
        if self.brain.get_action_space() is None:
            raise ValueError("Cannot find best action without an action space. Call create_action_space() first.")

        if self.brain.get_utility_function() is None:
            raise ValueError("Cannot find best action without a utility function. Call create_utility_function() first.")

        # Set the number of samples
        if hasattr(self.brain, 'decision_params'):
            self.brain.decision_params["num_samples"] = num_samples

        # Use the brain's find_best_action method
        return self.brain.find_best_action(num_samples=num_samples, use_gradient=use_gradient)
    except Exception as e:
        print(f"Error finding best action: {e}")
        return None, 0.0
```

### 4. Add `perceive_and_decide()` Method

Reimplement this convenience method:

```python
def perceive_and_decide(self, environment=None):
    """
    Perceive the environment and make a decision.

    This convenience method updates from all sensors and then finds the best action.

    Args:
        environment: Environment data to pass to sensors

    Returns:
        Tuple of (best_action, expected_utility)
    """
    if not self.brain:
        print("BayesBrain not available, cannot perceive and decide")
        return None, 0.0

    try:
        # Update from all sensors
        if hasattr(self, 'update_from_all_sensors'):
            self.update_from_all_sensors(environment)

        # Find the best action
        return self.find_best_action()
    except Exception as e:
        print(f"Error in perceive_and_decide: {e}")
        return None, 0.0
```

## Future Enhancements

1. **More Utility Function Types**:

   - Multi-objective utility functions
   - Time-discounted utility functions
   - Risk-sensitive utility functions

2. **Better Integration with TensorFlow**:

   - More efficient vectorized implementations
   - Support for automatic differentiation
   - Integration with JAX for high-performance Bayesian computation

3. **Custom Constraints**:

   - Support for user-defined constraint functions
   - Better handling of hard vs. soft constraints

4. **Documentation and Examples**:
   - Clear documentation for all utility function types
   - Examples of common use cases
   - Tutorials on extending the framework with custom utility functions
