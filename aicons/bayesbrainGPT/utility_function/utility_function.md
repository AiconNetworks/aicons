# Creating Custom Utility Functions in BayesBrainGPT

This guide explains how to create custom utility functions in BayesBrainGPT. Utility functions are essential components that evaluate the "goodness" of actions given specific states.

## Basic Structure

Every utility function must inherit from the `UtilityFunction` base class and implement the required `evaluate_tf` method. Here's a basic template:

```python
from aicons.bayesbrainGPT.utility_function import UtilityFunction
import tensorflow as tf
from typing import Dict, Any

class MyCustomUtility(UtilityFunction):
    def __init__(self, name: str, param1: float, param2: float = 0.5,
                 description: str = "", action_space=None):
        """
        Initialize the utility function.

        Args:
            name: Name of the utility function
            param1: First parameter
            param2: Second parameter (optional)
            description: Description of what this utility function measures
            action_space: Optional action space that this utility function will evaluate
        """
        super().__init__(name=name, description=description, action_space=action_space)
        self.param1 = param1
        self.param2 = param2

    def __str__(self) -> str:
        """Show exactly what this utility function computes."""
        return f"MyCustomUtility: {self.param1} * value + {self.param2}"

    def evaluate_tf(self, action: tf.Tensor,
                   state_samples: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Evaluate the utility using TensorFlow.

        Args:
            action: Tensor containing action values
            state_samples: Dictionary mapping state factor names to tensors

        Returns:
            Tensor of utility values
        """
        # Your TensorFlow implementation here
        return result
```

## Key Components

### 1. Class Definition

- Inherit from `UtilityFunction`
- Import necessary dependencies (tensorflow, typing)

### 2. Constructor

- Call `super().__init__()` with name and description
- Store any parameters needed for evaluation
- Handle optional action_space parameter

### 3. evaluate_tf Method

- Must be implemented (it's an abstract method)
- Takes action tensor and state samples as input
- Returns a tensor of utility values
- Use TensorFlow operations for efficient computation

### 4. String Representation

- Implement `__str__` method to show exactly what the utility function computes
- This is crucial for debugging and understanding the utility function's behavior
- Should show the mathematical formula or computation being performed

## Example Implementations

### 1. Linear Utility (Current Implementation)

```python
class LinearUtility(UtilityFunction):
    def __init__(self, name: str, weights: Dict[str, float],
                 description: str = "", action_space=None):
        super().__init__(name=name, description=description, action_space=action_space)
        self.weights = weights

        # Convert weights to tensor if it's a dictionary
        if isinstance(weights, dict):
            if action_space is not None and action_space.dimensions is not None:
                action_names = [dim.name for dim in action_space.dimensions]
                self.weights = [weights.get(name, 0.0) for name in action_names]
            else:
                self.weights = list(weights.values())

        self.weights = tf.convert_to_tensor(self.weights, dtype=tf.float32)

    def __str__(self) -> str:
        """Show exactly what this utility function computes."""
        if isinstance(self.weights, dict):
            weights_str = ", ".join(f"{k}: {v}" for k, v in self.weights.items())
        else:
            weights_str = ", ".join(str(w) for w in self.weights)
        return f"LinearUtility: Σ(weights * values) where weights = [{weights_str}]"

    def evaluate_tf(self, action_values: tf.Tensor,
                   state_samples: Optional[tf.Tensor] = None,
                   **kwargs) -> tf.Tensor:
        """
        Evaluate the utility function using TensorFlow.

        For marketing campaigns, this computes:
        U(Ad) = Expected Revenue - Cost + λ ⋅ Brand Impact

        Where:
        - Expected Revenue = budget * conversion_rate
        - Cost = budget
        - Brand Impact = budget * brand_factor
        """
        # These would come from historical data or LLM inference
        conversion_rate = 0.1  # Example: 10% conversion rate
        brand_factor = 0.05   # Example: 5% brand impact per dollar

        # Compute components
        expected_revenue = action_values * conversion_rate
        cost = action_values
        brand_impact = action_values * brand_factor

        # Compute final utility
        utility = expected_revenue - cost + brand_impact

        # Apply weights to each campaign's utility
        return tf.reduce_sum(utility * self.weights, axis=-1)
```

### 2. State-Dependent Utility

```python
class StateDependentUtility(UtilityFunction):
    def __init__(self, name: str, sensitivity: float,
                 description: str = "", action_space=None):
        super().__init__(name=name, description=description, action_space=action_space)
        self.sensitivity = sensitivity

    def __str__(self) -> str:
        """Show exactly what this utility function computes."""
        return f"StateDependentUtility: value * (1 + {self.sensitivity} * state)"

    def evaluate_tf(self, action: tf.Tensor,
                   state_samples: Dict[str, tf.Tensor]) -> tf.Tensor:
        # Get state values
        state_value = state_samples.get('state_factor', tf.constant(0.0))

        # Combine action and state
        combined = action * (1.0 + self.sensitivity * state_value)
        return tf.reduce_sum(combined, axis=-1)
```

### 3. Constrained Utility

```python
class ConstrainedUtility(UtilityFunction):
    def __init__(self, name: str, min_value: float, max_value: float,
                 description: str = "", action_space=None):
        super().__init__(name=name, description=description, action_space=action_space)
        self.min_value = min_value
        self.max_value = max_value

    def __str__(self) -> str:
        """Show exactly what this utility function computes."""
        return f"ConstrainedUtility: clip(value, {self.min_value}, {self.max_value})"

    def evaluate_tf(self, action: tf.Tensor,
                   state_samples: Dict[str, tf.Tensor]) -> tf.Tensor:
        # Apply constraints
        constrained_action = tf.clip_by_value(
            action,
            clip_value_min=self.min_value,
            clip_value_max=self.max_value
        )
        return tf.reduce_sum(constrained_action, axis=-1)
```

## Best Practices

1. **Use TensorFlow Operations**

   - Always use TensorFlow operations for computation
   - Avoid Python loops when possible
   - Use vectorized operations for efficiency

2. **Handle Action Space**

   - Check for action space dimensions
   - Map parameters to correct dimensions
   - Provide fallback behavior when dimensions aren't available

3. **State Sample Handling**

   - Use `.get()` with default values for state samples
   - Handle missing state factors gracefully
   - Consider state dependencies in computation

4. **Error Handling**

   - Validate input parameters
   - Provide meaningful error messages
   - Handle edge cases gracefully

5. **Documentation**

   - Document all parameters
   - Explain the utility function's purpose
   - Provide usage examples

6. **String Representation**
   - Always implement `__str__` method
   - Show the mathematical formula or computation
   - Include key parameters in the string representation
   - Make it clear what the utility function is computing

## Integration

To use your custom utility function:

1. Add it to the `UTILITY_FACTORIES` dictionary in `__init__.py`:

```python
UTILITY_FACTORIES = {
    # ... existing utilities ...
    "my_custom": MyCustomUtility
}
```

2. Use it through the factory function:

```python
from aicons.bayesbrainGPT.utility_function import create_utility

utility = create_utility(
    utility_type="my_custom",
    param1=0.7,
    param2=0.3,
    action_space=your_action_space
)
```

3. Or instantiate directly:

```python
from aicons.bayesbrainGPT.utility_function import MyCustomUtility

utility = MyCustomUtility(
    name="my_utility",
    param1=0.7,
    param2=0.3,
    action_space=your_action_space
)
```

## Testing

Always test your utility function with:

1. Different action values
2. Various state samples
3. Edge cases and boundary conditions
4. TensorFlow gradient computation
5. Integration with other components
6. String representation output
