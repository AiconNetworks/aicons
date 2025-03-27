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

## Example Implementations

### 1. Simple Linear Utility

```python
class LinearUtility(UtilityFunction):
    def __init__(self, name: str, weights: Dict[str, float],
                 description: str = "", action_space=None):
        super().__init__(name=name, description=description, action_space=action_space)
        self.weights = weights

    def evaluate_tf(self, action: tf.Tensor,
                   state_samples: Dict[str, tf.Tensor]) -> tf.Tensor:
        # Map weights to action dimensions
        if hasattr(self, 'dimensions') and self.dimensions is not None:
            weight_list = [self.weights.get(dim.name, 0.0) for dim in self.dimensions]
        else:
            weight_list = list(self.weights.values())

        weights_tensor = tf.constant(weight_list, dtype=tf.float32)
        return tf.reduce_sum(action * weights_tensor, axis=-1)
```

### 2. State-Dependent Utility

```python
class StateDependentUtility(UtilityFunction):
    def __init__(self, name: str, sensitivity: float,
                 description: str = "", action_space=None):
        super().__init__(name=name, description=description, action_space=action_space)
        self.sensitivity = sensitivity

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
