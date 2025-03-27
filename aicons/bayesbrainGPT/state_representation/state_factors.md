# State Factors and Hierarchical Modeling

## Overview

The state representation system implements a hierarchical Bayesian model where factors (latent variables) can have dependencies on other factors. This follows the Bayesian brain hypothesis where higher-level factors influence lower-level ones, creating a generative model of the world.

## Creating Hierarchical Factors

### The Correct Way

All factors should be created with explicit hierarchical relationships. There are two main approaches:

1. **Using AIcon (Recommended)**

```python
# First create root factors (no dependencies)
aicon.add_state_factor(
    name='weather',
    factor_type='categorical',
    value='clear',
    categories=['clear', 'cloudy', 'stormy'],
    probs=[0.5, 0.3, 0.2]
)

# Then create dependent factors with explicit relationships
aicon.add_state_factor(
    name='temperature',
    factor_type='continuous',
    value=25.0,
    uncertainty=2.0,
    relationships={
        'depends_on': ['weather'],
        'type': 'conditional',
        'conditional_probs': {
            'clear': {'mean': 25.0, 'std': 2.0},
            'cloudy': {'mean': 20.0, 'std': 2.0},
            'stormy': {'mean': 15.0, 'std': 2.0}
        }
    }
)
```

2. **Using BayesianState Directly**

```python
# First create root factors
state.add_categorical_latent(
    name='weather',
    initial_value='clear',
    possible_values=['clear', 'cloudy', 'stormy'],
    probs=[0.5, 0.3, 0.2]
)

# Then create dependent factors with relationships
state.add_continuous_latent(
    name='temperature',
    mean=25.0,
    uncertainty=2.0,
    relationships={
        'depends_on': ['weather'],
        'type': 'conditional',
        'conditional_probs': {
            'clear': {'mean': 25.0, 'std': 2.0},
            'cloudy': {'mean': 20.0, 'std': 2.0},
            'stormy': {'mean': 15.0, 'std': 2.0}
        }
    }
)
```

### Common Mistakes to Avoid

1. **Creating Factors Without Relationships**

```python
# DON'T DO THIS - Missing hierarchical structure
aicon.add_state_factor(
    name='temperature',
    factor_type='continuous',
    value=25.0,
    uncertainty=2.0
)
```

2. **Inconsistent Relationship Definition**

```python
# DON'T DO THIS - Inconsistent relationship definition
state.add_continuous_latent(
    name='temperature',
    mean=25.0,
    uncertainty=2.0,
    relationships={'depends_on': ['weather']}  # Missing type and conditional_probs
)
```

3. **Creating Factors in Wrong Order**

```python
# DON'T DO THIS - Creating dependent factor before its parent
aicon.add_state_factor(
    name='temperature',  # Depends on weather
    factor_type='continuous',
    value=25.0,
    uncertainty=2.0,
    relationships={'depends_on': ['weather']}  # weather doesn't exist yet!
)

aicon.add_state_factor(  # weather should be created first
    name='weather',
    factor_type='categorical',
    value='clear',
    categories=['clear', 'cloudy', 'stormy']
)
```

### Best Practices for Hierarchical Creation

1. **Always Define Relationships**

   - Every factor should have explicit relationships
   - Root factors should have empty relationships
   - Dependent factors must specify their parents

2. **Create in Correct Order**

   - Create root factors first
   - Then create factors that depend on them
   - Follow the hierarchical structure

3. **Use Consistent Relationship Format**

   ```python
   relationships={
       'depends_on': ['parent_factor1', 'parent_factor2'],
       'type': 'conditional',
       'conditional_probs': {
           # Define probabilities/parameters for each parent combination
       }
   }
   ```

4. **Validate Dependencies**
   - Check that parent factors exist
   - Ensure no circular dependencies
   - Verify relationship format

## Factor Types

### 1. Continuous Factors

- Represent continuous variables (e.g., temperature, budget)
- Can have constraints (lower/upper bounds)
- Use TensorFlow Probability distributions:
  - Normal for unconstrained
  - TruncatedNormal for bounded
  - TransformedDistribution for lower/upper bounded

```python
# Example: Adding a continuous factor
state.add_continuous_latent(
    name='budget',
    mean=1000.0,
    uncertainty=100.0,
    lower_bound=0.0,  # Budget can't be negative
    upper_bound=5000.0  # Maximum budget
)
```

### 2. Categorical Factors

- Represent discrete categories (e.g., weather, ad channel)
- Have a finite set of possible values
- Use Categorical distribution

```python
# Example: Adding a categorical factor
state.add_categorical_latent(
    name='weather',
    initial_value='clear',
    possible_values=['clear', 'cloudy', 'stormy'],
    probs=[0.5, 0.3, 0.2]  # Prior probabilities
)
```

### 3. Discrete Factors

- Represent integer values (e.g., counts, indices)
- Can be bounded or unbounded
- Use Poisson or Categorical distribution

```python
# Example: Adding a discrete factor
state.add_discrete_latent(
    name='clicks',
    initial_value=100,
    min_value=0,
    rate=100.0  # For Poisson distribution
)
```

## Hierarchical Relationships

### 1. Defining Dependencies

Factors can depend on other factors through the `relationships` parameter:

```python
# Example: Temperature depends on weather
state.add_continuous_latent(
    name='temperature',
    mean=25.0,
    uncertainty=2.0,
    relationships={
        'depends_on': ['weather'],
        'type': 'conditional',
        'conditional_probs': {
            'clear': {'mean': 25.0, 'std': 2.0},
            'cloudy': {'mean': 20.0, 'std': 2.0},
            'stormy': {'mean': 15.0, 'std': 2.0}
        }
    }
)
```

### 2. Joint Distribution

The system automatically creates a hierarchical joint distribution respecting dependencies. The joint distribution is built using TensorFlow Probability's `JointDistributionNamed` with proper topological ordering of factors:

```python
# The joint distribution P(temperature, weather) is created as:
P(temperature, weather) = P(weather) * P(temperature | weather)

# For more complex hierarchies:
P(temperature, weather, clicks) = P(weather) * P(temperature | weather) * P(clicks | weather, temperature)
```

The joint distribution is created by:

1. Building a dictionary of distributions in topological order
2. Handling both root factors (no parents) and conditional factors
3. Using lambda functions for conditional distributions
4. Maintaining proper parent-child relationships

Example of how the joint distribution is built internally:

```python
# For a factor with dependencies:
if factor_name in self.hierarchical_relations:
    # This is a conditional factor
    parent_names = self.hierarchical_relations[factor_name]["parents"]

    # Create lambda function that depends on parent values
    dist_dict[factor_name] = lambda *args, fn=factor_name: self._get_conditional_dist(fn, args)
else:
    # This is a root factor (no parents)
    dist_dict[factor_name] = factor.tf_distribution
```

The joint distribution ensures that:

1. Parent factors are sampled before their children
2. Conditional dependencies are properly respected
3. The hierarchical structure is maintained during sampling
4. All factors are properly integrated into the generative model

This hierarchical structure allows for:

- Proper modeling of complex dependencies
- Efficient sampling respecting the hierarchy
- Correct inference of conditional relationships
- Accurate representation of the Bayesian brain's generative model

## Implementation Details

### 1. Factor Storage

- Factors are stored in `self.factors` dictionary
- Each factor has its own TensorFlow distribution
- Relationships are stored in `self.hierarchical_relations`

### 2. Topological Order

- Factors are ordered based on dependencies
- Parent factors are evaluated before dependent factors
- Ensures proper sampling and inference

### 3. Conditional Distributions

- Stored in `self.conditional_distributions`
- Functions that return distributions based on parent values
- Used in joint distribution creation

## Usage in AIcon

### 1. Adding State Factors

```python
# Add a categorical factor (weather)
aicon.add_state_factor(
    name='weather',
    factor_type='categorical',
    value='clear',
    categories=['clear', 'cloudy', 'stormy'],
    probs=[0.5, 0.3, 0.2]
)

# Add a continuous factor that depends on weather
aicon.add_state_factor(
    name='temperature',
    factor_type='continuous',
    value=25.0,
    uncertainty=2.0,
    relationships={
        'depends_on': ['weather'],
        'type': 'conditional',
        'conditional_probs': {
            'clear': {'mean': 25.0, 'std': 2.0},
            'cloudy': {'mean': 20.0, 'std': 2.0},
            'stormy': {'mean': 15.0, 'std': 2.0}
        }
    }
)
```

### 2. Sampling from the Model

```python
# Sample from the joint distribution
samples = aicon.sample_from_hierarchical_prior(n_samples=1000)

# Update beliefs based on observations
aicon.update_beliefs(observations={
    'temperature': (22.0, 0.5),  # (value, reliability)
    'weather': ('cloudy', 1.0)
})
```

## Best Practices

1. **Factor Naming**

   - Use clear, descriptive names
   - Follow consistent naming conventions

2. **Dependencies**

   - Define dependencies explicitly
   - Avoid circular dependencies
   - Keep the hierarchy as shallow as possible

3. **Constraints**

   - Use appropriate constraints for continuous variables
   - Ensure categorical probabilities sum to 1
   - Set reasonable bounds for discrete variables

4. **Uncertainty**
   - Set appropriate uncertainty values
   - Consider sensor reliability
   - Update uncertainties based on observations

## Common Pitfalls

1. **Circular Dependencies**

   ```python
   # DON'T DO THIS
   factor1 = state.add_factor(..., relationships={'depends_on': ['factor2']})
   factor2 = state.add_factor(..., relationships={'depends_on': ['factor1']})
   ```

2. **Missing Dependencies**

   ```python
   # DON'T FORGET TO SPECIFY DEPENDENCIES
   # If temperature depends on weather, specify it:
   temperature = state.add_factor(..., relationships={'depends_on': ['weather']})
   ```

3. **Invalid Probabilities**
   ```python
   # DON'T USE INVALID PROBABILITIES
   # Probabilities must sum to 1
   state.add_categorical_latent(..., probs=[0.3, 0.4, 0.4])  # Sums to 1.1
   ```

## Validation Rules

### 1. Relationship Format Validation

The system enforces strict validation of relationship formats:

```python
# This will fail - relationships must be a dictionary
aicon.add_state_factor(
    name='temperature',
    factor_type='continuous',
    value=25.0,
    relationships='weather'  # Must be a dictionary
)

# This will fail - depends_on must be a list
aicon.add_state_factor(
    name='temperature',
    factor_type='continuous',
    value=25.0,
    relationships={'depends_on': 'weather'}  # Must be a list
)

# This will fail - all elements in depends_on must be strings
aicon.add_state_factor(
    name='temperature',
    factor_type='continuous',
    value=25.0,
    relationships={'depends_on': [1, 2, 3]}  # Must be strings
)

# This will work
aicon.add_state_factor(
    name='temperature',
    factor_type='continuous',
    value=25.0,
    relationships={'depends_on': ['weather']}
)
```

### 2. Parent Factor Existence

The system ensures that parent factors exist before creating dependent factors:

```python
# This will fail - weather doesn't exist yet
aicon.add_state_factor(
    name='temperature',
    factor_type='continuous',
    value=25.0,
    relationships={'depends_on': ['weather']}  # weather doesn't exist yet
)

# This will work
aicon.add_state_factor(
    name='weather',
    factor_type='categorical',
    value='clear',
    categories=['clear', 'cloudy']
)

aicon.add_state_factor(
    name='temperature',
    factor_type='continuous',
    value=25.0,
    relationships={'depends_on': ['weather']}  # weather exists now
)
```

### 3. Circular Dependency Detection

The system prevents circular dependencies between factors:

```python
# This will fail - creates a cycle
aicon.add_state_factor(
    name='A',
    factor_type='continuous',
    value=1.0,
    relationships={'depends_on': ['B']}
)

aicon.add_state_factor(
    name='B',
    factor_type='continuous',
    value=2.0,
    relationships={'depends_on': ['A']}  # Creates a cycle!
)

# This will work - proper hierarchical structure
aicon.add_state_factor(
    name='A',
    factor_type='continuous',
    value=1.0,
    relationships={}  # Root factor
)

aicon.add_state_factor(
    name='B',
    factor_type='continuous',
    value=2.0,
    relationships={'depends_on': ['A']}  # B depends on A
)

aicon.add_state_factor(
    name='C',
    factor_type='continuous',
    value=3.0,
    relationships={'depends_on': ['B']}  # C depends on B
)
```

### 4. Factor Type Validation

Each factor type has its own validation rules:

```python
# Categorical factors require categories and probabilities
aicon.add_state_factor(
    name='weather',
    factor_type='categorical',
    value='clear'  # Missing categories and probs
)  # This will fail

# This will work
aicon.add_state_factor(
    name='weather',
    factor_type='categorical',
    value='clear',
    categories=['clear', 'cloudy'],
    probs=[0.7, 0.3]
)

# Continuous factors validate numeric values
aicon.add_state_factor(
    name='temperature',
    factor_type='continuous',
    value='not a number'  # Must be numeric
)  # This will fail

# This will work
aicon.add_state_factor(
    name='temperature',
    factor_type='continuous',
    value=25.0,
    uncertainty=2.0
)
```

### 5. Validation Error Messages

The system provides clear error messages when validation fails:

```python
try:
    aicon.add_state_factor(
        name='temperature',
        factor_type='continuous',
        value=25.0,
        relationships={'depends_on': ['weather']}  # weather doesn't exist
    )
except ValueError as e:
    print(e)  # "Cannot create factor 'temperature' because parent factors do not exist: ['weather']"

try:
    aicon.add_state_factor(
        name='A',
        factor_type='continuous',
        value=1.0,
        relationships={'depends_on': ['B']}
    )
    aicon.add_state_factor(
        name='B',
        factor_type='continuous',
        value=2.0,
        relationships={'depends_on': ['A']}  # Creates cycle
    )
except ValueError as e:
    print(e)  # "Adding factor 'B' would create a circular dependency"
```

These validations ensure that:

1. All factors are created with proper relationships
2. The hierarchical structure is maintained
3. No circular dependencies can be created
4. Factor types have appropriate parameters
5. The system provides clear error messages for debugging
