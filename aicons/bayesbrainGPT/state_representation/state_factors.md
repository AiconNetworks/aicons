# State Factors and Hierarchical Modeling

## Overview

The state representation system implements a hierarchical Bayesian model where factors (latent variables) can have dependencies on other factors. This follows the Bayesian brain hypothesis where higher-level factors influence lower-level ones, creating a generative model of the world.

## Creating Hierarchical Factors

### The Correct Way

All factors should be created with explicit hierarchical relationships using the unified `add_state_factor` method:

```python
# First create root factors (no dependencies)
aicon.add_state_factor(
    name='market_size',
    factor_type='continuous',
    value=10000.0,
    params={
        'loc': 10000.0,
        'scale': 1000.0,
        'constraints': {'lower': 0}
    },
    relationships={
        'depends_on': []  # Empty list for root factor
    }
)

# Then create dependent factors with explicit relationships
aicon.add_state_factor(
    name='conversion_rate',
    factor_type='continuous',
    value=0.02,
    params={
        'loc': 0.02,
        'scale': 0.005,
        'constraints': {'lower': 0, 'upper': 1}
    },
    relationships={
        'depends_on': ['market_size', 'competition_level']
    }
)
```

### Common Mistakes to Avoid

1. **Creating Factors Without Relationships**

```python
# DON'T DO THIS - Missing relationships dictionary
aicon.add_state_factor(
    name='market_size',
    factor_type='continuous',
    value=10000.0,
    params={
        'loc': 10000.0,
        'scale': 1000.0
    }
    # Missing relationships!
)
```

2. **Inconsistent Relationship Definition**

```python
# DON'T DO THIS - depends_on must be a list
aicon.add_state_factor(
    name='conversion_rate',
    factor_type='continuous',
    value=0.02,
    params={
        'loc': 0.02,
        'scale': 0.005
    },
    relationships={
        'depends_on': 'market_size'  # Must be a list!
    }
)
```

3. **Creating Factors in Wrong Order**

```python
# DON'T DO THIS - Creating dependent factor before its parent
aicon.add_state_factor(
    name='conversion_rate',  # Depends on market_size
    factor_type='continuous',
    value=0.02,
    params={
        'loc': 0.02,
        'scale': 0.005
    },
    relationships={
        'depends_on': ['market_size']  # market_size doesn't exist yet!
    }
)

aicon.add_state_factor(  # market_size should be created first
    name='market_size',
    factor_type='continuous',
    value=10000.0,
    params={
        'loc': 10000.0,
        'scale': 1000.0
    },
    relationships={
        'depends_on': []
    }
)
```

### Best Practices for Hierarchical Creation

1. **Always Define Relationships**

   - Every factor must have a `relationships` dictionary
   - Root factors should have empty `depends_on` list
   - Dependent factors must specify their parent factors

2. **Create in Correct Order**

   - Create root factors first
   - Then create factors that depend on them
   - Follow the hierarchical structure

3. **Use Consistent Parameter Format**

   ```python
   params={
       'loc': value,  # For continuous
       'scale': uncertainty,  # For continuous
       'constraints': {'lower': 0, 'upper': 1},  # For continuous
       'categories': ['low', 'medium', 'high'],  # For categorical
       'probs': [0.2, 0.5, 0.3],  # For categorical
       'rate': 5.0  # For Poisson discrete
   }
   ```

4. **Validate Dependencies**
   - Check that parent factors exist
   - Ensure no circular dependencies
   - Verify relationship format

## Factor Types

### 1. Continuous Factors

- Represent continuous variables (e.g., market_size, conversion_rate)
- Can have constraints (lower/upper bounds)
- Use TensorFlow Probability distributions:
  - Normal for unconstrained
  - TruncatedNormal for bounded
  - TransformedDistribution for lower/upper bounded

```python
# Example: Adding a continuous factor
aicon.add_state_factor(
    name='market_size',
    factor_type='continuous',
    value=10000.0,
    params={
        'loc': 10000.0,
        'scale': 1000.0,
        'constraints': {'lower': 0}
    },
    relationships={
        'depends_on': []
    }
)
```

### 2. Categorical Factors

- Represent discrete categories (e.g., competition_level)
- Have a finite set of possible values
- Use Categorical distribution

```python
# Example: Adding a categorical factor
aicon.add_state_factor(
    name='competition_level',
    factor_type='categorical',
    value='medium',
    params={
        'categories': ['low', 'medium', 'high'],
        'probs': [0.2, 0.5, 0.3]
    },
    relationships={
        'depends_on': []
    }
)
```

### 3. Discrete Factors

- Represent integer values (e.g., num_competitors)
- Can be Poisson or Categorical distributed
- Use appropriate distribution parameters

```python
# Example: Adding a Poisson discrete factor
aicon.add_state_factor(
    name='num_competitors',
    factor_type='discrete',
    value=5,
    params={
        'rate': 5.0  # For Poisson distribution
    },
    relationships={
        'depends_on': []
    }
)
```

## Hierarchical Relationships

### 1. Defining Dependencies

Factors can depend on other factors through the `relationships` parameter:

```python
# Example: Conversion rate depends on market_size and competition_level
aicon.add_state_factor(
    name='conversion_rate',
    factor_type='continuous',
    value=0.02,
    params={
        'loc': 0.02,
        'scale': 0.005,
        'constraints': {'lower': 0, 'upper': 1}
    },
    relationships={
        'depends_on': ['market_size', 'competition_level']
    }
)
```

### 2. Joint Distribution

The system automatically creates a hierarchical joint distribution respecting dependencies. The joint distribution is built using TensorFlow Probability's `JointDistributionSequential` with proper topological ordering of factors:

```python
# The joint distribution P(conversion_rate, market_size, competition_level) is created as:
P(conversion_rate, market_size, competition_level) =
    P(market_size) * P(competition_level) * P(conversion_rate | market_size, competition_level)
```

The joint distribution ensures that:

1. Parent factors are sampled before their children
2. Conditional dependencies are properly respected
3. The hierarchical structure is maintained during sampling
4. All factors are properly integrated into the generative model

## Validation Rules

### 1. Relationship Format Validation

The system enforces strict validation of relationship formats:

```python
# This will fail - relationships must be a dictionary
aicon.add_state_factor(
    name='conversion_rate',
    factor_type='continuous',
    value=0.02,
    relationships='market_size'  # Must be a dictionary
)

# This will fail - depends_on must be a list
aicon.add_state_factor(
    name='conversion_rate',
    factor_type='continuous',
    value=0.02,
    relationships={'depends_on': 'market_size'}  # Must be a list
)

# This will work
aicon.add_state_factor(
    name='conversion_rate',
    factor_type='continuous',
    value=0.02,
    relationships={'depends_on': ['market_size']}
)
```

### 2. Parent Factor Existence

The system ensures that parent factors exist before creating dependent factors:

```python
# This will fail - market_size doesn't exist yet
aicon.add_state_factor(
    name='conversion_rate',
    factor_type='continuous',
    value=0.02,
    relationships={'depends_on': ['market_size']}  # market_size doesn't exist yet
)

# This will work
aicon.add_state_factor(
    name='market_size',
    factor_type='continuous',
    value=10000.0,
    relationships={'depends_on': []}
)

aicon.add_state_factor(
    name='conversion_rate',
    factor_type='continuous',
    value=0.02,
    relationships={'depends_on': ['market_size']}  # market_size exists now
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
    relationships={'depends_on': []}  # Root factor
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
    name='competition_level',
    factor_type='categorical',
    value='medium'  # Missing categories and probs
)  # This will fail

# This will work
aicon.add_state_factor(
    name='competition_level',
    factor_type='categorical',
    value='medium',
    params={
        'categories': ['low', 'medium', 'high'],
        'probs': [0.2, 0.5, 0.3]
    },
    relationships={'depends_on': []}
)

# Continuous factors validate numeric values
aicon.add_state_factor(
    name='market_size',
    factor_type='continuous',
    value='not a number'  # Must be numeric
)  # This will fail

# This will work
aicon.add_state_factor(
    name='market_size',
    factor_type='continuous',
    value=10000.0,
    params={
        'loc': 10000.0,
        'scale': 1000.0
    },
    relationships={'depends_on': []}
)
```

### 5. Validation Error Messages

The system provides clear error messages when validation fails:

```python
try:
    aicon.add_state_factor(
        name='conversion_rate',
        factor_type='continuous',
        value=0.02,
        relationships={'depends_on': ['market_size']}  # market_size doesn't exist
    )
except ValueError as e:
    print(e)  # "Cannot create factor 'conversion_rate' because parent factors do not exist: ['market_size']"

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
