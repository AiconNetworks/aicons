# Decision Making in BayesBrain

## Action Space Creation

When creating an action space in BayesBrain, there are several key components to consider:

### 1. Basic Structure

An action space consists of:

- Dimensions (variables that can be adjusted)
- Constraints (rules that must be satisfied)
- Utility functions (to evaluate actions)

### 2. Creating an Action Space

Action spaces are created through the AIcon class:

```python
# Create an AIcon instance
aicon = AIcon(name="marketing_aicon")

# Create a marketing action space
action_space = aicon.define_action_space(
    space_type='marketing',
    total_budget=1000.0,
    num_ads=2,
    budget_step=100.0,
    ad_names=['google', 'facebook']
)

# Create a budget allocation space
action_space = aicon.define_action_space(
    space_type='budget_allocation',
    total_budget=1000.0,
    num_ads=3,
    budget_step=10.0,
    min_budget=0.0,
    ad_names=['campaign1', 'campaign2', 'campaign3']
)

# Create a time budget space
action_space = aicon.define_action_space(
    space_type='time_budget',
    total_budget=1000.0,
    num_ads=2,
    num_days=7,
    budget_step=10.0,
    min_budget=0.0
)
```

### 3. Types of Constraints

The action space automatically includes constraints based on the type:

#### A. Marketing Space Constraints

```python
# When using space_type='marketing':
# 1. Total budget constraint
# 2. Minimum budget per ad constraint
# 3. Budget step size constraint
```

#### B. Budget Allocation Space Constraints

```python
# When using space_type='budget_allocation':
# 1. Total budget constraint
# 2. Minimum budget per item constraint
# 3. Budget step size constraint
```

#### C. Time Budget Space Constraints

```python
# When using space_type='time_budget':
# 1. Total budget constraint
# 2. Minimum budget per ad per day constraint
# 3. Budget step size constraint
```

### 4. When Constraints Are Checked

Constraints are automatically checked in several scenarios:

1. **Sampling Actions**

   - When calling `action_space.sample()`
   - Ensures sampled actions are valid

2. **Validating Actions**

   - When calling `action_space.contains(action)`
   - Checks if a specific action satisfies all constraints

3. **Evaluating Actions**
   - When calling `action_space.evaluate_actions(actions)`
   - Validates actions before computing their utility

### 5. Best Practices

1. **Always Define Through AIcon**

   - Use `aicon.define_action_space()` instead of creating ActionSpace directly
   - This ensures proper integration with the AIcon's brain

2. **Choose Appropriate Space Type**

   - Use 'marketing' for simple ad budget allocation
   - Use 'budget_allocation' for general budget distribution
   - Use 'time_budget' for time-based budget allocation

3. **Set Reasonable Parameters**

   - Set appropriate budget_step for your use case
   - Consider min_budget based on your requirements
   - Use meaningful ad_names for better readability

4. **Test the Space**
   - Sample actions to verify constraints
   - Check that valid actions are accepted
   - Ensure invalid actions are rejected

### 6. Example Usage

```python
# Create an AIcon
aicon = AIcon(name="marketing_aicon")

# Create a marketing action space
action_space = aicon.define_action_space(
    space_type='marketing',
    total_budget=1000.0,
    num_ads=2,
    budget_step=100.0,
    ad_names=['google', 'facebook']
)

# Print the action space
print(action_space.pprint())  # Human-readable format
print(action_space.raw_print())  # Mathematical format

# Sample a valid action
action = action_space.sample()

# Check if an action is valid
is_valid = action_space.contains(action)

# Evaluate multiple actions
actions = [action_space.sample() for _ in range(5)]
utilities = action_space.evaluate_actions(actions)
```

### 7. Printing Action Spaces

There are several ways to print information about an action space:

#### A. Default Representation

```python
print(action_space)  # Shows basic info: ActionSpace(dimensions=2, constraints=2)
```

#### B. Human-Readable Format

```python
print(action_space.pprint())
# Output example:
# Action Space:
# - Dimensions:
#   * google_budget: continuous [0.0, 1000.0] step=100.0
#   * facebook_budget: continuous [0.0, 1000.0] step=100.0
# - Constraints:
#   * Total budget: sum = 1000.0
#   * Min budget per ad: 0.0
#   * Custom constraint: google_budget >= 300.0
# - Sample Actions:
#   * [500.0, 500.0]
#   * [300.0, 700.0]
```

#### C. Raw Format

```python
print(action_space.raw_print())
# Output example:
# dimensions: [
#   'google_budget: [0.0, 1000.0] step=100.0',
#   'facebook_budget: [0.0, 1000.0] step=100.0'
# ]
# constraints: [
#   {'type': 'total_budget', 'dimensions': [0, 1], 'target': 1000.0}
# ]
# actions: [
#   {'google_budget': 0.0, 'facebook_budget': 1000.0},
#   {'google_budget': 100.0, 'facebook_budget': 900.0},
#   {'google_budget': 200.0, 'facebook_budget': 800.0},
#   ...
# ]
```

#### D. Getting Specific Information

```python
# Get dimension names
print(action_space.get_dimension_names())  # ['google_budget', 'facebook_budget']

# Get number of dimensions
print(len(action_space.dimensions))  # 2

# Get number of constraints
print(len(action_space.constraints))  # 3

# Get sample actions
print(action_space.sample())  # [500.0, 500.0]
print(action_space.sample())  # [300.0, 700.0]
```

Choose the printing method based on your needs:

- Use `pprint()` for human-readable output with examples
- Use `raw_print()` for raw data structure output
- Use specific getter methods for particular pieces of information
