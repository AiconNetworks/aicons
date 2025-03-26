# Decision Making in BayesBrain

## Action Space Creation

When creating an action space in BayesBrain, there are several key components to consider:

### 1. Basic Structure

An action space consists of:

- Dimensions (variables that can be adjusted)
- Constraints (rules that must be satisfied)
- Utility functions (to evaluate actions)

### 2. Creating an Action Space

There are two main ways to create an action space:

#### A. Using Factory Functions

```python
# Example: Marketing Ads Space
action_space = create_marketing_ads_space(
    total_budget=1000.0,
    num_ads=2,
    budget_step=100.0,
    ad_names=['google', 'facebook']
)

# Example: Budget Allocation Space
action_space = create_budget_allocation_space(
    total_budget=1000.0,
    num_ads=3,
    budget_step=10.0,
    min_budget=0.0,
    ad_names=['campaign1', 'campaign2', 'campaign3']
)

# Example: Time Budget Space
action_space = create_time_budget_allocation_space(
    total_budget=1000.0,
    num_ads=2,
    num_days=7,
    budget_step=10.0,
    min_budget=0.0
)
```

#### B. Creating Custom Action Space

```python
# Define dimensions
dimensions = [
    ActionDimension(
        name="ad1_budget",
        dim_type="continuous",
        min_value=0.0,
        max_value=1000.0,
        step=10.0
    ),
    ActionDimension(
        name="ad2_budget",
        dim_type="continuous",
        min_value=0.0,
        max_value=1000.0,
        step=10.0
    )
]

# Define constraints
constraints = [
    # Dictionary constraint for total budget
    {
        "type": "total_budget",
        "dimensions": [0, 1],
        "target": 1000.0
    },
    # Function constraint for minimum budget
    lambda x: x[0] >= 200.0  # First ad must get at least 200
]

# Create the action space
action_space = ActionSpace(
    dimensions=dimensions,
    constraints=constraints
)
```

### 3. Types of Constraints

#### A. Dictionary Constraints

These are predefined constraint types that are checked using the `check_constraints` method:

1. **Total Budget Constraint**

```python
{
    "type": "total_budget",
    "dimensions": list(range(num_ads)),
    "target": total_budget
}
```

- Ensures the sum of all ad budgets equals the specified total budget

2. **Minimum Budget Constraint**

```python
{
    "type": "min_budget",
    "dimensions": list(range(num_ads)),
    "target": min_budget
}
```

- Ensures each ad gets at least the minimum budget specified

#### B. Function Constraints

These are custom functions that take an action array and return a boolean:

```python
# Example: Ensure first ad gets at least 200
lambda x: x[0] >= 200.0

# Example: Ensure second ad gets at least 30% of total
lambda x: x[1] >= 0.3 * sum(x)

# Example: Ensure ads are within 20% of each other
lambda x: max(x) <= 1.2 * min(x)
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

1. **Always Define Constraints**

   - Even if using factory functions, review the default constraints
   - Add custom constraints if needed

2. **Use Appropriate Constraint Types**

   - Use dictionary constraints for common cases (budget limits)
   - Use function constraints for complex rules

3. **Test Constraints**

   - Verify constraints work as expected
   - Check that valid actions are accepted
   - Ensure invalid actions are rejected

4. **Document Constraints**
   - Add comments explaining complex constraints
   - Document any assumptions about the constraints

### 6. Example Usage

```python
# Create a marketing action space
action_space = create_marketing_ads_space(
    total_budget=1000.0,
    num_ads=2,
    budget_step=100.0,
    ad_names=['google', 'facebook']
)

# Add a custom constraint
action_space.constraints.append(
    lambda x: x[0] >= 300.0  # Google ad must get at least 300
)

# Sample a valid action
action = action_space.sample()

# Check if an action is valid
is_valid = action_space.contains(action)

# Evaluate multiple actions
actions = [action_space.sample() for _ in range(5)]
utilities = action_space.evaluate_actions(actions)
```
