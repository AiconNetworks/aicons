# AIcons Framework: Tools and Limbs

## Core Concepts

### State Representation

The AIcons framework uses a Bayesian brain-inspired state representation that consists of:

1. **Factor Types**:

   - `ContinuousLatentVariable`: For continuous values (e.g., market_size, conversion_rate)
   - `CategoricalLatentVariable`: For categorical values (e.g., competition_level)
   - `DiscreteLatentVariable`: For discrete values (e.g., num_competitors)

2. **Factor Relationships**:

   - Root factors: Independent factors with no dependencies
   - Dependent factors: Factors that depend on other factors
   - Hierarchical structure: Parent-child relationships between factors

3. **Distribution Parameters**:
   - Continuous: location, scale, and constraints
   - Categorical: categories and probabilities
   - Discrete: either categories/probs or Poisson rate

Example State:

```
State Representation:
market_size: 10000.00 (constraints: {'lower': 0}) (uncertainty: 1000.00)
competition_level: medium (probs: {'low': 0.2, 'medium': 0.5, 'high': 0.3})
conversion_rate: 0.02 (constraints: {'lower': 0, 'upper': 1}) (uncertainty: 0.01)
customer_acquisition_cost: 50.00 (constraints: {'lower': 0}) (uncertainty: 10.00)
num_competitors: 5 (Poisson rate: 5.0)

Factor Relationships:
market_size:
  Root factor (no dependencies)
competition_level:
  Root factor (no dependencies)
conversion_rate:
  Depends on: ['market_size', 'competition_level']
customer_acquisition_cost:
  Depends on: ['competition_level']
num_competitors:
  Root factor (no dependencies)
```

### Tools

A tool in the AIcons framework is an external piece of software that helps the AIcon to change the environment somewhere else. Tools are specific implementations that interact with external services or systems. They are the concrete actions that an AIcon can take to affect change in the world.

Examples of tools:

- Meta Ads Creation Tool (creates ads using Facebook Marketing API)
- Twitter Post Tool (posts tweets using Twitter API)
- Email Sending Tool (sends emails using SMTP)

## BaseAIcon

The BaseAIcon class serves as the foundation for all agent implementations in the AIcons framework. It provides core functionality that all specialized agents can build upon.

### Key Features

- **Bayesian Brain Integration**: Integrates with BayesBrain for probabilistic reasoning
- **State Factor Management**: Handles continuous, categorical, and discrete state factors
- **Hierarchical Modeling**: Supports hierarchical relationships between state factors
- **Sensor Integration**: Connects with sensors for gathering data from the environment
- **Decision Making**: Provides utilities for action space definition and utility optimization

### Core Methods

#### Initialization and Serialization

- `__init__(name, aicon_type="base", capabilities=None)`: Initialize a BaseAIcon with identity and metadata
- `to_dict()`: Convert AIcon to a dictionary representation for serialization
- `from_dict(data)`: Create an AIcon from a dictionary representation
- `save_state(filepath=None, format="json")`: Save the AIcon's state to a file
- `load_state(filepath, format=None)`: Load an AIcon's state from a file

#### State Factor Management

- `add_state_factor(name, factor_type, value, params, relationships=None)`: Unified method for adding any type of state factor
  - `factor_type`: 'continuous', 'categorical', or 'discrete'
  - `params`: Type-specific parameters:
    - For continuous: {'loc': float, 'scale': float, 'constraints': {'lower': float, 'upper': float}}
    - For categorical: {'categories': List[str], 'probs': List[float]}
    - For discrete: {'categories': List[int], 'probs': List[float]} or {'rate': float}
  - `relationships`: Optional hierarchical relationships with other factors
- `define_factor_dependency(name, parent_factors, relation_type="linear", parameters=None, uncertainty=1.0, description="")`: Define a factor that depends on other factors through a specific relationship
- `compile_probabilistic_model()`: Compile all factors and their relationships into a coherent joint distribution for inference

> **Note on State Factor Management**: The AIcons system uses a unified `add_state_factor` method for adding all types of factors. This provides a consistent interface while maintaining the flexibility to handle different factor types through their specific parameters. The method automatically handles the creation of appropriate latent variables based on the factor type and parameters provided.

- `sample_from_prior(n_samples=1)`: Sample from the prior distribution of the hierarchical model
- `get_state(format_nicely=False)`: Get the current state factors and their values

#### Sensor Integration

- `add_sensor(name, sensor=None, factor_mapping=None)`: Add a sensor to collect observations
- `update_from_sensor(sensor_name, environment=None, factor_mapping=None)`: Update beliefs using data from a specific sensor
- `update_from_all_sensors(environment=None)`: Update beliefs using data from all registered sensors

#### Decision Making

- `create_action_space(space_type='budget_allocation', **kwargs)`: Create an action space for decision-making
- `create_utility_function(utility_type='marketing_roi', **kwargs)`: Create a utility function for evaluating actions
- `find_best_action(num_samples=100, use_gradient=False)`: Find the best action based on current state and utility
- `perceive_and_decide(environment=None)`: Process environmental data and make optimal decisions in one step

> **What is "perceive_and_decide"?**: This method implements the Bayesian brain's perception-action cycle:
>
> 1. **Perceive**: First collects sensory information from the environment to update beliefs about the current state of the world
> 2. **Decide**: Then, based on these updated beliefs, evaluates possible actions to find the one with highest expected utility
>
> It's analogous to how humans process sensory information and make decisions. For example, a marketing AI would first perceive current market conditions and then decide how to allocate its budget across different campaigns.

#### State Tracking

- `mark_state_changed()`: Mark the AIcon state as changed, needing persistence
- `mark_brain_changed()`: Mark the brain as changed, needing persistence
- `record_update(source="manual", success=True, metadata=None)`: Record an update to the AIcon's state
- `get_metadata()`: Get basic metadata about this AIcon

### Action Space Types

The framework supports multiple types of action spaces:

- **budget_allocation**: For allocating budget across items
- **marketing**: For marketing campaign optimization
- **time_budget**: For time-based budget allocation
- **multi_campaign**: For multi-campaign budget allocation
- **custom**: For custom action spaces with specific dimensions

### Utility Function Types

Various utility functions are supported:

- **marketing_roi**: For marketing return on investment calculations
- **constrained_marketing_roi**: Marketing ROI with business constraints
- **weighted_sum**: Combination of multiple utility functions
- **multiobjective**: For multi-objective optimization

## Usage Example

```python
from aicons.definitions.base_aicon import BaseAIcon

# Create a marketing optimization agent
marketing_agent = BaseAIcon(
    name="Marketing Optimizer",
    aicon_type="marketing",
    capabilities=["budget_allocation", "ad_optimization"]
)

# Add state factors for marketing context
marketing_agent.add_state_factor(
    name="base_conversion_rate",
    factor_type="continuous",
    value=0.05,
    params={
        "loc": 0.05,
        "scale": 0.01,
        "constraints": {"lower": 0.0, "upper": 1.0}
    },
    relationships={"depends_on": []}
)

marketing_agent.add_state_factor(
    name="season",
    factor_type="categorical",
    value="summer",
    params={
        "categories": ["spring", "summer", "fall", "winter"],
        "probs": [0.25, 0.25, 0.25, 0.25]
    },
    relationships={"depends_on": []}
)

# Add a hierarchical factor
marketing_agent.define_factor_dependency(
    name="expected_roi",
    parent_factors=["base_conversion_rate", "season"],
    relation_type="linear",
    uncertainty=0.2,
    description="Expected ROI considering multiple factors"
)

# Add a sensor to collect marketing data
marketing_agent.add_sensor("marketing_data")

# Create an action space for budget allocation
marketing_agent.create_action_space(
    space_type="budget_allocation",
    total_budget=1000.0,
    items=["ad_1", "ad_2", "ad_3"],
    budget_step=10.0
)

# Create a utility function
marketing_agent.create_utility_function(
    utility_type="marketing_roi",
    revenue_per_conversion=50.0
)

# After defining all factors and their relationships
marketing_agent.compile_probabilistic_model()

# Run the decision cycle
environment_data = {
    "market_condition": "growing",
    "competitor_activity": "low"
}

best_action, expected_utility = marketing_agent.perceive_and_decide(environment_data)
print(f"Best action: {best_action}")
print(f"Expected utility: {expected_utility}")

# Save the state for later
marketing_agent.save_state("marketing_agent_state.json")
```

## Integration with BayesBrain

The BaseAIcon delegates core functionality to the BayesBrain component, which provides:

1. Bayesian inference for updating beliefs
2. Action space representation for decision-making
3. Utility optimization for finding the best actions
4. Perception mechanisms for processing sensor data

This delegation ensures that the BaseAIcon remains focused on providing a clean interface while leveraging the powerful Bayesian reasoning capabilities of BayesBrain.

## Utility Functions and Action Spaces

Utility functions and action spaces are two essential components of the decision-making process in AIcons:

### Action Spaces

- Define the set of possible actions an agent can take
- Structure the dimensions and constraints of decision variables
- Provide a search space for optimization algorithms

### Utility Functions

- Evaluate how "good" an action is given a specific state
- Transform state-action pairs into scalar values
- Enable finding optimal decisions by maximizing expected utility

### Relationship Between Them

The utility function needs to understand the structure of the action space to properly evaluate actions. Therefore:

1. When creating a utility function, it's provided with the action space
2. The utility function stores the dimensions of the action space for later use
3. This creates a dependency where the action space should be created before the utility function

```python
# Correct order of operations
aicon.create_action_space(...)        # First define what actions are possible
aicon.create_utility_function(...)    # Then define how to evaluate actions
best_action = aicon.find_best_action()  # Finally, find the best action
```

Behind the scenes, the utility function's `set_action_space` method is called to give it access to the dimensions it needs to evaluate actions. This coupling ensures that the utility function can correctly interpret and evaluate actions from the defined action space.

## Hierarchical Bayesian Model

The AICON system implements a **Hierarchical Bayesian Model** based on the Bayesian brain hypothesis. In this model:

1. **Everything is Hierarchical**: The entire state representation is structured as a hierarchical model where higher-level factors can influence or explain lower-level factors.

2. **Factors and Relationships**:

   - Basic factors (continuous, categorical, discrete) represent foundational beliefs
   - Hierarchical relationships define how factors influence each other
   - Together they form a complete probabilistic model of the world

3. **Perception as Inference**:

   - Sensory data updates beliefs via Bayesian inference
   - The system recursively updates all dependent factors when new data arrives

4. **Decision Making via Expected Utility**:
   - The hierarchical model produces predictions about outcomes of different actions
   - The utility function evaluates these outcomes based on goals
   - The system selects actions with highest expected utility

### Example of Hierarchical Structure

```
Expected ROI (Top-level factor)
  ├── Conversion Rate (Mid-level factor)
  │     ├── Ad Quality (Base factor)
  │     └── Market Conditions (Base factor)
  └── Cost Structure (Mid-level factor)
        ├── Ad Placement Cost (Base factor)
        └── Seasonality (Base factor)
```

In this example, the Expected ROI depends on both Conversion Rate and Cost Structure, which themselves depend on more fundamental factors. The hierarchical model captures these dependencies mathematically.
