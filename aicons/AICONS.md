# Aicon Framework: Tools

## Core Concepts

### State Representation

The AIcons framework uses a Bayesian brain-inspired state representation that consists of:

1. **Factor Types**:

   - `ContinuousLatentVariable`: For continuous values (e.g., market_size, conversion_rate)
   - `CategoricalLatentVariable`: For categorical values (e.g., competition_level)
   - `DiscreteLatentVariable`: For discrete values (e.g., num_competitors)
   - `HierarchicalLatentVariable`: A latent variable with hierarchical dependencies on other latent variables.

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

## BaseAicon

The BaseAicon class serves as the foundation for all agent implementations in the AIcons framework. It provides core functionality that all specialized agents can build upon.

### Key Features

- **Bayesian Brain Integration**: Integrates with BayesBrain for probabilistic reasoning
- **State Factor Management**: Handles continuous, categorical, and discrete state factors
- **Hierarchical Modeling**: Supports hierarchical relationships between state factors
- **Sensor Integration**: Connects with sensors for gathering data from the environment
- **Decision Making**: Provides utilities for action space definition and utility optimization

## Zero Model

The Zero model is a simplified version of the AIcon system that focuses on definition and storage rather than active computation. It serves as a foundational model for defining and storing key components without the full computational capabilities of the main AIcon system.

### Key Characteristics

- **Definition-Focused**: Primarily used for defining and storing components rather than active computation
- **Memory Storage**: Maintains state in the form of priors and posteriors
- **Core Components**:
  - Action space definitions
  - Utility function specifications
  - Prior and posterior distributions
  - State factor definitions

### Differences from Full AIcon System

While the full AIcon system includes active decision-making and sensor integration, the Zero model:

- Does not include decision-making capabilities
- Lacks sensor integration
- Focuses on storing definitions and distributions rather than active computation
- Serves as a reference model for defining the structure of more complex AIcon implementations

The Zero model is particularly useful for:

- Defining the structure of more complex AIcon implementations
- Storing and managing state definitions
- Maintaining reference distributions and utility functions
- Providing a foundation for building more sophisticated AIcon systems
