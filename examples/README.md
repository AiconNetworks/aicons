# BayesBrain Examples

This directory contains examples demonstrating how to use the BayesBrain module within the BadAIcon framework.

## Examples

### 1. Empty BayesBrain Example (`empty_bayes_brain_example.py`)

This example demonstrates how to create a BadAIcon with an empty BayesBrain and use it with minimal configuration. It shows:

- Creating a BadAIcon with an empty BayesBrain
- Checking the initial state of the BayesBrain
- Adding campaigns
- Setting up a minimal action space
- Sampling allocations from a minimally configured BayesBrain
- Setting a simple utility function
- Finding the best allocation with minimal configuration

Run this example with:

```bash
python examples/empty_bayes_brain_example.py
```

### 2. Manual BayesBrain Configuration Example (`manual_bayes_brain_example.py`)

This more comprehensive example demonstrates how to create a BadAIcon with an empty BayesBrain and then manually configure each component step by step. It shows:

- Creating a BadAIcon with an empty BayesBrain
- Checking the initial state of the BayesBrain
- Manually adding campaigns
- Manually creating and setting the action space
- Manually setting state factors
- Manually creating and setting posterior samples
- Manually creating and setting a utility function
- Manually adding sensors
- Sampling allocations from the manually configured BayesBrain
- Finding the best allocation using the utility function

Run this example with:

```bash
python examples/manual_bayes_brain_example.py
```

## Key Concepts

### BayesBrain Components

The BayesBrain consists of several key components:

1. **Action Space**: Defines the possible actions the AIcon can take
2. **State Factors**: Represents the current state of the world
3. **Posterior Samples**: Probabilistic beliefs about the world
4. **Utility Function**: Evaluates the expected utility of actions
5. **Sensors**: Collect data from the environment

### Minimal Configuration

At minimum, a BayesBrain needs:

- An action space to define possible actions
- A utility function to evaluate actions

With just these components, the BayesBrain can sample random actions and find the best action based on the utility function.

### Full Configuration

For full functionality, a BayesBrain should have:

- An action space to define possible actions
- State factors to represent the current state
- Posterior samples to represent beliefs about the world
- A utility function to evaluate actions
- Sensors to collect data from the environment

With all components configured, the BayesBrain can perform Bayesian inference, update its beliefs based on new data, and make decisions that maximize expected utility.
