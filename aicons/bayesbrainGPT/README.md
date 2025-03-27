# bayesbrainGPT

## State Representation

### Factor Structure

The state representation now uses a unified factor structure with explicit relationships:

1. **Factor Types**:

   - `ContinuousLatentVariable`: For continuous values with constraints and uncertainty
   - `CategoricalLatentVariable`: For categorical values with prior probabilities
   - `DiscreteLatentVariable`: For discrete values with either categorical or Poisson distributions

2. **Factor Relationships**:

   - Each factor has an explicit `relationships` dictionary with `depends_on` list
   - Root factors have empty `depends_on` list
   - Dependent factors list their parent factors in `depends_on`

3. **Distribution Parameters**:
   - Continuous: `loc`, `scale`, and `constraints` (e.g., lower/upper bounds)
   - Categorical: `categories` and `probs` for prior probabilities
   - Discrete: either `categories`/`probs` or `rate` for Poisson

Example:

```python
# Root factor (continuous)
add_state_factor(
    name="market_size",
    factor_type="continuous",
    value=10000.0,
    params={
        "loc": 10000.0,
        "scale": 1000.0,
        "constraints": {"lower": 0}
    },
    relationships={
        "depends_on": []  # Empty list for root factor
    }
)

# Dependent factor (continuous)
add_state_factor(
    name="conversion_rate",
    factor_type="continuous",
    value=0.02,
    params={
        "loc": 0.02,
        "scale": 0.005,
        "constraints": {"lower": 0, "upper": 1}
    },
    relationships={
        "depends_on": ["market_size", "competition_level"]
    }
)
```

### LLM Integration

Gemini for factor extraction. We are getting all the values from gemini LLM.
In a Bayesian brain–inspired system, the state representation is refreshed or reinitialized based on certain triggers or intervals. For your marketing analysis agent, you would typically update the state when:

Initial Startup:
When the agent first starts, it creates the full state representation from the priors and any initial sensor data (e.g., historical data from Meta, initial impressions, etc.).

Periodic Updates:
For example, at the beginning of each day (or at any predefined interval), the agent refreshes its state with the latest sensor data—like the day of the week, yesterday's impressions, add-to-cart numbers, etc. This ensures that the decision-making (e.g., choosing which ad to publish or how much budget to allocate) is based on the most current context.

Event-Driven Triggers:
The state can be updated whenever a significant change in sensor data is detected—for instance, if there's a sudden spike or drop in engagement, conversion rates, or any other key metric. This could be determined by predefined thresholds or "surprise" detection mechanisms.

Feedback After Action:
After the agent publishes an ad and collects outcomes (like performance metrics), that feedback is used to update the state further. This continuous learning loop adjusts the priors over time.

### Action Triggering

The system supports two main ways to trigger action making:

1. **Manual Triggering (AIcon)**

   - Direct call to action making from AIcon
   - Useful for scheduled or on-demand decisions
   - Example: Daily budget allocation at specific times

2. **Sensor-Based Triggering**
   - Automatic triggering based on sensor data fluctuations
   - Can be selective (not all sensors trigger actions)
   - Example: Sudden drop in conversion rate triggering budget reallocation

Example of manual triggering:

```python
# In AIcon
def make_decision(self):
    """
    Manually trigger action making.
    """
    # Get current state
    current_state = self.get_state()

    # Calculate expected utilities
    expected_utilities = self.calculate_expected_utilities()

    # Choose best action
    best_action = self.select_best_action(expected_utilities)

    return best_action

# Usage
action = aicon.make_decision()
```

Example of sensor-based triggering:

```python
# In AIcon
def on_sensor_update(self, sensor_name, data):
    """
    Handle sensor updates and potentially trigger actions.
    """
    # Update perception with new sensor data
    self.perception.update_from_sensor(sensor_name, data)

    # Check if this sensor should trigger action
    if self.should_trigger_action(sensor_name, data):
        return self.make_decision()

    return None
```

The system allows for flexible configuration of which sensors trigger actions and under what conditions, while maintaining the ability to manually trigger decisions when needed.
