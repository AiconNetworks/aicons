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
