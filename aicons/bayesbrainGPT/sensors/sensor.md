# Sensors in BayesBrain

## Overview

Sensors are the primary interface between the external world and the BayesBrain system. They provide observations about various state factors that the brain uses to maintain and update its beliefs about the world.

## Core Concepts

### What is a Sensor?

A sensor is a component that can:

1. Actively fetch information when requested (pull pattern)
2. Passively receive information (push pattern)
3. Stream continuous information
4. Have different reliability scores for different factors
5. Map between sensor factor names and state factor names

### Sensor Types

1. **Base Sensor Class**

   - Abstract base class that defines the interface for all sensors
   - Handles common functionality like reliability and factor mapping
   - Provides methods for data fetching and observation

2. **Specialized Sensors**
   - MarketingSensor: For marketing campaign data
   - WeatherSensor: For weather data
   - MetaAdsSalesSensor: For Meta (Facebook) ad campaign data
   - Custom sensors can be created by inheriting from the base Sensor class

## Sensor-Brain Integration

### Data Flow

1. **Observation Collection**

   - Sensors collect raw data from their sources
   - Data is formatted into (value, reliability) tuples
   - Factor names are mapped to state factor names

2. **Perception Processing**

   - The brain's perception system receives sensor data
   - Updates beliefs using Bayesian inference
   - Maintains uncertainty through reliability scores

3. **State Updates**
   - Updated beliefs are stored in the brain's state
   - State factors are updated with new values and uncertainties
   - Posterior distributions are maintained

### Factor Management

- Sensors define expected factors through `get_expected_factors()`
- The brain automatically creates missing factors
- Factor types (continuous, categorical, discrete) are validated
- Factor mappings allow flexible naming conventions

## Implementation Details

### Sensor Interface

```python
class Sensor(ABC):
    def __init__(self, name: str, reliability: float = 1.0, factor_mapping: Optional[Dict[str, str]] = None):
        self.name = name
        self.default_reliability = reliability
        self.factor_mapping = factor_mapping

    @abstractmethod
    def _setup_observable_factors(self):
        """Define which factors this sensor can observe and their reliabilities."""
        pass

    @abstractmethod
    def fetch_data(self, environment: Any = None) -> Dict[str, TensorType]:
        """Actively fetch data from source."""
        pass

    def get_data(self, environment: Any = None) -> Dict[str, ObservationType]:
        """Get data with reliability scores."""
        pass

    def get_expected_factors(self) -> Dict[str, Dict[str, Any]]:
        """Return information about expected factors."""
        pass
```

### Key Methods

1. **fetch_data()**

   - Pulls data from external sources
   - Returns raw observations

2. **get_data()**

   - Returns observations with reliability scores
   - Handles factor name mapping

3. **get_expected_factors()**
   - Defines factor structure for auto-creation
   - Specifies types, bounds, and default values

## Reliability System

The sensor system implements a dual-level reliability scoring:

1. **Sensor-Level Reliability**

   - Each sensor has a `default_reliability` (0.0 to 1.0)
   - Set when creating the sensor: `reliability: float = 1.0`
   - Used as a fallback when factor-specific reliability is not defined

2. **Factor-Level Reliability**
   - Each sensor maintains a `factor_reliabilities` dictionary
   - Maps specific factors to their individual reliability scores
   - Set up in the `_setup_observable_factors()` method
   - Takes precedence over sensor-level reliability

Example:

```python
class CustomSensor(Sensor):
    def __init__(self, name: str = "custom", reliability: float = 0.8):
        super().__init__(name, reliability)  # Sets default_reliability = 0.8

    def _setup_observable_factors(self):
        self.observable_factors = ["factor1", "factor2"]
        # Set specific reliabilities for each factor
        self.factor_reliabilities = {
            "factor1": 0.9,  # More reliable factor
            "factor2": 0.7   # Less reliable factor
        }
```

When getting data, the system:

1. First checks for factor-specific reliability
2. Falls back to sensor-level reliability if not found
3. Returns (value, reliability) tuples for each observation

## Usage Examples

### Creating a Custom Sensor

```python
class CustomSensor(Sensor):
    def _setup_observable_factors(self):
        self.observable_factors = ["factor1", "factor2"]
        self.factor_reliabilities = {
            "factor1": 0.9,
            "factor2": 0.8
        }

    def fetch_data(self, environment=None):
        # Implement data fetching logic
        return {
            "factor1": 42.0,
            "factor2": "category_a"
        }

    def get_expected_factors(self):
        return {
            "factor1": {
                "type": "continuous",
                "default_value": 0.0,
                "uncertainty": 1.0
            },
            "factor2": {
                "type": "categorical",
                "default_value": "category_a",
                "categories": ["category_a", "category_b"]
            }
        }
```

### Adding a Sensor to the Brain

```python
# Create sensor
sensor = CustomSensor(
    name="custom_sensor",
    reliability=0.85,  # Default reliability for factors without specific reliability
    factor_mapping={"factor1": "state_factor1"}
)

# Add to brain
brain.add_sensor("custom_sensor", sensor)
```

## Best Practices

1. **Reliability Scoring**

   - Use appropriate reliability scores (0.0 to 1.0)
   - Set sensor-level reliability as a reasonable default
   - Override with factor-specific reliabilities when needed
   - Document reliability assumptions and rationale

2. **Factor Mapping**

   - Use clear, consistent naming conventions
   - Document factor mappings
   - Handle missing or invalid mappings gracefully

3. **Error Handling**

   - Implement robust error handling
   - Provide meaningful error messages
   - Handle missing or invalid data gracefully

4. **Performance**
   - Optimize data fetching
   - Cache data when appropriate
   - Handle streaming data efficiently

## Common Issues and Solutions

1. **Missing Factors**

   - Ensure `get_expected_factors()` is implemented
   - Check factor type compatibility
   - Validate factor constraints

2. **Data Format Issues**

   - Validate data types
   - Handle missing values
   - Convert data to appropriate formats

3. **Reliability Issues**
   - Validate reliability scores
   - Handle zero reliability cases
   - Consider temporal reliability changes

## Future Improvements

1. **Planned Enhancements**

   - Streaming data support
   - Real-time reliability updates
   - Advanced factor mapping

2. **Integration Features**

   - Better error reporting
   - Performance monitoring
   - Debugging tools

3. **Documentation**
   - More examples
   - API documentation
   - Troubleshooting guides
