"""
TensorFlow-compatible sensors for Bayesian inference.

This module provides sensor classes that work with TensorFlow Probability for
collecting observations about state factors.
"""

from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union

TensorType = Union[tf.Tensor, np.ndarray, float, int, str]
ObservationType = Tuple[TensorType, float]  # (value, reliability)

class TFSensor(ABC):
    """
    Base class for TensorFlow-compatible sensors.
    
    A sensor can:
    1. Actively fetch information when requested (pull)
    2. Passively receive information (push)
    3. Stream continuous information
    4. Have different reliability scores for different factors
    5. Map between sensor factor names and state factor names
    """
    def __init__(self, name: str, reliability: float = 1.0, factor_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize a sensor.
        
        Args:
            name: Identifier for the sensor
            reliability: Default reliability for this sensor (0.0 to 1.0)
            factor_mapping: Optional mapping from sensor factor names to state factor names
                           Example: {"base_conversion_rate": "conversion_rate"}
        """
        self.name = name
        self.default_reliability = max(0.0, min(1.0, reliability))
        self.observable_factors: List[str] = []  # Factors this sensor can observe
        self.factor_reliabilities: Dict[str, float] = {}  # Reliability for each factor
        self.streaming: bool = False
        self.latest_data: Dict[str, TensorType] = {}
        self.factor_mapping: Dict[str, str] = factor_mapping or {}  # Initialize factor mapping
        
        # Set up which factors this sensor can observe
        self._setup_observable_factors()
    
    @abstractmethod
    def _setup_observable_factors(self):
        """Define which factors this sensor can observe and their reliabilities."""
        pass

    @abstractmethod
    def fetch_data(self, environment: Any = None) -> Dict[str, TensorType]:
        """
        Actively fetch data from source (pull pattern).
        
        Args:
            environment: Optional environment data to use for fetching
            
        Returns:
            Dictionary mapping factor names to observed values
        """
        pass

    def receive_data(self, data: Dict[str, TensorType]):
        """
        Receive data from external source (push pattern).
        
        Args:
            data: Dictionary mapping factor names to observed values
        """
        # Store data for factors this sensor can observe
        self.latest_data.update({
            factor: value for factor, value in data.items()
            if factor in self.observable_factors
        })

    def start_streaming(self):
        """Start continuous data stream."""
        self.streaming = True

    def stop_streaming(self):
        """Stop continuous data stream."""
        self.streaming = False
        
    def add_factor_mapping(self, sensor_factor_name: str, state_factor_name: str):
        """
        Add a mapping between a sensor factor name and a state factor name
        
        Args:
            sensor_factor_name: The name of the factor as known by the sensor
            state_factor_name: The name of the factor as known by the state/brain
        """
        self.factor_mapping[sensor_factor_name] = state_factor_name
        print(f"Sensor '{self.name}' added mapping: {sensor_factor_name} → {state_factor_name}")
    
    def _map_factor_name(self, sensor_factor_name: str) -> str:
        """
        Map a sensor factor name to a state factor name using the mapping.
        
        Args:
            sensor_factor_name: Factor name from the sensor
            
        Returns:
            Corresponding state factor name, or the input name if no mapping exists
        """
        return self.factor_mapping.get(sensor_factor_name, sensor_factor_name)

    def get_data(self, environment: Any = None) -> Dict[str, ObservationType]:
        """
        Get current sensor data with reliability scores.
        This is what the perception system calls to get information.
        
        Args:
            environment: Optional environment data to use for fetching
            
        Returns:
            Dictionary mapping factor names to (value, reliability) tuples
        """
        # If streaming, use latest data, otherwise fetch new data
        data = self.latest_data if self.streaming else self.fetch_data(environment)
        
        # Apply factor mapping and add reliability scores
        mapped_data = {}
        for factor, value in data.items():
            if factor in self.observable_factors:
                # Map the sensor factor name to state factor name
                state_factor_name = self._map_factor_name(factor)
                mapped_data[state_factor_name] = (value, self.factor_reliabilities.get(factor, self.default_reliability))
        
        return mapped_data


class MarketingSensor(TFSensor):
    """
    Sensor for marketing campaign data.
    
    This sensor can observe:
    - conversion_rate: Continuous factor for conversion rate
    - best_channel: Categorical factor for best marketing channel
    - ad_count: Discrete factor for number of ads
    """
    def __init__(self, name: str = "marketing", reliability: float = 0.8, factor_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize a marketing sensor.
        
        Args:
            name: Identifier for the sensor
            reliability: Default reliability for this sensor (0.0 to 1.0)
            factor_mapping: Optional mapping from sensor factor names to state factor names
        """
        super().__init__(name, reliability, factor_mapping)
        
    def _setup_observable_factors(self):
        """Define which factors this sensor can observe and their reliabilities."""
        # Factors this sensor can observe
        self.observable_factors = [
            "base_conversion_rate",
            "primary_channel",
            "optimal_daily_ads"
        ]
        
        # Reliability for each factor (can vary by factor)
        self.factor_reliabilities = {
            "base_conversion_rate": 0.9,  # High reliability for conversion rate
            "primary_channel": 0.7,       # Medium reliability for channel attribution
            "optimal_daily_ads": 0.8      # Good reliability for ad count
        }
    
    def fetch_data(self, environment: Any = None) -> Dict[str, TensorType]:
        """
        Fetch marketing campaign data.
        
        In a real application, this would connect to ad platforms like
        Facebook Ads, Google Ads, etc. For simulation, we generate
        realistic data with noise.
        
        Args:
            environment: Optional environment data providing true values
            
        Returns:
            Dictionary mapping factor names to observed values
        """
        # In a real application, we would fetch this data from ad platforms
        # Here we simulate it with realistic values and noise
        
        # True values (from environment if provided, otherwise defaults)
        if environment is not None and isinstance(environment, dict):
            true_values = environment
        else:
            # Default true values for simulation
            true_values = {
                "base_conversion_rate": 0.063,  # 6.3% conversion rate
                "primary_channel": "google",    # Google is best channel
                "optimal_daily_ads": 8          # 8 ads per day is optimal
            }
        
        # Add realistic noise to observations
        observations = {}
        
        # Add noise to conversion rate
        if "base_conversion_rate" in self.observable_factors:
            true_rate = true_values.get("base_conversion_rate", 0.05)
            conv_noise = np.random.normal(0, 0.005)  # Small Gaussian noise
            observed_rate = min(max(true_rate + conv_noise, 0), 1)
            observations["base_conversion_rate"] = float(observed_rate)
            
        # Add noise to best channel
        if "primary_channel" in self.observable_factors:
            true_channel = true_values.get("primary_channel", "facebook")
            channel_options = ["facebook", "google", "tiktok", "instagram"]
            
            # 80% chance of correct observation, 20% chance of error
            if np.random.random() < 0.8:
                observations["primary_channel"] = true_channel
            else:
                # Pick a random incorrect channel
                other_channels = [c for c in channel_options if c != true_channel]
                observations["primary_channel"] = np.random.choice(other_channels)
                
        # Add noise to ad count
        if "optimal_daily_ads" in self.observable_factors:
            true_count = true_values.get("optimal_daily_ads", 5)
            # Add Poisson noise to count (typical for count data)
            count_noise = np.random.poisson(1) - 1  # -1, 0, 1, 2...
            observed_count = max(true_count + count_noise, 1)  # At least 1 ad
            observations["optimal_daily_ads"] = int(observed_count)
            
        return observations


class WeatherSensor(TFSensor):
    """
    Sensor for weather data.
    
    This sensor can observe:
    - weather: Categorical factor for weather condition
    - temperature: Continuous factor for temperature
    """
    def __init__(self, name: str = "weather_station", reliability: float = 0.9, factor_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize a weather sensor.
        
        Args:
            name: Identifier for the sensor
            reliability: Default reliability for this sensor (0.0 to 1.0)
            factor_mapping: Optional mapping from sensor factor names to state factor names
        """
        super().__init__(name, reliability, factor_mapping)
        
    def _setup_observable_factors(self):
        """Define which factors this sensor can observe and their reliabilities."""
        # Factors this sensor can observe
        self.observable_factors = [
            "weather",
            "temperature"
        ]
        
        # Reliability for each factor
        self.factor_reliabilities = {
            "weather": 0.8,      # Good reliability for weather condition
            "temperature": 0.95  # Very high reliability for temperature
        }
    
    def fetch_data(self, environment: Any = None) -> Dict[str, TensorType]:
        """
        Fetch weather data.
        
        In a real application, this would connect to a weather API.
        For simulation, we generate realistic data with noise.
        
        Args:
            environment: Optional environment data providing true values
            
        Returns:
            Dictionary mapping factor names to observed values
        """
        # In a real application, we would fetch from a weather API
        # Here we simulate it with realistic values and noise
        
        # True values (from environment if provided, otherwise defaults)
        if environment is not None and isinstance(environment, dict):
            true_values = environment
        else:
            # Default true values for simulation
            true_values = {
                "weather": "cloudy",   # True weather condition
                "temperature": 15.0    # True temperature in Celsius
            }
        
        # Add realistic noise to observations
        observations = {}
        
        # Add noise to weather condition
        if "weather" in self.observable_factors:
            true_weather = true_values.get("weather", "clear")
            conditions = ["clear", "cloudy", "stormy"]
            
            # Sometimes we might observe the wrong condition
            weather_noise = np.random.random()
            if weather_noise < self.factor_reliabilities.get("weather", 0.8):
                # Correct observation
                observations["weather"] = true_weather
            else:
                # Observation error - randomly pick a different condition
                other_conditions = [c for c in conditions if c != true_weather]
                observations["weather"] = np.random.choice(other_conditions)
        
        # Add noise to temperature
        if "temperature" in self.observable_factors:
            true_temp = true_values.get("temperature", 20.0)
            temp_noise = np.random.normal(0, 0.5)  # Small Gaussian noise
            observed_temp = true_temp + temp_noise
            observations["temperature"] = float(observed_temp)
        
        return observations 