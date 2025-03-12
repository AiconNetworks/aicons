from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional

class Sensor(ABC):
    """
    Base interface for all sensors.
    A sensor can:
    1. Actively fetch information when requested (pull)
    2. Passively receive information (push)
    3. Stream continuous information
    4. Have different reliability scores
    5. Map between sensor factor names and state factor names
    """
    def __init__(self, name: str, reliability: float = 1.0, factor_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize a sensor
        Args:
            name: Identifier for the sensor
            reliability: How reliable is the information from this sensor (0.0 to 1.0)
            factor_mapping: Optional mapping from sensor factor names to state factor names
                           Example: {"temperature_celsius": "temperature"}
        """
        self.name = name
        self.reliability = max(0.0, min(1.0, reliability))  # Clamp between 0 and 1
        self.observable_factors: List[str] = []  # Factors this sensor can observe
        self.streaming: bool = False
        self.factor_mapping: Dict[str, str] = factor_mapping or {}  # Initialize factor mapping
        self._setup_observable_factors()
    
    @abstractmethod
    def _setup_observable_factors(self):
        """Define which factors this sensor can observe"""
        pass

    @abstractmethod
    def fetch_data(self, environment: Any = None) -> Dict[str, Any]:
        """
        Actively fetch data from source (pull pattern)
        Example: API call to weather service
        
        Args:
            environment: Optional environment data to use for fetching
            
        Returns:
            Dictionary mapping factor names to observed values
        """
        pass

    @abstractmethod
    def receive_data(self, data: Dict[str, Any]):
        """
        Receive data from external source (push pattern)
        Example: Webhook from a service
        """
        pass

    def start_streaming(self):
        """
        Start continuous data stream
        Example: Real-time sensor readings
        """
        self.streaming = True

    def stop_streaming(self):
        """Stop continuous data stream"""
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

    def get_data(self, environment: Any = None) -> Dict[str, Tuple[Any, float]]:
        """
        Get current sensor data with reliability scores.
        This is what the brain calls to get information.
        
        Args:
            environment: Optional environment data to use for fetching
            
        Returns:
            Dictionary mapping state factor names to (value, reliability) tuples
        """
        raw_data = self.fetch_data(environment)  # Pass environment to fetch_data
        
        # Apply factor mapping and filter by observable factors
        mapped_data = {}
        for sensor_factor_name, value in raw_data.items():
            if sensor_factor_name in self.observable_factors:
                # Map the sensor factor name to state factor name
                state_factor_name = self._map_factor_name(sensor_factor_name)
                mapped_data[state_factor_name] = (value, self.reliability)
        
        return mapped_data 