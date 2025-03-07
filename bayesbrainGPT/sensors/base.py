from abc import ABC, abstractmethod
import torch
from typing import Dict, Any, List, Tuple, Optional
from ..state_representation.state import EnvironmentState
from ..state_representation.factors import ContinuousFactor, CategoricalFactor, DiscreteFactor

class Sensor(ABC):
    """
    Base interface for all sensors.
    A sensor can:
    1. Actively fetch information when requested (pull)
    2. Passively receive information (push)
    3. Stream continuous information
    4. Have different reliability scores
    """
    def __init__(self, name: str, state: EnvironmentState, reliability: float = 1.0):
        """
        Initialize a sensor
        Args:
            name: Identifier for the sensor
            state: The environment state this sensor observes
            reliability: How reliable is the information from this sensor (0.0 to 1.0)
        """
        self.name = name
        self.state = state
        self.reliability = max(0.0, min(1.0, reliability))  # Clamp between 0 and 1
        self.observable_factors: List[str] = []  # Factors this sensor can observe
        self.streaming: bool = False
        self._setup_observable_factors()
    
    @abstractmethod
    def _setup_observable_factors(self):
        """Define which factors this sensor can observe"""
        pass

    @abstractmethod
    def fetch_data(self) -> Dict[str, torch.Tensor]:
        """
        Actively fetch data from source (pull pattern)
        Example: API call to weather service
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

    def get_data(self) -> Dict[str, Tuple[torch.Tensor, float]]:
        """
        Get current sensor data with reliability scores.
        This is what the brain calls to get information.
        """
        raw_data = self.fetch_data()  # Could also be latest from stream
        return {
            factor: (value, self.reliability) 
            for factor, value in raw_data.items()
            if factor in self.observable_factors
        } 