from abc import ABC, abstractmethod
import torch
from typing import Dict, Any

class Sensor(ABC):
    """Base class for all sensors"""
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def get_data(self) -> Dict[str, torch.Tensor]:
        """Get observation data from the sensor"""
        pass

class WeatherSensor(Sensor):
    """Sensor for weather-related observations"""
    def __init__(self):
        super().__init__("weather_sensor")
        
    def get_data(self) -> Dict[str, torch.Tensor]:
        # Simulate getting weather data
        return {
            "T_obs": torch.tensor(25.0),
            "r_obs": torch.tensor(12.0)
        }

class TrafficSensor(Sensor):
    """Sensor for traffic-related observations"""
    def __init__(self):
        super().__init__("traffic_sensor")
        
    def get_data(self) -> Dict[str, torch.Tensor]:
        # Simulate getting traffic data
        return {
            "d_obs": torch.tensor(2.0)
        } 