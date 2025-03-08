from typing import Dict, Any, Optional
import torch
from .base import Sensor
from ..state_representation.state import EnvironmentState
from ..state_representation.factors import DiscreteFactor

class StreamingTrafficSensor(Sensor):
    """Traffic sensor with continuous data stream"""
    def __init__(self, state: EnvironmentState, reliability: float = 0.95):
        super().__init__("traffic_stream", state, reliability)
        self._latest_data: Optional[Dict[str, torch.Tensor]] = None
        
    def _setup_observable_factors(self):
        """Define which traffic factors this sensor can observe"""
        self.observable_factors = [
            name for name, factor in self.state.factors.items()
            if name in ["traffic_density", "average_speed", "vehicle_count"]
        ]
    
    def fetch_data(self) -> Dict[str, torch.Tensor]:
        """Get current traffic data"""
        if self.streaming and self._latest_data is not None:
            return self._latest_data
            
        # Mock data when not streaming or no latest data
        return {
            "traffic_density": torch.tensor(3.0),  # Scale 1-5
            "average_speed": torch.tensor(45.0),   # mph
            "vehicle_count": torch.tensor(150.0)   # vehicles per hour
        }

    def receive_data(self, data: Dict[str, Any]):
        """Handle real-time traffic updates"""
        processed_data = {}
        if "density" in data:
            processed_data["traffic_density"] = torch.tensor(float(data["density"]))
        if "speed" in data:
            processed_data["average_speed"] = torch.tensor(float(data["speed"]))
        if "count" in data:
            processed_data["vehicle_count"] = torch.tensor(float(data["count"]))
            
        self._latest_data = processed_data

class StaticTrafficSensor(Sensor):
    """Traffic sensor at a fixed location (like a traffic camera)"""
    def __init__(self, state: EnvironmentState, location: str, reliability: float = 0.90):
        super().__init__(f"traffic_static_{location}", state, reliability)
        self.location = location
        
    def _setup_observable_factors(self):
        """Define observable factors for this location"""
        self.observable_factors = [
            name for name, factor in self.state.factors.items()
            if name in ["traffic_density", "vehicle_count"]
        ]
    
    def fetch_data(self) -> Dict[str, torch.Tensor]:
        """Get traffic data from this location"""
        # Mock data for different locations
        mock_data = {
            "downtown": {
                "traffic_density": torch.tensor(4.0),
                "vehicle_count": torch.tensor(200.0)
            },
            "suburb": {
                "traffic_density": torch.tensor(1.0),
                "vehicle_count": torch.tensor(50.0)
            },
            "highway": {
                "traffic_density": torch.tensor(3.0),
                "vehicle_count": torch.tensor(300.0)
            }
        }
        return mock_data.get(self.location, mock_data["suburb"])

    def receive_data(self, data: Dict[str, Any]):
        """Static sensors typically don't receive push updates"""
        raise NotImplementedError("Static sensors don't support push updates")

    def get_data(self):
        """Return static traffic data for testing"""
        return {
            "traffic_density": (
                torch.tensor(4.0),  # Value
                0.9  # Reliability
            )
        } 