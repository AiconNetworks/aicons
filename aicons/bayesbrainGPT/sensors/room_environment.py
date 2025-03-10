from typing import Dict, Any, Tuple
import torch
from .base import Sensor
from ..state_representation.state import EnvironmentState

class RoomEnvironmentSensor(Sensor):
    """Sensor for monitoring room environmental conditions including temperature, occupancy, and comfort levels"""
    def __init__(self, state: EnvironmentState, room_id: str = "main", reliability: float = 0.9):
        """
        Args:
            state: The environment state to observe
            room_id: Identifier for the room being monitored
            reliability: Base reliability of the sensor readings
        """
        super().__init__(f"room_env_{room_id}", state)
        self.room_id = room_id
        self.reliability = reliability
        self._latest_data = None
        
    def _setup_observable_factors(self):
        """Define which room environment factors this sensor can observe"""
        self.observable_factors = [
            name for name, factor in self.state.factors.items()
            if name in ["room_temperature", "room_occupancy", "comfort_level"]
        ]
    
    def fetch_data(self) -> Dict[str, torch.Tensor]:
        """Get current room environmental data"""
        if self._latest_data is not None:
            return self._latest_data

        # Get comfort factor to access its possible values
        comfort_factor = self.state.factors["comfort_level"]
        comfort_probs = torch.zeros(len(comfort_factor.possible_values))
        # Set high probability for "comfortable"
        comfort_probs[comfort_factor.possible_values.index("comfortable")] = 0.8
        # Distribute remaining probability
        remaining_prob = 0.2 / (len(comfort_factor.possible_values) - 1)
        for i in range(len(comfort_factor.possible_values)):
            if i != comfort_factor.possible_values.index("comfortable"):
                comfort_probs[i] = remaining_prob
        
        # Return mock readings that match the factor types
        return {
            "room_temperature": torch.tensor(24.0, dtype=torch.float32),  # Celsius
            "room_occupancy": torch.tensor(2, dtype=torch.float32),      # Integer value as tensor
            "comfort_level": comfort_probs.to(torch.float32)             # Categorical probabilities
        }

    def receive_data(self, data: Dict[str, Any]):
        """Handle real-time updates from room sensors"""
        processed_data = {}
        
        if "room_temperature" in data:
            processed_data["room_temperature"] = torch.tensor(float(data["room_temperature"]), dtype=torch.float32)
        
        if "room_occupancy" in data:
            processed_data["room_occupancy"] = torch.tensor(int(data["room_occupancy"]), dtype=torch.float32)
            
        if "comfort_level" in data:
            comfort_factor = self.state.factors["comfort_level"]
            if isinstance(data["comfort_level"], str):
                # Convert single value to probability distribution
                probs = torch.zeros(len(comfort_factor.possible_values), dtype=torch.float32)
                idx = comfort_factor.possible_values.index(data["comfort_level"])
                probs[idx] = 1.0
                processed_data["comfort_level"] = probs
            elif isinstance(data["comfort_level"], (list, torch.Tensor)):
                # Already a probability distribution
                processed_data["comfort_level"] = torch.tensor(data["comfort_level"], dtype=torch.float32)
        
        self._latest_data = processed_data

    def get_data(self) -> Dict[str, Tuple[torch.Tensor, float]]:
        """Return room environmental data with reliability scores"""
        data = self.fetch_data()
        return {
            name: (value, self.reliability) 
            for name, value in data.items()
        } 