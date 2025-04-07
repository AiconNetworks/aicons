import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
test_dir = Path(__file__).parent
project_root = test_dir.parent
sys.path.append(str(project_root))

from aicons.definitions.zero import ZeroAIcon
from aicons.bayesbrainGPT.sensors import Sensor

def print_beliefs(aicon):
    """Print current beliefs."""
    state = aicon.brain.state.get_beliefs()
    for key, value in state.items():
        print(f"{key}: {value:.4f}")

def main():
    # Create AIcon
    aicon = ZeroAIcon(name="my_aicon", description="My AIcon", model_name="deepseek-r1:7b")
    
    print("\n=== Initial State (Before Adding Factors) ===")
    print("Brain Uncertainty:", f"{aicon.brain.uncertainty:.1%}")
    print("State Factors:", aicon.brain.state.factors)
    
    # Add state factors
    print("\n=== Adding State Factors ===")
    aicon.add_state_factor(
        name="market_size",
        factor_type="continuous",
        value=10000.0,
        params={"loc": 10000.0, "scale": 1000.0}
    )
    print("\nAdded market_size factor")
    print_beliefs(aicon)
    
    aicon.add_state_factor(
        name="conversion_rate",
        factor_type="continuous",
        value=0.02,
        params={"loc": 0.02, "scale": 0.005}
    )
    print("\nAdded conversion_rate factor")
    print_beliefs(aicon)
    
    # Create a test sensor
    class TestSensor(Sensor):
        def _setup_observable_factors(self):
            self.observable_factors = ["market_size", "conversion_rate"]
            self.factor_reliabilities = {
                "market_size": 0.9,
                "conversion_rate": 0.8
            }
            
        def fetch_data(self, environment=None):
            data = {
                "market_size": 9500.0,
                "conversion_rate": 0.021
            }
            print("\n=== Sensor Data ===")
            for factor, value in data.items():
                reliability = self.factor_reliabilities.get(factor, self.default_reliability)
                print(f"{factor}: {value:.4f} (reliability: {reliability:.2f})")
            return data
    
    test_sensor = TestSensor(name="test_sensor")
    aicon.brain.perception.register_sensor("test_sensor", test_sensor)
    
    print("\n=== Starting Perception Update ===")
    print("Current brain uncertainty:", f"{aicon.brain.uncertainty:.1%}")
    
    # Update beliefs through sensor
    print("\nUpdating beliefs from sensor...")
    aicon.update_from_sensor("test_sensor")
    
    # Print final beliefs after perception
    print_beliefs(aicon)
    
    # Print update history
    print("\n=== Update History ===")
    for update in aicon.brain.state.update_history:
        print(f"\nTime: {update['timestamp']}")
        if 'posterior_stats' in update:
            for factor_name, stats in update['posterior_stats'].items():
                print(f"{factor_name}:")
                print(f"  Mean: {stats['mean']:.4f}")
                print(f"  Std: {stats['std']:.4f}")
        if 'sensor_data' in update:
            print("\nSensor Data Used:")
            for factor, (value, reliability) in update['sensor_data'].items():
                print(f"{factor}: {value:.4f} (reliability: {reliability:.2f})")

if __name__ == '__main__':
    main() 