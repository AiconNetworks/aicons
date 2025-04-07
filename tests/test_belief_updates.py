import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
test_dir = Path(__file__).parent
project_root = test_dir.parent
sys.path.append(str(project_root))

from aicons.definitions.zero import ZeroAIcon
from aicons.bayesbrainGPT.sensors import Sensor

def print_beliefs(aicon, label):
    """Helper function to print current beliefs"""
    beliefs = aicon.get_state()
    print(f"\n=== {label} ===")
    for factor_name, value in beliefs.items():
        factor = aicon.brain.state.factors[factor_name]
        print(f"{factor_name}: {value:.4f} ± {factor._uncertainty:.4f}")
        
    # Print brain's uncertainty
    print(f"\nBrain Uncertainty: {aicon.brain.uncertainty:.1%}")
        
    # Print detailed state representation
    print("\n=== Detailed State Representation ===")
    state_factors = aicon.brain.state.get_state_factors()
    for name, factor in state_factors.items():
        print(f"\n{name}:")
        print(f"  Type: {factor['type']}")
        print(f"  Value: {factor['value']}")
        print(f"  Distribution: {factor['distribution']}")
        print(f"  Parameters: {factor['params']}")
        print(f"  Relationships: {factor['relationships']}")
        print(f"  Uncertainty: {factor['uncertainty']}")

def main():
    # Create AIcon
    aicon = ZeroAIcon(name="my_aicon", description="My AIcon", model_name="deepseek-r1:7b")
    
    # Add state factors
    aicon.add_state_factor(
        name="market_size",
        factor_type="continuous",
        value=10000.0,
        params={"loc": 10000.0, "scale": 1000.0}
    )
    
    aicon.add_state_factor(
        name="conversion_rate",
        factor_type="continuous",
        value=0.02,
        params={"loc": 0.02, "scale": 0.005}
    )
    
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
    
    # Print initial beliefs
    print_beliefs(aicon, "Initial Beliefs")
    
    # Update beliefs through sensor
    print("\nUpdating beliefs from sensor...")
    
    # Get current posterior samples
    current_samples = aicon.brain.state.get_posterior_samples()
    if current_samples:
        print("\n=== Current Posterior Samples ===")
        for factor, samples in current_samples.items():
            print(f"{factor}:")
            print(f"  Mean: {np.mean(samples):.4f}")
            print(f"  Std: {np.std(samples):.4f}")
    
    # Update from sensor
    aicon.update_from_sensor("test_sensor")
    
    # Print updated beliefs
    print_beliefs(aicon, "Updated Beliefs")
    
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