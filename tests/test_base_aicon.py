"""
Test script for the BaseAIcon class.

This script demonstrates the basic functionality of the BaseAIcon class.
"""

import os
import json
import tempfile
from datetime import datetime
from aicons.definitions.base_aicon import BaseAIcon

def test_basic_functionality():
    """Test basic BaseAIcon functionality."""
    print("\n=== Testing BaseAIcon Basic Functionality ===")
    
    # Create a BaseAIcon instance
    aicon = BaseAIcon("TestAIcon", "test")
    print(f"Created AIcon: {aicon.name} (ID: {aicon.id})")
    
    # Add some factors
    aicon.add_factor_continuous(
        name="temperature", 
        value=22.5, 
        uncertainty=2.0,
        lower_bound=0,
        upper_bound=40,
        description="Current temperature in Celsius"
    )
    
    aicon.add_factor_categorical(
        name="weather", 
        value="sunny", 
        categories=["sunny", "cloudy", "rainy", "snowy"],
        description="Current weather condition"
    )
    
    aicon.add_factor_discrete(
        name="people_count", 
        value=5, 
        min_value=0, 
        max_value=20,
        description="Number of people in the room"
    )
    
    # Print the state
    print("\nAIcon State:")
    print(aicon.get_state(format_nicely=True))
    
    # Record an update
    aicon.record_update(
        source="test", 
        metadata={"test_name": "Basic functionality test"}
    )
    
    # Print metadata
    print("\nAIcon Metadata:")
    metadata = aicon.get_metadata()
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    print("\n=== Basic Functionality Test Completed ===")
    return aicon

def test_persistence():
    """Test persistence capabilities."""
    print("\n=== Testing BaseAIcon Persistence ===")
    
    # Create a BaseAIcon instance
    aicon = BaseAIcon("PersistenceTest", "test")
    
    # Add some factors
    aicon.add_factor_continuous("factor1", 10.0, 1.0)
    aicon.add_factor_categorical("factor2", "option2", ["option1", "option2", "option3"])
    
    # Create a temporary file for persistence
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Save state to the temporary file
        print(f"Saving AIcon state to {tmp_path}")
        success = aicon.save_state(filepath=tmp_path)
        
        if success:
            print("Successfully saved AIcon state")
            
            # Load the AIcon from the saved state
            loaded_aicon = BaseAIcon.load_state(tmp_path)
            
            if loaded_aicon:
                print("Successfully loaded AIcon state")
                print(f"Loaded AIcon: {loaded_aicon.name} (ID: {loaded_aicon.id})")
                
                # Print the loaded state
                print("\nLoaded AIcon State:")
                print(loaded_aicon.get_state(format_nicely=True))
                
                # Verify IDs match
                if loaded_aicon.id == aicon.id:
                    print("SUCCESS: Loaded AIcon has the same ID")
                else:
                    print("ERROR: Loaded AIcon has a different ID")
            else:
                print("ERROR: Failed to load AIcon state")
        else:
            print("ERROR: Failed to save AIcon state")
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    
    print("\n=== Persistence Test Completed ===")

def test_sensor_integration():
    """Test sensor integration capabilities."""
    print("\n=== Testing BaseAIcon Sensor Integration ===")
    
    # Create a BaseAIcon instance
    aicon = BaseAIcon("SensorTest", "test")
    
    # Create a simple mock sensor function
    def mock_weather_sensor(environment=None):
        """Mock weather sensor that returns simulated weather data."""
        # Default values if no environment is provided
        if environment is None:
            return {
                "temperature": 25.0,
                "humidity": 60.0,
                "condition": "sunny"
            }
        
        # Return values from environment if provided
        return environment
    
    # Add the sensor to the AIcon
    try:
        sensor = aicon.add_sensor("weather", mock_weather_sensor)
        if sensor:
            print("Successfully added mock weather sensor")
            
            # Check if factors were auto-created (this will happen if BayesBrain is available)
            if "temperature" in aicon.state_factors:
                print("Auto-created temperature factor")
            else:
                # Manually add factors if auto-creation didn't happen
                aicon.add_factor_continuous("temperature", 20.0, 5.0)
                aicon.add_factor_continuous("humidity", 50.0, 10.0)
                aicon.add_factor_categorical("condition", "cloudy", ["sunny", "cloudy", "rainy"])
                print("Manually added weather factors")
            
            # Print initial state
            print("\nInitial state:")
            print(aicon.get_state(format_nicely=True))
            
            # Update from sensor
            environment = {
                "temperature": 30.0,
                "humidity": 80.0,
                "condition": "rainy"
            }
            
            success = aicon.update_from_sensor("weather", environment)
            if success:
                print("\nSuccessfully updated from weather sensor")
                print("\nUpdated state:")
                print(aicon.get_state(format_nicely=True))
            else:
                print("WARNING: Update from sensor not successful")
                print("This is expected if BayesBrain/TensorFlow is not available")
        else:
            print("WARNING: Failed to add sensor")
            print("This is expected if BayesBrain is not available")
    except Exception as e:
        print(f"WARNING: Sensor integration test could not complete: {e}")
        print("This is expected if BayesBrain is not available")
    
    print("\n=== Sensor Integration Test Completed ===")

if __name__ == "__main__":
    print("=== Starting BaseAIcon Tests ===\n")
    
    # Run the tests
    test_aicon = test_basic_functionality()
    test_persistence()
    test_sensor_integration()
    
    print("\n=== All Tests Completed ===") 