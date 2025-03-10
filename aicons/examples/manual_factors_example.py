import torch
from aicons.bayesbrainGPT.state_representation import EnvironmentState
from aicons.bayesbrainGPT.perception import BayesianPerception
from aicons.bayesbrainGPT.sensors.room_environment import RoomEnvironmentSensor

def test_manual_factors():
    # Create state with manually added factors
    state = EnvironmentState()

    # Add a continuous factor for room temperature (like "temperature" in perception_example)
    state.add_continuous_factor(
        name="room_temperature",
        initial_value=20.0,
        description="Room temperature in Celsius"
    )

    # Add a categorical factor for comfort level (like "weather" in perception_example)
    state.add_categorical_factor(
        name="comfort_level",
        initial_value="comfortable",
        possible_values=["cold", "comfortable", "hot"],
        description="Comfort level based on temperature"
    )

    # Add a continuous factor for occupancy (like "traffic_density" in perception_example)
    state.add_continuous_factor(
        name="room_occupancy",
        initial_value=2.0,
        description="Number of people in the room (0-10)",
        relationships={
            "depends_on": ["comfort_level"],
            "model": {
                "comfort_level": {
                    "type": "categorical_effect",
                    "effects": {
                        "cold": 0.0,
                        "comfortable": 2.0,
                        "hot": 1.0
                    }
                }
            }
        }
    )

    # Add a continuous factor for CO2 (like "average_speed" in perception_example)
    state.add_continuous_factor(
        name="co2_level",
        initial_value=400.0,
        description="CO2 concentration in ppm",
        relationships={
            "depends_on": ["room_occupancy", "comfort_level"],
            "model": {
                "room_occupancy": {
                    "type": "linear",
                    "base": 400.0,  # Base CO2 level
                    "coefficient": 50.0  # CO2 increase per person
                },
                "comfort_level": {
                    "type": "categorical_effect",
                    "effects": {
                        "cold": 0.0,
                        "comfortable": 0.0,
                        "hot": 50.0
                    }
                }
            }
        }
    )

    print("\nInitial state:")
    print(f"Room Temperature: {state.factors['room_temperature'].value}")
    print(f"Comfort Level: {state.factors['comfort_level'].value}")
    print(f"Room Occupancy: {state.factors['room_occupancy'].value}")
    print(f"CO2 Level: {state.factors['co2_level'].value}")

    # Create perception system
    perception = BayesianPerception(state)

    # Create and register room environment sensor
    room_sensor = RoomEnvironmentSensor(state, room_id="office", reliability=0.9)
    perception.register_sensor(room_sensor)

    # Update state through perception
    print("\nUpdating state with room sensor data...")
    perception.update_all()

if __name__ == "__main__":
    test_manual_factors() 