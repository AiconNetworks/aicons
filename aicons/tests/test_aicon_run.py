"""
Test the AIcon run functionality with and without priors.

This test ensures that AIcon's run method doesn't execute perception updates
when there are no priors (state factors) defined.
"""

import unittest
import time
from unittest.mock import patch, MagicMock

from aicons.definitions.simple_bad_aicon import SimpleBadAIcon
from aicons.bayesbrainGPT.sensors.meta_s.meta_ads_sales_sensor import MetaAdsSalesSensor

class TestAIconRun(unittest.TestCase):
    """Tests for the AIcon run method."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a clean AIcon for each test
        self.aicon = SimpleBadAIcon(name="Test AIcon")
    
    def test_run_without_factors(self):
        """Test that run doesn't proceed when no factors are defined."""
        # Create a mock sensor
        sensor = MetaAdsSalesSensor(
            name="test_sensor",
            reliability=0.9,
        )
        
        # Register the sensor
        self.aicon.add_sensor("test_sensor", sensor)
        
        # Mock the update_from_sensor method to track if it's called
        self.aicon.update_from_sensor = MagicMock(return_value=True)
        
        # Run the AIcon
        run_stats = self.aicon.run(mode='once', sensor_name="test_sensor")
        
        # Verify that update_from_sensor was not called
        self.aicon.update_from_sensor.assert_not_called()
        
        # Verify that run_stats indicates 0 iterations
        self.assertEqual(run_stats["iterations"], 0)
        
        # Verify that the warning message is logged
        # (Would need to capture stdout to test this properly)
    
    def test_run_with_factors(self):
        """Test that run proceeds normally when factors are defined."""
        # Add factors (priors)
        self.aicon.add_factor_continuous(
            name="purchases",
            value=20.0,
            uncertainty=10.0,
            lower_bound=0.0
        )
        
        # Create a mock sensor
        sensor = MetaAdsSalesSensor(
            name="test_sensor",
            reliability=0.9,
        )
        
        # Register the sensor with factor mapping
        self.aicon.add_sensor("test_sensor", sensor, {"purchases": "purchases"})
        
        # Mock the update_from_sensor method to track if it's called
        self.aicon.update_from_sensor = MagicMock(return_value=True)
        
        # Run the AIcon
        run_stats = self.aicon.run(mode='once', sensor_name="test_sensor")
        
        # Verify that update_from_sensor was called exactly once
        self.aicon.update_from_sensor.assert_called_once()
        
        # Verify that run_stats indicates 1 iteration
        self.assertEqual(run_stats["iterations"], 1)
    
    def test_finite_run_stops_after_duration(self):
        """Test that finite run stops after specified duration."""
        # Add factors (priors)
        self.aicon.add_factor_continuous(
            name="purchases",
            value=20.0,
            uncertainty=10.0,
            lower_bound=0.0
        )
        
        # Create a mock sensor
        sensor = MetaAdsSalesSensor(
            name="test_sensor",
            reliability=0.9,
        )
        
        # Register the sensor
        self.aicon.add_sensor("test_sensor", sensor, {"purchases": "purchases"})
        
        # Mock the update_from_sensor method for faster testing
        self.aicon.update_from_sensor = MagicMock(return_value=True)
        
        # Run the AIcon in finite mode with 3 iterations
        run_stats = self.aicon.run(
            mode='finite', 
            sensor_name="test_sensor", 
            interval=0.01,  # Very short interval for faster testing
            duration=3      # 3 iterations
        )
        
        # Verify that update_from_sensor was called exactly 3 times
        self.assertEqual(self.aicon.update_from_sensor.call_count, 3)
        
        # Verify that run_stats indicates 3 iterations
        self.assertEqual(run_stats["iterations"], 3)

if __name__ == "__main__":
    unittest.main() 