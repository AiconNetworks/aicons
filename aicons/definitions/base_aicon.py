import os
import json
import uuid
import pickle
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Union
import sys
import importlib.util
import numpy as np

class BaseAIcon:
    """
    Base class for all AIcon implementations.
    
    Provides core functionality including:
    - Identity management (ID, name, type)
    - State factor management (continuous, categorical, discrete)
    - Basic persistence (save/load)
    - Sensor integration
    - Decision making (action space, utility function, optimization)
    """
    
    def __init__(self, name: str, aicon_type: str = "base", capabilities: List[str] = None):
        """Initialize a BaseAIcon with identity and metadata."""
        # Generate a unique identifier
        self.id = str(uuid.uuid4())
        
        # Basic metadata
        self.name = name
        self.aicon_type = aicon_type
        self.capabilities = capabilities or []
        self.created_at = datetime.now().isoformat()
        
        # Running state
        self.is_running = False
        self.run_stats = {
            "iterations": 0,
            "start_time": None,
            "last_update_time": None,
            "updates": []
        }
        
        # State tracking
        self._state_dirty = False
        self._brain_dirty = False
        self._last_persisted = None
        
        # Persistence settings
        self._persistence_dir = os.path.join(os.path.expanduser("~"), ".aicon", "states")
        os.makedirs(self._persistence_dir, exist_ok=True)
        
        # Initialize state factors
        self.state_factors = {}
        
        # Initialize brain - directly access brain methods instead of wrapping them
        try:
            from aicons.bayesbrainGPT.bayes_brain import BayesBrain
            self.brain = BayesBrain()
            
            # Initialize perception if possible
            try:
                from aicons.bayesbrainGPT.perception.perception import BayesianPerception
                self.brain.perception = BayesianPerception(self.brain)
            except ImportError:
                print("BayesianPerception module not found, perception features will be limited")
                
            # Try to initialize BayesianState for advanced state representation
            try:
                from aicons.bayesbrainGPT.state_representation.bayesian_state import BayesianState
                self.bayesian_state = BayesianState()
            except ImportError:
                print("BayesianState module not found, advanced state representation will be limited")
                self.bayesian_state = None
        except ImportError:
            print("Warning: BayesBrain module not found. Limited functionality available.")
            self.brain = None
            self.bayesian_state = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert AIcon to a dictionary representation for serialization."""
        data = {
            "id": self.id,
            "name": self.name,
            "aicon_type": self.aicon_type,
            "capabilities": self.capabilities,
            "created_at": self.created_at,
            "run_stats": self.run_stats,
            "is_running": self.is_running
        }
        
        # Include state factors if they exist
        if hasattr(self, 'state_factors') and self.state_factors:
            # Create a simplified version of state factors for serialization
            serializable_factors = {}
            for name, factor in self.state_factors.items():
                # Remove TensorFlow objects which can't be serialized
                factor_copy = {}
                for key, value in factor.items():
                    if key != 'tf_distribution':
                        factor_copy[key] = value
                serializable_factors[name] = factor_copy
            
            data["state_factors"] = serializable_factors
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create an AIcon from a dictionary representation."""
        # Create base instance
        aicon = cls(
            name=data.get("name", "LoadedAIcon"),
            aicon_type=data.get("aicon_type", "base"),
            capabilities=data.get("capabilities", [])
        )
        
        # Restore metadata
        aicon.id = data.get("id", aicon.id)
        aicon.created_at = data.get("created_at", aicon.created_at)
        
        # Restore state
        aicon.is_running = data.get("is_running", False)
        aicon.run_stats = data.get("run_stats", aicon.run_stats)
        
        # Restore state factors if present
        if "state_factors" in data and aicon.brain:
            aicon.state_factors = data["state_factors"]
            aicon.brain.set_state_factors(aicon.state_factors)
        
        return aicon
    
    def mark_state_changed(self):
        """Mark the AIcon state as changed, needing persistence."""
        self._state_dirty = True
    
    def mark_brain_changed(self):
        """Mark the brain as changed, needing persistence."""
        self._brain_dirty = True
    
    # Core state factor methods
    def add_factor_continuous(self, name: str, value: float, uncertainty: float = 0.1, 
                             lower_bound: Optional[float] = None, upper_bound: Optional[float] = None,
                             description: str = ""):
        """
        Add a continuous factor to the brain.
        
        Args:
            name: Factor name
            value: Initial value (mean)
            uncertainty: Standard deviation
            lower_bound: Optional lower bound
            upper_bound: Optional upper bound
            description: Optional description
            
        Returns:
            The created factor
        """
        # Create constraints dictionary
        constraints = {}
        if lower_bound is not None:
            constraints["lower"] = lower_bound
        if upper_bound is not None:
            constraints["upper"] = upper_bound
            
        # Create basic factor (without TensorFlow distribution)
        factor = {
            "type": "continuous",
            "distribution": "normal",
            "params": {"loc": float(value), "scale": float(uncertainty)},
            "shape": [],
            "value": float(value),
            "constraints": constraints if constraints else None,
            "description": description or f"Continuous factor: {name}"
        }
        
        # Add TensorFlow distribution if available
        try:
            import tensorflow as tf
            import tensorflow_probability as tfp
            tfd = tfp.distributions
            tfb = tfp.bijectors
            
            # Create appropriate TensorFlow distribution
            tf_dist = None
            if lower_bound is not None and upper_bound is not None:
                # Truncated normal for bounded variables
                tf_dist = tfd.TruncatedNormal(
                    loc=float(value), 
                    scale=float(uncertainty),
                    low=float(lower_bound),
                    high=float(upper_bound)
                )
            elif lower_bound is not None:
                # Transformed distribution for lower-bounded variables
                shift = float(lower_bound)
                bijector = tfb.Chain([tfb.Softplus(), tfb.Shift(shift=shift)])
                tf_dist = tfd.TransformedDistribution(
                    distribution=tfd.Normal(loc=float(value)-shift, scale=float(uncertainty)),
                    bijector=bijector
                )
            elif upper_bound is not None:
                # Transformed distribution for upper-bounded variables
                shift = float(upper_bound)
                bijector = tfb.Chain([tfb.Softplus(), tfb.Scale(-1.0), tfb.Shift(shift=shift)])
                tf_dist = tfd.TransformedDistribution(
                    distribution=tfd.Normal(loc=shift-float(value), scale=float(uncertainty)),
                    bijector=bijector
                )
            else:
                # Unconstrained normal distribution
                tf_dist = tfd.Normal(loc=float(value), scale=float(uncertainty))
                
            # Add the TensorFlow distribution
            factor["tf_distribution"] = tf_dist
            
        except ImportError:
            # TensorFlow/TensorFlow Probability not available
            print("TensorFlow Probability not found, running in limited mode.")
            pass
        
        # Store the factor in state_factors
        self.state_factors[name] = factor
        
        # Update the brain's state with this factor
        if self.brain:
            current_state = self.brain.get_state_factors() or {}
            current_state[name] = factor
            self.brain.set_state_factors(current_state)
        
        # Also add to bayesian_state if available
        if self.bayesian_state:
            try:
                # Add as continuous latent variable
                self.bayesian_state.add_continuous_latent(
                    name=name,
                    mean=float(value),
                    uncertainty=float(uncertainty),
                    description=description or f"Continuous factor: {name}",
                    lower_bound=lower_bound,
                    upper_bound=upper_bound
                )
            except Exception as e:
                print(f"Failed to add to BayesianState: {e}")
        
        # Mark state as changed
        self.mark_state_changed()
        
        return factor
    
    def add_factor_categorical(self, name: str, value: str, categories: List[str], 
                              probs: Optional[List[float]] = None,
                              description: str = ""):
        """
        Add a categorical factor to the brain.
        
        Args:
            name: Factor name
            value: Current value (must be in categories)
            categories: List of possible categories
            probs: Optional probabilities for each category (should sum to 1)
            description: Optional description
            
        Returns:
            The created factor
        """
        if value not in categories:
            raise ValueError(f"Value '{value}' not in provided categories: {categories}")
        
        if probs is None:
            # Equal probability for all categories
            probs = [1.0 / len(categories)] * len(categories)
        
        # Create basic factor (without TensorFlow distribution)
        factor = {
            "type": "categorical",
            "distribution": "categorical",
            "params": {"probs": probs},
            "categories": categories,
            "value": value,
            "description": description or f"Categorical factor: {name}"
        }
        
        # Add TensorFlow distribution if available
        try:
            import tensorflow as tf
            import tensorflow_probability as tfp
            tfd = tfp.distributions
            
            # Convert probs to tensor
            probs_tensor = tf.constant(probs, dtype=tf.float32)
            
            # Create TensorFlow categorical distribution
            tf_dist = tfd.Categorical(probs=probs_tensor)
            
            # Add the TensorFlow distribution
            factor["tf_distribution"] = tf_dist
            
        except ImportError:
            # TensorFlow/TensorFlow Probability not available
            pass
        
        # Store the factor in state_factors
        self.state_factors[name] = factor
        
        # Update the brain's state with this factor
        if self.brain:
            current_state = self.brain.get_state_factors() or {}
            current_state[name] = factor
            self.brain.set_state_factors(current_state)
        
        # Also add to bayesian_state if available
        if self.bayesian_state:
            try:
                # Add as categorical latent variable
                self.bayesian_state.add_categorical_latent(
                    name=name,
                    initial_value=value,
                    possible_values=categories,
                    description=description or f"Categorical factor: {name}",
                    probs=probs
                )
            except Exception as e:
                print(f"Failed to add to BayesianState: {e}")
        
        # Mark state as changed
        self.mark_state_changed()
        
        return factor
    
    def add_factor_discrete(self, name: str, value: int, min_value: int = 0, 
                           max_value: Optional[int] = None,
                           description: str = ""):
        """
        Add a discrete integer factor to the brain.
        
        Args:
            name: Factor name
            value: Current integer value
            min_value: Minimum possible value
            max_value: Maximum possible value
            description: Optional description
            
        Returns:
            The created factor
        """
        # Create basic factor structure based on bounds
        if max_value is None:
            # Unbounded discrete factor
            factor = {
                "type": "discrete",
                "distribution": "poisson",
                "params": {"rate": float(value)},
                "value": int(value),
                "constraints": {"lower": min_value},
                "description": description or f"Discrete factor: {name}"
            }
        else:
            # Bounded discrete factor (categorical representation)
            num_values = max_value - min_value + 1
            categories = list(range(min_value, max_value + 1))
            index = categories.index(value)
            
            probs = [0.0] * num_values
            probs[index] = 1.0
            
            factor = {
                "type": "discrete",
                "distribution": "categorical",
                "params": {"probs": probs},
                "categories": categories,
                "value": int(value),
                "description": description or f"Discrete factor: {name}"
            }
        
        # Add TensorFlow distribution if available
        try:
            import tensorflow as tf
            import tensorflow_probability as tfp
            tfd = tfp.distributions
            
            # Create appropriate TensorFlow distribution
            if max_value is None:
                # Poisson distribution for unbounded discrete values
                tf_dist = tfd.Poisson(rate=float(value))
            else:
                # Categorical distribution for bounded discrete values
                probs_tensor = tf.constant(factor["params"]["probs"], dtype=tf.float32)
                tf_dist = tfd.Categorical(probs=probs_tensor)
            
            # Add the TensorFlow distribution
            factor["tf_distribution"] = tf_dist
            
        except ImportError:
            # TensorFlow/TensorFlow Probability not available
            pass
        
        # Store the factor in state_factors
        self.state_factors[name] = factor
        
        # Update the brain's state with this factor
        if self.brain:
            current_state = self.brain.get_state_factors() or {}
            current_state[name] = factor
            self.brain.set_state_factors(current_state)
            
        # Also add to bayesian_state if available
        if self.bayesian_state:
            try:
                # Add as discrete latent variable
                self.bayesian_state.add_discrete_latent(
                    name=name,
                    initial_value=int(value),
                    description=description or f"Discrete factor: {name}",
                    min_value=min_value,
                    max_value=max_value
                )
            except Exception as e:
                print(f"Failed to add to BayesianState: {e}")
        
        # Mark state as changed
        self.mark_state_changed()
        
        return factor
    
    def add_sensor(self, name, sensor=None, factor_mapping=None):
        """
        Add a sensor to collect observations for the AIcon.
        
        Args:
            name: Name of the sensor
            sensor: Sensor object or function that returns observations
            factor_mapping: Optional mapping between sensor and state factor names
            
        Returns:
            The sensor for convenience
        """
        if not self.brain:
            print("BayesBrain not available, cannot add sensor")
            return None
            
        # Initialize perception if not already done
        if not hasattr(self.brain, 'perception'):
            try:
                from aicons.bayesbrainGPT.perception.perception import BayesianPerception
                self.brain.perception = BayesianPerception(self.brain)
            except ImportError:
                print("BayesianPerception module not found, cannot add sensor")
                return None
        
        # Check if sensor is None
        if sensor is None:
            # Try to create a sensor based on name
            try:
                from aicons.bayesbrainGPT.sensors.tf_sensors import MarketingSensor, WeatherSensor
                
                if name.lower() in ["marketing", "campaign", "ad"]:
                    sensor = MarketingSensor()
                elif name.lower() in ["weather", "weather_station"]:
                    sensor = WeatherSensor()
                else:
                    # Create a default sensor function that returns no data
                    sensor = lambda env=None: {}
            except ImportError:
                # Create a simple lambda function as a placeholder
                sensor = lambda env=None: {}
        
        # Auto-create required factors if possible
        if hasattr(sensor, 'get_expected_factors'):
            expected_factors = sensor.get_expected_factors()
            
            # Create any missing factors with default values
            for factor_name, factor_info in expected_factors.items():
                # Map the factor name if a mapping exists
                if factor_mapping and factor_name in factor_mapping:
                    mapped_name = factor_mapping[factor_name]
                else:
                    mapped_name = factor_name
                
                # Check if the factor already exists
                if mapped_name not in self.state_factors and factor_name not in self.state_factors:
                    # Extract factor properties from factor_info
                    factor_type = factor_info.get('type', 'continuous')
                    default_value = factor_info.get('default_value', 0.0)
                    uncertainty = factor_info.get('uncertainty', 0.1)
                    lower_bound = factor_info.get('lower_bound', None)
                    upper_bound = factor_info.get('upper_bound', None)
                    categories = factor_info.get('categories', None)
                    description = factor_info.get('description', f"Factor from {name} sensor")
                    
                    # Create the appropriate type of factor
                    if factor_type == 'continuous':
                        self.add_factor_continuous(
                            mapped_name, default_value, uncertainty,
                            lower_bound=lower_bound, upper_bound=upper_bound,
                            description=description
                        )
                    elif factor_type == 'categorical' and categories:
                        self.add_factor_categorical(
                            mapped_name, default_value, categories,
                            description=description
                        )
                    elif factor_type == 'discrete':
                        self.add_factor_discrete(
                            mapped_name, default_value, 
                            min_value=lower_bound, max_value=upper_bound,
                            description=description
                        )
        
        # Register the sensor with perception
        self.brain.perception.register_sensor(name, sensor, factor_mapping)
        return sensor
    
    def update_from_sensor(self, sensor_name, environment=None, factor_mapping=None):
        """
        Update beliefs based on data from a specific sensor.
        
        Args:
            sensor_name: Name of the sensor to use
            environment: Optional environment data to pass to the sensor
            factor_mapping: Optional one-time mapping of sensor factor names to state factor names
            
        Returns:
            True if update was successful
        """
        if not self.brain or not hasattr(self.brain, 'perception'):
            print("BayesianPerception not initialized, cannot update from sensor")
            return False
        
        try:
            # Update beliefs using the brain's perception component
            success = self.brain.perception.update_from_sensor(sensor_name, environment, factor_mapping)
            
            if success:
                # Mark state as changed
                self.mark_state_changed()
                
                # Update hierarchical factors if available
                if self.bayesian_state:
                    # Update any hierarchical factors based on their parents
                    for name, factor in self.state_factors.items():
                        if "parent_factors" in factor:
                            try:
                                # Get current values of parent factors
                                parent_values = {
                                    p: self.state_factors[p]["value"] 
                                    for p in factor["parent_factors"]
                                    if p in self.state_factors
                                }
                                
                                # Predict new value based on parents
                                if name in self.bayesian_state.factors:
                                    hierarchical_factor = self.bayesian_state.factors[name]
                                    updated_value = hierarchical_factor.predict_from_parents(parent_values)
                                    
                                    # Update in state_factors
                                    factor["value"] = updated_value
                                    factor["params"]["loc"] = updated_value
                                    
                                    # Update in brain
                                    if self.brain:
                                        current_state = self.brain.get_state_factors() or {}
                                        if name in current_state:
                                            current_state[name]["value"] = updated_value
                                            current_state[name]["params"]["loc"] = updated_value
                                            self.brain.set_state_factors(current_state)
                                            
                            except Exception as e:
                                print(f"Failed to update hierarchical factor {name}: {e}")
            
            return success
            
        except Exception as e:
            print(f"Error updating from sensor: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def update_from_all_sensors(self, environment=None):
        """
        Update beliefs based on data from all sensors.
        
        Args:
            environment: Optional environment data to pass to sensors
            
        Returns:
            True if update was successful
        """
        if not self.brain or not hasattr(self.brain, 'perception'):
            print("BayesianPerception not initialized, cannot update from sensors")
            return False
        
        try:
            # Update beliefs using the brain's perception component
            success = self.brain.perception.update_all(environment)
            
            if success:
                # Mark state as changed
                self.mark_state_changed()
                
                # Update hierarchical factors if available
                if self.bayesian_state:
                    # Update any hierarchical factors based on their parents
                    for name, factor in self.state_factors.items():
                        if "parent_factors" in factor:
                            try:
                                # Get current values of parent factors
                                parent_values = {
                                    p: self.state_factors[p]["value"] 
                                    for p in factor["parent_factors"]
                                    if p in self.state_factors
                                }
                                
                                # Predict new value based on parents
                                if name in self.bayesian_state.factors:
                                    hierarchical_factor = self.bayesian_state.factors[name]
                                    updated_value = hierarchical_factor.predict_from_parents(parent_values)
                                    
                                    # Update in state_factors
                                    factor["value"] = updated_value
                                    factor["params"]["loc"] = updated_value
                                    
                                    # Update in brain
                                    if self.brain:
                                        current_state = self.brain.get_state_factors() or {}
                                        if name in current_state:
                                            current_state[name]["value"] = updated_value
                                            current_state[name]["params"]["loc"] = updated_value
                                            self.brain.set_state_factors(current_state)
                                            
                            except Exception as e:
                                print(f"Failed to update hierarchical factor {name}: {e}")
            
            return success
            
        except Exception as e:
            print(f"Error updating from sensors: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_state(self, format_nicely: bool = False):
        """
        Get the current state (prior distributions) from the brain.
        
        Args:
            format_nicely: Whether to return a formatted human-readable representation
            
        Returns:
            Dictionary of state factors, or a formatted string if format_nicely is True
        """
        if self.brain:
            state = self.brain.get_state_factors()
        else:
            state = self.state_factors
        
        if not format_nicely:
            return state
        
        # Format nicely for human readability
        formatted = []
        formatted.append(f"AIcon State ({len(state)} factors):")
        
        for name, factor in state.items():
            factor_type = factor["type"]
            factor_value = factor["value"]
            distribution = factor["distribution"]
            
            # Basic factor info
            factor_str = f"\n{name}:"
            factor_str += f"\n  Type: {factor_type}"
            factor_str += f"\n  Distribution: {distribution}"
            factor_str += f"\n  Current value: {factor_value}"
            
            # Distribution parameters
            params = factor["params"]
            if factor_type == "continuous":
                factor_str += f"\n  Mean: {params.get('loc', 'N/A')}"
                factor_str += f"\n  Uncertainty: {params.get('scale', 'N/A')}"
                
                if "constraints" in factor and factor["constraints"]:
                    constraints = factor["constraints"]
                    bounds = []
                    if "lower" in constraints:
                        bounds.append(f">= {constraints['lower']}")
                    if "upper" in constraints:
                        bounds.append(f"<= {constraints['upper']}")
                    factor_str += f"\n  Constraints: {', '.join(bounds)}"
                    
            elif factor_type == "categorical":
                categories = factor.get("categories", [])
                probs = params.get("probs", [])
                
                # Format categories and their probabilities
                if categories and probs and len(categories) == len(probs):
                    factor_str += "\n  Categories (probability):"
                    for cat, prob in zip(categories, probs):
                        factor_str += f"\n    {cat}: {prob:.2f}"
                else:
                    factor_str += f"\n  Categories: {categories}"
                    factor_str += f"\n  Probabilities: {probs}"
                    
            elif factor_type == "discrete":
                if "categories" in factor:
                    factor_str += f"\n  Possible values: {factor['categories']}"
                elif "rate" in params:
                    factor_str += f"\n  Rate parameter: {params['rate']}"
                    
                if "constraints" in factor and factor["constraints"]:
                    constraints = factor["constraints"]
                    factor_str += f"\n  Constraints: {constraints}"
            
            # Add description if available
            if "description" in factor and factor["description"]:
                factor_str += f"\n  Description: {factor['description']}"
                
            formatted.append(factor_str)
            
        return "\n".join(formatted)
    
    def save_state(self, filepath: str = None, format: str = "json") -> bool:
        """
        Save the AIcon's state to a file.
        
        Args:
            filepath: Path to save the state file (optional)
            format: Format to use ("json" or "pickle")
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        if filepath is None:
            # Default path based on AIcon ID
            filename = f"{self.id}.{format}"
            filepath = os.path.join(self._persistence_dir, filename)
        
        try:
            if format.lower() == "json":
                with open(filepath, 'w') as f:
                    json.dump(self.to_dict(), f, indent=2)
            elif format.lower() == "pickle":
                with open(filepath, 'wb') as f:
                    pickle.dump(self, f)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Reset dirty flag
            self._state_dirty = False
            self._last_persisted = datetime.now().isoformat()
            
            return True
        except Exception as e:
            print(f"Failed to save AIcon state: {e}")
            return False
    
    @classmethod
    def load_state(cls, filepath: str, format: str = None) -> 'BaseAIcon':
        """
        Load an AIcon's state from a file.
        
        Args:
            filepath: Path to the state file
            format: Format of the file ("json" or "pickle"), inferred from extension if None
            
        Returns:
            BaseAIcon: Loaded AIcon instance or None if loading failed
        """
        if format is None:
            # Infer format from file extension
            format = filepath.split('.')[-1]
        
        try:
            if format.lower() == "json":
                with open(filepath, 'r') as f:
                    data = json.load(f)
                aicon = cls.from_dict(data)
            elif format.lower() in ["pickle", "pkl"]:
                with open(filepath, 'rb') as f:
                    aicon = pickle.load(f)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            return aicon
        except Exception as e:
            print(f"Failed to load AIcon state: {e}")
            return None
    
    def record_update(self, source: str = "manual", success: bool = True, 
                     metadata: Dict = None):
        """
        Record an update to the AIcon's state.
        
        Args:
            source: Source of the update (e.g., "sensor", "manual", "scheduled")
            success: Whether the update was successful
            metadata: Additional information about the update
        """
        now = datetime.now()
        
        # Update run stats
        self.run_stats["iterations"] += 1
        self.run_stats["last_update_time"] = now.isoformat()
        
        if self.run_stats["start_time"] is None:
            self.run_stats["start_time"] = now.isoformat()
        
        # Record the update
        update = {
            "time": now.isoformat(),
            "source": source,
            "success": success
        }
        
        # Add any additional metadata
        if metadata:
            update["metadata"] = metadata
        
        self.run_stats["updates"].append(update)
        
        # Mark state as changed
        self.mark_state_changed()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get basic metadata about this AIcon."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.aicon_type,
            "capabilities": self.capabilities,
            "created_at": self.created_at,
            "is_running": self.is_running,
            "iterations": self.run_stats["iterations"],
            "last_update": self.run_stats["last_update_time"],
            "last_persisted": self._last_persisted,
            "factor_count": len(self.state_factors) if hasattr(self, 'state_factors') else 0
        }
    
    def create_action_space(self, space_type: str = 'budget_allocation', **kwargs):
        """
        Create an action space for decision-making.
        
        Args:
            space_type: Type of action space to create. Options include:
                - 'budget_allocation': For allocating budget across items
                - 'marketing': For marketing campaign optimization
                - 'time_budget': For time-based budget allocation
                - 'multi_campaign': For multi-campaign budget allocation
                - 'custom': For custom action spaces with specific dimensions
            **kwargs: Additional parameters specific to the action space type
                
        Returns:
            The created action space
        """
        if not self.brain:
            print("BayesBrain not available, cannot create action space")
            return None
        
        try:
            # Import action space components
            from aicons.bayesbrainGPT.decision_making.action_space import (
                ActionSpace, 
                ActionDimension,
                create_budget_allocation_space,
                create_time_budget_allocation_space,
                create_multi_campaign_action_space,
                create_marketing_ads_space
            )
            
            # Create the appropriate action space based on space_type
            action_space = None
            
            if space_type == 'custom':
                # Get dimension specifications
                dimensions_specs = kwargs.get('dimensions_specs', [])
                constraints = kwargs.get('constraints', [])
                
                if not dimensions_specs:
                    raise ValueError("Custom action space requires dimensions_specs list")
                
                # Create ActionDimension objects
                dimensions = []
                for spec in dimensions_specs:
                    # Extract dimension parameters
                    name = spec.get('name')
                    dim_type = spec.get('type', 'continuous')
                    
                    if not name:
                        raise ValueError("Each dimension spec must include a 'name'")
                    
                    # Create appropriate dimension based on type
                    if dim_type == 'discrete':
                        values = spec.get('values')
                        if values is None:
                            raise ValueError(f"Discrete dimension '{name}' must specify 'values'")
                        dimensions.append(ActionDimension(
                            name=name,
                            dim_type='discrete',
                            values=values
                        ))
                    elif dim_type == 'continuous':
                        min_value = spec.get('min_value')
                        max_value = spec.get('max_value')
                        step = spec.get('step')
                        
                        if min_value is None or max_value is None:
                            raise ValueError(f"Continuous dimension '{name}' must specify 'min_value' and 'max_value'")
                        
                        dimensions.append(ActionDimension(
                            name=name,
                            dim_type='continuous',
                            min_value=min_value,
                            max_value=max_value,
                            step=step
                        ))
                    else:
                        raise ValueError(f"Unknown dimension type: {dim_type}")
                
                # Create the ActionSpace
                action_space = ActionSpace(dimensions=dimensions, constraints=constraints)
                
            elif space_type == 'budget_allocation':
                # Budget allocation across items
                total_budget = kwargs.get('total_budget', 1000.0)
                items = kwargs.get('items', [])
                budget_step = kwargs.get('budget_step', 10.0)
                min_budget = kwargs.get('min_budget', 0.0)
                
                action_space = create_budget_allocation_space(
                    total_budget=total_budget,
                    num_ads=len(items),
                    budget_step=budget_step,
                    min_budget=min_budget,
                    ad_names=items
                )
                
            elif space_type == 'marketing':
                # Marketing optimization space
                total_budget = kwargs.get('total_budget', 1000.0)
                num_ads = kwargs.get('num_ads', 3)
                budget_step = kwargs.get('budget_step', 10.0)
                min_budget = kwargs.get('min_budget', 0.0)
                ad_names = kwargs.get('ad_names', None)
                
                action_space = create_marketing_ads_space(
                    total_budget=total_budget,
                    num_ads=num_ads,
                    budget_step=budget_step,
                    min_budget=min_budget,
                    ad_names=ad_names
                )
                
            elif space_type == 'time_budget':
                # Time and budget allocation
                total_budget = kwargs.get('total_budget', 1000.0)
                num_days = kwargs.get('num_days', 7)
                num_ads = kwargs.get('num_ads', 2)
                budget_step = kwargs.get('budget_step', 10.0)
                min_budget = kwargs.get('min_budget', 0.0)
                
                action_space = create_time_budget_allocation_space(
                    total_budget=total_budget,
                    num_ads=num_ads,
                    num_days=num_days,
                    budget_step=budget_step,
                    min_budget=min_budget
                )
                
            elif space_type == 'multi_campaign':
                # Multi-campaign allocation
                campaigns = kwargs.get('campaigns', {})
                budget_step = kwargs.get('budget_step', 10.0)
                
                action_space = create_multi_campaign_action_space(
                    campaigns=campaigns,
                    budget_step=budget_step
                )
                
            else:
                print(f"Unknown action space type: {space_type}")
                return None
            
            # Set the action space in the brain
            if action_space and hasattr(self.brain, 'set_action_space'):
                self.brain.set_action_space(action_space)
                print(f"Created {space_type} action space with {len(action_space.dimensions)} dimensions")
            
            return action_space
            
        except Exception as e:
            print(f"Could not create action space: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_utility_function(self, utility_type='marketing_roi', **kwargs):
        """
        Create a utility function for evaluating actions.

        Args:
            utility_type (str): Type of utility function to create
                - 'marketing_roi': For marketing return on investment calculations
                - 'constrained_marketing_roi': Marketing ROI with business constraints
                - 'weighted_sum': Combination of multiple utility functions
                - 'multiobjective': For multi-objective optimization
                - 'custom': For custom utility functions
            **kwargs: Additional arguments for the specific utility function

        Returns:
            A utility function object that can evaluate actions given state
        """
        if not self.brain:
            print("BayesBrain not available, cannot create utility function")
            return None
        
        try:
            # Check for action space
            action_space = self.brain.get_action_space()
            if not action_space:
                print("Warning: No action space defined. Create an action space before defining utility.")
            
            # Import utility components
            try:
                from aicons.bayesbrainGPT.utility_function import create_utility
                
                # Create utility function
                utility = create_utility(
                    utility_type=utility_type, 
                    action_space=action_space,
                    **kwargs
                )
                
                # Set in brain
                self.brain.set_utility_function(utility)
                print(f"Created utility function: {utility.name if hasattr(utility, 'name') else utility_type}")
                return utility
                
            except ImportError:
                print("BayesBrainGPT utility module not available. Using simplified utility function.")
                
                # Simple fallback utility function for basic cases
                if utility_type == 'marketing_roi':
                    def simple_utility(action, state):
                        # Extract budget values from action
                        total_budget = sum(value for key, value in action.items() if key.endswith('_budget'))
                        # Simple ROI calculation
                        roi_multiplier = kwargs.get('roi_multiplier', 0.2)
                        return total_budget * roi_multiplier
                    
                    # Set in brain
                    self.brain.set_utility_function(simple_utility)
                    print(f"Created simple {utility_type} utility function")
                    return simple_utility
                else:
                    print(f"Utility type '{utility_type}' not available in fallback mode")
                    return None
        
        except Exception as e:
            print(f"Error creating utility function: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def find_best_action(self, num_samples: int = 100, use_gradient: bool = False):
        """
        Find the best action based on the current state and utility function.
        
        Args:
            num_samples: Number of samples to use for the search
            use_gradient: Whether to use gradient-based optimization (requires TensorFlow)
            
        Returns:
            Tuple of (best_action, expected_utility)
        """
        if not self.brain:
            print("BayesBrain not available, cannot find best action")
            return None, 0.0
        
        try:
            # Make sure we have an action space and utility function
            action_space = self.brain.get_action_space()
            if action_space is None:
                print("Cannot find best action without an action space. Call create_action_space() first.")
                return None, 0.0
            
            utility_function = self.brain.get_utility_function()
            if utility_function is None:
                print("Cannot find best action without a utility function. Call create_utility_function() first.")
                return None, 0.0
            
            # Set the number of samples in the brain's decision parameters
            if hasattr(self.brain, 'decision_params'):
                self.brain.decision_params["num_samples"] = num_samples
            
            # Simply delegate to the brain's find_best_action method
            return self.brain.find_best_action(num_samples=num_samples, use_gradient=use_gradient)
        
        except Exception as e:
            print(f"Error finding best action: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0
    
    def add_hierarchical_factor(self, name: str, parent_factors: List[str], 
                               relation_type: str = "linear", 
                               parameters: Dict = None,
                               uncertainty: float = 1.0,
                               description: str = ""):
        """
        Add a hierarchical factor that depends on other factors.
        
        Args:
            name: Factor name
            parent_factors: List of parent factor names this factor depends on
            relation_type: Type of relationship (linear, nonlinear)
            parameters: Parameters defining the relationship
            uncertainty: Uncertainty in the relationship
            description: Optional description
            
        Returns:
            The created hierarchical factor
        """
        if not self.bayesian_state:
            print("BayesianState not available, cannot add hierarchical factor")
            return None
            
        try:
            # Create parent factors dictionary
            parent_dict = {}
            for parent in parent_factors:
                if parent in self.state_factors:
                    parent_dict[parent] = self.state_factors[parent]["value"]
                else:
                    raise ValueError(f"Parent factor {parent} not found")
            
            # Set default parameters if none provided
            if parameters is None:
                # Default to small random weights
                parameters = {
                    "weights": {p: np.random.normal(0, 0.1) for p in parent_factors},
                    "intercept": np.random.normal(0, 0.1)
                }
            
            # Add hierarchical latent variable to BayesianState
            self.bayesian_state.add_hierarchical_latent(
                name=name,
                parents=parent_dict,
                relation_type=relation_type,
                parameters=parameters,
                uncertainty=uncertainty,
                description=description or f"Hierarchical factor: {name}"
            )
            
            # Get the current value prediction based on parents
            initial_value = self.bayesian_state.factors[name].predict_from_parents(parent_dict)
            
            # Create simplified version for standard state factors
            factor = {
                "type": "continuous",
                "distribution": "normal",
                "params": {"loc": float(initial_value), "scale": float(uncertainty)},
                "value": float(initial_value),
                "parent_factors": parent_factors,
                "relationship": {"type": relation_type, "parameters": parameters},
                "description": description or f"Hierarchical factor: {name}"
            }
            
            # Store in state_factors
            self.state_factors[name] = factor
            
            # Update the brain's state
            if self.brain:
                current_state = self.brain.get_state_factors() or {}
                current_state[name] = factor
                self.brain.set_state_factors(current_state)
            
            # Mark state as changed
            self.mark_state_changed()
            
            return factor
            
        except Exception as e:
            print(f"Failed to add hierarchical factor: {e}")
            return None

    
    def create_hierarchical_model(self):
        """
        Create a hierarchical generative model from the current state factors.
        
        This enables more sophisticated Bayesian inference by capturing
        dependencies between factors.
        
        Returns:
            True if the model was created successfully
        """
        if not self.bayesian_state:
            print("BayesianState not available, cannot create hierarchical model")
            return False
            
        try:
            # Create joint distribution based on state factors
            self.bayesian_state.create_joint_distribution()
            print("Created hierarchical model with joint distribution")
            
            # Mark brain as changed
            self.mark_brain_changed()
            
            return True
        except Exception as e:
            print(f"Failed to create hierarchical model: {e}")
            return False
    
    def sample_from_prior(self, n_samples: int = 1):
        """
        Sample from the prior distribution of the hierarchical model.
        
        Args:
            n_samples: Number of samples to draw
            
        Returns:
            Dictionary of samples from prior distribution
        """
        if not self.bayesian_state:
            print("BayesianState not available, cannot sample from prior")
            return None
            
        try:
            # Sample from the BayesianState's prior
            return self.bayesian_state.sample_from_prior(n_samples)
        except Exception as e:
            print(f"Failed to sample from prior: {e}")
            return None
    
    def perceive_and_decide(self, environment=None):
        """
        Perceive the environment and make a decision.
        
        This convenience method updates from all sensors and then finds the best action.
        
        Args:
            environment: Environment data to pass to sensors
            
        Returns:
            Tuple of (best_action, expected_utility)
        """
        if not self.brain:
            print("BayesBrain not available, cannot perceive and decide")
            return None, 0.0
        
        try:
            # Step 1: Update beliefs based on sensor data
            if hasattr(self, 'update_from_all_sensors'):
                success = self.update_from_all_sensors(environment)
                if not success:
                    print("Warning: Failed to update from sensors")
            
            # Step 2: Find the best action based on updated beliefs
            return self.find_best_action()
        
        except Exception as e:
            print(f"Error in perceive_and_decide: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0
    
    # Provide method aliases with the new names that better reflect their roles
    # These maintain backward compatibility while adopting clearer naming
    def define_factor_dependency(self, name, parent_factors, relation_type="linear", 
                              parameters=None, uncertainty=1.0, description=""):
        """
        Define a factor that depends on other factors through a specific relationship.
        
        This is an alias for add_hierarchical_factor with a more descriptive name that
        better reflects the inherently hierarchical nature of the system.
        
        Args:
            name: Factor name
            parent_factors: List of parent factor names this factor depends on
            relation_type: Type of relationship (linear, nonlinear)
            parameters: Parameters defining the relationship
            uncertainty: Uncertainty in the relationship
            description: Optional description
            
        Returns:
            The created hierarchical factor
        """
        return self.add_hierarchical_factor(
            name=name,
            parent_factors=parent_factors,
            relation_type=relation_type,
            parameters=parameters,
            uncertainty=uncertainty,
            description=description
        )
    
    def compile_probabilistic_model(self):
        """
        Compile all factors and their relationships into a coherent joint distribution for inference.
        
        This is an alias for create_hierarchical_model with a more descriptive name that
        better reflects the inherently hierarchical nature of the system.
        
        Returns:
            True if the model was compiled successfully
        """
        return self.create_hierarchical_model() 