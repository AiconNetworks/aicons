"""
Base AIcon Module

This module provides the base AIcon class that serves as the foundation for all AIcon implementations.
The AIcon class handles:
1. Defining and managing action spaces
2. Creating and configuring utility functions
3. Coordinating with BayesBrain for Bayesian processing
4. Managing state factors and sensors
5. Managing and executing tools
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
from abc import ABC, abstractmethod

# Import BayesBrain and its components
from ..bayesbrainGPT.bayes_brain_ref import BayesBrain
from ..bayesbrainGPT.decision_making.action_space import (
    ActionSpace,
    ActionDimension,
    create_budget_allocation_space,
    create_time_budget_allocation_space,
    create_multi_campaign_action_space,
    create_marketing_ads_space
)
from ..bayesbrainGPT.utility_function import create_utility

# Import tools
from ..tools.ask_question import AskQuestionTool
from ..tools.speak_out_loud import SpeakOutLoudTool

class AIcon(ABC):
    """
    Base class for all AIcon implementations.
    
    This class provides the core functionality for:
    1. Defining and managing action spaces
    2. Creating and configuring utility functions
    3. Coordinating with BayesBrain
    4. Managing state factors and sensors
    5. Managing and executing tools
    
    Each AIcon instance has its own BayesBrain that handles the Bayesian processing,
    while the AIcon handles the high-level components and coordination.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize an AIcon instance.
        
        Args:
            name: Name of the AIcon
            description: Description of this AIcon's purpose
        """
        self.name = name
        self.description = description
        
        # Initialize BayesBrain
        self.brain = BayesBrain(
            name=f"{name}_brain",
            description=f"Bayesian brain for {name}"
        )
        self.brain.set_aicon(self)  # Set reference to this AIcon
        
        # State management
        self.state_factors = {}
        self.sensors = {}
        
        # Tool management
        self.tools = {}
        self._initialize_tools()
        
        # Initialize the AIcon
        self._initialize()
    
    def _initialize(self):
        """Initialize the AIcon's components. Override this method in subclasses."""
        pass
    
    def _initialize_tools(self):
        """Initialize the default tools."""
        # Add ask_question tool
        self.add_tool(AskQuestionTool())
        
        # Add speak_out_loud tool
        self.add_tool(SpeakOutLoudTool())
    
    def add_tool(self, tool: Any) -> None:
        """
        Add a tool to this AIcon.
        
        Args:
            tool: The tool to add
        """
        if hasattr(tool, 'name'):
            self.tools[tool.name] = tool
        else:
            print(f"Warning: Tool {tool} has no name attribute")
    
    def get_tool(self, tool_name: str) -> Optional[Any]:
        """
        Get a tool by name.
        
        Args:
            tool_name: Name of the tool to get
            
        Returns:
            The tool if found, None otherwise
        """
        return self.tools.get(tool_name)
    
    def make_decision(self, action: Dict[str, Any]) -> bool:
        """
        Make a decision based on the action from the brain.
        
        This method evaluates the action and uses appropriate tools to execute it.
        
        Args:
            action: The action to evaluate and execute
            
        Returns:
            True if the decision was executed successfully
        """
        try:
            # Get the action type and parameters
            action_type = action.get('type', 'unknown')
            action_params = action.get('params', {})
            
            # Execute the appropriate tool based on action type
            if action_type == 'ask_question':
                tool = self.get_tool('ask_question')
                if tool:
                    return tool.execute(
                        question=action_params.get('question', ''),
                        context=action_params.get('context')
                    )
                    
            elif action_type == 'speak':
                tool = self.get_tool('speak_out_loud')
                if tool:
                    return tool.execute(
                        statement=action_params.get('statement', ''),
                        context=action_params.get('context')
                    )
            
            print(f"Unknown action type: {action_type}")
            return False
            
        except Exception as e:
            print(f"Error making decision: {e}")
            return False
    
    def define_action_space(self, space_type: str, **kwargs) -> ActionSpace:
        """
        Define an action space for this AIcon.
        
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
        try:
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
                raise ValueError(f"Unknown action space type: {space_type}")
            
            # Set it in the brain
            self.brain.set_action_space(action_space)
            
            return action_space
            
        except Exception as e:
            print(f"Error creating action space: {e}")
            return None
    
    def define_utility_function(self, utility_type: str, **kwargs) -> Any:
        """
        Define a utility function for this AIcon.
        
        Args:
            utility_type: Type of utility function to create
                - 'marketing_roi': For marketing return on investment calculations
                - 'constrained_marketing_roi': Marketing ROI with business constraints
                - 'weighted_sum': Combination of multiple utility functions
                - 'multiobjective': For multi-objective optimization
                - 'custom': For custom utility functions
            **kwargs: Additional arguments for the specific utility function
            
        Returns:
            The created utility function
        """
        try:
            # Make sure we have an action space in the brain
            if not self.brain.action_space:
                raise ValueError("Must define an action space before defining utility function")
            
            # Create utility function
            utility = create_utility(
                utility_type=utility_type,
                action_space=self.brain.action_space,
                **kwargs
            )
            
            # Set it in the brain
            self.brain.set_utility_function(utility)
            
            return utility
            
        except Exception as e:
            print(f"Error creating utility function: {e}")
            return None
    
    def add_state_factor(self, name: str, factor_type: str, **kwargs) -> Dict[str, Any]:
        """
        Add a state factor to this AIcon.
        
        Args:
            name: Name of the factor
            factor_type: Type of factor ('continuous', 'categorical', or 'discrete')
            **kwargs: Additional parameters for the factor
            
        Returns:
            The created factor dictionary
        """
        try:
            factor = None
            
            if factor_type == 'continuous':
                value = kwargs.get('value', 0.0)
                uncertainty = kwargs.get('uncertainty', 0.1)
                lower_bound = kwargs.get('lower_bound')
                upper_bound = kwargs.get('upper_bound')
                
                factor = {
                    "type": "continuous",
                    "value": value,
                    "params": {"scale": uncertainty},
                    "constraints": {}
                }
                
                if lower_bound is not None:
                    factor["constraints"]["lower"] = lower_bound
                if upper_bound is not None:
                    factor["constraints"]["upper"] = upper_bound
                    
            elif factor_type == 'categorical':
                value = kwargs.get('value')
                categories = kwargs.get('categories', [])
                probs = kwargs.get('probs', None)
                
                if not value or not categories:
                    raise ValueError("Categorical factor requires value and categories")
                    
                if value not in categories:
                    raise ValueError(f"Value '{value}' not in categories: {categories}")
                    
                if probs is None:
                    # Equal probability for all categories
                    probs = [1.0 / len(categories)] * len(categories)
                    
                factor = {
                    "type": "categorical",
                    "value": value,
                    "categories": categories,
                    "params": {"probs": probs}
                }
                
            elif factor_type == 'discrete':
                value = kwargs.get('value', 0)
                min_value = kwargs.get('min_value', 0)
                max_value = kwargs.get('max_value')
                
                factor = {
                    "type": "discrete",
                    "value": value,
                    "constraints": {"lower": min_value}
                }
                
                if max_value is not None:
                    factor["constraints"]["upper"] = max_value
                    
            else:
                raise ValueError(f"Unknown factor type: {factor_type}")
            
            # Add description if provided
            if 'description' in kwargs:
                factor["description"] = kwargs['description']
            
            # Store the factor
            self.state_factors[name] = factor
            
            # Update the brain's state
            self.brain.set_state_factors(self.state_factors)
            
            return factor
            
        except Exception as e:
            print(f"Error adding state factor: {e}")
            return None
    
    def add_sensor(self, name: str, sensor: Any, factor_mapping: Optional[Dict[str, str]] = None) -> Any:
        """
        Add a sensor to this AIcon.
        
        Args:
            name: Name of the sensor
            sensor: The sensor object or function
            factor_mapping: Optional mapping between sensor outputs and state factors
            
        Returns:
            The sensor for convenience
        """
        self.sensors[name] = {
            "sensor": sensor,
            "factor_mapping": factor_mapping or {}
        }
        return sensor
    
    def update_from_sensor(self, sensor_name: str, environment: Any = None) -> bool:
        """
        Update beliefs based on data from a specific sensor.
        
        Args:
            sensor_name: Name of the sensor to use
            environment: Optional environment data to pass to the sensor
            
        Returns:
            True if update was successful
        """
        if sensor_name not in self.sensors:
            print(f"Unknown sensor: {sensor_name}")
            return False
            
        try:
            # Get sensor data
            sensor = self.sensors[sensor_name]["sensor"]
            factor_mapping = self.sensors[sensor_name]["factor_mapping"]
            
            # Get sensor data
            sensor_data = sensor(environment) if callable(sensor) else sensor.get_data(environment)
            
            # Map sensor data to state factors
            mapped_data = {}
            for factor_name, (value, reliability) in sensor_data.items():
                # Use mapping if available, otherwise use original name
                mapped_name = factor_mapping.get(factor_name, factor_name)
                mapped_data[mapped_name] = (value, reliability)
            
            # Update beliefs in the brain
            self.brain.update_beliefs(mapped_data)
            return True
            
        except Exception as e:
            print(f"Error updating from sensor: {e}")
            return False
    
    def update_from_all_sensors(self, environment: Any = None) -> bool:
        """
        Update beliefs based on data from all sensors.
        
        Args:
            environment: Optional environment data to pass to sensors
            
        Returns:
            True if update was successful
        """
        success = True
        for sensor_name in self.sensors:
            if not self.update_from_sensor(sensor_name, environment):
                success = False
        return success
    
    def find_best_action(self, num_samples: Optional[int] = None) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Find the best action based on current beliefs.
        
        Args:
            num_samples: Number of actions to sample (for Monte Carlo methods)
            
        Returns:
            Tuple of (best_action, expected_utility)
        """
        return self.brain.find_best_action(num_samples)
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state beliefs."""
        return self.brain.get_state()
    
    def get_posterior_samples(self) -> Dict[str, Any]:
        """Get samples from the posterior distribution."""
        return self.brain.get_posterior_samples() 