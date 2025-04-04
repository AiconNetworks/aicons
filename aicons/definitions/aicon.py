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
import tensorflow_probability as tfp
import time
import logging
from datetime import datetime

# Import BayesBrain and its components
from ..bayesbrainGPT.bayes_brain_ref import BayesBrain
from ..bayesbrainGPT.decision_making.action_space import (
    ActionSpace,
    ActionDimension
)
from ..bayesbrainGPT.decision_making.marketing_action_spaces import (
    create_budget_allocation_space,
    create_time_budget_allocation_space,
    create_multi_campaign_action_space,
    create_marketing_ads_space
)
from ..bayesbrainGPT.utility_function import create_utility
from ..bayesbrainGPT.state_representation.latent_variables import (
    LatentVariable,
    ContinuousLatentVariable,
    CategoricalLatentVariable,
    DiscreteLatentVariable
)
from ..bayesbrainGPT.state_representation import BayesianState

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
        # Implementation in subclasses
        pass
    
    def add_state_factor(self, name: str, factor_type: str, value: Any,
                        params: Dict[str, Any], relationships: Optional[Dict[str, Any]] = None) -> Union[ContinuousLatentVariable, CategoricalLatentVariable, DiscreteLatentVariable]:
        """
        Add a state factor through the brain.
        
        This method delegates to the brain's state management to ensure proper
        hierarchical relationships and consistency.
        
        Args:
            name: Name of the factor
            factor_type: Type of factor ('continuous', 'categorical', 'discrete')
            value: Initial value
            params: Type-specific parameters:
                - For continuous: {'scale': float, 'lower_bound': float, 'upper_bound': float}
                - For categorical: {'categories': List[str], 'probs': List[float]}
                - For discrete: {'categories': List[int], 'probs': List[float]} or {'rate': float}
            relationships: Optional hierarchical relationships with other factors
            
        Returns:
            The created latent variable
        """
        # Delegate to brain's state
        factor = self.brain.state.add_factor(
            name=name,
            factor_type=factor_type,
            value=value,
            params=params,
            relationships=relationships
        )
        
        # Verify that the returned factor is a LatentVariable
        assert isinstance(factor, LatentVariable), f"Expected LatentVariable, got {type(factor)}"
        
        return factor
    
    def get_state_factors(self) -> Dict[str, Any]:
        """
        Get state factors from the brain.
        
        Returns:
            Dictionary of state factors
        """
        return self.brain.get_state_factors()
    
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
        self.brain.add_sensor(name, sensor, factor_mapping)
        return sensor
    
    def update_from_sensor(self, sensor_name: Optional[Union[str, List[str]]] = None, environment: Any = None) -> bool:
        """
        Update beliefs based on data from sensors.
        
        Args:
            sensor_name: Name of the sensor to use, list of sensor names, or None to use all sensors
            environment: Optional environment data to pass to the sensor
            
        Returns:
            True if update was successful
        """
        if sensor_name is None:
            # If no sensor specified, use update_all which will use priors if no sensors
            return self.brain.perception.update_all(environment)
        elif isinstance(sensor_name, list):
            # If list of sensors provided, update from each one
            success = True
            for name in sensor_name:
                if not self.brain.update_from_sensor(name, environment):
                    success = False
            return success
        else:
            # Single sensor case
            return self.brain.update_from_sensor(sensor_name, environment)
    
    def update_from_all_sensors(self, environment: Any = None) -> bool:
        """
        Update beliefs based on data from all sensors.
        
        Args:
            environment: Optional environment data to pass to sensors
            
        Returns:
            True if update was successful
        """
        return self.brain.update_from_all_sensors(environment)
    
    def find_best_action(self, num_samples: Optional[int] = None) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Find the best action based on current beliefs.
        
        Args:
            num_samples: Number of actions to sample (for Monte Carlo methods)
            
        Returns:
            Tuple of (best_action, expected_utility)
        """
        start_time = time.time()
        logger = logging.getLogger(__name__)
        logger.info(f"Starting find_best_action with {num_samples if num_samples else 'default'} samples")

        try:
            # Get the result from brain
            result = self.brain.find_best_action(num_samples)
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result:
                best_action, expected_utility = result
                logger.info(f"find_best_action completed in {duration:.2f} seconds")
                logger.info(f"Best action found: {best_action}")
                logger.info(f"Expected utility: {expected_utility}")
            else:
                logger.warning(f"find_best_action completed in {duration:.2f} seconds but returned no result")
            
            return result
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            logger.error(f"find_best_action failed after {duration:.2f} seconds with error: {str(e)}")
            raise
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state beliefs."""
        return self.brain.get_state()
    
    def get_posterior_samples(self) -> Dict[str, Any]:
        """Get samples from the posterior distribution."""
        return self.brain.state.get_posterior_samples()
    
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
            
            # Print the actual utility function
            print(f"\nUtility Function: {utility}")
            
            return utility
            
        except Exception as e:
            print(f"Error creating utility function: {e}")
            return None 