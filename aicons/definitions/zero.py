"""
Zero AIcon Module

A simplified version of AIcon that uses Gemini for LLM integration and tracks context window usage.
The context window is split into four parts:
1. State Representation (priors, sensors, posteriors)
2. Utility Function
3. Action Space
4. Inference
"""

from typing import Dict, Any, Optional, List, Tuple, Literal, Callable, Union
import json
import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging
import numpy as np
import tensorflow_probability as tfp
import time
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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define available models and their context window sizes
MODEL_CONFIGS = {
    "gemini-1.5-flash": {
        "context_window": 1_000_000,  # 1M tokens
        "description": "Fast and versatile performance across a diverse variety of tasks"
    },
    "gemini-1.5-pro": {
        "context_window": 2_000_000,  # 2M tokens
        "description": "Complex reasoning tasks requiring more intelligence"
    }
}

class ZeroAIcon:
    """
    A simplified AIcon implementation that uses Gemini and tracks context window usage.
    """
    
    def __init__(self, name: str, description: str, model_name: str = "gemini-1.5-flash"):
        """Initialize ZeroAIcon."""
        self.name = name
        self.description = description
        self.model_name = model_name
        self.context_window_size = 1000000  # 1M tokens for Flash model
        
        # Initialize token usage tracking
        self.token_usage = {
            "state_representation": 0,
            "utility_function": 0,
            "action_space": 0,
            "inference": 0
        }
        
        # Initialize brain
        self.brain = BayesBrain()
        
        # Initialize LLM
        self.llm = None
        self._initialize_llm()
        
        # Validate model name
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Invalid model name. Must be one of: {list(MODEL_CONFIGS.keys())}")
        
        # Initialize Gemini client
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(model_name)
        
        # Initialize context window tracking
        self.model_config = MODEL_CONFIGS[model_name]
        self.context_window = self.model_config["context_window"]
        
        # Initialize BayesBrain
        self.brain = BayesBrain(
            name=f"{name}_brain",
            description=f"Bayesian brain for {name}"
        )
        self.brain.set_aicon(self)  # Set reference to this AIcon
        
        # Setup logging
        logger.info(f"Initialized {self.name} with {model_name} (context window: {self.context_window:,} tokens)")
        logger.info(f"Model description: {self.model_config['description']}")
        
    def _initialize_llm(self):
        # Implementation of _initialize_llm method
        pass
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using Gemini's token counter.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return self.model.count_tokens(text).total_tokens
    
    def _update_token_usage(self, component: str, text: str) -> None:
        """
        Update token usage for a specific component.
        
        Args:
            component: Component name ('state_representation', 'utility_function', etc.)
            text: Text to count tokens for
        """
        tokens = self._count_tokens(text)
        self.token_usage[component] += tokens
        remaining = self.context_window - sum(self.token_usage.values())
        
        if remaining < self.context_window * 0.1:  # Less than 10% remaining
            logger.warning(f"Low remaining tokens: {remaining:,} ({remaining/self.context_window:.1%} of context window)")
        
    def get_remaining_tokens(self) -> int:
        """
        Get remaining tokens in context window.
        
        Returns:
            Number of remaining tokens
        """
        total_used = sum(self.token_usage.values())
        remaining = self.context_window - total_used
        logger.info(f"Remaining tokens: {remaining:,}")
        return remaining
    
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
        # Update token usage for state representation
        self._update_token_usage("state_representation", str(sensor))
        logger.info(f"Added sensor: {name}")
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
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state beliefs."""
        return self.brain.get_state()
    
    def get_posterior_samples(self) -> Dict[str, Any]:
        """Get samples from the posterior distribution."""
        return self.brain.get_posterior_samples()
    
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
            
            # Update token usage
            self._update_token_usage("action_space", str(action_space))
            
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
            
            # Update token usage
            self._update_token_usage("utility_function", str(utility))
            
            # Print the actual utility function
            print(f"\nUtility Function: {utility}")
            
            return utility
            
        except Exception as e:
            print(f"Error creating utility function: {e}")
            return None
    
    def get_state_representation(self) -> Dict[str, Any]:
        """Get the current state representation."""
        return {
            "state": self.brain.state.get_state(),
            "posteriors": self.brain.state.get_posterior_samples(),
            "utility_function": str(self.brain.utility_function),
            "action_space": str(self.brain.action_space)
        }
    
    def make_inference(self, prompt: str) -> str:
        """
        Make inference using Gemini.
        
        Args:
            prompt: Prompt for inference
            
        Returns:
            Inference result
        """
        # Prepare context with all components
        context = {
            "state": self.get_state_representation(),
            "utility_function": str(self.brain.utility_function) if self.brain.utility_function else None,
            "action_space": str(self.brain.action_space),
            "prompt": prompt
        }
        
        # Update inference token usage
        self._update_token_usage("inference", json.dumps(context))
        
        # Get remaining tokens
        remaining = self.get_remaining_tokens()
        if remaining < 1000:  # Safety margin
            logger.warning("Low remaining tokens in context window")
            
        # Make inference using Gemini
        response = self.model.generate_content(json.dumps(context))
        
        return response.text
        
    def get_token_usage_report(self) -> Dict[str, Any]:
        """Get a report of token usage across components."""
        return {
            "state_representation": self.token_usage["state_representation"],
            "utility_function": self.token_usage["utility_function"],
            "action_space": self.token_usage["action_space"],
            "inference": self.token_usage["inference"],
            "total_used": sum(self.token_usage.values()),
            "remaining": self.context_window - sum(self.token_usage.values())
        } 