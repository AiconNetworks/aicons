"""
Zero AIcon Module

A simplified version of AIcon that uses various LLM backends for integration and tracks context window usage.
The context window is split into four parts:
1. State Representation (priors, sensors, posteriors)
2. Utility Function
3. Action Space
4. Inference
"""

from typing import Dict, Any, Optional, List, Tuple, Literal, Callable, Union, AsyncGenerator
import json
import os
from dotenv import load_dotenv
import logging
import numpy as np
import tensorflow_probability as tfp
import time
from datetime import datetime
import asyncio
import aiohttp

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

# Import LLM implementations
from .llms import create_llm, BaseLLM

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZeroAIcon:
    """
    A simplified AIcon implementation that supports multiple LLM backends.
    """
    
    def __init__(self, name: str, description: str, model_name: str = "deepseek-r1:7b"):
        """Initialize ZeroAIcon."""
        self.name = name
        self.description = description
        self.model_name = model_name
        
        # Initialize token usage tracking
        self.token_usage = {
            "state_representation": 0,
            "utility_function": 0,
            "action_space": 0,
            "inference": 0
        }
        
        # Initialize LLM
        self.llm = create_llm(model_name)
        
        # Initialize brain
        self.brain = BayesBrain(
            name=f"{name}_brain",
            description=f"Bayesian brain for {name}"
        )
        self.brain.set_aicon(self)  # Set reference to this AIcon
        
        # Setup logging
        logger.info(f"Initialized {self.name} with {model_name}")
        logger.info(f"Context window: {self.llm.context_window:,} tokens")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using LLM's token counter."""
        return self.llm.count_tokens(text)
    
    def _update_token_usage(self, component: str, text: str) -> None:
        """Update token usage for a specific component."""
        tokens = self._count_tokens(text)
        self.token_usage[component] += tokens
        remaining = self.llm.context_window - sum(self.token_usage.values())
        
        if remaining < self.llm.context_window * 0.1:  # Less than 10% remaining
            logger.warning(f"Low remaining tokens: {remaining:,} ({remaining/self.llm.context_window:.1%} of context window)")
    
    def get_remaining_tokens(self) -> int:
        """Get remaining tokens in context window."""
        total_used = sum(self.token_usage.values())
        remaining = self.llm.context_window - total_used
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
        return self.brain.state.get_state_factors()
    
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
        return self.brain.state.get_beliefs()
    
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
            "state": self.brain.state.get_beliefs(),
            "posteriors": self.brain.state.get_posterior_samples(),
            "utility_function": str(self.brain.utility_function),
            "action_space": str(self.brain.action_space)
        }
    
    async def make_inference(self, prompt: str) -> str:
        """Make inference using configured LLM."""
        try:
            logger.info(f"Starting inference with prompt: {prompt[:50]}...")
            
            # Prepare context with all components
            context = {
                "state": self.get_state_representation(),
                "utility_function": str(self.brain.utility_function) if self.brain.utility_function else None,
                "action_space": str(self.brain.action_space),
                "prompt": prompt
            }
            
            # Update inference token usage
            context_str = json.dumps(context)
            self._update_token_usage("inference", context_str)
            logger.info(f"Inference context prepared, size: {len(context_str)} characters")
            
            # Get remaining tokens
            remaining = self.get_remaining_tokens()
            if remaining < 1000:  # Safety margin
                logger.warning(f"Low remaining tokens in context window: {remaining}")
            
            # Make inference using LLM
            logger.info(f"Calling LLM ({self.model_name}) for inference")
            full_response = ""
            accumulated_chunk = ""
            start_time = time.time()
            
            try:
                # Set a timeout for the entire operation
                timeout_seconds = 60  # Adjust as needed
                logger.info(f"Setting request timeout to {timeout_seconds} seconds")
                
                # Create a task with timeout
                async def generate_with_timeout():
                    nonlocal full_response, accumulated_chunk
                    chunk_count = 0
                    
                    async for chunk in self.llm.generate(context_str):
                        chunk_count += 1
                        full_response += chunk
                        accumulated_chunk += chunk
                        
                        # Print meaningful segments (sentences, paragraphs) or after accumulating enough content
                        if '\n' in accumulated_chunk or len(accumulated_chunk) > 50 or '.' in accumulated_chunk:
                            logger.info(f"Message segment: {accumulated_chunk}")
                            accumulated_chunk = ""
                        
                        # Log progress periodically
                        if len(full_response) % 200 == 0:
                            logger.info(f"Received {len(full_response)} characters so far")
                
                try:
                    # Run with timeout
                    await asyncio.wait_for(generate_with_timeout(), timeout=timeout_seconds)
                    
                    # Log any remaining accumulated chunk
                    if accumulated_chunk:
                        logger.info(f"Final segment: {accumulated_chunk}")
                        
                except asyncio.TimeoutError:
                    logger.error(f"LLM request timed out after {timeout_seconds} seconds")
                    logger.info(f"Partial response at timeout: {full_response}")
                    return f"Error: Request timed out after {timeout_seconds} seconds. Partial response: {full_response}"
                except asyncio.CancelledError as ce:
                    partial_response = full_response if full_response else "No response received"
                    logger.error(f"Request was cancelled. Partial response: {partial_response[:100]}...")
                    logger.info(f"Full partial response: {partial_response}")
                    raise RuntimeError(f"Request cancelled. Partial: {partial_response[:100]}...") from ce
                
            except Exception as e:
                logger.error(f"Error during LLM generation: {str(e)}", exc_info=True)
                # Try to return any partial response
                if full_response:
                    logger.info(f"Returning partial response of {len(full_response)} characters: {full_response}")
                    return f"Error occurred, but partial response available: {full_response}"
                raise
            
            elapsed_time = time.time() - start_time
            logger.info(f"Inference completed in {elapsed_time:.2f}s, response length: {len(full_response)} chars")
            logger.info(f"Complete response: {full_response}")
            
            # Process the response to remove the thinking part
            processed_response = full_response
            
            # Remove the <think>...</think> section if present
            think_start = processed_response.find("<think>")
            think_end = processed_response.find("</think>")
            
            if think_start != -1 and think_end != -1 and think_end > think_start:
                # Extract the content before <think> and after </think>
                before_think = processed_response[:think_start].strip()
                after_think = processed_response[think_end + 8:].strip()  # 8 is the length of "</think>"
                
                # Combine the parts, with space in between if both exist
                if before_think and after_think:
                    processed_response = before_think + " " + after_think
                else:
                    processed_response = before_think + after_think
                
                logger.info(f"Removed thinking process. Final response: {processed_response}")
            
            return processed_response.strip()
        except Exception as e:
            logger.error(f"Error in make_inference: {str(e)}", exc_info=True)
            return f"Error: {str(e)}"
    
    def get_state_representation_text(self) -> str:
        """Get the actual text content of the state representation."""
        if not self.brain:
            return "No brain defined"
        
        # Get current state and posterior samples
        state = self.brain.state.get_beliefs()
        posteriors = self.brain.state.get_posterior_samples()
        
        # Create output format
        output = []
        
        # Add state factors
        if state:
            output.append("=== State Factors ===")
            for name, value in state.items():
                # Get the corresponding posterior samples if available
                if name in posteriors:
                    samples = posteriors[name]
                    if isinstance(samples, np.ndarray) and len(samples) > 0:
                        mean = np.mean(samples)
                        std = np.std(samples)
                        output.append(f"{name:20s}:")
                        output.append(f"  Current: {value:10.2f}")
                        output.append(f"  Mean:    {mean:10.2f}")
                        output.append(f"  Std:     {std:10.2f}")
                    else:
                        output.append(f"{name:20s}: {value:10.2f}")
                else:
                    output.append(f"{name:20s}: {value:10.2f}")
        
        # Add belief history if available
        if hasattr(self.brain, 'update_history') and self.brain.update_history:
            if output:  # Add a newline if we already have content
                output.append("")
            output.append("=== Belief Update History ===")
            for update in self.brain.update_history:
                output.append(f"\nUpdate at {update.get('timestamp', 'unknown time')}:")
                for name, value in update.get('values', {}).items():
                    output.append(f"  {name}: {value}")
        
        # If we have no content at all, return a message indicating empty state
        if not output:
            return "No state factors defined"
            
        return "\n".join(output)

    def get_utility_function_text(self) -> str:
        """Get the actual text content of the utility function."""
        if not self.brain or not self.brain.utility_function:
            return "No utility function defined"
        
        return str(self.brain.utility_function)

    def get_action_space_text(self) -> str:
        """Get the actual text content of the action space."""
        if not self.brain or not self.brain.action_space:
            return "No action space defined"
        
        return self.brain.action_space.raw_print()

    def get_inference_text(self) -> str:
        """Get the actual text content of the inference queries."""
        if not hasattr(self, 'inference_queries'):
            return "No inference queries yet"
        
        return json.dumps(self.inference_queries, indent=2)

    def get_token_usage_report(self) -> Dict[str, Any]:
        """Get a report of token usage across components with actual content."""
        # Get the raw content first
        state_repr = self.get_state_representation()
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
            
        state_repr = convert_numpy(state_repr)
        state_repr_str = json.dumps(state_repr)
        
        action_space_str = str(self.brain.action_space) if self.brain.action_space else ""
        utility_str = str(self.brain.utility_function) if self.brain.utility_function else ""
        
        # Count tokens and compare with estimation
        def count_and_compare(text: str, component: str) -> int:
            tokens = self._count_tokens(text)
            estimated = len(text) // 4
            print(f"{component}: {tokens} tokens (est: {estimated})")
            return tokens
        
        # Count tokens for each component
        state_repr_tokens = count_and_compare(state_repr_str, "State Representation")
        action_space_tokens = count_and_compare(action_space_str, "Action Space")
        utility_tokens = count_and_compare(utility_str, "Utility Function")
        
        return {
            "state_representation": {
                "tokens": state_repr_tokens,
                "content": self.get_state_representation_text()
            },
            "utility_function": {
                "tokens": utility_tokens,
                "content": self.get_utility_function_text()
            },
            "action_space": {
                "tokens": action_space_tokens,
                "content": self.get_action_space_text()
            },
            "inference": {
                "tokens": self.token_usage["inference"],
                "content": self.get_inference_text()
            },
            "total_used": state_repr_tokens + action_space_tokens + utility_tokens + self.token_usage["inference"],
            "remaining": self.llm.context_window - (state_repr_tokens + action_space_tokens + utility_tokens + self.token_usage["inference"])
        } 