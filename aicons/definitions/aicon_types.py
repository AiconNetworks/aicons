from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import sys
import os
from pathlib import Path
import numpy as np

# Add parent directory to sys.path to allow importing bayesbrainGPT
sys.path.append(str(Path(__file__).parent.parent))
from bayesbrainGPT.state_representation.state import EnvironmentState
from aicons.bayesbrainGPT.decision_making.action_space import (
    ActionDimension,
    ActionSpace,
    create_budget_allocation_space
)

# Import the BayesBrain class
from aicons.bayesbrainGPT.bayes_brain import BayesBrain

# Base AIcon class (Abstract Base Class)
class BaseAIcon(ABC):
    """Base class for all AI Icons (AIcons)"""
    
    def __init__(self, name: str, aicon_type: str, capabilities: List[str]):
        self.name = name
        self.type = aicon_type
        self.capabilities = capabilities
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the AIcon"""
        pass
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return results"""
        pass
    
    @abstractmethod
    def update_state(self, state_data: Dict[str, Any]) -> None:
        """Update internal state with new data"""
        pass

@dataclass
class Campaign:
    """Represents an advertising campaign"""
    id: str
    name: str
    platform: str  # e.g., "facebook", "google", "meta"
    total_budget: float
    daily_budget: float
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    status: str = "active"  # active, paused, completed
    
    def update_metrics(self, new_metrics: Dict[str, float]) -> None:
        """Update performance metrics for the campaign"""
        self.performance_metrics.update(new_metrics)

class BadAIcon(BaseAIcon):
    """Budget Allocation Decision (BAD) AIcon for managing ad campaigns"""
    
    def __init__(self, name: str, capabilities: List[str]):
        super().__init__(name, "bad", capabilities)
        self.campaigns: Dict[str, Campaign] = {}
        self.meta_campaign: Optional[Campaign] = None
        self.budget_increment = 100.0  # Budget allocation increment in dollars
        
        # Initialize an empty BayesBrain
        self.brain = BayesBrain()
    
    def initialize(self) -> None:
        """Initialize the BAD AIcon with a meta campaign and empty BayesBrain"""
        # Create the meta campaign that will manage budget for all campaigns
        self.meta_campaign = Campaign(
            id="meta_campaign_001",
            name="Meta Budget Allocation Campaign",
            platform="meta",
            total_budget=10000.0,
            daily_budget=500.0,
            performance_metrics={
                "roi": 1.0,
                "impressions": 0,
                "clicks": 0,
                "conversions": 0
            }
        )
        
        # Initialize state factors
        self._initialize_state_factors()
        
        # Initialize posterior samples
        self._initialize_posterior_samples()
        
        # Initialize the action space
        self._initialize_action_space()
        
        # Set the utility function
        self._initialize_utility_function()
        
        # Add sensors
        self._initialize_sensors()
        
        print(f"Initialized BadAIcon '{self.name}' with {len(self.campaigns)} campaigns")
    
    def _initialize_state_factors(self) -> None:
        """Initialize state factors for the BayesBrain"""
        state_factors = {
            "roi": {
                "type": "continuous",
                "value": 1.0,
                "description": "Return on Investment"
            },
            "ctr": {
                "type": "continuous",
                "value": 0.02,
                "description": "Click-through Rate"
            },
            "cpc": {
                "type": "continuous",
                "value": 1.0,
                "description": "Cost Per Click"
            },
            "conversion_rate": {
                "type": "continuous",
                "value": 0.05,
                "description": "Conversion Rate"
            }
        }
        
        # Set state factors in the BayesBrain
        self.brain.set_state_factors(state_factors)
    
    def _initialize_sensors(self) -> None:
        """Initialize sensors for the BayesBrain"""
        # Define a sensor function to collect campaign performance data
        def campaign_performance_sensor(environment):
            """Sensor for collecting campaign performance data"""
            # In a real implementation, this would collect data from the environment
            # For now, we'll just return an empty dictionary
            return {}
        
        # Add the sensor to the BayesBrain
        self.brain.add_sensor(campaign_performance_sensor)
    
    def _initialize_action_space(self) -> None:
        """Initialize the action space for budget allocation decisions"""
        num_ads = max(1, len(self.campaigns))
        total_budget = self.meta_campaign.daily_budget if self.meta_campaign else 1000.0
        
        # Create a budget allocation action space
        action_space = create_budget_allocation_space(
            total_budget=total_budget,
            num_ads=num_ads,
            budget_step=self.budget_increment,
            min_budget=0.0
        )
        
        # Set the action space in the BayesBrain
        self.brain.set_action_space(action_space)
        
        print(f"Initialized action space with {num_ads} ad dimensions")
        print(f"Total budget: ${total_budget:.2f}, increment: ${self.budget_increment:.2f}")
        
        # Print the dimensions of the action space
        for dim in action_space.dimensions:
            print(f"- {dim.name}: {dim.dim_type}, range: {dim.min_value} to {dim.max_value}, step: {dim.step}")
    
    def _initialize_posterior_samples(self) -> None:
        """Initialize posterior samples with reasonable priors"""
        num_ads = max(1, len(self.campaigns))
        num_days = 3  # Default to 3-day effect modifiers
        num_samples = 1000  # Number of samples to maintain
        
        # Conversion rate samples (phi) - around 5% conversion rate with small variance
        phi_samples = np.random.normal(
            0.05, 0.01, size=(num_samples, num_ads)
        )
        
        # Cost per click samples (c) - around $0.70 per click with gamma distribution
        c_samples = np.random.gamma(
            5.0, 1/7.0, size=(num_samples, num_ads)
        )
        
        # Day effect multiplier samples (delta) - log-normal distribution centered at 1.0
        delta_samples = np.exp(np.random.normal(
            0, 0.3, size=(num_samples, num_days)
        ))
        
        # Set the posterior samples in the BayesBrain
        self.brain.set_posterior_samples({
            "phi": phi_samples,
            "c": c_samples,
            "delta": delta_samples
        })
    
    def _initialize_utility_function(self) -> None:
        """Initialize the utility function for decision-making"""
        # Define a utility function based on our posterior samples
        def calculate_expected_roi(action: Dict[str, float]) -> float:
            """Calculate expected ROI for a given budget allocation"""
            # Get campaign IDs in the right order
            campaign_ids = list(self.campaigns.keys())
            
            # Convert action dict to numpy array in the right order
            budgets = np.array([action[f"ad{i+1}_budget"] for i in range(len(campaign_ids))])
            
            # Get posterior samples
            posterior_samples = self.brain.posterior_samples
            phi_samples = posterior_samples["phi"]  # Conversion rate samples
            c_samples = posterior_samples["c"]      # Cost per click samples
            
            # Calculate expected sales for each sample
            # Simple model: sales = budget * conversion_rate / cost_per_click
            expected_clicks = budgets / c_samples
            expected_sales = expected_clicks * phi_samples
            
            # Calculate ROI: (revenue - cost) / cost
            revenue_per_sale = 10.0  # Assume $10 revenue per conversion
            expected_revenue = np.sum(expected_sales * revenue_per_sale, axis=1)
            total_cost = np.sum(budgets)
            
            if total_cost == 0:
                return 0.0
                
            expected_roi = (expected_revenue - total_cost) / total_cost
            
            # Return mean ROI across all samples
            return float(np.mean(expected_roi))
        
        # Set the utility function in the BayesBrain
        self.brain.set_utility_function(calculate_expected_roi)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data (campaign performance) and make budget allocation decisions
        Returns updated budget allocations
        """
        # Update internal state with new data
        self.update_state(input_data)
        
        # Create a simple environment object from the input data
        environment = {
            "campaigns": self.campaigns,
            "meta_campaign": self.meta_campaign,
            "input_data": input_data
        }
        
        # Use BayesBrain to perceive the environment and decide on the best action
        best_action, expected_roi = self.brain.perceive_and_decide(environment)
        
        # Convert action space allocation to campaign_id dictionary
        allocations = {}
        if best_action:
            campaign_ids = list(self.campaigns.keys())
            for i, campaign_id in enumerate(campaign_ids):
                allocations[campaign_id] = best_action[f"ad{i+1}_budget"]
        else:
            # Fallback to equal allocation if no valid action found
            total_budget = self.meta_campaign.daily_budget if self.meta_campaign else 1000.0
            equal_share = total_budget / max(1, len(self.campaigns))
            allocations = {campaign_id: equal_share for campaign_id in self.campaigns.keys()}
            expected_roi = 0.0
        
        # Update campaign budgets based on allocations
        if self.meta_campaign and self.meta_campaign.status == "active":
            self._apply_budget_allocations(allocations)
            
        # Get current state factors
        state_factors = self.brain.get_state_factors()
            
        return {
            "budget_allocations": allocations,
            "expected_roi": expected_roi,
            "meta_campaign_status": self.meta_campaign.status if self.meta_campaign else "not_initialized",
            "action_space_dimensions": len(self.brain.get_action_space().dimensions) if self.brain.get_action_space() else 0,
            "state_factors": state_factors
        }
    
    def update_state(self, state_data: Dict[str, Any]) -> None:
        """Update internal state with new campaign performance data"""
        # Update state factors in the BayesBrain
        for key, value in state_data.items():
            if key in self.brain.get_state_factors():
                self.brain.update_state_factor(key, value)
        
        # Update meta campaign metrics if applicable
        if self.meta_campaign and "campaign_metrics" in state_data:
            self.meta_campaign.update_metrics(state_data["campaign_metrics"])
            
        # Update posterior samples if campaign performance data is provided
        if "campaign_performance" in state_data:
            self._update_posterior_samples(state_data["campaign_performance"])
    
    def _update_posterior_samples(self, performance_data: Dict[str, Any]) -> None:
        """
        Update posterior samples based on new campaign performance data
        
        In a full implementation, this would run a proper Bayesian inference step
        using TensorFlow Probability, PyMC, or similar
        """
        # Define an update function for the posterior samples
        def update_function(posterior_samples, performance_data):
            """Update posterior samples based on new performance data"""
            # For now, we'll use a simplified update that adjusts the existing samples
            # In a real implementation, this would be replaced with proper Bayesian inference
            
            # Get observed data
            observed_sales = np.array(performance_data["observed_sales"])  # Shape: [days, ads]
            observed_cpc = np.array(performance_data["observed_cpc"])  # Shape: [ads]
            budgets = np.array(performance_data.get("budgets", np.ones_like(observed_sales)))  # Shape: [days, ads]
            
            # Simple Bayesian update simulation (not real inference)
            # Shift phi samples toward observed conversion rates
            num_days, num_ads = observed_sales.shape
            
            # Make a copy of the posterior samples to avoid modifying the original
            updated_samples = {
                "phi": posterior_samples["phi"].copy(),
                "c": posterior_samples["c"].copy(),
                "delta": posterior_samples["delta"].copy()
            }
            
            # Ensure our posterior samples match the current number of ads
            if updated_samples["phi"].shape[1] != num_ads:
                # If the number of ads has changed, we need to reinitialize
                return None
            
            # Update phi (conversion rate) - simplified
            for d in range(num_days):
                for i in range(num_ads):
                    if budgets[d, i] > 0:
                        observed_rate = observed_sales[d, i] / budgets[d, i]
                        updated_samples["phi"][:, i] = (
                            0.95 * updated_samples["phi"][:, i] + 
                            0.05 * observed_rate
                        )
            
            # Update c (cost per click) - simplified
            for i in range(num_ads):
                updated_samples["c"][:, i] = (
                    0.95 * updated_samples["c"][:, i] + 
                    0.05 * observed_cpc[i]
                )
            
            return updated_samples
        
        # Use the BayesBrain to update the posterior samples
        self.brain.update_posterior_samples(performance_data, update_function)
        
        # If the update function returned None, we need to reinitialize
        if self.brain.get_posterior_samples() is None:
            self._initialize_posterior_samples()
    
    def _apply_budget_allocations(self, allocations: Dict[str, float]) -> None:
        """Apply the calculated budget allocations to the campaigns"""
        for campaign_id, budget in allocations.items():
            if campaign_id in self.campaigns:
                self.campaigns[campaign_id].daily_budget = budget
    
    def add_campaign(self, campaign: Campaign) -> None:
        """Add a campaign to the BAD AIcon"""
        self.campaigns[campaign.id] = campaign
        print(f"Added campaign: {campaign.name} (ID: {campaign.id})")
        
        # Reinitialize posterior samples and action space with the new campaign count
        self._initialize_posterior_samples()
        self._initialize_action_space()
        self._initialize_utility_function()
    
    def remove_campaign(self, campaign_id: str) -> bool:
        """Remove a campaign from the BAD AIcon"""
        if campaign_id in self.campaigns:
            campaign = self.campaigns.pop(campaign_id)
            print(f"Removed campaign: {campaign.name} (ID: {campaign_id})")
            
            # Reinitialize posterior samples and action space with the updated campaign count
            self._initialize_posterior_samples()
            self._initialize_action_space()
            self._initialize_utility_function()
            return True
        else:
            print(f"Campaign with ID {campaign_id} not found")
            return False
    
    def set_budget_increment(self, increment: float) -> None:
        """Set the budget allocation increment"""
        self.budget_increment = increment
        print(f"Set budget increment to ${increment:.2f}")
        
        # Reinitialize the action space with the new increment
        self._initialize_action_space()
        
    def get_action_space(self):
        """Get the current action space"""
        return self.brain.get_action_space()
        
    def sample_allocation(self):
        """Sample a random allocation from the action space"""
        return self.brain.sample_action()

@dataclass
class AgentDefinition:
    """Defines an AI agent's characteristics and capabilities"""
    name: str
    type: str
    capabilities: List[str]
    state_factors: Dict[str, Any]
    decision_rules: List[str]

# Example agent definitions
MARKETING_AGENT = AgentDefinition(
    name="marketing_specialist",
    type="marketing",
    capabilities=[
        "campaign_optimization",
        "budget_management",
        "audience_targeting"
    ],
    state_factors={
        "campaign_roi": {
            "type": "continuous",
            "initial_value": 1.0
        },
        "engagement_rate": {
            "type": "continuous",
            "initial_value": 0.5
        }
    },
    decision_rules=[
        "evaluate_campaign_performance",
        "get_budget_adjustment"
    ]
)

ROOM_ENVIRONMENT_AGENT = AgentDefinition(
    name="room_environment_monitor",
    type="room_environment",
    capabilities=[
        "temperature_monitoring",
        "occupancy_tracking",
        "comfort_assessment",
        "air_quality_monitoring"
    ],
    state_factors={
        "room_temperature": {
            "type": "continuous",
            "initial_value": 20.0,
            "description": "Room temperature in Celsius"
        },
        "comfort_level": {
            "type": "categorical",
            "initial_value": "comfortable",
            "possible_values": ["cold", "comfortable", "hot"],
            "description": "Comfort level based on temperature"
        },
        "room_occupancy": {
            "type": "continuous",
            "initial_value": 2.0,
            "description": "Number of people in the room (0-10)",
            "relationships": {
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
        },
        "co2_level": {
            "type": "continuous",
            "initial_value": 400.0,
            "description": "CO2 concentration in ppm",
            "relationships": {
                "depends_on": ["room_occupancy", "comfort_level"],
                "model": {
                    "room_occupancy": {
                        "type": "linear",
                        "base": 400.0,
                        "coefficient": 50.0
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
        }
    },
    decision_rules=[
        "evaluate_comfort_conditions",
        "suggest_ventilation_actions",
        "monitor_air_quality"
    ]
)

AIQUON_AGENT = AgentDefinition(
    name="aiquon",
    type="aiquon",
    capabilities=[
        "bayesian_inference",
        "factor_management",
        "state_tracking",
        "relationship_learning"
    ],
    state_factors={},  # Empty by default since factors will be defined dynamically
    decision_rules=[
        "update_state",
        "infer_relationships",
        "predict_changes"
    ]
)

# Create a BAD AIcon instance for ad budget management
BAD_AICON = BadAIcon(
    name="ad_budget_allocator",
    capabilities=[
        "budget_optimization",
        "roi_analysis",
        "campaign_performance_tracking",
        "bayesian_decision_making"
    ]
)

# Updated AGENT_TYPES dictionary
AGENT_TYPES = {
    "marketing": MARKETING_AGENT,
    "room_environment": ROOM_ENVIRONMENT_AGENT,
    "aiquon": AIQUON_AGENT,
    "bad": BAD_AICON  # Add the BAD AIcon to the available agent types
} 