# Budget Allocation Demo using Bayesian Decision Making

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import time

# Fix import path
project_root = "/Users/infa/Documents/Babel"
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to sys.path")

# Import modules
from aicons.definitions.simple_bad_aicon import SimpleBadAIcon

# Define a realistic sensor for ad performance data
class AdPerformanceSensor:
    """Sensor that provides simulated ad performance data."""
    
    def __init__(self, num_ads=3, reliability=0.9, ad_names=None):
        self.num_ads = num_ads
        self.reliability = reliability
        self.ad_names = ad_names or [f"ad{i+1}" for i in range(num_ads)]
        
    def get_expected_factors(self):
        """Define the factors this sensor provides."""
        factors = {}
        
        for ad_name in self.ad_names:
            # Conversion rate factors
            factors[f"{ad_name}_conversion_rate"] = {
                "type": "continuous",
                "default_value": 0.02,  # 2% base conversion rate
                "uncertainty": 0.01,
                "lower_bound": 0.0,
                "upper_bound": 0.2,
                "description": f"Conversion rate for {ad_name}"
            }
            
            # Cost per click factors
            factors[f"{ad_name}_cost_per_click"] = {
                "type": "continuous",
                "default_value": 0.5,  # $0.50 per click
                "uncertainty": 0.2,
                "lower_bound": 0.1,
                "description": f"Cost per click for {ad_name}"
            }
            
            # Click-through rate factors
            factors[f"{ad_name}_ctr"] = {
                "type": "continuous", 
                "default_value": 0.01,  # 1% click-through rate
                "uncertainty": 0.005,
                "lower_bound": 0.0,
                "upper_bound": 0.1,
                "description": f"Click-through rate for {ad_name}"
            }
        
        # Add weather factor
        factors["weather_quality"] = {
            "type": "continuous",
            "default_value": 0.5,  # Neutral weather
            "uncertainty": 0.2,
            "lower_bound": 0.0,
            "upper_bound": 1.0,
            "description": "Weather quality (0=bad, 1=excellent)"
        }
        
        return factors
        
    def run(self, environment=None):
        """Generate ad performance data."""
        # Default ground truth values for each ad
        true_values = {}
        
        # Ad-specific conversion rates (some ads perform better than others)
        base_conversion_rates = {
            "ad1": 0.025,  # Good performer
            "ad2": 0.015,  # Average performer
            "ad3": 0.035   # Best performer
        }
        
        # Ad-specific costs per click (some ads are more expensive)
        base_costs = {
            "ad1": 0.45,   # Cheaper
            "ad2": 0.65,   # Average cost
            "ad3": 0.80    # More expensive
        }
        
        # Ad-specific CTRs
        base_ctrs = {
            "ad1": 0.012,  # Average CTR
            "ad2": 0.008,  # Lower CTR
            "ad3": 0.015   # Higher CTR
        }
        
        # Weather quality affects performance
        weather = environment.get("weather_quality", 0.5) if environment else 0.5
        
        # Generate true values with weather effects
        for ad_name in self.ad_names:
            if ad_name in base_conversion_rates:
                # Weather affects conversion rates differently for different ads
                weather_effect = 1.0 + (weather - 0.5) * 0.4  # +/- 20% effect
                
                # Apply weather effect
                true_values[f"{ad_name}_conversion_rate"] = base_conversion_rates[ad_name] * weather_effect
                true_values[f"{ad_name}_cost_per_click"] = base_costs[ad_name]
                true_values[f"{ad_name}_ctr"] = base_ctrs[ad_name] * weather_effect
        
        # Add weather
        true_values["weather_quality"] = weather
        
        # Apply noise based on reliability
        observations = {}
        for factor, true_value in true_values.items():
            noise = np.random.normal(0, (1.0 - self.reliability) * true_value)
            observations[factor] = max(0.001, true_value + noise)
            
        return observations


# MAIN DEMO SCRIPT

print("=== BAYESIAN BUDGET ALLOCATION DEMO ===")
print("This demo shows the process of making budget allocation decisions using Bayesian principles.")

# Step 1: Define Prior Beliefs
print("\n--- STEP 1: DEFINE PRIOR BELIEFS ---")
print("First, we define our prior beliefs about the performance of different ads")

aicon = SimpleBadAIcon("budget_allocation_demo")

# Define the ad names we'll use
ad_names = ["ad1", "ad2", "ad3"]
num_ads = len(ad_names)

# Set priors for each ad
for ad_name in ad_names:
    # Prior belief about conversion rate
    aicon.add_factor_continuous(
        name=f"{ad_name}_conversion_rate", 
        value=0.02,  # Initial belief: 2% conversion rate
        uncertainty=0.01,  # High uncertainty
        lower_bound=0,
        upper_bound=0.2,
        description=f"Conversion rate for {ad_name}"
    )
    
    # Prior belief about cost per click
    aicon.add_factor_continuous(
        name=f"{ad_name}_cost_per_click", 
        value=0.5,  # Initial belief: $0.50 per click
        uncertainty=0.2,  # High uncertainty
        lower_bound=0.1,
        description=f"Cost per click for {ad_name}"
    )
    
    # Prior belief about click-through rate
    aicon.add_factor_continuous(
        name=f"{ad_name}_ctr", 
        value=0.01,  # Initial belief: 1% CTR
        uncertainty=0.005,  # Moderate uncertainty
        lower_bound=0,
        upper_bound=0.1,
        description=f"Click-through rate for {ad_name}"
    )

# Prior belief about weather (which affects performance)
aicon.add_factor_continuous(
    name="weather_quality", 
    value=0.5,  # Neutral weather
    uncertainty=0.2,
    lower_bound=0,
    upper_bound=1,
    description="Weather quality (0=bad, 1=excellent)"
)

# Print prior beliefs
print("\nPrior beliefs set.")
aicon.print_current_state()

# Step 2: Define Action Space
print("\n--- STEP 2: DEFINE ACTION SPACE ---")
print("Next, we define the possible budget allocations we can choose from")

# Total marketing budget
total_budget = 1000.0  # $1000

# Create the action space
action_space = aicon.create_action_space(
    space_type='marketing',
    total_budget=total_budget,
    num_ads=num_ads,
    budget_step=100.0,  # $100 increments
    min_budget=0.0,     # Can allocate $0 to an ad
    ad_names=ad_names
)

# Print action space info
print(f"\nAction space created with {len(action_space.dimensions)} dimensions")
print(f"Total budget: ${total_budget}")
print(f"Budget step: ${action_space.dimensions[0].step}")
print(f"This creates {action_space.dimensions[0].size} possible allocations per ad")

# Step 3: Define Utility Function
print("\n--- STEP 3: DEFINE UTILITY FUNCTION ---")
print("Now, we define how to evaluate different budget allocations")

# Create utility function to maximize ROI
utility_function = aicon.create_utility_function(
    utility_type='weather_dependent_marketing_roi',  # Weather affects performance
    revenue_per_sale=50.0,  # Each conversion generates $50 in revenue
    num_ads=num_ads,
    ad_names=ad_names,
    # Weather affects different ads differently
    weather_effects={
        'sunny': {'ad1': 1.2, 'ad2': 0.9, 'ad3': 1.1},  # Ad1 does better in good weather
        'rainy': {'ad1': 0.8, 'ad2': 1.1, 'ad3': 0.9}   # Ad2 does better in bad weather
    }
)

print(f"\nUtility function created: {utility_function.name}")
print(f"Description: {utility_function.description}")

# Step 4: Add Sensor and Update Beliefs
print("\n--- STEP 4: UPDATE BELIEFS WITH SENSOR DATA ---")
print("We'll now update our beliefs using data from our ad performance sensor")

# Create and add the sensor
sensor = AdPerformanceSensor(num_ads=num_ads, reliability=0.8, ad_names=ad_names)
aicon.add_sensor("ad_performance", sensor)
print("\nSensor added. Now updating beliefs...")

# Environment with good weather
environment = {"weather_quality": 0.7}  # Good weather 

# Update beliefs based on sensor data
aicon.update_from_sensor("ad_performance", environment)

# Show posterior distribution
print("\nBeliefs updated based on sensor data")
aicon.print_posterior()

# Step 5: Compute Expected Utility
print("\n--- STEP 5: COMPUTE EXPECTED UTILITY ---")
print("Now we evaluate the expected utility of different budget allocations")

# Sample a few random allocations and compute their expected utility
print("\nEvaluating some sample budget allocations:")
for i in range(3):
    random_action = aicon.sample_action()
    expected_utility = 0
    
    # Compute expected utility across 50 posterior samples
    posterior_samples = aicon.get_posterior_samples()
    if posterior_samples:
        first_factor = next(iter(posterior_samples.values()))
        num_samples = min(50, len(first_factor))
        
        for j in range(num_samples):
            # Extract the j-th sample from each factor
            sample = {k: v[j] if isinstance(v, np.ndarray) else v for k, v in posterior_samples.items()}
            
            # Compute utility for this sample
            utility = utility_function.evaluate(random_action, sample)
            expected_utility += utility
            
        expected_utility /= num_samples
    
    # Print the allocation and its expected utility
    print(f"\nSample allocation #{i+1}:")
    for name, budget in random_action.items():
        percentage = (budget / total_budget) * 100
        print(f"- {name}: ${budget:.2f} ({percentage:.1f}%)")
    print(f"Expected utility: ${expected_utility:.2f}")

# Step 6: Make Decision
print("\n--- STEP 6: MAKE BUDGET ALLOCATION DECISION ---")
print("Finally, we find the budget allocation that maximizes expected utility")

# Find the best allocation
best_action, best_utility = aicon.find_best_action(num_samples=200)

# Print the optimal allocation
print(f"\nOptimal budget allocation found with expected utility: ${best_utility:.2f}")
print("Budget allocation:")
for name, budget in best_action.items():
    percentage = (budget / total_budget) * 100
    print(f"- {name}: ${budget:.2f} ({percentage:.1f}%)")

# Show what happens with different weather
print("\n--- BONUS: TESTING DIFFERENT WEATHER CONDITIONS ---")

# Test different weather values
weather_values = [0.2, 0.5, 0.8]  # Bad, neutral, good
weather_labels = ["Bad", "Neutral", "Good"]

for weather, label in zip(weather_values, weather_labels):
    print(f"\nTesting {label.lower()} weather (weather_quality = {weather})")
    
    # Update environment
    environment = {"weather_quality": weather}
    
    # Update beliefs
    aicon.update_from_sensor("ad_performance", environment)
    
    # Find best allocation
    weather_action, weather_utility = aicon.find_best_action(num_samples=200)
    
    # Print allocation
    print(f"Optimal allocation for {label.lower()} weather - Utility: ${weather_utility:.2f}")
    for name, budget in weather_action.items():
        percentage = (budget / total_budget) * 100
        print(f"- {name}: ${budget:.2f} ({percentage:.1f}%)")

print("\n--- DEMO COMPLETE ---")
print("This demonstration showed the Bayesian decision-making process:")
print("1. We defined prior beliefs about ad performance")
print("2. We specified an action space of possible budget allocations")
print("3. We defined a utility function to evaluate allocations")
print("4. We updated our beliefs using sensor data (Bayesian inference)")
print("5. We computed expected utility across posterior samples")
print("6. We selected the allocation that maximizes expected utility")
print("\nThe process adapts to changing conditions, as shown by the weather tests.") 