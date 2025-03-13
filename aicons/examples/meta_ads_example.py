"""
Meta Ads Example for SimpleBadAIcon

This example shows how to use the MetaAdsSalesSensor with SimpleBadAIcon
to perform perception and update the AIcon's beliefs based on Meta Ads data.
"""

import tensorflow as tf
import numpy as np
from aicons.definitions.simple_bad_aicon import SimpleBadAIcon
from aicons.bayesbrainGPT.sensors.meta_s.meta_ads_sales_sensor import MetaAdsSalesSensor

def run_meta_ads_example():
    """Run the Meta Ads example with SimpleBadAIcon"""
    
    print("=== Meta Ads Example with SimpleBadAIcon ===")
    
    # Step 1: Create the AIcon
    aicon = SimpleBadAIcon(name="Meta Ads Campaign")
    print(f"Created AIcon: {aicon.name}")
    
    # Step 2: Add factors (priors)
    aicon.add_factor_continuous(
        name="purchases",
        value=20.0,
        uncertainty=10.0,
        lower_bound=0.0,
        description="Number of purchases from ads"
    )
    
    aicon.add_factor_continuous(
        name="purchase_roas",
        value=2.0,
        uncertainty=0.5,
        lower_bound=0.0,
        description="Return on ad spend for purchases"
    )
    
    aicon.add_factor_continuous(
        name="add_to_carts",
        value=100.0,
        uncertainty=30.0,
        lower_bound=0.0,
        description="Number of add to cart events"
    )
    
    aicon.add_factor_continuous(
        name="initiated_checkouts",
        value=40.0,
        uncertainty=15.0,
        lower_bound=0.0,
        description="Number of checkout initiated events"
    )
    
    # Print the prior state
    print("\nPRIOR STATE:")
    print(aicon.get_state(format_nicely=True))
    
    # Step 3: Create and register the Meta Ads sensor
    # In a real scenario, you would provide your access token and account IDs
    sensor = MetaAdsSalesSensor(
        name="meta_ads",
        reliability=0.9,
        access_token=None,  # Replace with your token if available
        ad_account_id=None,  # Replace with your account ID if available
        campaign_id=None     # Replace with your campaign ID if available
    )
    
    # Define factor mapping - this maps sensor factor names to state factor names
    # In this case, they're the same, but this mapping would be useful if they were different
    factor_mapping = {
        "purchases": "purchases",
        "add_to_carts": "add_to_carts", 
        "initiated_checkouts": "initiated_checkouts",
        "purchase_roas": "purchase_roas"
    }
    
    # Register sensor with perception system
    aicon.add_sensor("meta_ads", sensor, factor_mapping)
    print(f"\nSensor added: {sensor.name}")
    print(f"Observable factors: {sensor.observable_factors}")
    
    # Step 4: Run perception once - update from sensor
    print("\nUpdating beliefs from Meta Ads sensor...")
    update_result = aicon.update_from_sensor("meta_ads")
    print(f"Update result: {update_result}")
    
    # Step 5: Examine the posterior samples
    posterior_samples = aicon.get_posterior_samples()
    print("\nPosterior samples keys:", list(posterior_samples.keys()))
    print("Number of samples:", {k: len(v) if hasattr(v, '__len__') else 'N/A' for k, v in posterior_samples.items()})
    
    # Print sample statistics for each factor if available
    for name, samples in posterior_samples.items():
        if isinstance(samples, np.ndarray):
            print(f"\n{name} posterior:")
            print(f"  Mean: {np.mean(samples):.4f}")
            print(f"  Std: {np.std(samples):.4f}")
            print(f"  Min: {np.min(samples):.4f}")
            print(f"  Max: {np.max(samples):.4f}")
    
    # Print the posterior state
    print("\nPOSTERIOR STATE:")
    print(aicon.get_state(format_nicely=True))
    
    return aicon

# Run the example if this script is executed directly
if __name__ == "__main__":
    aicon = run_meta_ads_example() 