import importlib
import sys
import os
import json
from pprint import pprint

# Add the project root to the Python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the Meta Ads Sales Sensor
from aicons.bayesbrainGPT.sensors.meta_s.meta_ads_sales_sensor import MetaAdsSalesSensor
from aicons.definitions.simple_bad_aicon import SimpleBadAIcon

# Create a sensor instance (without real credentials)
sensor = MetaAdsSalesSensor(
    name="meta_ads_test",
    reliability=0.9,
    # We're not providing access_token or ad_account_id, so it will use sample data
)

print("\n=== Meta Ads Sales Sensor Test ===")
print(f"Sensor name: {sensor.name}")
print(f"Observable factors: {sensor.observable_factors}")

# Run the sensor to get the data
print("\nFetching data from sensor...")
data = sensor.run()

print("\nAggregated Campaign Metrics:")
print(f"- Total purchases: {data['purchases']}")
print(f"- Total add to carts: {data['add_to_carts']}")
print(f"- Total initiated checkouts: {data['initiated_checkouts']}")
print(f"- Average cost per result: {data['cost_per_result']}")
print(f"- Best ROAS: {data['purchase_roas']}")
print(f"- Best performing ad ID: {data['best_performing_ad_id']}")

print("\nIndividual Ad Performances:")
for ad_id, ad_data in data['ad_performances'].items():
    print(f"\nAd ID: {ad_id} - {ad_data['ad_name']}")
    print(f"- Purchases: {ad_data['purchases']}")
    print(f"- Add to carts: {ad_data['add_to_carts']}")
    print(f"- Checkouts: {ad_data['initiated_checkouts']}")
    print(f"- Cost per result: ${ad_data['cost_per_result']:.2f}")
    print(f"- ROAS: {ad_data['purchase_roas']:.2f}x")
    print(f"- Total spend: ${ad_data['spend']:.2f}")

# Test integration with AIcon
print("\n=== Testing Integration with AIcon ===")
aicon = SimpleBadAIcon(name="Ad Performance Analyzer")

# Add the Meta Ads sensor to the AIcon
aicon.add_sensor("meta_ads", sensor)

# Check if the factors were created
print("\nFactor Map after adding Meta Ads sensor:")
for factor_name in sensor.observable_factors:
    if factor_name in aicon.brain.get_state_factors():
        print(f"✅ Factor created: {factor_name}")
    else:
        print(f"❌ Factor missing: {factor_name}")

# Test fetch_data for tensor compatibility
print("\nTesting fetch_data for tensor compatibility:")
tensor_data = sensor.fetch_data()
print(f"- Keys in tensor data: {list(tensor_data.keys())}")
print(f"- ad_performances data type: {type(tensor_data['ad_performances'])}")

# Test conversion back from JSON
if "ad_performances" in tensor_data:
    try:
        ad_perf_dict = json.loads(tensor_data["ad_performances"])
        print(f"- Successfully converted ad_performances back from JSON, {len(ad_perf_dict)} ads")
    except Exception as e:
        print(f"- Error converting ad_performances from JSON: {e}")

print("\nTest completed successfully!") 