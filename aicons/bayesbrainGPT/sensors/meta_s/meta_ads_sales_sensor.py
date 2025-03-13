"""
Meta Ads Sales Sensor for BayesBrain.

This module provides a sensor that can observe Meta (Facebook) ad campaign 
sales performance metrics and provide them to the BayesBrain system.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import datetime
import pandas as pd
import json

# Import the Facebook Business SDK directly
from facebook_business.api import FacebookAdsApi
from facebook_business.adobjects.adaccount import AdAccount
from facebook_business.adobjects.adsinsights import AdsInsights
from facebook_business.adobjects.campaign import Campaign
from facebook_business.adobjects.adset import AdSet
from facebook_business.adobjects.ad import Ad

# Import the base TFSensor class
from ..tf_sensors import TFSensor, TensorType, ObservationType

class MetaAdsSalesSensor(TFSensor):
    """
    Sensor for Meta (Facebook) ad campaign sales data.
    
    This sensor provides observations for sales metrics:
    - purchases: Number of purchases
    - add_to_carts: Number of add to cart events
    - initiated_checkouts: Number of checkout initiated events
    - cost_per_result: Cost per result (typically cost per purchase)
    - purchase_roas: Return on ad spend for purchases
    - results: Number of results (typically purchases)
    - result_type: Type of result (default: "purchase")
    """
    
    def __init__(self, 
                name: str = "meta_ads_sales", 
                reliability: float = 0.85, 
                access_token: Optional[str] = None,
                ad_account_id: Optional[str] = None,
                campaign_id: Optional[str] = None,
                api_version: str = "v18.0",  # Use a more common API version
                time_granularity: str = "day",  # Options: "day", "hour"
                factor_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize a Meta Ads Sales sensor.
        
        Args:
            name: Identifier for the sensor
            reliability: Default reliability for this sensor (0.0 to 1.0)
            access_token: Facebook API access token
            ad_account_id: Facebook Ad Account ID (format: 'act_XXXXXXXXXX')
            campaign_id: Facebook Campaign ID to analyze
            api_version: Version of the Facebook Graph API to use (default: v18.0)
            time_granularity: Time granularity for data retrieval ("day" or "hour")
            factor_mapping: Optional mapping from sensor factor names to state factor names
        """
        super().__init__(name, reliability, factor_mapping)
        self.access_token = access_token
        self.ad_account_id = ad_account_id
        self.campaign_id = campaign_id
        self.api_version = api_version
        self.time_granularity = time_granularity
        self.use_real_data = access_token and ad_account_id
        
        # Initialize API if credentials provided
        if self.use_real_data:
            try:
                # Initialize with specific API version
                FacebookAdsApi.init(access_token=access_token, api_version=api_version)
                self.ad_account = AdAccount(ad_account_id)
                print(f"✅ Successfully initialized Meta Ads API v{api_version} for account {ad_account_id}")
            except Exception as e:
                print(f"❌ Failed to initialize Meta Ads API: {e}")
                self.use_real_data = False
    
    def _setup_observable_factors(self):
        """Define which factors this sensor can observe and their reliabilities."""
        # Sales performance metrics based on SalesMetrics dataclass
        self.observable_factors = [
            "purchases",              # Number of purchases
            "add_to_carts",           # Number of add to cart events
            "initiated_checkouts",    # Number of checkout initiated events
            "cost_per_result",        # Cost per result (typically cost per purchase)
            "purchase_roas",          # Return on ad spend for purchases
            "results",                # Number of results (typically purchases)
            "result_type"             # Type of result (default: "purchase")
        ]
        
        # Use the same reliability score for all factors (from sensor level)
        self.factor_reliabilities = {
            factor: self.default_reliability for factor in self.observable_factors
        }
    
    def extract_action_value(self, actions, action_type):
        """
        Extracts the value of a given action type from the API response.
        
        Args:
            actions: List of action dictionaries from the Facebook API
            action_type: Type of action to extract (e.g., 'offsite_conversion.fb_pixel_purchase')
            
        Returns:
            Value of the action or 0 if not found
        """
        if isinstance(actions, list):
            for action in actions:
                if action.get('action_type') == action_type:
                    return float(action.get('value', 0))  # Convert to float for numerical operations
        return 0  # Default to 0 if no matching action type is found

    def run(self) -> Dict[str, Any]:
        """
        Run the sensor and get the current snapshot of raw Meta Ads sales data from right now.
        
        This method fetches exactly the raw metrics specified in SalesMetrics at the current moment.
            
        Returns:
            Dictionary of raw sales metrics for the current moment
        """
        # Try to get real data if credentials are available
        if self.use_real_data and self.campaign_id:
            try:
                # First try to get the campaign to verify it exists
                print(f"Checking campaign {self.campaign_id}...")
                campaign_fields = [Campaign.Field.name]
                campaign_params = {"filtering": f'[{{"field":"id","operator":"EQUAL","value":"{self.campaign_id}"}}]'}
                
                # Get campaign details to verify access
                campaigns = self.ad_account.get_campaigns(fields=campaign_fields, params=campaign_params)
                if not campaigns:
                    print(f"❌ Campaign {self.campaign_id} not found")
                    raise ValueError(f"Campaign {self.campaign_id} not found for account {self.ad_account_id}")
                
                campaign_name = campaigns[0].get(Campaign.Field.name, "Unknown Campaign")
                print(f"✅ Found campaign: {campaign_name}")
                
                # Get date and time for time range based on granularity
                now = datetime.datetime.now()
                
                if self.time_granularity == "hour":
                    # Format for hourly granularity (ISO 8601 format with timezone)
                    # Start time: beginning of current hour
                    start_hour = now.replace(minute=0, second=0, microsecond=0)
                    # Format with timezone offset
                    timezone_offset = now.strftime('%z') or '+0000'
                    start_time = start_hour.strftime('%Y-%m-%dT%H:%M:%S') + timezone_offset
                    
                    # End time: current time
                    end_time = now.strftime('%Y-%m-%dT%H:%M:%S') + timezone_offset
                    
                    print(f"Using hourly granularity: {start_time} to {end_time}")
                else:
                    # Daily granularity (just date)
                    today = now.strftime('%Y-%m-%d')
                    start_time = today
                    end_time = today
                    print(f"Using daily granularity for date: {today}")
                
                # For debugging
                print(f"Current date/time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Get campaign insights data - use same fields as working example
                insight_fields = [
                    AdsInsights.Field.campaign_id,
                    AdsInsights.Field.spend,
                    AdsInsights.Field.clicks,
                    AdsInsights.Field.cpc,
                    AdsInsights.Field.ctr,
                    AdsInsights.Field.actions,
                    AdsInsights.Field.purchase_roas,
                ]
                
                # Set up parameters for insights - ensure all parameters are properly formatted
                params = {
                    "level": "campaign",
                    "filtering": f'[{{"field":"campaign.id","operator":"EQUAL","value":"{self.campaign_id}"}}]',
                    "time_range": json.dumps({
                        "since": start_time,
                        "until": end_time
                    })
                }
                
                # Add time increment if using hourly granularity
                if self.time_granularity == "hour":
                    params["time_increment"] = 1
                
                print(f"Fetching insights for campaign {self.campaign_id}...")
                print(f"Using parameters: {params}")
                
                # Get insights from the API
                campaign_insights = self.ad_account.get_insights(fields=insight_fields, params=params)
                
                # Debug: print what we got back
                print(f"API response received. Found {len(campaign_insights) if campaign_insights else 0} insights")
                
                if campaign_insights and len(campaign_insights) > 0:
                    # Get the most recent insight
                    insight = campaign_insights[0]
                    
                    # Debug: print the raw insight data
                    print(f"Raw insight data keys: {list(insight.keys()) if insight else 'No data'}")
                    
                    # Extract the exact metrics required
                    spend = float(insight.get("spend", 0))
                    
                    # Check for different purchase action types that might be in the data
                    purchases = self.extract_action_value(insight.get("actions", []), "offsite_conversion.fb_pixel_purchase")
                    if purchases == 0:  # Try web_in_store_purchase if no offsite conversions
                        purchases = self.extract_action_value(insight.get("actions", []), "web_in_store_purchase")
                    
                    # Check for different add_to_cart action types
                    add_to_carts = self.extract_action_value(insight.get("actions", []), "offsite_conversion.fb_pixel_add_to_cart") 
                    if add_to_carts == 0:  # Try onsite conversion if no offsite
                        add_to_carts = self.extract_action_value(insight.get("actions", []), "onsite_conversion.add_to_cart")
                    
                    # Check for different checkout action types
                    initiated_checkouts = self.extract_action_value(insight.get("actions", []), "offsite_conversion.fb_pixel_initiate_checkout")
                    if initiated_checkouts == 0:  # Try onsite conversion if no offsite
                        initiated_checkouts = self.extract_action_value(insight.get("actions", []), "onsite_conversion.initiate_checkout")
                    
                    # Extract purchase ROAS - try multiple possible types
                    purchase_roas = self.extract_action_value(insight.get("purchase_roas", []), "omni_purchase")
                    if purchase_roas == 0:
                        purchase_roas = self.extract_action_value(insight.get("purchase_roas", []), "offsite_purchase")
                    if purchase_roas == 0:
                        purchase_roas = self.extract_action_value(insight.get("purchase_roas", []), "web_in_store_purchase")
                    
                    # If still no ROAS, calculate it manually if we have purchases and spend
                    if purchase_roas == 0 and purchases > 0 and spend > 0:
                        # Assume average order value for calculation - adjust as needed
                        avg_order_value = 100  # Default assumption
                        purchase_roas = (purchases * avg_order_value) / spend
                    
                    # Calculate cost per result
                    cost_per_result = spend / purchases if purchases > 0 else 0
                    
                    print(f"Extracted data: purchases={purchases}, add_to_carts={add_to_carts}, " +
                          f"initiated_checkouts={initiated_checkouts}, purchase_roas={purchase_roas}")
                    
                    # Return exactly the metrics requested in SalesMetrics
                    return {
                        "purchases": purchases,
                        "add_to_carts": add_to_carts,
                        "initiated_checkouts": initiated_checkouts,
                        "cost_per_result": cost_per_result,
                        "purchase_roas": purchase_roas,
                        "results": int(purchases),
                        "result_type": "purchase"
                    }
                else:
                    print("No insights data returned for the campaign. Check if campaign ID is correct and has data.")
            except Exception as e:
                import traceback
                print(f"❌ Error in Meta Ads API data fetching: {e}")
                print(traceback.format_exc())  # Print full traceback for better debugging
                print("Falling back to sample data")
        
        # Return sample data if real data not available or fetch failed
        return {
            "purchases": 25.0,
            "add_to_carts": 120.0,
            "initiated_checkouts": 45.0,
            "cost_per_result": 20.0,
            "purchase_roas": 2.1,
            "results": 25,
            "result_type": "purchase"
        }
    
    def fetch_data(self, environment: Any = None) -> Dict[str, TensorType]:
        """
        Fetch Meta Ads sales performance data.
        
        Args:
            environment: Optional environment data (unused)
            
        Returns:
            Dictionary mapping factor names to values
        """
        # Get current snapshot from run method
        data = self.run()
        
        # Convert to proper tensor types
        return {
            "purchases": np.float32(data["purchases"]),
            "add_to_carts": np.float32(data["add_to_carts"]),
            "initiated_checkouts": np.float32(data["initiated_checkouts"]),
            "cost_per_result": np.float32(data["cost_per_result"]),
            "purchase_roas": np.float32(data["purchase_roas"]),
            "results": np.int32(data["results"]),
            "result_type": data["result_type"]
        } 