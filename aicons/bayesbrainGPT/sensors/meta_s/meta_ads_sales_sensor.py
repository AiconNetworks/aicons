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
    
    This sensor provides observations for sales metrics directly from Meta API:
    - ad_performances: Dictionary of ad IDs to performance metrics
    - adset_performances: Dictionary of adset IDs to performance metrics
    - purchases: Total number of purchases for all ads
    - add_to_carts: Total number of add to cart events
    - initiated_checkouts: Total number of checkout initiated events
    - results: Total number of results (typically purchases)
    - result_type: Type of result (default: "purchase")
    """
    
    def __init__(self, 
                name: str = "meta_ads_sales", 
                reliability: float = 0.85, 
                access_token: Optional[str] = None,
                ad_account_id: Optional[str] = None,
                campaign_id: Optional[str] = None,
                api_version: str = "v18.0",
                time_granularity: str = "hour",  # Options: "day", "hour" - default to hourly
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
            time_granularity: Time granularity for data retrieval ("hour" or "day"), defaults to hourly
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
        # Raw metrics directly from Meta API
        self.observable_factors = [
            "ad_performances",        # Dictionary of ad IDs to performance metrics
            "adset_performances",     # Dictionary of adset IDs to performance metrics
            "purchases",              # Total number of purchases for all ads
            "add_to_carts",           # Total number of add to cart events for all ads
            "initiated_checkouts",    # Total number of checkout initiated events for all ads
            "results",                # Total number of results (typically purchases) for all ads
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

    def get_adsets_for_campaign(self, campaign_id):
        """
        Fetches all ad sets belonging to a campaign.
        
        Args:
            campaign_id: ID of the campaign
            
        Returns:
            List of dictionaries containing ad set information
        """
        if not self.use_real_data:
            return []  # Return empty list if not using real data
            
        try:
            adset_fields = [
                AdSet.Field.id,
                AdSet.Field.name,
                AdSet.Field.status
            ]

            params = {
                "filtering": [{"field": "campaign.id", "operator": "EQUAL", "value": campaign_id}]
            }

            adsets = self.ad_account.get_ad_sets(fields=adset_fields, params=params)

            adset_list = []
            for adset in adsets:
                adset_list.append({
                    "adset_id": adset[AdSet.Field.id],
                    "adset_name": adset[AdSet.Field.name],
                    "campaign_id": campaign_id
                })

            print(f"Found {len(adset_list)} ad sets in campaign {campaign_id}")
            return adset_list
            
        except Exception as e:
            print(f"❌ Error getting ad sets for campaign {campaign_id}: {e}")
            return []

    def get_ads_for_adsets(self, adsets):
        """
        Fetches all ads inside the given ad sets.
        
        Args:
            adsets: List of dictionaries containing ad set information
            
        Returns:
            List of dictionaries containing ad information
        """
        if not self.use_real_data:
            return []  # Return empty list if not using real data
            
        try:
            ad_list = []

            for adset in adsets:
                adset_id = adset["adset_id"]

                ad_fields = [
                    Ad.Field.id,
                    Ad.Field.name,
                    Ad.Field.status,
                    Ad.Field.effective_status  # Includes more detailed status info
                ]
                ad_params = {"filtering": [{"field": "adset.id", "operator": "EQUAL", "value": adset_id}]}
                ads = self.ad_account.get_ads(fields=ad_fields, params=ad_params)

                for ad in ads:
                    ad_list.append({
                        "ad_id": ad[Ad.Field.id],
                        "ad_name": ad[Ad.Field.name],
                        "adset_id": adset_id,
                        "adset_name": adset["adset_name"],
                        "campaign_id": adset["campaign_id"],
                        "status": ad.get(Ad.Field.status, "UNKNOWN"),
                        "effective_status": ad.get(Ad.Field.effective_status, "UNKNOWN")
                    })

            print(f"Found {len(ad_list)} ads in {len(adsets)} ad sets")
            return ad_list
            
        except Exception as e:
            print(f"❌ Error getting ads for ad sets: {e}")
            return []

    def get_insights_for_ads(self, ad_list, start_date=None, end_date=None):
        """
        Gets insights data for a list of ads with hourly or daily breakdown.
        
        Args:
            ad_list: List of dictionaries containing ad information
            start_date: Optional start date for insights data (format: YYYY-MM-DD)
            end_date: Optional end date for insights data (format: YYYY-MM-DD)
            
        Returns:
            List of dictionaries containing insights data
        """
        if not self.use_real_data or not ad_list:
            return []  # Return empty list if not using real data or no ads
            
        try:
            # Extract ad IDs
            ad_ids = [ad["ad_id"] for ad in ad_list]
            
            # Fields to retrieve from insights
            insight_fields = [
                AdsInsights.Field.ad_id,
                AdsInsights.Field.ad_name,
                AdsInsights.Field.adset_id,
                AdsInsights.Field.adset_name,
                AdsInsights.Field.campaign_id,
                AdsInsights.Field.date_start,
                AdsInsights.Field.impressions,
                AdsInsights.Field.clicks,
                AdsInsights.Field.spend,
                AdsInsights.Field.cpc,
                AdsInsights.Field.ctr,
                AdsInsights.Field.actions,
                AdsInsights.Field.purchase_roas,
            ]
            
            # Let's try different date presets to find data
            date_presets_to_try = ["today", "yesterday", "last_3d"]
            
            # Track if we found hourly data
            found_hourly_data = False
            last_hour_insights = []
            latest_hour = None
            
            for date_preset in date_presets_to_try:
                print(f"Trying date_preset: {date_preset}")
                
                # Set up parameters for data retrieval
                params = {
                    "level": "ad",
                    "date_preset": date_preset,
                    "limit": 1000
                }
                
                # Add hourly breakdown for hourly data
                if self.time_granularity == "hour":
                    params["breakdowns"] = ["hourly_stats_aggregated_by_advertiser_time_zone"]
                
                # Add filtering for ad IDs
                if ad_ids:
                    params["filtering"] = [{"field": "ad.id", "operator": "IN", "value": ad_ids}]
                else:
                    params["filtering"] = [{"field": "campaign.id", "operator": "EQUAL", "value": self.campaign_id}]
                
                print(f"Fetching data for {len(ad_ids)} ads with {date_preset} preset...")
                
                # Get insights from the API
                insights = self.ad_account.get_insights(fields=insight_fields, params=params)
                
                # Convert insights to a list (it's an iterator)
                insight_list = list(insights)
                print(f"Retrieved {len(insight_list)} records")
                
                # If we have data and are looking for hourly
                if insight_list and self.time_granularity == "hour":
                    # Find latest hour in data
                    for insight in insight_list:
                        hour = insight.get("hourly_stats_aggregated_by_advertiser_time_zone", "")
                        if hour and (latest_hour is None or hour > latest_hour):
                            latest_hour = hour
                    
                    if latest_hour:
                        print(f"Found latest hour: {latest_hour}")
                        found_hourly_data = True
                        
                        # Filter to only include the latest hour
                        for insight in insight_list:
                            hour = insight.get("hourly_stats_aggregated_by_advertiser_time_zone", "")
                            if hour == latest_hour:
                                last_hour_insights.append({
                                    "date": insight.get("date_start", "N/A"),
                                    "hour_range": hour,
                                    "campaign_id": insight.get("campaign_id"),
                                    "adset_id": insight.get("adset_id"),
                                    "adset_name": insight.get("adset_name"),
                                    "ad_id": insight.get("ad_id"),
                                    "ad_name": insight.get("ad_name"),
                                    "impressions": int(insight.get("impressions", 0)),
                                    "clicks": int(insight.get("clicks", 0)),
                                    "ctr": float(insight.get("ctr", 0)),
                                    "cpc": float(insight.get("cpc", 0)),
                                    "spend": float(insight.get("spend", 0)),
                                    "add_to_carts": self.extract_action_value(insight.get("actions", []), "offsite_conversion.fb_pixel_add_to_cart"),
                                    "initiated_checkouts": self.extract_action_value(insight.get("actions", []), "offsite_conversion.fb_pixel_initiate_checkout"),
                                    "purchases": self.extract_action_value(insight.get("actions", []), "offsite_conversion.fb_pixel_purchase"),
                                    "purchase_roas": self.extract_action_value(insight.get("purchase_roas", []), "omni_purchase"),
                                })
                        
                        print(f"Filtered to {len(last_hour_insights)} records for hour {latest_hour}")
                        break
                
                # If we're looking for daily data or couldn't find hourly data but have some insights
                if not found_hourly_data and insight_list:
                    # Fallback to daily data
                    print(f"Using daily data from {date_preset}")
                    for insight in insight_list:
                        last_hour_insights.append({
                            "date": insight.get("date_start", "N/A"),
                            "hour_range": "daily",
                            "campaign_id": insight.get("campaign_id"),
                            "adset_id": insight.get("adset_id"),
                            "adset_name": insight.get("adset_name"),
                            "ad_id": insight.get("ad_id"),
                            "ad_name": insight.get("ad_name"),
                            "impressions": int(insight.get("impressions", 0)),
                            "clicks": int(insight.get("clicks", 0)),
                            "ctr": float(insight.get("ctr", 0)),
                            "cpc": float(insight.get("cpc", 0)),
                            "spend": float(insight.get("spend", 0)),
                            "add_to_carts": self.extract_action_value(insight.get("actions", []), "offsite_conversion.fb_pixel_add_to_cart"),
                            "initiated_checkouts": self.extract_action_value(insight.get("actions", []), "offsite_conversion.fb_pixel_initiate_checkout"),
                            "purchases": self.extract_action_value(insight.get("actions", []), "offsite_conversion.fb_pixel_purchase"),
                            "purchase_roas": self.extract_action_value(insight.get("purchase_roas", []), "omni_purchase"),
                        })
                    
                    # If we found data, stop trying other date presets
                    if last_hour_insights:
                        break
            
            if last_hour_insights:
                data_type = "hourly" if found_hourly_data else "daily"
                print(f"Found {len(last_hour_insights)} {data_type} records")
                return last_hour_insights
            else:
                print("No data found with any date preset")
                return []
            
        except Exception as e:
            import traceback
            print(f"❌ Error getting insights for ads: {e}")
            print(traceback.format_exc())
            return []

    def process_insights_data(self, insights):
        """
        Process insights data without adding derived metrics.
        
        Args:
            insights: List of dictionaries containing insights data
            
        Returns:
            Dictionary containing raw data
        """
        if not insights:
            return {
                "ad_performances": {},
                "adset_performances": {},
                "purchases": 0,
                "add_to_carts": 0,
                "initiated_checkouts": 0,
                "results": 0,
                "result_type": "purchase"
            }
        
        # Organize data by ad and adset
        ad_performances = {}
        adset_performances = {}
        
        # Total metrics
        total_purchases = 0
        total_add_to_carts = 0
        total_initiated_checkouts = 0
        
        # Process each insight - only organizing data, no computation
        for insight in insights:
            # Get or create ad record
            ad_id = insight["ad_id"]
            adset_id = insight["adset_id"]
            
            # Create ad performance entry if it doesn't exist
            if ad_id not in ad_performances:
                ad_performances[ad_id] = {
                    "ad_id": ad_id,
                    "ad_name": insight["ad_name"],
                    "adset_id": adset_id,
                    "adset_name": insight.get("adset_name", ""),
                    "status": insight.get("status", "UNKNOWN"),  # Include status if available
                    "effective_status": insight.get("effective_status", "UNKNOWN"),  # Include effective status if available
                    "purchases": 0,
                    "add_to_carts": 0,
                    "initiated_checkouts": 0,
                    "impressions": 0,
                    "clicks": 0,
                    "spend": 0,
                    "hourly_data" if self.time_granularity == "hour" else "daily_data": []
                }
            
            # Create adset performance entry if it doesn't exist
            if adset_id not in adset_performances:
                adset_performances[adset_id] = {
                    "adset_id": adset_id,
                    "adset_name": insight.get("adset_name", ""),
                    "purchases": 0,
                    "add_to_carts": 0,
                    "initiated_checkouts": 0,
                    "impressions": 0,
                    "clicks": 0,
                    "spend": 0,
                    "ads": []
                }
            
            # Add this insight to hourly/daily data
            data_key = "hourly_data" if self.time_granularity == "hour" else "daily_data"
            
            ad_performances[ad_id][data_key].append({
                "date": insight["date"],
                "hour_range": insight.get("hour_range", ""),
                "purchases": insight["purchases"],
                "add_to_carts": insight["add_to_carts"],
                "initiated_checkouts": insight["initiated_checkouts"],
                "impressions": insight["impressions"],
                "clicks": insight["clicks"],
                "spend": insight["spend"],
                "purchase_roas": insight["purchase_roas"]
            })
            
            # Update ad totals (simple summation)
            ad_performances[ad_id]["purchases"] += insight["purchases"]
            ad_performances[ad_id]["add_to_carts"] += insight["add_to_carts"]
            ad_performances[ad_id]["initiated_checkouts"] += insight["initiated_checkouts"]
            ad_performances[ad_id]["impressions"] += insight["impressions"]
            ad_performances[ad_id]["clicks"] += insight["clicks"]
            ad_performances[ad_id]["spend"] += insight["spend"]
            
            # Update adset totals (simple summation)
            adset_performances[adset_id]["purchases"] += insight["purchases"]
            adset_performances[adset_id]["add_to_carts"] += insight["add_to_carts"]
            adset_performances[adset_id]["initiated_checkouts"] += insight["initiated_checkouts"]
            adset_performances[adset_id]["impressions"] += insight["impressions"]
            adset_performances[adset_id]["clicks"] += insight["clicks"]
            adset_performances[adset_id]["spend"] += insight["spend"]
            
            # Track this ad in the adset's ads list if not already there
            if ad_id not in [a["ad_id"] for a in adset_performances[adset_id].get("ads", [])]:
                adset_performances[adset_id]["ads"].append({
                    "ad_id": ad_id,
                    "ad_name": insight["ad_name"],
                    "status": insight.get("status", "UNKNOWN"),
                    "effective_status": insight.get("effective_status", "UNKNOWN")
                })
            
            # Update campaign totals
            total_purchases += insight["purchases"]
            total_add_to_carts += insight["add_to_carts"]
            total_initiated_checkouts += insight["initiated_checkouts"]
        
        return {
            "ad_performances": ad_performances,
            "adset_performances": adset_performances,
            "purchases": total_purchases,
            "add_to_carts": total_add_to_carts,
            "initiated_checkouts": total_initiated_checkouts,
            "results": int(total_purchases),
            "result_type": "purchase"
        }

    def get_campaign_details(self, campaign_id):
        """
        Fetches campaign details including the objective.
        
        Args:
            campaign_id: ID of the campaign
            
        Returns:
            Dictionary containing campaign information
        """
        if not self.use_real_data:
            return None
            
        try:
            campaign_fields = [
                Campaign.Field.id,
                Campaign.Field.name,
                Campaign.Field.status,
                Campaign.Field.objective
            ]

            campaign = Campaign(campaign_id).api_get(fields=campaign_fields)
            
            return {
                "campaign_id": campaign.get(Campaign.Field.id),
                "campaign_name": campaign.get(Campaign.Field.name),
                "campaign_status": campaign.get(Campaign.Field.status),
                "campaign_objective": campaign.get(Campaign.Field.objective)
            }
            
        except Exception as e:
            print(f"❌ Error getting campaign details for campaign {campaign_id}: {e}")
            return None

    def run(self) -> Dict[str, Any]:
        """
        Run the sensor and get the current snapshot of raw Meta Ads data for all ads in the campaign.
        
        This method fetches ad-level metrics for every ad in the specified campaign.
            
        Returns:
            Dictionary containing metrics for each ad and aggregate metrics for the campaign
        """
        # Try to get real data if credentials are available
        if self.use_real_data and self.campaign_id:
            try:
                # Step 0: Verify campaign objective
                campaign_details = self.get_campaign_details(self.campaign_id)
                
                if campaign_details:
                    campaign_objective = campaign_details.get("campaign_objective")
                    print(f"Campaign objective: {campaign_objective}")
                    
                    # Check if campaign is for sales objective
                    if campaign_objective != "OUTCOME_SALES":
                        print(f"Campaign {self.campaign_id} objective is {campaign_objective}, not OUTCOME_SALES. Returning empty data.")
                        return self.get_empty_data()
                
                # Step 1: Get all ad sets in the campaign
                adsets = self.get_adsets_for_campaign(self.campaign_id)
                
                if not adsets:
                    print(f"No ad sets found in campaign {self.campaign_id}. Returning empty data.")
                    return self.get_empty_data()
                
                # Step 2: Get all ads inside those ad sets
                ads = self.get_ads_for_adsets(adsets)
                
                if not ads:
                    print(f"No ads found in campaign {self.campaign_id}. Returning empty data.")
                    return self.get_empty_data()
                
                # Step 3: Get insights for all ads
                now = datetime.datetime.now()
                
                # Use dates according to time granularity
                if self.time_granularity == "hour":
                    # For hourly, use the current day
                    start_date = now.strftime('%Y-%m-%d')
                    end_date = start_date
                else:
                    # For daily, use yesterday
                    yesterday = now - datetime.timedelta(days=1)
                    start_date = yesterday.strftime('%Y-%m-%d')
                    end_date = start_date
                
                insights = self.get_insights_for_ads(ads, start_date, end_date)
                
                if not insights:
                    print(f"No insights data found for ads in campaign {self.campaign_id}. Returning empty data.")
                    return self.get_empty_data()
                
                # Step 4: Process insights data without adding derived metrics
                return self.process_insights_data(insights)
                
            except Exception as e:
                import traceback
                print(f"❌ Error in Meta Ads API data fetching: {e}")
                print(traceback.format_exc())  # Print full traceback for better debugging
                print("Returning empty data")
                return self.get_empty_data()
        
        # If no real data credentials, return empty data
        return self.get_empty_data()
    
    def get_empty_data(self) -> Dict[str, Any]:
        """
        Returns an empty data structure when no real data is available.
        
        Returns:
            Dictionary with empty structures for all factors
        """
        return {
            "ad_performances": {},
            "adset_performances": {},
            "purchases": 0.0,
            "add_to_carts": 0.0,
            "initiated_checkouts": 0.0,
            "results": 0,
            "result_type": "purchase"
        }
    
    def extract_ad_factors(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract individual factors for each ad to better integrate with perception system.
        
        Args:
            data: Raw data from run() method
            
        Returns:
            Dictionary where each key is a factor name and each value is (value, reliability)
        """
        factors = {}
        reliability = self.default_reliability
        
        # Add campaign-level metrics
        factors["purchases"] = (data["purchases"], reliability)
        factors["add_to_carts"] = (data["add_to_carts"], reliability)
        factors["initiated_checkouts"] = (data["initiated_checkouts"], reliability)
        
        # Collect information about processed ads for debugging
        processed_ads = []
        
        # Add individual ad metrics as separate factors
        for ad_id, ad_data in data["ad_performances"].items():
            # Use clean ad_id without any 'ad_' prefix to ensure consistent naming
            if ad_id.startswith('ad_'):
                ad_id = ad_id[3:]  # Remove 'ad_' prefix if present
                
            ad_prefix = f"ad_{ad_id}"
            processed_ads.append(ad_id)
            
            # Only include numerical metrics as factors
            factors[f"{ad_prefix}_purchases"] = (ad_data["purchases"], reliability)
            factors[f"{ad_prefix}_add_to_carts"] = (ad_data["add_to_carts"], reliability)
            factors[f"{ad_prefix}_initiated_checkouts"] = (ad_data["initiated_checkouts"], reliability)
            factors[f"{ad_prefix}_impressions"] = (ad_data["impressions"], reliability)
            factors[f"{ad_prefix}_clicks"] = (ad_data["clicks"], reliability)
            factors[f"{ad_prefix}_spend"] = (ad_data["spend"], reliability)
        
        # Print debug information about processed ads
        print(f"Extracted factors for {len(processed_ads)} ads")
        if processed_ads:
            print(f"Example ad IDs: {', '.join(processed_ads[:3])}")
            print(f"Example factor names: {', '.join(list(factors.keys())[:10])}")
        
        return factors
    
    def fetch_data(self, environment: Any = None) -> Dict[str, TensorType]:
        """
        Fetch Meta Ads sales performance data for all ads in the campaign.
        
        Args:
            environment: Optional environment data (unused)
            
        Returns:
            Dictionary mapping factor names to values
        """
        # Get current snapshot from run method
        data = self.run()
        
        # For compatibility with both approaches
        if environment and isinstance(environment, dict) and environment.get("use_individual_factors", False):
            # Extract individual factors for better perception integration
            extracted_factors = self.extract_ad_factors(data)
            
            # Convert to proper tensor types
            result = {}
            for factor_name, (value, reliability) in extracted_factors.items():
                if isinstance(value, (int, float)):
                    # Use the appropriate numeric type
                    tensor_value = np.float32(value) if isinstance(value, float) else np.int32(value)
                else:
                    # Keep strings as is
                    tensor_value = value
                
                # Store the value (not the tuple) for TF compatibility
                result[factor_name] = tensor_value
            
            return result
        else:
            # Use the original approach for backward compatibility
            # Convert dictionaries to string representations for tensor compatibility
            ad_performances_str = json.dumps(data["ad_performances"])
            adset_performances_str = json.dumps(data["adset_performances"])
            
            # Convert to proper tensor types
            return {
                "ad_performances": ad_performances_str,
                "adset_performances": adset_performances_str,
                "purchases": np.float32(data["purchases"]),
                "add_to_carts": np.float32(data["add_to_carts"]),
                "initiated_checkouts": np.float32(data["initiated_checkouts"]),
                "results": np.int32(data["results"]),
                "result_type": data["result_type"]
            }
    
    def get_data(self, environment: Any = None) -> Dict[str, Tuple[Any, float]]:
        """
        Get sensor data in the format expected by perception system.
        
        Args:
            environment: Optional environment data
            
        Returns:
            Dictionary mapping factor names to (value, reliability) tuples
        """
        # Get raw data
        data = self.run()
        
        # Extract individual factors with reliability values
        return self.extract_ad_factors(data)
        
    def observe(self, environment: Any = None) -> Dict[str, ObservationType]:
        """
        Observe the environment and return observations.
        
        This method is called by the perception system to get observations
        from this sensor. It returns a dictionary mapping factor names to
        (value, reliability) tuples.
        
        Args:
            environment: Optional environment data to use when observing
            
        Returns:
            Dictionary mapping factor names to (value, reliability) tuples
        """
        # Check if we should use extracted factors
        use_individual_factors = True
        
        # Get data from sensor
        sensor_data = self.fetch_data(environment)
        
        if not sensor_data:
            print(f"No data fetched from {self.name} sensor")
            return {}
        
        # Extract individual factors if requested
        if use_individual_factors:
            # Run the extract method first
            extracted_factors = self.extract_ad_factors(sensor_data)
            
            # For any ad-specific factor, ensure proper creation if it doesn't exist
            observations = {}
            
            # For each extracted factor, ensure it's properly named and mapped
            for factor_name, (value, reliability) in extracted_factors.items():
                # Map sensor factor name to state factor name
                mapped_name = self._map_factor_name(factor_name)
                
                # Store observation with mapped name
                observations[mapped_name] = (value, reliability)
                
            print(f"Returning {len(observations)} individual observations from {self.name} sensor")
            return observations
        else:
            # Fall back to default behavior
            observations = {}
            
            # For each observable factor, check if it's in fetched data
            for factor_name in self.observable_factors:
                if factor_name in sensor_data:
                    # Convert to tensor if needed
                    value = sensor_data[factor_name]
                    reliability = self.factor_reliabilities.get(factor_name, self.default_reliability)
                    
                    # Map sensor factor name to state factor name
                    mapped_name = self._map_factor_name(factor_name)
                    
                    # Store observation with mapped name
                    observations[mapped_name] = (value, reliability)
            
            print(f"Returning {len(observations)} observations from {self.name} sensor")
            return observations
    
    def get_expected_factors(self) -> Dict[str, Dict[str, Any]]:
        """
        Return information about the factors this sensor expects to provide data for.
        
        This method is used for automatic factor creation when registering the sensor.
        
        Returns:
            Dictionary mapping factor names to information about each factor
        """
        # Fetch active ads first to get actual ad IDs
        active_ads = []
        if self.use_real_data:
            try:
                active_ads = self.get_active_ads()
                # Print active ads for debugging
                print(f"Found {len(active_ads)} active ads")
                for ad in active_ads:
                    print(f"- Ad ID: {ad['ad_id']}, Name: {ad['ad_name']}")
            except Exception as e:
                print(f"Error fetching active ads: {e}")
                active_ads = []
        
        # Define the base campaign-level factors with hierarchical relationships
        factors = {
            "purchases": {
                "type": "continuous",
                "default_value": 0.0,
                "uncertainty": 1.0,
                "lower_bound": 0.0,
                "description": "Total number of purchases from ads"
            },
            "add_to_carts": {
                "type": "continuous",
                "default_value": 0.0,
                "uncertainty": 5.0,
                "lower_bound": 0.0,
                "description": "Total number of add to cart events from ads"
            },
            "initiated_checkouts": {
                "type": "continuous",
                "default_value": 0.0,
                "uncertainty": 2.0,
                "lower_bound": 0.0,
                "description": "Total number of checkout initiations from ads"
            }
        }
        
        # Define the metrics we want to track for each ad
        ad_metric_specs = [
            # Conversion metrics
            {"suffix": "purchases", "type": "continuous", "default": 0.0, "uncertainty": 1.0, "description": "Number of purchases from this ad"},
            {"suffix": "add_to_carts", "type": "continuous", "default": 0.0, "uncertainty": 5.0, "description": "Number of add to cart events from this ad"},
            {"suffix": "initiated_checkouts", "type": "continuous", "default": 0.0, "uncertainty": 2.0, "description": "Number of checkout initiations from this ad"},
            # Impression and click metrics
            {"suffix": "impressions", "type": "continuous", "default": 0.0, "uncertainty": 100.0, "description": "Number of impressions for this ad"},
            {"suffix": "clicks", "type": "continuous", "default": 0.0, "uncertainty": 10.0, "description": "Number of clicks for this ad"},
            {"suffix": "spend", "type": "continuous", "default": 0.0, "uncertainty": 10.0, "description": "Amount spent on this ad"}
        ]
        
        # If we have actual ad IDs, create specific factors for them
        if active_ads:
            # Create specific factors for each active ad
            for ad in active_ads:
                ad_id = ad['ad_id']
                for metric in ad_metric_specs:
                    factor_name = f"ad_{ad_id}_{metric['suffix']}"
                    factors[factor_name] = {
                        "type": metric["type"],
                        "default_value": metric["default"],
                        "uncertainty": metric["uncertainty"],
                        "lower_bound": 0.0,
                        "description": f"{metric['description']} (ID: {ad_id})"
                    }
            # Print number of factors for debugging
            print(f"Number of factors: {len(factors)}")
            print(f"Factor names: {', '.join(list(factors.keys())[:10])}...")
        else:
            # If no active ads found, fall back to generic pattern with wildcard
            # This is a fallback for when we can't get real ad data
            for metric in ad_metric_specs:
                factor_name = f"ad_*_{metric['suffix']}"
                factors[factor_name] = {
                    "type": metric["type"],
                    "default_value": metric["default"],
                    "uncertainty": metric["uncertainty"],
                    "lower_bound": 0.0,
                    "description": f"{metric['description']} (generic pattern)"
                }
            print("Using generic ad patterns - WILL NEED UPDATE with real ad data")
            
        return factors

    def get_all_ads(self) -> List[Dict[str, Any]]:
        """
        Get a simple list of all ads in the campaign with their names and status.
        
        Returns:
            List of dictionaries containing ad information
        """
        # Get all adsets first
        adsets = self.get_adsets_for_campaign(self.campaign_id)
        if not adsets:
            print(f"No ad sets found in campaign {self.campaign_id}")
            return []
            
        # Get all ads in those adsets
        ads = self.get_ads_for_adsets(adsets)
        
        # Return the list of ads (already contains status information)
        return ads
        
    def get_active_ads(self) -> List[Dict[str, Any]]:
        """
        Get a list of only the active/running ads in the campaign.
        
        Returns:
            List of dictionaries containing active ad information
        """
        all_ads = self.get_all_ads()
        
        # Filter to only include active ads
        # Effective statuses that indicate the ad is running include:
        # ACTIVE, CAMPAIGN_PAUSED, etc. - but we just want the ones that are fully active
        active_ads = [ad for ad in all_ads if ad.get("effective_status", "") == "ACTIVE"]
        
        print(f"Found {len(active_ads)} active ads out of {len(all_ads)} total ads")
        return active_ads

    def mock_run(self, num_adsets=3, num_ads_per_adset=5, include_performance_data=True) -> Dict[str, Any]:
        """
        Generate mock data for testing without requiring API access.
        
        Args:
            num_adsets: Number of ad sets to generate
            num_ads_per_adset: Number of ads per ad set
            include_performance_data: Whether to include mock performance metrics
            
        Returns:
            Dictionary containing mock metrics for testing
        """
        import random
        
        # Generate mock adsets
        adset_performances = {}
        ad_performances = {}
        
        # Possible statuses for variety
        ad_statuses = ["ACTIVE", "PAUSED"]
        effective_statuses = ["ACTIVE", "PAUSED", "ADSET_PAUSED", "CAMPAIGN_PAUSED", "DISAPPROVED"]
        
        total_purchases = 0
        total_add_to_carts = 0
        total_initiated_checkouts = 0
        
        # Create mock adsets
        for adset_index in range(1, num_adsets + 1):
            adset_id = f"{adset_index}"
            adset_name = f"Mock Ad Set {adset_index}"
            
            # Create empty adset performance entry
            adset_performances[adset_id] = {
                "adset_id": adset_id,
                "adset_name": adset_name,
                "purchases": 0,
                "add_to_carts": 0,
                "initiated_checkouts": 0,
                "impressions": 0,
                "clicks": 0,
                "spend": 0,
                "ads": []
            }
            
            # Create mock ads for this adset
            for ad_index in range(1, num_ads_per_adset + 1):
                ad_id = f"{adset_index}_{ad_index}"  # Remove the "ad_" prefix here
                ad_name = f"Mock Ad {adset_index}.{ad_index}"
                
                # Randomly assign status
                status = random.choice(ad_statuses)
                effective_status = status if status == "ACTIVE" else random.choice(effective_statuses)
                
                # Create a reference to this ad in the adset
                adset_performances[adset_id]["ads"].append({
                    "ad_id": ad_id,
                    "ad_name": ad_name,
                    "status": status,
                    "effective_status": effective_status
                })
                
                # Only add performance data if requested and ad could be active
                if include_performance_data:
                    # Generate mock metrics with some randomness but reasonable values
                    impressions = random.randint(100, 5000) if effective_status != "PAUSED" else 0
                    clicks = min(impressions, random.randint(0, impressions // 10)) if impressions > 0 else 0
                    ctr = clicks / impressions if impressions > 0 else 0
                    spend = round(clicks * random.uniform(0.5, 2.0), 2) if clicks > 0 else 0
                    
                    # Conversion metrics - more rare than clicks
                    add_to_carts = min(clicks, random.randint(0, clicks // 3)) if clicks > 0 else 0
                    initiated_checkouts = min(add_to_carts, random.randint(0, add_to_carts)) if add_to_carts > 0 else 0
                    purchases = min(initiated_checkouts, random.randint(0, initiated_checkouts)) if initiated_checkouts > 0 else 0
                    
                    # Hourly data (for mock, we'll just use one hour)
                    hourly_data = [{
                        "date": "2023-04-15",
                        "hour_range": "00-01",
                        "purchases": purchases,
                        "add_to_carts": add_to_carts,
                        "initiated_checkouts": initiated_checkouts,
                        "impressions": impressions,
                        "clicks": clicks,
                        "spend": spend,
                        "purchase_roas": round(purchases * 50 / spend, 2) if spend > 0 and purchases > 0 else 0
                    }]
                    
                    # Create ad performance entry
                    ad_performances[ad_id] = {
                        "ad_id": ad_id,
                        "ad_name": ad_name,
                        "adset_id": adset_id,
                        "adset_name": adset_name,
                        "status": status,
                        "effective_status": effective_status,
                        "purchases": purchases,
                        "add_to_carts": add_to_carts,
                        "initiated_checkouts": initiated_checkouts,
                        "impressions": impressions,
                        "clicks": clicks,
                        "spend": spend,
                        "hourly_data": hourly_data
                    }
                    
                    # Update adset totals
                    adset_performances[adset_id]["purchases"] += purchases
                    adset_performances[adset_id]["add_to_carts"] += add_to_carts
                    adset_performances[adset_id]["initiated_checkouts"] += initiated_checkouts
                    adset_performances[adset_id]["impressions"] += impressions
                    adset_performances[adset_id]["clicks"] += clicks
                    adset_performances[adset_id]["spend"] += spend
                    
                    # Update campaign totals
                    total_purchases += purchases
                    total_add_to_carts += add_to_carts
                    total_initiated_checkouts += initiated_checkouts
                    
        return {
            "ad_performances": ad_performances,
            "adset_performances": adset_performances,
            "purchases": total_purchases,
            "add_to_carts": total_add_to_carts,
            "initiated_checkouts": total_initiated_checkouts,
            "results": total_purchases,
            "result_type": "purchase"
        } 