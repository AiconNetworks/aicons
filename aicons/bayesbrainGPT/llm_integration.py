# llm_integration.py
import json
from dotenv import load_dotenv
import os
from google import genai
import enum
from typing import Dict, List
from pydantic import BaseModel  # We still need this for the schema

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EXAMPLE_PROMPT = """
**I. Understanding the Foundation: The Marketing Mix (4Ps + Others)**

*   **Product:**
    *   *Features & Benefits:* How clearly are your ads highlighting the *core value* of the product? Is the focus on features or the benefits they deliver to the customer? A hypothesis might be: "Ads that emphasize the benefit of 'saving time' for busy professionals will generate higher click-through rates than ads focused on the product's technical specifications."
    *   *Product Life Cycle:* Is your product new, mature, or declining? This impacts messaging.  A new product needs awareness, while a mature product might need reminders or differentiation. Hypothesis: "For a mature product category, ads highlighting a unique product feature will result in higher conversion rates than general awareness ads."
    *   *Brand Perception:*  What's the existing perception of your brand? Does your ad align with this perception, or is it trying to change it? Misalignment can lead to confusion. Hypothesis: "Ads that align with our brand's established reputation for 'reliability' will have higher click-through rates among existing customers compared to ads focused on 'innovation'."

*   **Price:**
    *   *Price Point:* How does your product's price compare to competitors? Is it premium, competitive, or budget-friendly? Ads should reflect this. Hypothesis: "Ads highlighting a 'limited-time discount' will be more effective at driving conversions for budget-conscious customers during the holiday season."
    *   *Value Proposition:* Is the price justified by the perceived value? Ads need to clearly communicate this value. Hypothesis: "For a premium product, ads that showcase the 'superior quality' and 'craftsmanship' will lead to higher purchase values compared to ads that solely focus on price."

*   **Place (Distribution):**
    *   *Where are you selling the product?*: Online, retail stores, through partners? Ads need to guide the user to the relevant purchase location. Hypothesis: "Ads targeting users within a 5-mile radius of our retail locations will have higher click-through rates when including store hours and directions in the ad copy."
    *   *Channel Considerations:* Different channels (social media, search, display) have different user intents and formats. Hypothesis: "Video ads showcasing product demonstrations will perform better on YouTube compared to static image ads on the same platform."

*   **Promotion (Marketing Communication):**
    *   *Advertising Message:* Is it clear, concise, and compelling? Does it resonate with the target audience? Hypothesis: "Ads that use a problem-solution framework, directly addressing the user's pain points, will have higher engagement rates compared to ads that only highlight product features."
    *   *Call to Action (CTA):* Is it strong and specific? Does it tell the user exactly what you want them to do? Hypothesis: "Ads with a CTA of 'Shop Now' will generate higher conversion rates compared to ads with a generic CTA like 'Learn More'."
    *   *Ad Creative (Image/Video/Copy):* Are the visuals high-quality and relevant? Is the copy persuasive and engaging? Hypothesis: "Ads featuring user-generated content (reviews, testimonials) will have higher click-through rates and conversion rates compared to ads with solely brand-created content."

*   **People (Target Audience):**
    *   *Demographics:* Age, gender, location, income, education.
    *   *Psychographics:* Interests, values, lifestyle, attitudes.
    *   *Behavioral Data:* Past purchases, website visits, online activity.
        *   Hypothesis: "Ads targeting users interested in 'fitness and wellness' will have higher engagement rates with promotions for athletic apparel."
        *   Hypothesis: "Re-targeting ads to users who have previously visited our product pages will have a higher conversion rate than ads shown to new users."

*   **Process:**
    *   *Customer Journey:* Where does the ad fit into the overall customer journey? Are you targeting users at the awareness, consideration, or decision stage? Hypothesis: "Ads offering a free trial will be more effective for users in the consideration stage compared to those in the awareness stage."
    *   *Landing Page Experience:* Does the landing page match the ad's message and offer? Is it easy to navigate and convert? Hypothesis: "Ads that direct users to a landing page with a clear and concise product description will have higher conversion rates compared to ads that direct users to a generic homepage."

*   **Physical Evidence:**
    *   *Reviews and Ratings:* Positive reviews build trust and credibility. Consider incorporating testimonials in your ads. Hypothesis: "Ads displaying a 4.5-star rating or higher will have a higher click-through rate compared to ads without rating information."
    *   *Guarantees and Warranties:*  Reducing risk can increase conversions. Highlight guarantees in your ads. Hypothesis: "Ads that highlight a 'money-back guarantee' will have higher conversion rates, especially for new customers."

**II. Campaign-Specific Information**

*   **Campaign Objectives:** What are you trying to achieve with your ad campaign (awareness, leads, sales, etc.)?  The KPIs (Key Performance Indicators) you track should align with these objectives.
*   **Budget Allocation:**  How much are you spending on each ad campaign and channel?  A higher budget might lead to greater reach and impressions, but it doesn't guarantee better performance.
*   **Targeting Settings:**  What are your targeting criteria (keywords, interests, demographics, etc.)? Are you using broad or narrow targeting? Hypothesis: "Refining our targeting to include users who have actively searched for 'specific problem' will significantly improve the relevance of our ads and increase conversion rates."
*   **Bidding Strategy:** Are you using manual bidding or automated bidding? Are you optimizing for clicks, conversions, or impressions?
*   **A/B Testing:** What variations are you testing (ad copy, images, landing pages, etc.)? What are the results of your A/B tests?

**III. External Factors**

*   **Competitor Activity:** What are your competitors doing? Are they running similar ads or offering similar promotions?
*   **Seasonality:** Is your product affected by seasonal trends? Are you running ads during peak seasons or off-seasons?
*   **Economic Conditions:** Are consumers spending more or less money? How might this affect your ad performance?
*   **Cultural Events:** Are there any major cultural events or holidays that might affect your ad performance?
*   **Platform Changes:** Algorithm updates, new ad features, or policy changes from advertising platforms (Google, Facebook, etc.) can significantly impact ad performance.

**Example Hypothesis Construction:**

Let's say you're selling organic coffee online.  You've noticed your Facebook ad conversions are lower than expected.  Here's how you might build a hypothesis:

1.  **Observation:** Facebook ad conversions for organic coffee are low.
2.  **Marketing Information:**
    *   *Target Audience:* Currently targeting broad "coffee lovers" demographic.
    *   *Product:** Organic coffee, premium priced.
    *   *Competitor Activity:* Competitors are emphasizing ethical sourcing and sustainability.
    *   *Ad Copy:*  Currently focuses on the "great taste" of the coffee.
3.  **Hypothesis:**  "Refining our Facebook ad targeting to include users who are interested in 'sustainable living' and 'ethical consumption,' combined with ad copy that highlights the ethical sourcing and sustainability practices of our organic coffee, will result in a higher conversion rate compared to our current ads targeting a broad audience and focusing solely on taste."

**Key Takeaways for Building Strong Hypotheses:**

*   **Specificity:** Be precise about what you're testing and what you expect to happen.
*   **Measurability:**  Choose metrics that you can track and analyze.
*   **Relevance:**  Base your hypotheses on solid marketing information and insights.
*   **Testability:**  Design your tests to clearly prove or disprove your hypotheses.

By carefully considering these marketing factors and applying them to your ad campaigns, you can develop insightful hypotheses t
"""

class FactorType(enum.Enum):
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    DISCRETE = "discrete"
    BAYESIAN_LINEAR = "bayesian_linear"

class Factor(BaseModel):
    type: FactorType
    value: str
    description: str

def fetch_state_context_from_llm(prompt: str) -> dict:
    """
    Get structured state information from LLM.
    """
    total_tokens = 0
    
    # First get marketing insights
    # prompt = """
    # Based on marketing point out information that is relevant to take in consideration for building hypothesis on how ads are performing.
    # """
    # response = client.models.generate_content(
    #     model="gemini-2.0-flash",
    #     contents=prompt,
    # )
    
    # context = response.text
    # print("Initial response:", context)
    # print(f"Initial prompt tokens: {response.prompt_feedback.token_count}")
    # print(f"Initial completion tokens: {response.candidates[0].token_count}")
    # total_tokens += response.prompt_feedback.token_count + response.candidates[0].token_count
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    # Split into 4 specific prompts
    prompts = [
        # Continuous Numerical Factors
        f"""You are a marketing analytics expert. Based on the following marketing context:
        
    {EXAMPLE_PROMPT}

    Please extract and list all continuous numerical factors. For each factor, provide:
    - The factor name
    - Its current value (as a number)
    - A brief description (e.g., "Budget in USD" or "Conversion rate as a decimal").
    Example output format:
        - Budget: 10000 (Marketing budget in USD)
        - ConversionRate: 0.05 (Current conversion rate)
    """,
        
        # Categorical Factors
        f"""You are a marketing analytics expert. Based on the following marketing context:
        
    {EXAMPLE_PROMPT}

    Please extract and list all categorical factors. For each factor, provide:
    - The factor name
    - Its current category (as a string)
    - A brief description (e.g., "Market segment" or "Region").
    Example output format:
        - MarketSegment: "Youth" (Target market segment)
        - Region: "North America" (Geographic market area)
    """,
        
        # Discrete Numerical Factors
        f"""You are a marketing analytics expert. Based on the following marketing context:
        
    {EXAMPLE_PROMPT}

    Please extract and list all discrete numerical factors. For each factor, provide:
    - The factor name
    - Its current value (as an integer)
    - A brief description (e.g., "Number of active campaigns" or "Customer count").
    Example output format:
        - CampaignCount: 3 (Number of active marketing campaigns)
        - CustomerCount: 1500 (Total number of customers)
    """,
        
        # Factors for Bayesian Analysis
    f"""You are a marketing analytics expert. Based on the following marketing context:

    {EXAMPLE_PROMPT}

    Identify factors in this context that would benefit from Bayesian analysis (for example, conversion rate predictions that depend on multiple variables). For each such factor, please provide a JSON object with the following details:
    - "factor_name": The name of the factor.
    - "description": A brief explanation of what the factor represents.
    - "explanatory_vars": A dictionary mapping each explanatory variable to its current value.
    - "theta_prior": A dictionary where each parameter (e.g., "theta0", "theta_budget", "theta_seasonality", etc.) is assigned a prior specified as an object with "mean" and "variance".
    - "variance": The noise variance for the linear regression model.
    - Optionally, any comments or expected ranges.

    For example, your output should be in the following format:

    {
    "factor_name": "ConversionRate",
    "description": "Predicted conversion rate modeled using budget and seasonality",
    "explanatory_vars": {"budget": 10000, "seasonality": 0.2},
    "theta_prior": {
        "theta0": {"mean": 0.0, "variance": 1.0},
        "theta_budget": {"mean": 0.0001, "variance": 0.00001},
        "theta_seasonality": {"mean": 0.01, "variance": 0.001}
    },
    "variance": 0.05
    }

    Please output one such JSON object for each Bayesian factor identified.
    """
    ]
    
    all_factors = []
    for prompt_type, type_prompt in zip(['continuous', 'categorical', 'discrete', 'bayesian_linear'], prompts):
        # Count prompt tokens first
        token_count = client.models.count_tokens(
            model="gemini-1.5-flash",
            contents=type_prompt
        )
        prompt_tokens = token_count.total_tokens
        total_prompt_tokens += prompt_tokens
        
        # Generate response
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=type_prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "factors": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": [prompt_type]
                                    },
                                    "value": {"type": "string"},
                                    "description": {"type": "string"}
                                },
                                "required": ["type", "value", "description"]
                            }
                        }
                    },
                    "required": ["prompt", "factors"]
                }
            }
        )
        
        # Count completion tokens
        completion_count = client.models.count_tokens(
            model="gemini-1.5-flash",
            contents=response.text
        )
        completion_tokens = completion_count.total_tokens
        total_completion_tokens += completion_tokens
        
        response_data = json.loads(response.text)
        all_factors.extend(response_data["factors"])
        
        print(f"\n{prompt_type.title()} response:")
        print(f"Prompt tokens: {prompt_tokens}")
        print(f"Completion tokens: {completion_tokens}")
        print(response.text)
    
    # Calculate costs based on Gemini 1.5 Flash pricing (per 1M tokens)
    prompt_cost = 0
    completion_cost = 0
    
    # Prompt cost ($0.075 or $0.15 per 1M tokens)
    if total_prompt_tokens <= 128000:
        prompt_cost = (total_prompt_tokens / 1_000_000) * 0.075
    else:
        prompt_cost = (total_prompt_tokens / 1_000_000) * 0.15
        
    # Completion cost ($0.30 or $0.60 per 1M tokens)
    if total_completion_tokens <= 128000:
        completion_cost = (total_completion_tokens / 1_000_000) * 0.30
    else:
        completion_cost = (total_completion_tokens / 1_000_000) * 0.60
    
    total_cost = prompt_cost + completion_cost
    
    print(f"\nTotal prompt tokens: {total_prompt_tokens}")
    print(f"Total completion tokens: {total_completion_tokens}")
    print(f"Prompt cost: ${prompt_cost:.8f}")  # More decimals since costs will be tiny
    print(f"Completion cost: ${completion_cost:.8f}")
    print(f"Total cost: ${total_cost:.8f}")
    
    return all_factors



# For testing purposes:
if __name__ == "__main__":
    prompt = "Based on current observations, provide environmental data including rain, temperature, traffic, and weather."
    sensor_data = fetch_state_context_from_llm(prompt)
    print("Fetched sensor data from LLM:")
    print(json.dumps(sensor_data, indent=2))

