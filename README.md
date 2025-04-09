# BayesBrain Budget Allocation AIcon

A Bayesian decision-making system for optimizing budget allocation across marketing campaigns.

## Security and Environment Variables

This project uses environment variables to store sensitive information like API keys. Never commit these values to version control.

1. Create a `.env` file in the project root with the following structure:

```bash
# API Keys
GEMINI_API_KEY=your_gemini_api_key_here
META_ACCESS_TOKEN=your_meta_access_token_here
```

2. Add `.env` to your `.gitignore` file (already included in this project)

3. Use environment variables in your code:

```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Access variables
access_token = os.getenv('META_ACCESS_TOKEN')
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Babel.git
cd Babel

# Install dependencies
pip install -r requirements.txt

# Optional: Install TensorFlow with GPU support for faster gradient optimization
pip install tensorflow-gpu
```

## Quick Start

Here's a minimal example to optimize budget allocation for Meta Ads:

```python
from aicons.definitions.simple_bad_aicon import SimpleBadAIcon
from aicons.bayesbrainGPT.sensors.meta_ads_sensor import MetaAdsSensor
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create AIcon
aicon = SimpleBadAIcon(name="BudgetOptimizer")

# Configure Meta Ads sensor
sensor = MetaAdsSensor(
    name="meta_ads",
    access_token=os.getenv('META_ACCESS_TOKEN'),  # Get from environment variables
    ad_account_id="act_your_account_id",
    campaign_id="your_campaign_id"
)

# Add sensor to AIcon
aicon.add_sensor("meta_ads", sensor)

# Get active ads
active_ads = sensor.get_active_ads()
ad_ids = [ad['ad_id'] for ad in active_ads]

# Create action space
action_space = aicon.create_action_space(
    space_type='budget_allocation',
    total_budget=1000,  # $1000 total budget
    items=ad_ids,
    step_size=0.01  # 1% steps
)

# Create utility function
utility_function = aicon.create_utility_function(
    utility_type='marketing_roi',
    revenue_per_sale=50.0  # $50 revenue per conversion
)

# Update beliefs with latest data
aicon.update_from_sensor("meta_ads")

# Find optimal budget allocation
best_action, expected_profit = aicon.find_best_action(
    num_samples=500,
    use_gradient=True  # Enable gradient-based optimization
)

# Print results
print("\nOPTIMAL BUDGET ALLOCATION:")
for ad_id, budget in best_action.items():
    ad_name = next((ad['ad_name'] for ad in active_ads if ad['ad_id'] == ad_id), ad_id)
    print(f"- {ad_name}: ${budget:.2f}")
print(f"\nExpected profit: ${expected_profit:.2f}")
```

## Detailed Usage Guide

### 1. Create the AIcon

Start by creating a SimpleBadAIcon instance:

```python
from aicons.definitions.simple_bad_aicon import SimpleBadAIcon
aicon = SimpleBadAIcon(name="BudgetOptimizer")
```

### 2. Define Priors (Optional)

You can explicitly define prior beliefs about ad performance:

```python
# Add priors for conversion rates and costs
aicon.add_factor_continuous("conversion_rate_ad1", 0.05, 0.01, lower_bound=0.0)
aicon.add_factor_continuous("cost_per_click_ad1", 0.5, 0.1, lower_bound=0.0)
```

### 3. Set Up Meta Ads Sensor

Configure the Meta Ads sensor with your credentials:

```python
from aicons.bayesbrainGPT.sensors.meta_ads_sensor import MetaAdsSensor

sensor = MetaAdsSensor(
    name="meta_ads",
    access_token="your_access_token",
    ad_account_id="act_your_account_id",
    campaign_id="your_campaign_id",
    api_version="v18.0",
    time_granularity="hour"  # Options: "hour", "day", "week"
)

aicon.add_sensor("meta_ads", sensor)
```

### 4. Create Action Space

Define the budget allocation action space:

```python
# Get active ads
active_ads = sensor.get_active_ads()
ad_ids = [ad['ad_id'] for ad in active_ads]

action_space = aicon.create_action_space(
    space_type='budget_allocation',
    total_budget=1000,  # Total budget to allocate
    items=ad_ids,       # Which ads to allocate budget to
    step_size=0.01,     # Step size (as percentage of total)
    ignore_validation=True  # Optional: ignore validation checks
)
```

### 5. Create Utility Function

Define how you measure success:

```python
utility_function = aicon.create_utility_function(
    utility_type='marketing_roi',
    revenue_per_sale=50.0,  # Revenue per conversion
    # Optional parameters:
    # min_roi=2.0,         # Minimum ROI constraint
    # brand_impact=0.2     # Weight for brand impact
)
```

### 6. Update Beliefs

Gather the latest data to update your beliefs:

```python
# Update with latest Meta Ads performance
aicon.update_from_sensor("meta_ads")

# Alternatively, provide custom data
aicon.update_from_sensor("meta_ads", environment={
    "ad1": {"clicks": 1000, "conversions": 50},
    "ad2": {"clicks": 800, "conversions": 32},
    "ad3": {"clicks": 1200, "conversions": 72}
})
```

### 7. Find Optimal Allocation

Use gradient-based optimization to find the best budget allocation:

```python
best_action, expected_profit = aicon.find_best_action(
    num_samples=500,     # Number of samples to evaluate
    use_gradient=True    # Use gradient-based optimization
)

# Print results
print("\nOPTIMAL BUDGET ALLOCATION:")
for ad_id, budget in best_action.items():
    ad_name = next((ad['ad_name'] for ad in active_ads if ad['ad_id'] == ad_id), ad_id)
    print(f"- {ad_name}: ${budget:.2f}")
print(f"\nExpected profit: ${expected_profit:.2f}")
```

## Advanced Usage

### Different Action Space Types

The system supports various action space types:

```python
# Simple marketing space
action_space = aicon.create_action_space(
    space_type='marketing',
    total_budget=1000,
    num_ads=3,
    budget_step=10
)

# Time-based budget allocation
action_space = aicon.create_action_space(
    space_type='time_budget',
    total_budget=3000,
    num_ads=3,
    num_days=7,
    budget_step=50
)

# Multi-campaign allocation
action_space = aicon.create_action_space(
    space_type='multi_campaign',
    campaigns={
        'campaign1': {'total_budget': 1000, 'ads': ['ad1', 'ad2'], 'days': 3},
        'campaign2': {'total_budget': 2000, 'ads': ['ad3', 'ad4'], 'days': 5}
    },
    budget_step=50
)
```

### Custom Utility Functions

Create custom utility functions:

```python
utility_function = aicon.create_utility_function(
    utility_type='custom_marketing',
    revenue_function=lambda conversions: conversions * 50,
    cost_function=lambda clicks, cpc: clicks * cpc,
    constraint_function=lambda allocation: allocation['ad1'] >= 100  # Minimum spend on ad1
)
```

### Continuous Updating

Run the system continuously to adapt to changing conditions:

```python
# Run continuously, updating every hour
aicon.run(
    mode='continuous',
    sensor_name="meta_ads",
    interval=3600,  # Update every hour
)

# Run for a specific duration
aicon.run(
    mode='finite',
    sensor_name="meta_ads",
    interval=3600,
    duration=86400  # Run for 24 hours
)
```

## Understanding the Bayesian Framework

This system uses a full Bayesian approach:

1. **Priors**: Initial beliefs about conversion rates, costs, etc.
2. **Likelihood**: Model of how data is generated given the true parameters
3. **Posterior**: Updated beliefs after observing data
4. **Expected Utility**: Integration over posterior to find optimal actions

The gradient-based optimization ensures proper integration over the posterior distribution, accounting for uncertainty in ad performance parameters.

## ZeroAIcon Chat Interface

The project includes an interactive chat interface that allows you to converse with the ZeroAIcon's reasoning process.

### Features

- Real-time streaming of AI responses
- Toggleable "Thinking Process" visualization
- Interactive collapsible sections for AI reasoning
- Clean web interface for easy interaction

### Running the Chat Server

To start the chat interface:

```bash
# Using Poetry (recommended)
poetry run python run_chat.py --port 8000

# Or with plain Python
python run_chat.py --port 8000
```

Then open your browser to http://localhost:8000 to access the chat interface.

### Chat Commands

The chat interface responds to natural language queries and shows both the final answer and the AI's reasoning process.

You can toggle the display of the thinking process in two ways:

1. Use the "Show Thinking Process" toggle switch before sending a message
2. Click the "Show Thinking Process" button that appears with any response that includes thinking steps

This interactive thinking visualization helps understand how the AI approaches problems and makes its decisions.
