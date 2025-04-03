import sys
import os
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from aicons.bayesbrainGPT.decision_making.marketing_action_spaces import create_budget_allocation_space

def test_budget_combinations():
    # Test case 1: Simple case with 2 ads
    print("\n=== Test Case 1: 2 ads, $100 total, $50 steps ===")
    space = create_budget_allocation_space(
        total_budget=100,
        num_ads=2,
        budget_step=50
    )
    print(f"Found {len(space.valid_actions)} combinations:")
    for action in space.valid_actions:
        print(f"  {action} (sum: {sum(action.values())})")

    # Test case 2: 3 ads with medium budget
    print("\n=== Test Case 2: 3 ads, $200 total, $50 steps ===")
    space = create_budget_allocation_space(
        total_budget=200,
        num_ads=3,
        budget_step=50
    )
    print(f"Found {len(space.valid_actions)} combinations:")
    for action in space.valid_actions[:5]:  # Print first 5
        print(f"  {action} (sum: {sum(action.values())})")
    if len(space.valid_actions) > 5:
        print(f"  ... and {len(space.valid_actions) - 5} more")

    # Test case 3: With minimum budget
    print("\n=== Test Case 3: 2 ads, $200 total, $50 steps, min $50 ===")
    space = create_budget_allocation_space(
        total_budget=200,
        num_ads=2,
        budget_step=50,
        min_budget=50
    )
    print(f"Found {len(space.valid_actions)} combinations:")
    for action in space.valid_actions:
        print(f"  {action} (sum: {sum(action.values())})")

    # Test case 4: Real use case
    print("\n=== Test Case 4: 7 ads, $1000 total, $100 steps ===")
    space = create_budget_allocation_space(
        total_budget=1000,
        num_ads=7,
        budget_step=100
    )
    print(f"Found {len(space.valid_actions)} combinations")
    print("\nFirst 5 combinations:")
    for action in space.valid_actions[:5]:
        print(f"  {action} (sum: {sum(action.values())})")
    if len(space.valid_actions) > 5:
        print(f"\n... and {len(space.valid_actions) - 5} more")

if __name__ == '__main__':
    test_budget_combinations() 