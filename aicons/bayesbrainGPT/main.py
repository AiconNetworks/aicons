# main.py
from .state_representation.state import EnvironmentState
from .state_representation.factors import ContinuousFactor, CategoricalFactor, DiscreteFactor, BayesianLinearFactor
from .perception import BayesianPerception
from .config import DEFAULT_STATE_CONFIG
from .llm_integration import fetch_state_context_from_llm
import torch

def create_state_from_config(config: dict) -> EnvironmentState:
    """
    Create an EnvironmentState from the configuration dictionary.
    Depending on the factor type, instantiate the appropriate Factor subclass.
    """
    factors = {}
    for key, params in config.items():
        factor_type = params.get("type", "continuous")
        if factor_type == "continuous":
            factors[key] = ContinuousFactor(
                name=key,
                value=params.get("value"),
                description=params.get("description", "")
            )
        elif factor_type == "categorical":
            factors[key] = CategoricalFactor(
                name=key,
                value=params.get("value"),
                description=params.get("description", "")
            )
        elif factor_type == "discrete":
            factors[key] = DiscreteFactor(
                name=key,
                value=params.get("value"),
                description=params.get("description", "")
            )
        elif factor_type == "bayesian_linear":
            factors[key] = BayesianLinearFactor(
                name=key,
                explanatory_vars=params.get("explanatory_vars", {}),
                theta_prior=params.get("theta_prior", {}),
                variance=params.get("variance", 1.0),
                description=params.get("description", "")
            )
        else:
            raise ValueError(f"Unknown factor type: {factor_type} for factor '{key}'")
    return EnvironmentState(factors=factors)

if __name__ == "__main__":
    # Instead of hardcoding sensor data, fetch state context via an LLM.
    prompt = "Based on current observations, provide environmental data including rain, temperature, traffic, and weather."
    llm_sensor_data = fetch_state_context_from_llm(prompt)
    
    # Create the initial state from configuration (which includes priors for factors).
    env_state = create_state_from_config(DEFAULT_STATE_CONFIG)
    perception = BayesianPerception(env_state)
    
    # Convert LLM data to observation format with reliability scores
    observations = {}
    for key, value in llm_sensor_data.items():
        if key in env_state.factors:
            if isinstance(env_state.factors[key], CategoricalFactor):
                # Convert categorical to one-hot with high reliability for LLM data
                possible_values = env_state.factors[key].possible_values
                one_hot = torch.zeros(len(possible_values))
                one_hot[possible_values.index(value)] = 1.0
                observations[key] = (one_hot, 0.8)  # 0.8 reliability for LLM data
            else:
                # Convert numerical with high reliability for LLM data
                observations[key] = (torch.tensor(float(value)), 0.8)
    
    # Update the state through Bayesian perception
    perception.update_all(observations=observations)
    
    print("State after LLM retrieval:")
    print(env_state)
