# main.py
from state_representation.state import EnvironmentState
from state_representation.factors import ContinuousFactor, CategoricalFactor, DiscreteFactor, BayesianLinearFactor
from config import DEFAULT_STATE_CONFIG
from llm_integration import fetch_state_context_from_llm

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
    
    # Update the state with the sensor data extracted from the LLM.
    env_state.update_state(llm_sensor_data)
    
    print("State after LLM retrieval:")
    print(env_state)
