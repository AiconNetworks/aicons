import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from typing import Dict, List, Optional
from ..state_representation.state import EnvironmentState
from ..state_representation.factors import BayesianLinearFactor, ContinuousFactor, CategoricalFactor, DiscreteFactor
from ..sensors import Sensor

class BayesianPerception:
    """
    Handles Bayesian inference for perception based on sensor data and prior beliefs.
    """
    def __init__(self, state: EnvironmentState):
        self.state = state
        self.sensors: Dict[str, Sensor] = {}
        self.setup_pyro_model()
        
    def setup_pyro_model(self):
        """Initialize dynamic Pyro model based on factor relationships"""
        def model(**observations):
            # First sample independent factors
            sampled_values = {}
            for name, factor in self.state.factors.items():
                if not self._has_dependencies(factor):
                    value = self._sample_factor(f"{name}_prior", factor)
                    sampled_values[name] = value

            # Then sample dependent factors in order
            dependency_order = self._get_dependency_order()
            for name in dependency_order:
                factor = self.state.factors[name]
                try:
                    value = self._sample_dependent_factor(
                        f"{name}_dependent", factor, sampled_values
                    )
                    sampled_values[name] = value
                except Exception as e:
                    print(f"Error sampling {name}: {str(e)}")
                    raise

            # Add observations
            for name, (value, reliability) in observations.items():
                if name in sampled_values:
                    noise = torch.tensor(1.0) / reliability
                    pyro.sample(
                        f"{name}_obs",
                        dist.Normal(sampled_values[name], noise),
                        obs=value
                    )

        self.model = model

    def _sample_dependent_factor(self, name: str, factor, sampled_values: dict):
        """Sample a factor considering its dependencies"""
        if not hasattr(factor, 'relationships') or not factor.relationships:
            return self._sample_factor(name, factor)

        # Calculate mean based on dependencies
        if isinstance(factor, ContinuousFactor):
            mean = self._compute_dependent_mean(factor, sampled_values)
            return pyro.sample(name, dist.Normal(mean, factor.uncertainty))
        elif isinstance(factor, CategoricalFactor):
            logits = self._compute_categorical_logits(factor, sampled_values)
            return pyro.sample(name, 
                dist.RelaxedOneHotCategorical(
                    temperature=torch.tensor(0.5),
                    logits=logits
                ))
        else:
            return self._sample_factor(name, factor)

    def _sample_factor(self, name: str, factor):
        """Sample from a factor"""
        if isinstance(factor, BayesianLinearFactor):
            return self._sample_bayesian_linear(name, factor)
        elif isinstance(factor, ContinuousFactor):
            return self._sample_continuous(name, factor)
        elif isinstance(factor, CategoricalFactor):
            return self._sample_categorical(name, factor)
        elif isinstance(factor, DiscreteFactor):
            return self._sample_discrete(name, factor)

    def _sample_bayesian_linear(self, name: str, factor: BayesianLinearFactor):
        """Sample from a Bayesian linear factor"""
        # Extract parameters from factor
        explanatory_vars = torch.tensor([float(v) for v in factor.explanatory_vars.values()])
        theta_prior = factor.theta_prior
        
        # Sample coefficients
        theta = pyro.sample(
            f"{name}_theta",
            dist.Normal(
                torch.tensor(theta_prior["mean"]),
                torch.tensor(theta_prior["variance"])
            ).to_event(1)
        )
        
        # Linear combination
        mean = torch.dot(theta, explanatory_vars)
        pyro.sample(name, dist.Normal(mean, torch.tensor(factor.variance)))

    def _sample_continuous(self, name: str, factor: ContinuousFactor):
        """Sample from a continuous factor"""
        # Create a prior distribution based on the current value
        prior_mean = torch.tensor(factor.value if factor.value is not None else 0.0)
        prior_std = torch.tensor(1.0)
        
        # Sample from the prior
        value = pyro.sample(
            name,
            dist.Normal(prior_mean, prior_std)
        )
        return value  # Make sure we return the sampled value

    def _sample_categorical(self, name: str, factor: CategoricalFactor):
        """Sample from a categorical factor using relaxation"""
        n_categories = len(factor.possible_values)
        # Create logits based on current value
        logits = torch.zeros(n_categories)
        current_idx = factor.possible_values.index(factor.value)
        logits[current_idx] = 1.0  # Slight preference for current value
        
        # Sample using RelaxedOneHotCategorical (Gumbel-Softmax)
        temperature = torch.tensor(0.5)  # Lower values -> more discrete
        return pyro.sample(name, 
            dist.RelaxedOneHotCategorical(temperature=temperature, logits=logits))

    def _sample_discrete(self, name: str, factor: DiscreteFactor):
        """Sample from a discrete factor using continuous relaxation"""
        # Use LogNormal as continuous approximation
        mean = torch.tensor(float(factor.value)) if factor.value is not None else torch.tensor(1.0)
        sigma = torch.tensor(0.1)
        mu = torch.log(mean) - (sigma**2)/2
        pyro.sample(name, dist.LogNormal(mu, sigma))

    def register_sensor(self, sensor: Sensor):
        """Register a new sensor"""
        self.sensors[sensor.name] = sensor

    def update_from_sensor(self, sensor_name: str):
        """Update state using data from a specific sensor"""
        if sensor_name not in self.sensors:
            raise ValueError(f"No sensor registered with name {sensor_name}")
        
        # Get data and separate values from reliabilities
        sensor_data = self.sensors[sensor_name].get_data()
        
        print("\nPERCEPTION UPDATE:")
        # Process each observation
        for factor, (value, reliability) in sensor_data.items():
            if factor in self.state.factors:
                factor_obj = self.state.factors[factor]
                print(f"\nFactor: {factor}")
                print(f"Prior: Normal(μ={factor_obj.value}, σ={factor_obj.uncertainty})")
                print(f"Observation: value={value.item()}, reliability={reliability}")
                
                # Update value and uncertainty
                factor_obj.value = value.item()
                new_uncertainty = 1.0 / reliability if reliability > 0 else float('inf')
                factor_obj.update_uncertainty(new_uncertainty)
                
                print(f"Posterior: Normal(μ={factor_obj.value}, σ={factor_obj.uncertainty})")
        
        # TODO: Later we'll implement proper Bayesian updates considering reliability

    def update_all(self):
        """Run full Bayesian update considering all relationships"""
        print("\n=== Starting Bayesian Update ===")
        
        # 1. Collect sensor data
        all_sensor_data = {}
        for sensor in self.sensors.values():
            sensor_data = sensor.get_data()
            for factor, (value, reliability) in sensor_data.items():
                all_sensor_data[factor] = (value, reliability)
                print(f"\nReceived sensor data:")
                print(f"  {factor}: value={value}, reliability={reliability}")
        
        print("\nInitial state:")
        for name, factor in self.state.factors.items():
            print(f"  {name}: {factor.value}")
            if hasattr(factor, 'relationships'):
                print(f"  {name} relationships: {factor.relationships}")
        
        # 2. Run MCMC
        print("\nRunning MCMC sampling...")
        nuts_kernel = NUTS(self.model)
        mcmc = MCMC(nuts_kernel, num_samples=50, warmup_steps=25)
        mcmc.run(**all_sensor_data)
        
        # 3. Get samples and prepare for update
        samples = mcmc.get_samples()
        final_samples = {name: samples[name][-1] for name in samples}
        
        print("\nFinal samples obtained:")
        for name, sample in final_samples.items():
            print(f"  {name}: {sample}")
        
        # 4. Update factors in correct order
        print("\nUpdating factors in dependency order:")
        
        # First independent factors
        updated_values = {}
        for name, factor in self.state.factors.items():
            if not self._has_dependencies(factor):
                if name in final_samples:
                    print(f"\nUpdating independent factor {name}:")
                    old_value = factor.value
                    self._update_factor_from_sample(name, factor, final_samples[name])
                    updated_values[name] = factor.value
                    print(f"  {name} changed: {old_value} -> {factor.value}")
        
        # Then dependent factors
        dependency_order = self._get_dependency_order()
        print(f"\nDependency order: {dependency_order}")
        
        for name in dependency_order:
            if name in final_samples:
                factor = self.state.factors[name]
                print(f"\nUpdating dependent factor {name}:")
                old_value = factor.value
                
                if self._has_dependencies(factor):
                    print(f"  Computing dependent value for {name}")
                    print(f"  Using updated values: {updated_values}")
                    mean = self._compute_dependent_mean(factor, updated_values)
                    factor.value = float(mean)
                    print(f"  Computed new value from dependencies: {factor.value}")
                else:
                    self._update_factor_from_sample(name, factor, final_samples[name])
                
                updated_values[name] = factor.value
                print(f"  {name} changed: {old_value} -> {factor.value}")
        
        print("\n=== Final State ===")
        for name, factor in self.state.factors.items():
            print(f"{name}: {factor.value}")

    def _update_factor_from_sample(self, name: str, factor, sample):
        """Update a factor based on the final MCMC sample"""
        if isinstance(factor, CategoricalFactor):
            if len(sample.shape) > 0:  # One-hot encoded
                category_idx = sample.argmax().item()
                old_value = factor.value
                factor.value = factor.possible_values[category_idx]
                print(f"{name}: {old_value} -> {factor.value}")
                print(f"  probabilities: {dict(zip(factor.possible_values, sample.tolist()))}")
        else:
            old_value = factor.value
            new_value = float(sample)
            factor.value = new_value
            print(f"{name}: {old_value:.2f} -> {new_value:.2f}")

    def _compute_dependent_mean(self, factor, sampled_values: dict):
        """Compute mean value based on factor relationships"""
        if isinstance(factor, CategoricalFactor):
            return factor.value

        base_value = float(factor.value)
        original_value = base_value  # Store original for logging
        
        if not hasattr(factor, 'relationships') or 'model' not in factor.relationships:
            return torch.tensor(base_value)

        # Process each dependency
        for dependency, model in factor.relationships["model"].items():
            if dependency not in sampled_values:
                continue
            
            dep_value = sampled_values[dependency]
            
            # Handle relaxed categorical values (one-hot vectors)
            if isinstance(dep_value, torch.Tensor) and len(dep_value.shape) > 0:
                if model["type"] == "categorical_effect":
                    dep_factor = self.state.factors[dependency]
                    effects = torch.tensor([
                        model["effects"].get(val, 0.0) 
                        for val in dep_factor.possible_values
                    ])
                    effect = (dep_value * effects).sum()
                    base_value += float(effect)
                    print(f"  {factor.name}: Categorical effect from {dependency}: {float(effect)}")
            # Handle numerical values
            else:
                dep_value = float(dep_value)
                if model["type"] == "linear":
                    effect = model["coefficient"] * dep_value
                    base = model.get("base", base_value)
                    base_value = base + effect
                    print(f"  {factor.name}: Linear effect from {dependency}: base={base}, coeff={model['coefficient']}, value={dep_value}")
                elif model["type"] == "exponential":
                    effect = model["base"] * torch.exp(model["scale"] * dep_value)
                    base_value *= float(effect)
                    print(f"  {factor.name}: Exponential effect from {dependency}: base={model['base']}, scale={model['scale']}, value={dep_value}")

        if base_value != original_value:
            print(f"  {factor.name}: Value changed {original_value:.2f} -> {base_value:.2f}")
        return torch.tensor(base_value)

    def _has_dependencies(self, factor) -> bool:
        """Check if a factor has dependencies on other factors"""
        return hasattr(factor, 'relationships') and \
               'depends_on' in factor.relationships

    def _get_dependency_order(self) -> List[str]:
        """Get factors in order of their dependencies"""
        order = []
        visited = set()
        temp_visited = set()  # For cycle detection
        
        def visit(name):
            if name in temp_visited:
                raise ValueError(f"Cyclic dependency detected involving {name}")
            if name in visited:
                return
            
            temp_visited.add(name)
            factor = self.state.factors[name]
            
            if self._has_dependencies(factor):
                for dep in factor.relationships['depends_on']:
                    if dep not in self.state.factors:
                        raise ValueError(f"Unknown dependency {dep} for factor {name}")
                    visit(dep)
                
            visited.add(name)
            temp_visited.remove(name)
            order.append(name)
        
        try:
            for name in self.state.factors:
                if name not in visited:
                    visit(name)
        except Exception as e:
            print(f"Error in dependency resolution: {str(e)}")
            raise
        
        return order

    def _compute_categorical_logits(self, factor, sampled_values: dict) -> torch.Tensor:
        """Compute logits for categorical factors based on dependencies"""
        if not hasattr(factor, 'possible_values'):
            return torch.ones(1)  # Default for unknown categories
        
        n_categories = len(factor.possible_values)
        logits = torch.ones(n_categories) / n_categories
        
        if self._has_dependencies(factor):
            # Modify logits based on dependencies
            for dep, model in factor.relationships['model'].items():
                if dep in sampled_values:
                    # Apply dependency effects to logits
                    if model['type'] == 'categorical_effect':
                        for i, val in enumerate(factor.possible_values):
                            logits[i] *= model['effects'].get(val, 1.0)
        
        return logits 