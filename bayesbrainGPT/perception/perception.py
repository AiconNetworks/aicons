import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from typing import Dict, List, Optional
from ..state_representation.state import EnvironmentState
from .sensor import Sensor

class BayesianPerception:
    """
    Handles Bayesian inference for perception based on sensor data and prior beliefs.
    """
    def __init__(self, state: EnvironmentState):
        self.state = state
        self.sensors: Dict[str, Sensor] = {}
        self.setup_pyro_model()
        
    def setup_pyro_model(self):
        """Initialize the Pyro model with current state priors"""
        # Define the model based on state factors
        def model():
            # Get priors from state
            for factor_name, factor in self.state.factors.items():
                if isinstance(factor, BayesianLinearFactor):
                    # Handle Bayesian linear factors
                    self._sample_bayesian_linear(factor_name, factor)
                elif isinstance(factor, ContinuousFactor):
                    # Handle continuous factors
                    self._sample_continuous(factor_name, factor)
                elif isinstance(factor, CategoricalFactor):
                    # Handle categorical factors with relaxation
                    self._sample_categorical(factor_name, factor)
                elif isinstance(factor, DiscreteFactor):
                    # Handle discrete factors with continuous relaxation
                    self._sample_discrete(factor_name, factor)
        
        self.model = model

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
        # Use current value as mean and add some uncertainty
        mean = torch.tensor(float(factor.value)) if factor.value is not None else torch.tensor(0.0)
        pyro.sample(name, dist.Normal(mean, torch.tensor(1.0)))

    def _sample_categorical(self, name: str, factor: CategoricalFactor):
        """Sample from a categorical factor using relaxation"""
        # Convert categorical to one-hot with relaxation
        categories = factor.possible_values if hasattr(factor, 'possible_values') else ['unknown']
        logits = torch.ones(len(categories)) / len(categories)
        temperature = torch.tensor(0.5)
        pyro.sample(name, dist.RelaxedOneHotCategorical(temperature=temperature, logits=logits))

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
        """Update perception based on data from a specific sensor"""
        if sensor_name not in self.sensors:
            raise ValueError(f"Unknown sensor: {sensor_name}")
        
        sensor = self.sensors[sensor_name]
        sensor_data = sensor.get_data()
        
        # Run inference with the sensor data
        nuts_kernel = NUTS(self.model)
        mcmc = MCMC(nuts_kernel, num_samples=2000, warmup_steps=1000)
        mcmc.run(**sensor_data)
        
        # Update state with posterior samples
        samples = mcmc.get_samples()
        self._update_state_from_samples(samples)

    def update_all(self):
        """Update perception using all registered sensors"""
        all_sensor_data = {}
        for sensor in self.sensors.values():
            all_sensor_data.update(sensor.get_data())
            
        # Run inference with all sensor data
        nuts_kernel = NUTS(self.model)
        mcmc = MCMC(nuts_kernel, num_samples=2000, warmup_steps=1000)
        mcmc.run(**all_sensor_data)
        
        # Update state with posterior samples
        samples = mcmc.get_samples()
        self._update_state_from_samples(samples)

    def _update_state_from_samples(self, samples: Dict[str, torch.Tensor]):
        """Update state factors with posterior samples"""
        for name, sample_tensor in samples.items():
            if name in self.state.factors:
                # Use mean of posterior samples as new value
                new_value = sample_tensor.mean(0)
                self.state.factors[name].value = new_value.item() 