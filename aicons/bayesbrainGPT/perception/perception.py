import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from typing import Dict, List, Optional, Any

# TFP shortcuts
tfd = tfp.distributions
tfb = tfp.bijectors

class BayesianPerception:
    """
    Handles Bayesian inference for perception based on sensor data and prior beliefs
    using TensorFlow Probability.
    """
    def __init__(self, brain):
        """
        Initialize the perception system with a reference to the brain.
        
        Args:
            brain: The BayesBrain instance
        """
        self.brain = brain
        self.sensors = {}
        self.posterior_samples = {}
        
    def register_sensor(self, name, sensor_function):
        """
        Register a new sensor that can provide observations.
        
        Args:
            name: Name of the sensor
            sensor_function: Function that returns sensor data
        """
        self.sensors[name] = sensor_function
        print(f"Registered sensor: {name}")
    
    def collect_sensor_data(self, environment=None):
        """
        Collect data from all registered sensors.
        
        Args:
            environment: Optional environment data to pass to sensors
            
        Returns:
            Dictionary of observations from all sensors
        """
        observations = {}
        
        for name, sensor_function in self.sensors.items():
            # Get data from sensor function
            data = sensor_function(environment) if environment else sensor_function()
            print(f"Collected data from sensor: {name}")
            
            # Add to observations
            for factor_name, observation in data.items():
                if isinstance(observation, tuple) and len(observation) == 2:
                    # If observation includes value and reliability
                    observations[factor_name] = observation
                else:
                    # If only value is provided, assume reliability = 1.0
                    observations[factor_name] = (observation, 1.0)
                    
                print(f"  Factor: {factor_name}, Value: {observations[factor_name][0]}, Reliability: {observations[factor_name][1]}")
                
        return observations
    
    def create_joint_prior(self):
        """
        Create a TensorFlow Probability joint distribution based on state factors.
        
        Returns:
            Joint distribution object
        """
        state_factors = self.brain.get_state_factors()
        prior_dict = {}
        
        for name, factor in state_factors.items():
            # Use the TensorFlow distribution directly if available
            if "tf_distribution" in factor:
                prior_dict[name] = factor["tf_distribution"]
                continue
                
            # Legacy fallback for state factors without TF distributions
            if "type" not in factor:
                continue
                
            factor_type = factor["type"]
            
            if factor_type == "continuous":
                # Continuous variable with normal distribution
                loc = float(factor["params"]["loc"])
                scale = float(factor["params"]["scale"])
                
                # Handle constraints if present
                if "constraints" in factor and factor["constraints"]:
                    constraints = factor["constraints"]
                    
                    if "lower" in constraints and "upper" in constraints:
                        # Bounded normal
                        lower = float(constraints["lower"])
                        upper = float(constraints["upper"])
                        prior_dict[name] = tfd.TruncatedNormal(
                            loc=loc, scale=scale, low=lower, high=upper
                        )
                    elif "lower" in constraints:
                        # Lower bounded (e.g., positive values)
                        lower = float(constraints["lower"])
                        # Use transformation to ensure lower bound
                        prior_dict[name] = tfd.TransformedDistribution(
                            distribution=tfd.Normal(loc=loc-lower, scale=scale),
                            bijector=tfb.Shift(shift=lower) @ tfb.Softplus()
                        )
                    elif "upper" in constraints:
                        # Upper bounded
                        upper = float(constraints["upper"])
                        # Use transformation to ensure upper bound
                        prior_dict[name] = tfd.TransformedDistribution(
                            distribution=tfd.Normal(loc=upper-loc, scale=scale),
                            bijector=tfb.Shift(shift=upper) @ tfb.Scale(-1.0) @ tfb.Softplus()
                        )
                else:
                    # Unconstrained normal
                    prior_dict[name] = tfd.Normal(loc=loc, scale=scale)
                    
            elif factor_type == "categorical":
                # Categorical variable
                categories = factor["categories"]
                probs = tf.constant(factor["params"]["probs"], dtype=tf.float32)
                prior_dict[name] = tfd.Categorical(probs=probs)
                
            elif factor_type == "discrete":
                # Discrete variable
                if "categories" in factor:
                    # Finite discrete values
                    categories = factor["categories"]
                    probs = tf.constant(factor["params"]["probs"], dtype=tf.float32)
                    prior_dict[name] = tfd.Categorical(probs=probs)
                else:
                    # Poisson distribution for counts
                    rate = float(factor["params"]["rate"])
                    prior_dict[name] = tfd.Poisson(rate=rate)
        
        # Create joint distribution
        if not prior_dict:
            # Return None if no valid priors
            return None
            
        return tfd.JointDistributionNamed(prior_dict)
    
    def create_likelihood_function(self, observations):
        """
        Create a function that computes the log likelihood of observations.
        
        Args:
            observations: Dictionary of (value, reliability) tuples
            
        Returns:
            Log likelihood function
        """
        state_factors = self.brain.get_state_factors()
        
        def log_likelihood(**state_values):
            total_ll = 0.0
            
            for factor_name, (obs_value, reliability) in observations.items():
                if factor_name not in state_values or factor_name not in state_factors:
                    continue
                    
                factor = state_factors[factor_name]
                factor_type = factor["type"]
                factor_value = state_values[factor_name]
                
                # Use reliability to scale noise
                noise_scale = 1.0 / reliability if reliability > 0 else 1.0
                
                if factor_type == "continuous":
                    # Continuous variable - normal likelihood
                    ll = tfd.Normal(loc=factor_value, scale=noise_scale).log_prob(obs_value)
                    total_ll += ll
                    
                elif factor_type == "categorical":
                    # Categorical variable
                    if isinstance(obs_value, str) and "categories" in factor:
                        # Convert category name to index
                        categories = factor["categories"]
                        if obs_value in categories:
                            index = categories.index(obs_value)
                            # One-hot encoding for categorical variables
                            one_hot = tf.one_hot(index, len(categories))
                            ll = tf.reduce_sum(tf.math.log(factor_value + 1e-10) * one_hot)
                            total_ll += ll
                    
                elif factor_type == "discrete":
                    # Discrete variable
                    if "categories" in factor:
                        # Finite discrete values - similar to categorical
                        if isinstance(obs_value, int) and 0 <= obs_value < len(factor["categories"]):
                            one_hot = tf.one_hot(obs_value, len(factor["categories"]))
                            ll = tf.reduce_sum(tf.math.log(factor_value + 1e-10) * one_hot)
                            total_ll += ll
                    else:
                        # Poisson distribution
                        ll = tfd.Poisson(rate=factor_value).log_prob(obs_value)
                        total_ll += ll
            
            return total_ll
            
        return log_likelihood
    
    def sample_posterior(self, observations, num_results=1000, num_burnin_steps=500):
        """
        Sample from the posterior distribution using MCMC.
        
        Args:
            observations: Dictionary of observations
            num_results: Number of samples to generate
            num_burnin_steps: Number of burn-in steps
            
        Returns:
            Dictionary of posterior samples
        """
        print("\nPERCEPTION UPDATE:")
        print(f"Observations: {observations}")
        
        # Create prior distribution
        prior = self.create_joint_prior()
        if prior is None:
            print("No valid prior distributions found")
            return {}
            
        # Create likelihood function
        likelihood_fn = self.create_likelihood_function(observations)
        
        # Create target log probability function
        def target_log_prob(**state_values):
            log_prior = prior.log_prob(state_values)
            log_likelihood = likelihood_fn(**state_values)
            return log_prior + log_likelihood
        
        # Initialize state for HMC
        state_factors = self.brain.get_state_factors()
        initial_state = []
        state_names = []
        
        for name, factor in state_factors.items():
            if "type" not in factor or "value" not in factor:
                continue
                
            state_names.append(name)
            initial_state.append(factor["value"])
        
        if not initial_state:
            print("No valid initial state for MCMC")
            return {}
            
        # Convert to TensorFlow tensors
        initial_state = [tf.constant(val, dtype=tf.float32) for val in initial_state]
        
        # Set up HMC kernel
        hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob,
            step_size=0.01,
            num_leapfrog_steps=3
        )
        
        # Add adaptive step size
        adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=hmc_kernel,
            num_adaptation_steps=int(num_burnin_steps * 0.8)
        )
        
        # Run HMC
        @tf.function
        def run_chain():
            return tfp.mcmc.sample_chain(
                num_results=num_results,
                num_burnin_steps=num_burnin_steps,
                current_state=initial_state,
                kernel=adaptive_hmc,
                trace_fn=None
            )
            
        try:
            samples = run_chain()
            
            # Convert samples to dictionary
            posterior_samples = {}
            for i, name in enumerate(state_names):
                posterior_samples[name] = samples[i]
                
            self.posterior_samples = posterior_samples
            print(f"Generated {num_results} posterior samples")
            return posterior_samples
            
        except Exception as e:
            print(f"Error in MCMC sampling: {str(e)}")
            return {}
    
    def update_state_from_posterior(self):
        """
        Update state factors based on posterior samples.
        
        Returns:
            True if update was successful
        """
        if not self.posterior_samples:
            print("No posterior samples available")
            return False
            
        state_factors = self.brain.get_state_factors().copy()
        updated_factors = state_factors.copy()
        
        print("\nUpdating state factors from posterior:")
        
        for name, samples in self.posterior_samples.items():
            if name not in state_factors:
                continue
                
            factor = state_factors[name]
            factor_type = factor["type"]
            
            if factor_type == "continuous":
                # Update continuous factor with posterior mean and std
                samples_np = samples.numpy()
                new_mean = float(np.mean(samples_np))
                new_std = float(np.std(samples_np))
                
                old_value = factor["value"]
                factor["value"] = new_mean
                factor["params"]["loc"] = new_mean
                factor["params"]["scale"] = new_std
                
                print(f"  {name}: {old_value:.4f} -> {new_mean:.4f} (std: {new_std:.4f})")
                
            elif factor_type == "categorical":
                # Update categorical factor
                if "categories" in factor:
                    samples_np = samples.numpy()
                    categories = factor["categories"]
                    
                    # For categorical, find the most probable category
                    if len(samples_np.shape) > 1:
                        # One-hot encoded samples
                        counts = np.sum(samples_np, axis=0)
                    else:
                        # Index-based samples
                        counts = np.bincount(samples_np, minlength=len(categories))
                        
                    new_probs = counts / np.sum(counts)
                    most_likely_idx = np.argmax(new_probs)
                    new_value = categories[most_likely_idx]
                    
                    old_value = factor["value"]
                    factor["value"] = new_value
                    factor["params"]["probs"] = new_probs.tolist()
                    
                    print(f"  {name}: {old_value} -> {new_value}")
                    print(f"    New probabilities: {dict(zip(categories, new_probs))}")
                    
            elif factor_type == "discrete":
                # Update discrete factor
                if "categories" in factor:
                    # Categorical-like discrete
                    samples_np = samples.numpy()
                    categories = factor["categories"]
                    
                    if len(samples_np.shape) > 1:
                        # One-hot encoded
                        counts = np.sum(samples_np, axis=0)
                    else:
                        # Index-based
                        counts = np.bincount(samples_np, minlength=len(categories))
                        
                    new_probs = counts / np.sum(counts)
                    most_likely_idx = np.argmax(new_probs)
                    new_value = categories[most_likely_idx]
                    
                    old_value = factor["value"]
                    factor["value"] = new_value
                    factor["params"]["probs"] = new_probs.tolist()
                    
                    print(f"  {name}: {old_value} -> {new_value}")
                else:
                    # Poisson distribution
                    samples_np = samples.numpy()
                    new_rate = float(np.mean(samples_np))
                    new_value = int(round(new_rate))
                    
                    old_value = factor["value"]
                    factor["value"] = new_value
                    factor["params"]["rate"] = new_rate
                    
                    print(f"  {name}: {old_value} -> {new_value} (rate: {new_rate:.2f})")
        
        # Update brain state
        self.brain.set_state_factors(updated_factors)
        return True
    
    def update_from_sensor(self, sensor_name: str, environment=None):
        """
        Update state using data from a specific sensor.
        
        Args:
            sensor_name: Name of the sensor
            environment: Optional environment data
            
        Returns:
            True if update was successful
        """
        if sensor_name not in self.sensors:
            print(f"No sensor registered with name {sensor_name}")
            return False
            
        # Get sensor data
        sensor_data = {}
        sensor_function = self.sensors[sensor_name]
        data = sensor_function(environment) if environment else sensor_function()
        
        # Process data to ensure (value, reliability) tuples
        for factor_name, observation in data.items():
            if isinstance(observation, tuple) and len(observation) == 2:
                sensor_data[factor_name] = observation
            else:
                sensor_data[factor_name] = (observation, 1.0)
        
        # Sample posterior
        self.sample_posterior(sensor_data)
        
        # Update state
        return self.update_state_from_posterior()
    
    def update_all(self, environment=None):
        """
        Update state using data from all sensors.
        
        Args:
            environment: Optional environment data
            
        Returns:
            True if update was successful
        """
        # Collect data from all sensors
        observations = self.collect_sensor_data(environment)
        
        if not observations:
            print("No observations from sensors")
            return False
            
        # Sample posterior
        posterior_samples = self.sample_posterior(observations)
        
        # Update state
        return self.update_state_from_posterior() 