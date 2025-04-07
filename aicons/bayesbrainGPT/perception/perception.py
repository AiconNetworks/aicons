import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from typing import Dict, List, Optional, Any
import time

# TFP shortcuts
tfd = tfp.distributions
tfb = tfp.bijectors

# Import the base Sensor class
from ..sensors import Sensor

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
        # Factor name mapping to support different naming conventions between sensors and state factors
        self.factor_name_mapping = {}  # Maps from sensor factor names to state factor names
        
    def register_sensor(self, name, sensor, factor_mapping=None):
        """
        Register a new sensor that can provide observations.
        
        Args:
            name: Name of the sensor
            sensor: Sensor object or function that returns sensor data
            factor_mapping: Optional dictionary mapping sensor factor names to state factor names
                            For example: {"base_conversion_rate": "conversion_rate"}
        """
        # Check if this is a Sensor object or a function
        if isinstance(sensor, Sensor):
            # Store Sensor object
            self.sensors[name] = sensor
            
            # Add factor mapping for this sensor
            if factor_mapping:
                for sensor_name, state_name in factor_mapping.items():
                    self.factor_name_mapping[sensor_name] = state_name
        else:
            # Backward compatibility for sensor functions
            self.sensors[name] = sensor
            
            # Add factor mapping for this sensor
            if factor_mapping:
                for sensor_name, state_name in factor_mapping.items():
                    self.factor_name_mapping[sensor_name] = state_name
    
    def add_factor_mapping(self, sensor_factor_name, state_factor_name):
        """
        Add a single factor name mapping.
        
        Args:
            sensor_factor_name: Factor name used by sensors
            state_factor_name: Corresponding factor name in the state
        """
        self.factor_name_mapping[sensor_factor_name] = state_factor_name
    
    def _map_factor_name(self, sensor_factor_name):
        """
        Map a sensor factor name to a state factor name using the mapping.
        
        Args:
            sensor_factor_name: Factor name from a sensor
            
        Returns:
            Corresponding state factor name, or the input name if no mapping exists
        """
        return self.factor_name_mapping.get(sensor_factor_name, sensor_factor_name)
    
    def collect_sensor_data(self, environment=None):
        """
        Collect data from all registered sensors.
        
        Args:
            environment: Optional environment data to pass to sensors
            
        Returns:
            Dictionary mapping factor names to (value, reliability) tuples
        """
        print("\n=== Collecting Sensor Data ===")
        all_sensor_data = {}
        
        for sensor_name, sensor in self.sensors.items():
            print(f"\nCollecting data from sensor: {sensor_name}")
            
            try:
                # Get data from sensor
                if isinstance(sensor, Sensor):
                    print("Using Sensor object's get_data method")
                    sensor_data = sensor.get_data(environment)
                else:
                    print("Using legacy sensor function")
                    data = sensor(environment) if environment else sensor()
                    
                    # Process data to ensure (value, reliability) tuples
                    sensor_data = {}
                    for factor_name, observation in data.items():
                        if isinstance(observation, tuple) and len(observation) == 2:
                            sensor_data[factor_name] = observation
                        else:
                            sensor_data[factor_name] = (observation, 1.0)
                
                print("Raw sensor data:")
                for factor_name, (value, reliability) in sensor_data.items():
                    print(f"  {factor_name}: value={value:.4f}, reliability={reliability:.2f}")
                
                # Map factor names to state factor names
                print("Mapping to state factors:")
                for sensor_factor_name, observation in sensor_data.items():
                    state_factor_name = self._map_factor_name(sensor_factor_name)
                    all_sensor_data[state_factor_name] = observation
                    print(f"  {sensor_factor_name} → {state_factor_name}: value={observation[0]:.4f}, reliability={observation[1]:.2f}")
                    
            except Exception as e:
                print(f"ERROR collecting data from sensor {sensor_name}: {str(e)}")
                continue
        
        if not all_sensor_data:
            print("WARNING: No sensor data collected")
        else:
            print("\nCollected data for factors:")
            for factor_name, (value, reliability) in all_sensor_data.items():
                print(f"  {factor_name}: value={value:.4f}, reliability={reliability:.2f}")
        
        return all_sensor_data
    
    def create_joint_prior(self):
        """
        Create a TensorFlow Probability joint distribution based on state factors.
        
        Returns:
            Joint distribution object from BayesianState
        """
        # Use the state's joint distribution directly
        return self.brain.state.create_joint_distribution()
    
    def create_likelihood_function(self, observations):
        """
        Create a function that computes the log likelihood of observations.
        
        Args:
            observations: Dictionary of (value, reliability) tuples
            
        Returns:
            Log likelihood function
        """
        state_factors = self.brain.state.get_state_factors()
        
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
                        rate = float(factor_value)
                        ll = tfd.Poisson(rate=rate).log_prob(obs_value)
                        total_ll += ll
            
            return total_ll
            
        return log_likelihood
    
    def sample_posterior(self, observations=None):
        """
        Sample from the posterior distribution given observations.
        If no observations are provided, samples from the prior distribution.
        Uses hierarchical sampling which can handle both dependent and independent priors.
        
        Args:
            observations: Optional dictionary mapping factor names to (value, reliability) tuples
            
        Returns:
            Dictionary of posterior samples for each factor
        """
        if not observations:
            print("\nNo observations available, using prior distributions")
            joint_dist = self.create_joint_prior()
            samples = joint_dist.sample(1000)
            
            self.posterior_samples = {}
            state_factors = self.brain.state.get_state_factors()
            
            if isinstance(samples, list):
                for i, name in enumerate(state_factors.keys()):
                    self.posterior_samples[name] = samples[i].numpy()
            else:
                self.posterior_samples = {k: v.numpy() for k, v in samples.items()}
            
            return self.posterior_samples
        
        # If observations are provided, use HMC for posterior sampling
        print("\nComputing posterior with observations using HMC")
        return self._sample_posterior_hierarchical(observations)
    
    def _sample_posterior_hierarchical(self, observations):
        """
        Sample from the posterior distribution using hierarchical sampling.
        
        Args:
            observations: Dictionary mapping factor names to (value, reliability) tuples
            
        Returns:
            Dictionary mapping factor names to posterior samples
        """
        # Convert observations to tensors
        observed_data = {}
        for name, (value, reliability) in observations.items():
            observed_data[name] = tf.convert_to_tensor(value, dtype=tf.float32)
        
        # Set up observed data with reliability
        def target_log_prob_fn(*args):
            # Combine prior and likelihood
            log_prob = 0.0
            
            # Add prior terms
            for i, (name, factor) in enumerate(self.brain.state.get_state_factors().items()):
                if factor["type"] == "continuous":
                    # Normal distribution
                    dist = tfd.Normal(
                        loc=factor["params"]["loc"],
                        scale=factor["params"]["scale"]
                    )
                    log_prob += dist.log_prob(args[i])
            
            # Add likelihood terms
            for i, (name, (value, reliability)) in enumerate(observations.items()):
                if name in self.brain.state.get_state_factors():
                    factor = self.brain.state.get_state_factors()[name]
                    if factor["type"] == "continuous":
                        # Normal distribution with reliability-weighted variance
                        dist = tfd.Normal(
                            loc=value,
                            scale=factor["params"]["scale"] / reliability
                        )
                        log_prob += dist.log_prob(args[i])
            
            return log_prob
        
        # Get HMC configuration from brain
        hmc_config = self.brain.hmc_config
        
        # Create HMC kernel
        hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=hmc_config['step_size'],
            num_leapfrog_steps=hmc_config['num_leapfrog_steps']
        )
        
        # Adaptive step size
        adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=hmc_kernel,
            num_adaptation_steps=int(hmc_config['num_burnin_steps'] * 0.8),
            target_accept_prob=hmc_config['target_accept_prob']
        )
        
        # Initial state
        initial_state = []
        for name, factor in self.brain.state.get_state_factors().items():
            if factor["type"] == "continuous":
                initial_state.append(tf.convert_to_tensor(factor["value"], dtype=tf.float32))
        
        # Run HMC
        @tf.function
        def run_chain():
            return tfp.mcmc.sample_chain(
                num_results=hmc_config['num_results'],
                num_burnin_steps=hmc_config['num_burnin_steps'],
                current_state=initial_state,
                kernel=adaptive_hmc,
                trace_fn=lambda _, pkr: pkr.inner_results.is_accepted
            )
        
        try:
            samples, is_accepted = run_chain()
            acceptance_rate = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float32))
            print(f"\nRunning HMC sampling...")
            print(f"HMC Acceptance Rate: {acceptance_rate:.2%}")
            
            # Process samples
            self.posterior_samples = {}  # Initialize posterior_samples
            for i, (name, factor) in enumerate(self.brain.state.get_state_factors().items()):
                if factor["type"] == "continuous":
                    samples_i = samples[i].numpy()
                    self.posterior_samples[name] = samples_i
            
            # If acceptance rate is low, increase uncertainty once for the entire run
            if acceptance_rate < 0.5:
                # Increase brain's uncertainty by 10% (0.1)
                new_uncertainty = min(1.0, self.brain.uncertainty + 0.1)
                # Let the brain track the uncertainty change
                self.brain.update_uncertainty(new_uncertainty, "low_acceptance")
                print(f"Low acceptance rate: Increased brain uncertainty to {new_uncertainty:.1%}")
            
            return self.posterior_samples
            
        except Exception as e:
            print(f"Error during HMC sampling: {str(e)}")
            # Even if sampling fails, increase uncertainty for all continuous factors
            for name, factor in self.brain.state.get_state_factors().items():
                if factor["type"] == "continuous":
                    factor_obj = self.brain.state.factors[name]
                    new_uncertainty = factor_obj._uncertainty * 1.2
                    factor_obj.update_uncertainty(new_uncertainty)
                    print(f"Sampling failed: Increased uncertainty for {name} to {new_uncertainty:.4f}")
            
            # Set posterior_samples to None to indicate sampling failed
            self.posterior_samples = None
            return None
    
    def update_state_from_posterior(self):
        """
        Update state factors based on posterior samples.
        
        Returns:
            True if update was successful
        """
        if not self.posterior_samples:
            print("No posterior samples available")
            return False
            
        state_factors = self.brain.state.get_state_factors().copy()
        updated_factors = state_factors.copy()
        
        print("\nUpdating state factors from posterior:")
        
        for name, samples in self.posterior_samples.items():
            if name not in state_factors:
                continue
                
            factor = state_factors[name]
            factor_type = factor["type"]
            
            if factor_type == "continuous":
                # Update continuous factor with posterior mean and std
                if hasattr(samples, 'numpy'):
                    samples_np = samples.numpy()
                else:
                    samples_np = samples  # Already a numpy array
                    
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
                    if hasattr(samples, 'numpy'):
                        samples_np = samples.numpy()
                    else:
                        samples_np = samples  # Already a numpy array
                        
                    categories = factor["categories"]
                    
                    # For categorical, find the most probable category
                    if isinstance(samples_np[0], (int, np.integer)):
                        # Index-based samples
                        counts = np.bincount(samples_np, minlength=len(categories))
                    elif isinstance(samples_np[0], str):
                        # String-based samples
                        counts = np.zeros(len(categories))
                        for s in samples_np:
                            if s in categories:
                                idx = categories.index(s)
                                counts[idx] += 1
                    else:
                        # One-hot encoded samples
                        counts = np.sum(samples_np, axis=0)
                        
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
                    if hasattr(samples, 'numpy'):
                        samples_np = samples.numpy()
                    else:
                        samples_np = samples  # Already a numpy array
                        
                    categories = factor["categories"]
                    
                    if isinstance(samples_np[0], (int, np.integer)):
                        # Index-based samples
                        counts = np.bincount(samples_np, minlength=len(categories))
                    else:
                        # One-hot encoded or any other type
                        counts = np.sum(samples_np, axis=0)
                        
                    new_probs = counts / np.sum(counts)
                    most_likely_idx = np.argmax(new_probs)
                    new_value = categories[most_likely_idx]
                    
                    old_value = factor["value"]
                    factor["value"] = new_value
                    factor["params"]["probs"] = new_probs.tolist()
                    
                    print(f"  {name}: {old_value} -> {new_value}")
                else:
                    # Poisson distribution
                    if hasattr(samples, 'numpy'):
                        samples_np = samples.numpy()
                    else:
                        samples_np = samples  # Already a numpy array
                        
                    new_rate = float(np.mean(samples_np))
                    new_value = int(round(new_rate))
                    
                    old_value = factor["value"]
                    factor["value"] = new_value
                    factor["params"]["rate"] = new_rate
                    
                    print(f"  {name}: {old_value} -> {new_value} (rate: {new_rate:.2f})")
        
        # Update brain state
        self.brain.state.set_state_factors(updated_factors)
        return True
    
    def update_from_sensor(self, sensor_name: str, environment=None, factor_mapping=None):
        """
        Update state using data from a specific sensor.
        
        Args:
            sensor_name: Name of the sensor
            environment: Optional environment data
            factor_mapping: Optional dictionary mapping sensor factor names to state factor names
                           For example: {"base_conversion_rate": "conversion_rate"}
            
        Returns:
            True if update was successful
        """
        print(f"\n=== BayesianPerception.update_from_sensor ===")
        print(f"Sensor name: {sensor_name}")
        
        if sensor_name not in self.sensors:
            print(f"ERROR: No sensor registered with name {sensor_name}")
            return False
        
        # Apply any one-time factor mappings for this update
        if factor_mapping:
            print("\nApplying factor mappings:")
            for sensor_name, state_name in factor_mapping.items():
                self.factor_name_mapping[sensor_name] = state_name
                print(f"  {sensor_name} → {state_name}")
        
        # Get sensor data
        print("\nGetting sensor data...")
        sensor = self.sensors[sensor_name]
        
        # Check if this is a Sensor object or a function
        if isinstance(sensor, Sensor):
            print("Using Sensor object's get_data method")
            try:
                sensor_data = sensor.get_data(environment)
                print(f"Raw sensor data returned: {sensor_data}")
            except Exception as e:
                print(f"ERROR: Failed to get data from sensor: {str(e)}")
                return False
        else:
            print("Using legacy sensor function")
            try:
                data = sensor(environment) if environment else sensor()
                print(f"Raw function data returned: {data}")
                
                # Process data to ensure (value, reliability) tuples
                sensor_data = {}
                for factor_name, observation in data.items():
                    if isinstance(observation, tuple) and len(observation) == 2:
                        sensor_data[factor_name] = observation
                    else:
                        sensor_data[factor_name] = (observation, 1.0)
            except Exception as e:
                print(f"ERROR: Failed to get data from sensor function: {str(e)}")
                return False
        
        if not sensor_data:
            print("ERROR: No sensor data received")
            return False
        
        print("\nProcessed sensor data:")
        for factor_name, (value, reliability) in sensor_data.items():
            print(f"  {factor_name}: value={value:.4f}, reliability={reliability:.2f}")
        
        # Map factor names and create observations dict
        print("\nMapping sensor factors to state factors:")
        mapped_sensor_data = {}
        for sensor_factor_name, observation in sensor_data.items():
            # Map the sensor factor name to state factor name
            state_factor_name = self._map_factor_name(sensor_factor_name)
            mapped_sensor_data[state_factor_name] = observation
            print(f"  {sensor_factor_name} → {state_factor_name}: value={observation[0]:.4f}, reliability={observation[1]:.2f}")
        
        if not mapped_sensor_data:
            print("ERROR: No mapped sensor data available")
            return False
        
        print("\nCalling sample_posterior with mapped observations...")
        # Sample posterior
        updated_samples = self.sample_posterior(mapped_sensor_data)
        
        if updated_samples:
            print("\nGot posterior samples:")
            for name, samples in updated_samples.items():
                if isinstance(samples, np.ndarray):
                    print(f"  {name}:")
                    print(f"    Mean: {np.mean(samples):.4f}")
                    print(f"    Std: {np.std(samples):.4f}")
        else:
            print("ERROR: No posterior samples generated")
            return False
        
        # Update state directly through brain.state
        print("\nUpdating brain state with posterior samples...")
        self.brain.state.set_posterior_samples(updated_samples)
        
        print("Perception update completed successfully")
        return True
    
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