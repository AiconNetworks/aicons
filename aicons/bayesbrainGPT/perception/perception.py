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
        # Check if this is a TFSensor object or a function
        from ..sensors.tf_sensors import TFSensor
        
        if isinstance(sensor, TFSensor):
            # Store TFSensor object
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
            Dictionary of observations from all sensors
        """
        observations = {}
        
        for name, sensor in self.sensors.items():
            # Check if this is a TFSensor object or a function
            from ..sensors.tf_sensors import TFSensor
            
            if isinstance(sensor, TFSensor):
                # Use TFSensor's get_data method
                data = sensor.get_data(environment)
            else:
                # Backward compatibility for sensor functions
                data = sensor(environment) if environment else sensor()
            
            # Add to observations with factor name mapping
            for sensor_factor_name, observation in data.items():
                # Map the sensor factor name to state factor name
                state_factor_name = self._map_factor_name(sensor_factor_name)
                
                if isinstance(observation, tuple) and len(observation) == 2:
                    # If observation includes value and reliability
                    observations[state_factor_name] = observation
                else:
                    # If only value is provided, assume reliability = 1.0
                    observations[state_factor_name] = (observation, 1.0)
                
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
                        rate = float(factor_value)
                        ll = tfd.Poisson(rate=rate).log_prob(obs_value)
                        total_ll += ll
            
            return total_ll
            
        return log_likelihood
    
    def sample_posterior(self, observations):
        """
        Sample from the posterior distribution given observations.
        
        Args:
            observations: Dictionary mapping factor names to (value, reliability) tuples
            
        Returns:
            Dictionary of posterior samples for each factor
        """
        print("Sampling posterior distribution...")
        state_factors = self.brain.get_state_factors()
        
        # Check if we can use hierarchical joint distribution
        if hasattr(self.brain, 'create_joint_distribution'):
            return self._sample_posterior_hierarchical(observations)
        
        # Create a joint prior distribution based on state factors
        joint_prior = self.create_joint_prior()
        
        if not joint_prior:
            print("No valid prior distributions found. Cannot sample posterior.")
            return {}
            
        # Set up observed data with reliability
        tf_observations = {}
        reliabilities = {}
        
        for name, (obs_value, reliability) in observations.items():
            if name not in state_factors:
                print(f"Warning: Factor {name} not in state. Skipping.")
                continue
                
            # Store observations and reliabilities
            tf_observations[name] = obs_value
            reliabilities[name] = reliability
        
        # Set up MCMC
        print(f"Setting up MCMC sampling for {len(tf_observations)} observations...")
        
        # Define target log probability function
        factor_names = list(tf_observations.keys())
        
        # Track which factors are discrete or categorical
        discrete_factors = {}
        for name in factor_names:
            factor = state_factors[name]
            factor_type = factor.get("type", "continuous")
            if factor_type in ["discrete", "categorical"]:
                discrete_factors[name] = factor_type
        
        def target_log_prob_fn(*args):
            # Convert positional arguments to dictionary
            state_dict = {}
            for i, name in enumerate(factor_names):
                state_dict[name] = args[i]
                
            # Apply continuous relaxation for all variables
            for name, factor_type in discrete_factors.items():
                if name in state_dict:
                    if factor_type == "discrete":
                        # For discrete variables, ensure positive values with softplus
                        # Add a small scale factor to make gradients smoother
                        state_dict[name] = tf.nn.softplus(state_dict[name]) * 0.9 + 0.1
            
            # Compute prior log probability with numerical safeguards
            try:
                prior_log_prob = joint_prior.log_prob(state_dict)
                # Prevent extreme values that could cause numerical issues
                prior_log_prob = tf.clip_by_value(prior_log_prob, -1000.0, 1000.0)
            except Exception as e:
                # If prior calculation fails, return a very low probability
                # This helps the sampler avoid problematic regions
                prior_log_prob = tf.constant(-1000.0, dtype=tf.float32)
            
            # Likelihood (observation model)
            likelihood_log_prob = tf.constant(0.0, dtype=tf.float32)
            
            for name, obs_value in tf_observations.items():
                if name not in state_dict:
                    continue
                    
                factor = state_factors[name]
                factor_type = factor.get("type", "continuous")
                reliability = reliabilities[name]
                
                # Use the continuously relaxed value
                value = state_dict[name]
                
                if factor_type == "continuous":
                    # Scale precision by reliability
                    variance = (1.0 / reliability) ** 2
                    # Add a small minimum variance for numerical stability
                    variance = tf.maximum(variance, 1e-6)
                    likelihood = tfd.Normal(loc=obs_value, scale=tf.sqrt(variance))
                    ll = likelihood.log_prob(value)
                    # Clip extreme values
                    ll = tf.clip_by_value(ll, -100.0, 100.0)
                    likelihood_log_prob += ll
                    
                elif factor_type == "categorical":
                    # Use categorical likelihood scaled by reliability
                    categories = factor.get("categories", [])
                    if not categories:
                        continue
                        
                    # Find index of observed value
                    try:
                        if isinstance(obs_value, str):
                            obs_idx = categories.index(obs_value)
                        else:
                            obs_idx = int(obs_value)
                    except ValueError:
                        print(f"Warning: Observed value {obs_value} not in categories for {name}")
                        continue
                        
                    # Create probability vector with reliability on observed category
                    # Ensure no zero probabilities for numerical stability
                    epsilon = 1e-6
                    base_prob = (1.0 - reliability - epsilon) / (len(categories) - 1)
                    probs = [max(base_prob, epsilon)] * len(categories)
                    probs[obs_idx] = max(reliability, epsilon)
                    
                    # Normalize to ensure probabilities sum to 1.0
                    probs = np.array(probs) / sum(probs)
                    
                    # For categorical variables, we need the probability mass function
                    # Use tf.cond instead of Python if for tensor operations
                    log_probs = tf.math.log(tf.constant(probs, dtype=tf.float32))
                    
                    # Define functions for both cases
                    def scalar_case():
                        # Create a soft one-hot encoding using softmax instead of sigmoid
                        # This ensures the distribution sums to 1.0
                        temp = 5.0  # Lower temperature for smoother gradients
                        indices = tf.range(len(categories), dtype=tf.float32)
                        logits = -temp * tf.square(indices - value)
                        soft_one_hot = tf.nn.softmax(logits)
                        return tf.reduce_sum(soft_one_hot * log_probs)
                    
                    def vector_case():
                        return tf.reduce_sum(value * log_probs)
                    
                    # Use tf.cond to choose between the two cases
                    rank = tf.rank(value)
                    log_prob = tf.cond(
                        tf.equal(rank, 0),  # condition
                        scalar_case,        # if true
                        vector_case         # if false
                    )
                    
                    # Scale by reliability to prevent overconfidence
                    log_prob = log_prob * tf.sqrt(reliability)
                    
                    # Clip to avoid extreme values
                    log_prob = tf.clip_by_value(log_prob, -100.0, 100.0)
                    likelihood_log_prob += log_prob
                    
                elif factor_type == "discrete":
                    # For discrete, use a continuous relaxation of Poisson likelihood
                    rate = tf.constant(float(obs_value), dtype=tf.float32)
                    # Ensure rate is positive and not too close to zero
                    rate = tf.maximum(rate, 0.1)
                    
                    # Use a softer penalty term
                    integer_penalty = tf.square(tf.sin(np.pi * value)) * 0.05
                    
                    # Compute Poisson log probability with continuous value
                    # We use a smooth approximation of factorial using lgamma
                    smooth_factorial = tf.math.lgamma(value + 1)
                    log_prob = value * tf.math.log(rate) - rate - smooth_factorial - integer_penalty
                    
                    # Clip log probability to avoid numerical issues
                    log_prob = tf.clip_by_value(log_prob, -100.0, 100.0)
                    
                    # Scale by reliability but don't let it become too small
                    reliability = tf.maximum(reliability, 0.1)
                    likelihood_log_prob += log_prob * reliability
            
            # Scale down the final log probability to make the posterior smoother
            total_log_prob = (prior_log_prob + likelihood_log_prob) * 0.1
            
            # Final numerical safeguard
            return tf.where(tf.math.is_finite(total_log_prob), total_log_prob, -1000.0)
            
        # Use HMC sampler
        num_results = 1000
        num_burnin_steps = 1000  # Increased burn-in steps
        
        # Initialize near the prior means but with some noise to encourage exploration
        initial_state = []
        for name in factor_names:
            factor = state_factors[name]
            factor_type = factor.get("type", "continuous")
            
            if factor_type == "continuous":
                # Add small noise to initial value to break symmetry
                factor_value = float(factor["value"])
                factor_std = float(factor.get("params", {}).get("scale", 0.01))
                noisy_value = factor_value + np.random.normal(0, factor_std * 0.1)
                initial_state.append(tf.constant(noisy_value, dtype=tf.float32))
            elif factor_type == "categorical":
                categories = factor.get("categories", [])
                if categories:
                    # For categorical, initialize to float index with small noise
                    try:
                        if isinstance(factor["value"], str):
                            idx = float(categories.index(factor["value"]))
                        else:
                            idx = float(factor["value"])
                        # Add small noise to break symmetry
                        noisy_idx = idx + np.random.uniform(-0.1, 0.1)
                        initial_state.append(tf.constant(noisy_idx, dtype=tf.float32))
                    except ValueError:
                        initial_state.append(tf.constant(0.0, dtype=tf.float32))
            elif factor_type == "discrete":
                # For discrete, initialize to float value with noise
                factor_value = float(factor["value"])
                noisy_value = factor_value + np.random.uniform(-0.2, 0.2)
                initial_state.append(tf.constant(noisy_value, dtype=tf.float32))
                
        # Use smaller step sizes for better acceptance rate
        step_size = 0.001  # Reduced step size
                
        try:
            print("Running MCMC sampling...")
            # Create HMC kernel with proper tensor handling
            hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=target_log_prob_fn,
                step_size=step_size,
                num_leapfrog_steps=2  # Reduced from 3 to 2
            )
            
            # Make sure we're handling adaptive step sizes correctly
            adaptive_hmc_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                inner_kernel=hmc_kernel,
                num_adaptation_steps=int(num_burnin_steps * 0.8),
                target_accept_prob=0.65,  # Target higher acceptance rate
                adaptation_rate=0.1       # Slower adaptation rate
            )
            
            # Fix: Use the @tf.function decorator correctly
            @tf.function(autograph=False)
            def run_mcmc():
                return tfp.mcmc.sample_chain(
                    num_results=num_results,
                    num_burnin_steps=num_burnin_steps,
                    current_state=initial_state,
                    kernel=adaptive_hmc_kernel,
                    trace_fn=lambda _, pkr: pkr.inner_results.is_accepted
                )
            
            # Run the sampler
            samples, is_accepted = run_mcmc()
            
            # Process samples
            acceptance_rate = tf.reduce_mean(tf.cast(is_accepted, tf.float32))
            print(f"Acceptance rate: {acceptance_rate:.2%}")
            
            # Analyze convergence
            print("Sample statistics:")
            for i, name in enumerate(factor_names):
                sample_values = samples[i].numpy()
                print(f"  {name}: mean={np.mean(sample_values):.4f}, std={np.std(sample_values):.4f}")
            
            # Store posterior samples
            self.posterior_samples = {}
            
            # Fix: Match samples to their corresponding factors
            for i, name in enumerate(factor_names):
                factor = state_factors[name]
                factor_type = factor.get("type", "continuous")
                sample_values = samples[i].numpy()
                
                if factor_type == "categorical":
                    # Convert continuous samples to categorical values
                    categories = factor.get("categories", [])
                    if categories:
                        # Round to nearest index
                        indices = np.round(sample_values).astype(int)
                        # Clip to valid range
                        indices = np.clip(indices, 0, len(categories) - 1)
                        # Convert indices to category values
                        self.posterior_samples[name] = np.array([categories[int(idx)] for idx in indices])
                elif factor_type == "discrete":
                    # Convert continuous samples to discrete values
                    # Round to nearest integer
                    self.posterior_samples[name] = np.round(sample_values).astype(int)
                else:
                    # Keep continuous values as is
                    self.posterior_samples[name] = sample_values
            
            return self.posterior_samples
            
        except Exception as e:
            print(f"Error in MCMC sampling: {e}")
            import traceback
            traceback.print_exc()
            return {}
            
    def _sample_posterior_hierarchical(self, observations):
        """
        Sample from the posterior using the hierarchical joint distribution.
        
        Args:
            observations: Dictionary mapping factor names to (value, reliability) tuples
            
        Returns:
            Dictionary of posterior samples for each factor
        """
        print("Using hierarchical joint distribution for posterior sampling...")
        
        # Get joint prior distribution
        joint_dist = self.brain.create_joint_distribution()
        
        # Set up observed data
        tf_observations = {}
        reliabilities = {}
        
        for name, (obs_value, reliability) in observations.items():
            # Convert observation values to appropriate tensor types
            if isinstance(obs_value, (int, np.int32, np.int64)):
                tf_observations[name] = tf.constant(obs_value, dtype=tf.int32)
            elif isinstance(obs_value, (float, np.float32, np.float64)):
                tf_observations[name] = tf.constant(obs_value, dtype=tf.float32)
            else:
                tf_observations[name] = obs_value
            reliabilities[name] = reliability
            
        # Define unnormalized posterior log probability function
        def target_log_prob_fn(sample_dict):
            # Prior log probability
            prior_log_prob = joint_dist.log_prob(sample_dict)
            
            # Likelihood (observation model)
            likelihood_log_prob = tf.constant(0.0, dtype=tf.float32)
            
            for name, obs_value in tf_observations.items():
                if name not in sample_dict:
                    continue
                    
                # Get the sampled value for this factor
                sampled_value = sample_dict[name]
                reliability = reliabilities[name]
                
                # Get the factor
                state_factors = self.brain.get_state_factors()
                if name not in state_factors:
                    continue
                    
                factor = state_factors[name]
                factor_type = factor.get("type", "continuous")
                
                if factor_type == "continuous":
                    # For continuous, use normal likelihood with reliability as precision
                    variance = (1.0 / reliability) ** 2
                    likelihood = tfd.Normal(loc=obs_value, scale=tf.sqrt(variance))
                    likelihood_log_prob += likelihood.log_prob(sampled_value)
                elif factor_type == "categorical":
                    # For categorical, create probability vector with reliability on observed value
                    categories = factor.get("categories", [])
                    if not categories:
                        continue
                        
                    # For categorical values, we need to compare strings, not indices
                    if isinstance(sampled_value, (int, np.integer)):
                        # Convert index to category value for comparison
                        category_value = categories[int(sampled_value)]
                    else:
                        category_value = sampled_value
                        
                    # Log probability is reliability if matching, (1-reliability)/(n-1) if not
                    if category_value == obs_value:
                        log_prob = tf.math.log(reliability)
                    else:
                        log_prob = tf.math.log((1.0 - reliability) / (len(categories) - 1))
                    likelihood_log_prob += log_prob
                elif factor_type == "discrete":
                    # For discrete, use Poisson likelihood
                    rate = float(obs_value)
                    likelihood = tfd.Poisson(rate=rate)
                    likelihood_log_prob += likelihood.log_prob(sampled_value) * reliability
            
            return prior_log_prob + likelihood_log_prob
            
        # Get bijectors for constrained variables
        bijectors = {}
        state_factors = self.brain.get_state_factors()
        
        for name, factor in state_factors.items():
            factor_type = factor.get("type", "continuous")
            constraints = factor.get("constraints", {})
            
            if factor_type == "continuous" and constraints:
                lower = constraints.get("lower")
                upper = constraints.get("upper")
                
                if lower is not None and upper is not None:
                    bijectors[name] = tfb.Sigmoid(low=float(lower), high=float(upper))
                elif lower is not None:
                    bijectors[name] = tfb.Softplus() + tfb.Shift(shift=float(lower))
                elif upper is not None:
                    bijectors[name] = -tfb.Softplus() + tfb.Shift(shift=float(upper))
                    
        # Initialize Hamiltonian Monte Carlo (HMC) kernel
        num_results = 1000
        num_burnin_steps = 500
        step_size = 0.01  # Use a single scalar step size
        num_leapfrog_steps = 10
        
        try:
            # Initial state based on prior samples
            initial_sample = joint_dist.sample()
            
            # Convert initial sample values to tensors if needed
            for name, value in initial_sample.items():
                if not tf.is_tensor(value):
                    if isinstance(value, (int, np.int32, np.int64)):
                        initial_sample[name] = tf.constant(value, dtype=tf.int32)
                    elif isinstance(value, (float, np.float32, np.float64)):
                        initial_sample[name] = tf.constant(value, dtype=tf.float32)
            
            # Create bijector dict with proper structure
            if bijectors:
                # Use empty dict if no bijectors needed
                transformed_bijectors = bijectors
            else:
                transformed_bijectors = {}
            
            # HMC transition kernel
            hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=target_log_prob_fn,
                step_size=step_size,
                num_leapfrog_steps=num_leapfrog_steps
            )
            
            # Add adaptation for step size
            adaptive_hmc_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                inner_kernel=hmc_kernel,
                num_adaptation_steps=int(num_burnin_steps * 0.8)
            )
            
            # Add bijector transformation if needed
            if transformed_bijectors:
                kernel = tfp.mcmc.TransformedTransitionKernel(
                    inner_kernel=adaptive_hmc_kernel,
                    bijector=transformed_bijectors
                )
            else:
                kernel = adaptive_hmc_kernel
            
            # Run the sampler
            @tf.function(autograph=False)
            def run_mcmc():
                return tfp.mcmc.sample_chain(
                    num_results=num_results,
                    num_burnin_steps=num_burnin_steps,
                    current_state=initial_sample,
                    kernel=kernel,
                    trace_fn=lambda _, pkr: pkr.inner_results.is_accepted if not transformed_bijectors else pkr.inner_results.inner_results.is_accepted
                )
            
            print("Running hierarchical MCMC sampling...")
            samples, is_accepted = run_mcmc()
            
            # Process and store posterior samples
            acceptance_rate = tf.reduce_mean(tf.cast(is_accepted, tf.float32))
            print(f"Acceptance rate: {acceptance_rate:.2%}")
            
            # Convert to numpy arrays and process categorical variables
            self.posterior_samples = {}
            for name in samples.keys():
                factor = state_factors.get(name, {})
                factor_type = factor.get("type", "continuous")
                
                # Convert samples to the right type
                if factor_type == "categorical":
                    # Convert indices to category values
                    categories = factor.get("categories", [])
                    if categories:
                        # Check if samples are indices or already category values
                        if tf.is_tensor(samples[name]) and samples[name].dtype in (tf.int32, tf.int64):
                            self.posterior_samples[name] = np.array([categories[int(idx)] 
                                                              for idx in samples[name].numpy()])
                        else:
                            self.posterior_samples[name] = samples[name].numpy()
                else:
                    self.posterior_samples[name] = samples[name].numpy()
            
            return self.posterior_samples
        except Exception as e:
            print(f"Error in hierarchical MCMC sampling: {e}")
            import traceback
            traceback.print_exc()
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
        self.brain.set_state_factors(updated_factors)
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
        if sensor_name not in self.sensors:
            print(f"No sensor registered with name {sensor_name}")
            return False
        
        # Apply any one-time factor mappings for this update
        if factor_mapping:
            for sensor_name, state_name in factor_mapping.items():
                self.factor_name_mapping[sensor_name] = state_name
                print(f"Applied temporary mapping: {sensor_name} → {state_name}")
            
        # Get sensor data
        sensor = self.sensors[sensor_name]
        
        # Check if this is a TFSensor object or a function
        from ..sensors.tf_sensors import TFSensor
        
        if isinstance(sensor, TFSensor):
            # Use TFSensor's get_data method
            sensor_data = sensor.get_data(environment)
        else:
            # Backward compatibility for sensor functions
            data = sensor(environment) if environment else sensor()
            
            # Process data to ensure (value, reliability) tuples
            sensor_data = {}
            for factor_name, observation in data.items():
                if isinstance(observation, tuple) and len(observation) == 2:
                    sensor_data[factor_name] = observation
                else:
                    sensor_data[factor_name] = (observation, 1.0)
        
        # Map factor names and create observations dict
        mapped_sensor_data = {}
        for sensor_factor_name, observation in sensor_data.items():
            # Map the sensor factor name to state factor name
            state_factor_name = self._map_factor_name(sensor_factor_name)
            mapped_sensor_data[state_factor_name] = observation
            
            print(f"Mapping observation: {sensor_factor_name} → {state_factor_name}")
            
        # Sample posterior
        self.sample_posterior(mapped_sensor_data)
        
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