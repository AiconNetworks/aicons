import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from typing import Dict, List, Optional, Any

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
            Dictionary of observations from all sensors
        """
        observations = {}
        
        for name, sensor in self.sensors.items():
            # Check if this is a Sensor object or a function
            if isinstance(sensor, Sensor):
                # Use Sensor's get_data method
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
        Uses hierarchical sampling which can handle both dependent and independent priors.
        
        Args:
            observations: Dictionary mapping factor names to (value, reliability) tuples
            
        Returns:
            Dictionary of posterior samples for each factor
        """
        print("Sampling posterior distribution...")
        
        # Always use hierarchical sampling
        return self._sample_posterior_hierarchical(observations)
    
    def _sample_posterior_hierarchical(self, observations):
        """
        Sample from the posterior using the hierarchical joint distribution.
        
        Args:
            observations: Dictionary mapping factor names to (value, reliability) tuples
            
        Returns:
            Dictionary of posterior samples for each factor
        """
        print("Using hierarchical joint distribution for posterior sampling...")
        state_factors = self.brain.get_state_factors()
        
        # Get joint prior distribution using the correct method
        joint_dist = self.create_joint_prior()
        
        # Set up observed data with reliability
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
        def target_log_prob_fn(*args):
            # Convert positional arguments to dictionary
            sample_dict = {name: value for name, value in zip(state_factors.keys(), args)}
            
            print("\nComputing target log probability...")
            print(f"Sample dict keys: {list(sample_dict.keys())}")
            print("\nSample dict values and types:")
            for name, value in sample_dict.items():
                print(f"{name}: {value} (type: {type(value)})")
                if tf.is_tensor(value):
                    print(f"  Tensor dtype: {value.dtype}")
                    print(f"  Tensor shape: {value.shape}")
            
            # Prior log probability
            try:
                print("\nComputing prior log probability...")
                print("Joint distribution type:", type(joint_dist))
                print("Joint distribution structure:", joint_dist)
                
                # Convert sample_dict to list in the correct order
                print("\nConverting sample_dict to list...")
                print("State factor keys:", list(state_factors.keys()))
                sample_list = []
                for name in state_factors.keys():
                    value = sample_dict[name]
                    print(f"\nProcessing {name}:")
                    print(f"  Value: {value}")
                    print(f"  Type: {type(value)}")
                    if tf.is_tensor(value):
                        print(f"  Tensor dtype: {value.dtype}")
                        print(f"  Tensor shape: {value.shape}")
                    sample_list.append(value)
                
                print("\nFinal sample list:")
                for i, value in enumerate(sample_list):
                    print(f"  {i}: {value} (type: {type(value)})")
                    if tf.is_tensor(value):
                        print(f"    Tensor dtype: {value.dtype}")
                        print(f"    Tensor shape: {value.shape}")
                
                print("\nComputing log probability...")
                prior_log_prob = joint_dist.log_prob(sample_list)
                print(f"Prior log probability: {prior_log_prob}")
            except Exception as e:
                print(f"\nError computing prior log probability: {e}")
                print("Error type:", type(e))
                print("Error args:", e.args)
                import traceback
                traceback.print_exc()
                # Return a very low probability with proper gradient information
                return tf.reduce_sum(tf.constant(-1e10, dtype=tf.float32) * tf.ones_like(args[0]))
            
            # Likelihood (observation model)
            likelihood_log_prob = tf.constant(0.0, dtype=tf.float32)
            
            for name, obs_value in tf_observations.items():
                if name not in sample_dict:
                    continue
                    
                print(f"\nProcessing factor: {name}")
                print(f"Observation value: {obs_value}")
                print(f"Observation type: {type(obs_value)}")
                if tf.is_tensor(obs_value):
                    print(f"  Observation tensor dtype: {obs_value.dtype}")
                    print(f"  Observation tensor shape: {obs_value.shape}")
                
                # Get the sampled value for this factor
                sampled_value = sample_dict[name]
                reliability = reliabilities[name]
                print(f"Sampled value: {sampled_value}")
                print(f"Sampled value type: {type(sampled_value)}")
                if tf.is_tensor(sampled_value):
                    print(f"  Sampled value tensor dtype: {sampled_value.dtype}")
                    print(f"  Sampled value tensor shape: {sampled_value.shape}")
                print(f"Reliability: {reliability}")
                
                # Get the factor
                factor = state_factors[name]
                factor_type = factor.get("type", "continuous")
                params = factor.get("params", {})
                print(f"Factor type: {factor_type}")
                print(f"Factor params: {params}")
                
                if factor_type == "continuous":
                    # For continuous, use normal likelihood with reliability as precision
                    variance = (1.0 / reliability) ** 2
                    # Convert observation to float32 to match scale dtype
                    obs_value_float = tf.cast(obs_value, tf.float32)
                    likelihood = tfd.Normal(loc=obs_value_float, scale=tf.sqrt(variance))
                    factor_ll = likelihood.log_prob(sampled_value)
                    likelihood_log_prob += factor_ll
                    print(f"Continuous factor log likelihood: {factor_ll}")
                    
                elif factor_type == "categorical":
                    # For categorical, use Gumbel-Softmax trick
                    categories = params.get("categories", [])
                    print(f"Categorical categories: {categories}")
                    
                    if not categories:
                        print("No categories found, skipping")
                        continue
                        
                    # Convert observation to one-hot encoding
                    if isinstance(obs_value, str):
                        if obs_value in categories:
                            obs_idx = categories.index(obs_value)
                            obs_one_hot = tf.one_hot(obs_idx, len(categories))
                            print(f"Observation index: {obs_idx}")
                            print(f"One-hot encoding: {obs_one_hot}")
                            print(f"One-hot encoding dtype: {obs_one_hot.dtype}")
                        else:
                            print(f"Observation {obs_value} not in categories, skipping")
                            continue
                    else:
                        print(f"Observation not a string, skipping")
                        continue
                        
                    # Use Gumbel-Softmax distribution for categorical variables
                    temperature = 0.1  # Controls the "softness" of the approximation
                    gumbel_dist = tfd.Gumbel(loc=0., scale=1.)
                    logits = sampled_value  # sampled_value is now the logits
                    print(f"Logits: {logits}")
                    print(f"Logits dtype: {logits.dtype}")
                    
                    # Compute Gumbel-Softmax probabilities
                    gumbel_noise = gumbel_dist.sample([len(categories)])
                    print(f"Gumbel noise: {gumbel_noise}")
                    print(f"Gumbel noise dtype: {gumbel_noise.dtype}")
                    softmax_probs = tf.nn.softmax((logits + gumbel_noise) / temperature)
                    print(f"Softmax probabilities: {softmax_probs}")
                    print(f"Softmax probabilities dtype: {softmax_probs.dtype}")
                    
                    # Compute log probability using cross-entropy
                    log_prob = tf.reduce_sum(obs_one_hot * tf.math.log(softmax_probs + 1e-10))
                    likelihood_log_prob += log_prob * reliability
                    print(f"Categorical factor log likelihood: {log_prob}")
                    
                elif factor_type == "discrete":
                    if "categories" in params:
                        # For categorical-like discrete, use same Gumbel-Softmax approach
                        categories = params["categories"]
                        print(f"Discrete categories: {categories}")
                        
                        if not categories:
                            print("No categories found, skipping")
                            continue
                            
                        # Convert observation to one-hot encoding
                        if isinstance(obs_value, str):
                            if obs_value in categories:
                                obs_idx = categories.index(obs_value)
                                obs_one_hot = tf.one_hot(obs_idx, len(categories))
                                print(f"Observation index: {obs_idx}")
                                print(f"One-hot encoding: {obs_one_hot}")
                                print(f"One-hot encoding dtype: {obs_one_hot.dtype}")
                            else:
                                print(f"Observation {obs_value} not in categories, skipping")
                                continue
                        else:
                            print(f"Observation not a string, skipping")
                            continue
                            
                        # Use Gumbel-Softmax distribution
                        temperature = 0.1
                        gumbel_dist = tfd.Gumbel(loc=0., scale=1.)
                        logits = sampled_value
                        print(f"Logits: {logits}")
                        print(f"Logits dtype: {logits.dtype}")
                        
                        # Compute Gumbel-Softmax probabilities
                        gumbel_noise = gumbel_dist.sample([len(categories)])
                        print(f"Gumbel noise: {gumbel_noise}")
                        print(f"Gumbel noise dtype: {gumbel_noise.dtype}")
                        softmax_probs = tf.nn.softmax((logits + gumbel_noise) / temperature)
                        print(f"Softmax probabilities: {softmax_probs}")
                        print(f"Softmax probabilities dtype: {softmax_probs.dtype}")
                        
                        # Compute log probability using cross-entropy
                        log_prob = tf.reduce_sum(obs_one_hot * tf.math.log(softmax_probs + 1e-10))
                        likelihood_log_prob += log_prob * reliability
                        print(f"Discrete categorical factor log likelihood: {log_prob}")
                    else:
                        # For Poisson distribution
                        rate = float(obs_value)
                        likelihood = tfd.Poisson(rate=rate)
                        factor_ll = likelihood.log_prob(sampled_value) * reliability
                        likelihood_log_prob += factor_ll
                        print(f"Poisson factor log likelihood: {factor_ll}")
            
            total_log_prob = prior_log_prob + likelihood_log_prob
            print(f"\nTotal log probability: {total_log_prob}")
            return total_log_prob
        
        # Get bijectors for constrained variables
        bijectors = {}
        print("\nSetting up bijectors...")
        for name, factor in state_factors.items():
            factor_type = factor.get("type", "continuous")
            params = factor.get("params", {})
            constraints = params.get("constraints", {})
            print(f"\nProcessing bijector for factor: {name}")
            print(f"Factor type: {factor_type}")
            
            if factor_type == "categorical" or (factor_type == "discrete" and "categories" in params):
                # For categorical variables, use unconstrained space for logits
                categories = params.get("categories", [])
                if categories:
                    bijectors[name] = tfb.Identity()  # No transformation needed for logits
                    print(f"Using Identity bijector for categorical variable with {len(categories)} categories")
            elif factor_type == "continuous" and constraints:
                # For continuous variables with constraints
                lower = constraints.get("lower")
                upper = constraints.get("upper")
                
                if lower is not None and upper is not None:
                    bijectors[name] = tfb.Sigmoid(low=float(lower), high=float(upper))
                    print(f"Using Sigmoid bijector for continuous variable with bounds [{lower}, {upper}]")
                elif lower is not None:
                    bijectors[name] = tfb.Softplus() + tfb.Shift(shift=float(lower))
                    print(f"Using Softplus+Shift bijector for continuous variable with lower bound {lower}")
                elif upper is not None:
                    bijectors[name] = -tfb.Softplus() + tfb.Shift(shift=float(upper))
                    print(f"Using -Softplus+Shift bijector for continuous variable with upper bound {upper}")
        
        # Initialize Hamiltonian Monte Carlo (HMC) kernel
        num_results = 1000
        num_burnin_steps = 500
        step_size = 0.01
        num_leapfrog_steps = 10
        
        print("\nInitializing MCMC sampling...")
        print(f"Number of results: {num_results}")
        print(f"Number of burnin steps: {num_burnin_steps}")
        print(f"Step size: {step_size}")
        print(f"Number of leapfrog steps: {num_leapfrog_steps}")
        
        try:
            # Initial state based on prior samples
            print("\nSampling initial state from prior...")
            initial_sample = joint_dist.sample()
            print(f"Initial sample type: {type(initial_sample)}")
            
            # Handle both list and dictionary returns from joint_dist.sample()
            if isinstance(initial_sample, list):
                # If it's a list, convert to dictionary using state factor names
                initial_sample = {name: value for name, value in zip(state_factors.keys(), initial_sample)}
                print("Converted list initial sample to dictionary")
            
            print("\nInitial sample values:")
            for name, value in initial_sample.items():
                print(f"{name}: {value} (type: {type(value)})")
            
            # Create bijector dict with proper structure
            transformed_bijectors = bijectors if bijectors else {}
            print(f"\nNumber of bijectors: {len(transformed_bijectors)}")
            
            # HMC transition kernel
            print("\nCreating HMC kernel...")
            hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=target_log_prob_fn,
                step_size=step_size,
                num_leapfrog_steps=num_leapfrog_steps
            )
            
            # Add adaptation for step size
            print("Adding step size adaptation...")
            adaptive_hmc_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                inner_kernel=hmc_kernel,
                num_adaptation_steps=int(num_burnin_steps * 0.8)
            )
            
            # Add bijector transformation if needed
            if transformed_bijectors:
                print("\n=== DETAILED STATE AND BIJECTOR DEBUG ===")
                print("\n1. STATE FACTORS STRUCTURE:")
                for name, factor in state_factors.items():
                    print(f"\nFactor: {name}")
                    print(f"  Type: {factor.get('type', 'unknown')}")
                    print(f"  Params: {factor.get('params', {})}")
                    print(f"  Value: {factor.get('value', 'No value')}")
                    if 'constraints' in factor.get('params', {}):
                        print(f"  Constraints: {factor['params']['constraints']}")
                
                print("\n2. TRANSFORMED BIJECTORS:")
                for name, bijector in transformed_bijectors.items():
                    print(f"- {name}: {type(bijector).__name__}")
                    print(f"  Bijector type: {type(bijector)}")
                    print(f"  Bijector parameters: {bijector.parameters}")
                
                # Create a list of bijectors in the same order as state factors
                bijector_list = []
                print("\n3. CREATING BIJECTOR LIST:")
                print("State factor order:")
                for i, name in enumerate(state_factors.keys()):
                    print(f"  {i}: {name}")
                    if name in transformed_bijectors:
                        bijector = transformed_bijectors[name]
                        print(f"    Using {type(bijector).__name__}")
                    else:
                        bijector = tfb.Identity()
                        print(f"    Using Identity (no transformation)")
                    bijector_list.append(bijector)
                
                print(f"\n4. COUNTS:")
                print(f"State Factor Count: {len(state_factors)}")
                print(f"Bijector List Length: {len(bijector_list)}")
                print(f"Transformed Bijectors Count: {len(transformed_bijectors)}")
                
                print("\n5. BLOCKWISE BIJECTOR SETUP:")
                print("Creating Blockwise bijector with list:")
                for i, bijector in enumerate(bijector_list):
                    print(f"  {i}: {type(bijector).__name__}")
                    print(f"    Parameters: {bijector.parameters}")
                
                # Create a Blockwise bijector from the list
                blockwise_bijector = tfb.Blockwise(bijector_list)
                print(f"\n6. BLOCKWISE BIJECTOR CREATED:")
                print(f"  Number of parts: {len(bijector_list)}")
                print(f"  Bijector type: {type(blockwise_bijector)}")
                print(f"  Bijector parameters: {blockwise_bijector.parameters}")
                
                print("\n7. CREATING TRANSFORMED TRANSITION KERNEL:")
                kernel = tfp.mcmc.TransformedTransitionKernel(
                    inner_kernel=adaptive_hmc_kernel,
                    bijector=blockwise_bijector
                )
                print("TransformedTransitionKernel created successfully")
                print(f"  Kernel type: {type(kernel)}")
                print(f"  Inner kernel type: {type(kernel.inner_kernel)}")
                print(f"  Bijector type: {type(kernel.bijector)}")
            else:
                kernel = adaptive_hmc_kernel
            
            # Run the sampler
            @tf.function(autograph=False)
            def run_mcmc():
                print("\nConverting initial state to tensors...")
                # Convert initial state to list of tensors for MCMC
                initial_state = []
                for name in state_factors.keys():
                    if name in initial_sample:
                        value = initial_sample[name]
                        factor = state_factors[name]
                        factor_type = factor.get("type", "continuous")
                        params = factor.get("params", {})
                        
                        print(f"\nProcessing initial state for factor: {name}")
                        print(f"Value: {value} (type: {type(value)})")
                        print(f"Factor type: {factor_type}")
                        
                        if factor_type == "categorical":
                            # For categorical variables, convert to index
                            categories = params.get("categories", [])
                            if isinstance(value, str):
                                initial_state.append(tf.constant(categories.index(value), dtype=tf.int32))
                                print(f"Converted string to index: {categories.index(value)}")
                            elif isinstance(value, (int, np.integer)):
                                initial_state.append(tf.constant(value, dtype=tf.int32))
                                print(f"Using integer value: {value}")
                            else:
                                initial_state.append(tf.constant(0, dtype=tf.int32))
                                print("Unexpected value type, using index 0")
                        elif factor_type == "discrete" and "categories" in params:
                            # For categorical-like discrete variables
                            categories = params["categories"]
                            if isinstance(value, str):
                                if value in categories:
                                    initial_state.append(tf.constant(categories.index(value), dtype=tf.int32))
                                    print(f"Converted string to index: {categories.index(value)}")
                                else:
                                    initial_state.append(tf.constant(0, dtype=tf.int32))
                                    print("Value not in categories, using index 0")
                            elif isinstance(value, (int, np.integer)):
                                initial_state.append(tf.constant(value, dtype=tf.int32))
                                print(f"Using integer value: {value}")
                            else:
                                initial_state.append(tf.constant(0, dtype=tf.int32))
                                print("Unexpected value type, using index 0")
                        else:
                            # For continuous and other discrete variables
                            if tf.is_tensor(value):
                                initial_state.append(value)
                                print("Using existing tensor")
                            else:
                                if isinstance(value, (int, np.int32, np.int64)):
                                    initial_state.append(tf.constant(value, dtype=tf.int32))
                                    print(f"Converting integer to tensor: {value}")
                                elif isinstance(value, (float, np.float32, np.float64)):
                                    initial_state.append(tf.constant(value, dtype=tf.float32))
                                    print(f"Converting float to tensor: {value}")
                                else:
                                    initial_state.append(tf.constant(value))
                                    print(f"Converting other type to tensor: {value}")
                    else:
                        # Default value if not in initial sample
                        initial_state.append(tf.constant(0.0, dtype=tf.float32))
                        print(f"Using default value for {name}")
                
                print("\nRunning MCMC chain...")
                return tfp.mcmc.sample_chain(
                    num_results=num_results,
                    num_burnin_steps=num_burnin_steps,
                    current_state=initial_state,
                    kernel=kernel,
                    trace_fn=lambda _, pkr: pkr.inner_results.is_accepted if not transformed_bijectors else pkr.inner_results.inner_results.is_accepted
                )
            
            print("\nStarting MCMC sampling...")
            samples, is_accepted = run_mcmc()
            
            # Process and store posterior samples
            acceptance_rate = tf.reduce_mean(tf.cast(is_accepted, tf.float32))
            print(f"\nAcceptance rate: {acceptance_rate:.2%}")
            
            # Convert samples to dictionary format
            print("\nProcessing posterior samples...")
            self.posterior_samples = {}
            
            # Convert samples to dictionary format
            for i, name in enumerate(state_factors.keys()):
                factor = state_factors.get(name, {})
                factor_type = factor.get("type", "continuous")
                params = factor.get("params", {})
                
                print(f"\nProcessing samples for factor: {name}")
                print(f"Factor type: {factor_type}")
                
                # Get samples for this factor
                factor_samples = samples[i]
                
                # Convert samples to the right type
                if factor_type == "categorical":
                    # Convert indices to category values
                    categories = params.get("categories", [])
                    if categories:
                        print(f"Categories: {categories}")
                        # Check if samples are indices or already category values
                        if tf.is_tensor(factor_samples) and factor_samples.dtype in (tf.int32, tf.int64):
                            print("Converting indices to category values")
                            self.posterior_samples[name] = np.array([categories[int(idx)] 
                                                              for idx in factor_samples.numpy()])
                        else:
                            print("Using samples as is")
                            self.posterior_samples[name] = factor_samples.numpy()
                else:
                    print("Converting samples to numpy array")
                    self.posterior_samples[name] = factor_samples.numpy()
            
            return self.posterior_samples
            
        except Exception as e:
            print(f"\nError in hierarchical MCMC sampling: {e}")
            import traceback
            traceback.print_exc()
            self.posterior_samples = {}  # Initialize empty posterior samples
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
        
        # Check if this is a Sensor object or a function
        if isinstance(sensor, Sensor):
            # Use Sensor's get_data method
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