Not so naive, gpt-enhanced bayes.
Generative bayesian inference process using LLMs as experience in prior construction for statistically self-controlled decision making.

Literature review
BAYESIAN INFERENCE
Incoming sensory data:
Every sensor has its context.
Right now sensors are api calls. So if looker renders dta that is incoming sensory data. If we talk with it then that is another sensor which will have different contextualization. But that is more application than logic.
Perception
The Bayesian approach helps explain how we interpret ambiguous sensory data. For example, in visual perception, our brains can resolve unclear images by integrating context and prior experiences. This is why optical illusions or ambiguous figures can sometimes be seen in multiple ways.
Decision-Making:
In uncertain situations, the brain weighs potential outcomes based on previous experiences and current sensory inputs. This probabilistic decision-making process is a cornerstone of the Bayesian brain model.

Neural Mechanisms
Distributed Coding:
Instead of having dedicated “Bayesian neurons,” many researchers believe that probabilistic information is distributed across networks of neurons. Different regions of the brain might encode various aspects of uncertainty and prediction, contributing to an overall probabilistic computation.

Cortical Hierarchies:
The brain’s layered structure, especially in the cortex, seems well-suited to the Bayesian framework. Higher-level areas might integrate complex contextual information and expectations, while lower-level areas process raw sensory inputs, with ongoing feedback loops helping to update predictions.

L’Esprit Prédictif : Introduction à la Théorie du Cerveau Bayésien

Bayesian brain theory is based on predictive processing and suggest that brain encode beliefs to generate predicitns about sensory input. when sensory experiences differ from thes predictions the brain updates its beliefs accordingly.

Uses probabilistic beliefs to predict sensory
Updates these beleids based on prediction errors.
Uses bayesian math to adjust therse beliefs dynamically

The framework has computation models that somualte perception, decision making, learning and psychtiatric symptoms

perceptin is an unconsuons inference beased on hemrnan von helmholtz

The bayesian frameworks evlolecec form probability theory allowing thebrain to calculate the likelihood of sensory events and ajdjustebeliefs accordingly

General principles of pred itv processing

Beliefs probaiistic estimations of the world
hIerarchy, belieds are structured hierarchically in the brain
Predicion the brain constantly geenrates expectations
Prediciton eros, difference between predictions and sensory input
Belief updating adjusting beliefs based on errors
Precssion weightnig, the brain assings different levels of importance to differenc predictions and errso.

How it works:
Brain crates hypotheses about incoming sensory inforatio
Compares prediciotns to reality
Errors are sent up the hierarchy to refine future predictions

Likelihood of a perception is detmeied by past experiences (bayesbrainGPT, the memory or experience allocator is the large language model)

Action as prediction
brain predicts movements and updates predictions based on feedback
Acitve inference the brain acts to confirm its predictions
reflex aciotn, can be explained by prediciton errors triggereng immediate motor responses.

Emotion as predictor errors
ther brain predicrts internal bodily states
emomotion emegres when these predictions are VIOLATED
example fear occurs when encountering an unexpected danger, seeing a snake at hme. Triggering a sudden hear rate increase.

Intelligence and uncertainty: Implications of hierarchical predictive processing for the neuroscience of cognitive ability
This “predictive mind” thesis holds that the
whole of cortical information processing ultimately serves to (1) implement and dynamically adjust a hierarchy of predictions about the
organism’s expected sensory inputs and internal states, (2) estimate and
tune the precision of those predictions, and (3) continuously minimize
the resulting “errors” or violations of those predictions, which arise
through environmental changes and the organism’s actions upon its
environment

In particular, it is argued that in the process of minimizing prediction errors, the brain implicitly minimizes the organism’s likelihood of finding itself in phenotypically unexpected (and thus probabilistically maladaptive) states, thereby making environmental prediction the key way that organisms maintain homeostasis

Frequentist and Bayesian Statistics Explained
Bayes is subjective probability
rules: probability cannot be divorces form background information
background information can be represented on donctionl probability
This include subjective beliefs but also data, research and experiences.
X is all the background information someone has about an event.
Probability that an event A happes is condiitined on X. X is represented in beliefs that they have other priors embedded inside the abstract representation of that belief.
If two people have same X, then P(A | X ) = P(A | X´)
So we are using the logical interpretation of probability; bayesian probability = logical interpretation

For bayesian analysis we need two things:
Likelihood of data
beliefs in term of a prior
Assumptions
We think that rather to store values in memory we can compute priors and update priors as we go along. This way we can do continuous learning through bayesian interpretation of actions.

We assume through bayesian brain hypothesis that the bayesbrainGPT builds probabilistic models that can explicitly represent dependencies, covariances and interaciotns among variables. We handle precision
We handle covariance and interrelations of factors by using precision precision adn incertany and modeling it as a hierarchical model .This would allow higher level priors rto capture global dependences. Fore example a high-level model might encode typical weather patterns taht affect multiple lower-level variables like temperature , humidity and rain.

The brain might encode this uncertainty using population codes, where neural activity patterns reflect variance (uncertainty about a single variable) and covariance (how two variables relate). Since we are using llms as proxy of experience data storage similar to what the brain does, we are going to assume this encoding as well for now.
State representation
In the bayesian brain hypothesis, the brain is thought to have higher cortical areas which encode abstract, high-level expectations like “its stormy”,; it has lower cortical areas which process sensory details, like local rain intensity. And it has feedback looks that carry precitns form higher levels to lower levels and send back errors signals when prec iotns dont match sensory input.

In our system, state representation is also the place where the system makes assumptions of data. These are latent variables.

Latent Varibales
For example we can assume there is a true temperature Ttrue that we want to infer. We dont observe Ttrue directly; instead we get a measurement of Tobs that is noisy.
Then we get measurement model (likelohood)

Tobs​=Ttrue​+ϵT​,
where ϵT​∼N(0,σT2​)

Thenw e specify a prior for Tture, for instance if we expect the temperature typically hovers around 25 witha astandar deviation of 5 we could define

Ttrue​∼N(25,52).

This is the prior belief before seeing the current observation. This is also where we can leverage large language models experience for prior modeling.

So even though temperature is coninoulsy observed, we predefine it by assuming a generative process: thre is a latent Ttrue that geenrates teh observed Tobs via noise.
Specifying a prior for Tture; this ecnodes initial belief about temperature before new data.
Incorporating measurement uncertainty: Through the noise model ϵT​∼N(0,σT2​).

Factors

In a bayesian framework, state s captures all relevant aspects of the environment and/or internal conditions that influence decisions.

Philosophically, what we catch through our senses might trigger some action or decision making process. This is a bottom-up decision making situation.

They can be discrete variables, have a finite set of possible values.
Examples:
Weather: sunny, cloudy, rainy
Traffic: light, moderate, heavy

Continuous variables:
rain amount
temperature
speed

In the bayesian brain perspective the model already holds some internal representation of how these factors behave or co-vary.

These are hypotheses or explanatory variables that could account for variations in features F.
Defining factors
List factors
Expectations

You might expect that “Rain amount” is typically low in certain seasons, high in others.
You might expect that “Heavy traffic” is more common at 5 PM than at 2 PM.

Specify their internal structure:

In the example of rain, we could use a linear model for r.

r=θ 0 ​ +θ 1 ​ x 1 ​ +θ 2 ​ x 2 ​ +θ 3 ​ x 3 ​ +θ 4 ​ x 4 ​ +θ 5 ​ x 5 ​ +ϵ

In a bayesian framework we treat these parameters as Random Variables with their own prior distributions:

θi​∼P(θi​)

Continuous Variables in bayesbrainGPT
Variables that can take any value in a range.
Example: Rain amount (in mm), temperature (in °C)
Probability Density Function
Often modeled with gaussian distribution

For rain we have r = theta0 + theta1\*x1 ... + e, e ~ N(0, sigma^2)

Conditionally, r is normally distributed with a mean given by the linear combination and variance sigma squared.

If its like temperature we introduce a latent variable. So we assunem there is a true temperateuer T_true that we want to infer. We dont observe T_true directly instead we get a measurement of T_obs that is noisy. So we model teh observed temperature as T_obs = T_true + e_T, where E_T = N(0, sigma^2_T). This says that the observed alue is drawn from a nrmal distribution centerd at T true with some measuemrnt error sigma T.
Assign Priors
We assign priors to any parameters involved. Also we might have hyperprios. Hyperpriors are for hyperparamers. So we assign priors to miu, sigma . They control the mean and variance of the prior distribution of theta. They deteimene how spread out or concentrated the distribution fo lower level parameters is. THey also capture aborerder expectations or uncertainteies about the parameters that govern the behavior of the system.

we specify a prior for Ttrue which might be 25 with standardd eviation of 5 ; then that is ourprior belief before seeing the current observation.

Discrete Variables in bayesbrainGPT
Variables that take on a countable set of values.
Example: Number of cars passing a street per minute, or the state of a switch (on/off)

Probability Mass Function (PMF)
If we have a discrete variable d, we might assume a Poisson distribution.

Priors
If discrete variable is part of a larger hiercahcial model we migh specify hyperparemeters that control its distribution,

Monte Carlo simulation
We had an issue using NUTS and HMC in general because nuts is a continuous inference algorithm, so it requirese all latent variables to be coninupus and differentiable. In our model we are going to have variables like weather and these acan be discrete due to how the LLM creates them. Instead of using NUTS we could use sequential monte carlo, that can naturally handle discrete variabesles, Or we can use discrete friendly MCMC. The trade off is that can be less efficient or require more tuning.

Categorical Variables in bayesbrainGPT
Variables that represent categories or labels which may not have a natural numeric ordering.

Categorical Distribution (or Multinomial if multiple outcomes)
P(w = Clear) = 0.4, P(w = Cloudy) = 0.4, P(w = rainy) = 0.2

Priors and hierarchy
We might have a hierarchical model where the probs are drawn from a dirichlet prior, allowing the system to learn or adjust these probs over time. Example latent dirichlet allocation.
Prior Distribution P(s)
Mathematically we can use a function (or a set of functions) encoding what the brain believes before seeing new evidence.

For mixed discrete/continuous states it can be a joint distribution.
It has probability mass (pmf)
It has probability density (pdf)

For purely discrete
Categorical distribution

For purely continuous
We can have a gaussian.
Other continuous prior.

Identify the objective and context
Objective what is the end foal of the model?
Context which aspects of weather are most relevant?

So for example, programming campaign ads.

List potential factors
Catgeroical variables, continuous, discrete
Decide on the internal structure
We eed to check how factors might be interrelated.
Hierarchical Relationships
Conditional Dependencies
Contextual Variations
Assign priors and hyperpriors

For each variable or parameter in the conditional distribution, we need to choose an appropriate prior distribution. The hyperparameters of the priors can reflect export knowledge or historical data.

Retrieval / Memory
In a biological Bayesian brain, these priors are shaped by experience—i.e., memory of past patterns. Think of it as a long-term representation that you “retrieve” whenever you need to infer or predict current conditions.

In our BayesbrainGPT, they are stored in neural nets in a transformer like architecture.
Modeling Structure:
Decide whether to define a single joint distribution over all factors or to use a hierarchical model that factors the joint distribution into manageable conditional components.

Hierarchical model
In the bayesia brain hypothesis evidence suggest that the brain uses a hierarchical generative model.

Top down predictions
Higher corical areas generate predictions about what teh sensory input should be

Bottom0up error signals
Lower level sensory areas compare actual input with these prediciots sending back errors that help update th emodel

Learning and Adaptation:
This layered structure allows the brain to learn and update its beliefs at multiple levels, from raw sensory data to abstract concepts.

Problems when defining factors
Model Complexity
Each additional factor adds complexity. In a Bayesian setting, the joint distribution P(s)P(\mathbf{s})P(s) can blow up combinatorially if you have many discrete categories or multi‐dimensional continuous variables. Balancing completeness (lots of variables) vs. tractability is part of the design

Relevance to Decisions
Usually, you pick variables that meaningfully impact your utility function or your predictions. If wind speed doesn’t change your decisions, maybe you skip it.

Categorical problems
Soltuion: One hot encoding

Discrete problem
Sincew we are using NUTS. We cannot use direct discrete variables in our maode so what we do is relation of factors wechih allow for usng themas continueposu variables. We need to determine a amtreco tp understand if this is wokeign orn not.

Relationships
The brain uses top-down prdcitions to infer sensory outputs. If you know from your past experience that heavy storms typically bring hight rain and lower temperata your model can predict what sensory data should look like. When the actual data deviates the predciotn error informs an update of a humans beliefs.

Relatinsohp can hep explain away correlated bariabilty. If two variables tend to co-vary liek temp adn rain storme, mdoeling their relaiosps reduces redundancy and improves inference.

Learnign and adaptations, I father erleaitonshop between weather and traffic changes, system can update its model accordingly. This is important because mandy prediction models before covid were awful predicting during coivd and even after covid.

Example: Modeling Weather and Related Factors
Let the overall state be
s=(r,T,ω,d),\mathbf{s} = (r, T, \omega, d),s=(r,T,ω,d),
where:
rrr (Rain Amount): A continuous variable (in millimeters).

TTT (Temperature): A continuous variable (in °C).

ω\omegaω (Weather Type): A categorical variable with levels, for example, {Clear,Cloudy,Stormy}\{\text{Clear}, \text{Cloudy}, \text{Stormy}\}{Clear,Cloudy,Stormy}.

ddd (Number of Traffic Incidents): A discrete variable (non-negative integer) that might depend on the weather.

Hierarchical Factorization
Instead of specifying one flat joint distribution P(r,T,ω,d)P(r, T, \omega, d)P(r,T,ω,d), we can factorize it to reflect the dependencies we believe exist:
P(r,T,ω,d)=P(ω)  P(T∣ω)  P(r∣T,ω)  P(d∣ω).P(r, T, \omega, d) = P(\omega) \; P(T \mid \omega) \; P(r \mid T, \omega) \; P(d \mid \omega).P(r,T,ω,d)=P(ω)P(T∣ω)P(r∣T,ω)P(d∣ω).
Explanation of Each Term
P(ω)P(\omega)P(ω) – Weather Type (Categorical):
This is the top-level prior. For example, we might assume:
P(ω=Clear)=0.5,P(ω=Cloudy)=0.3,P(ω=Stormy)=0.2.P(\omega = \text{Clear}) = 0.5,\quad P(\omega = \text{Cloudy}) = 0.3,\quad P(\omega = \text{Stormy}) = 0.2.P(ω=Clear)=0.5,P(ω=Cloudy)=0.3,P(ω=Stormy)=0.2.
P(T∣ω)P(T \mid \omega)P(T∣ω) – Temperature Given Weather (Continuous):
Temperature depends on the weather type. We might model this with a Gaussian distribution whose parameters (mean and variance) depend on ω\omegaω. For instance:
T∣ω=Clear∼N(μClear,σClear2),T \mid \omega = \text{Clear} \sim \mathcal{N}(\mu*{\text{Clear}}, \sigma*{\text{Clear}}^2),T∣ω=Clear∼N(μClear​,σClear2​), T∣ω=Cloudy∼N(μCloudy,σCloudy2),T \mid \omega = \text{Cloudy} \sim \mathcal{N}(\mu*{\text{Cloudy}}, \sigma*{\text{Cloudy}}^2),T∣ω=Cloudy∼N(μCloudy​,σCloudy2​), T∣ω=Stormy∼N(μStormy,σStormy2).T \mid \omega = \text{Stormy} \sim \mathcal{N}(\mu*{\text{Stormy}}, \sigma*{\text{Stormy}}^2).T∣ω=Stormy∼N(μStormy​,σStormy2​).
The parameters μ\muμ and σ2\sigma^2σ2 here can themselves have hyperpriors reflecting our higher-level uncertainty.

P(r∣T,ω)P(r \mid T, \omega)P(r∣T,ω) – Rain Amount Given Temperature and Weather (Continuous):
The amount of rain might depend both on the weather type and the temperature. Again, we can model rrr as Gaussian:
r∣T,ω∼N(θ0(ω)+θ1(ω) T,  σr2(ω)),r \mid T, \omega \sim \mathcal{N}(\theta*0(\omega) + \theta_1(\omega)\, T,\; \sigma_r^2(\omega)),r∣T,ω∼N(θ0​(ω)+θ1​(ω)T,σr2​(ω)),
where the regression coefficients θ0(ω)\theta_0(\omega)θ0​(ω) and θ1(ω)\theta_1(\omega)θ1​(ω) and the variance σr2(ω)\sigma_r^2(\omega)σr2​(ω) are specific to the weather type ω\omegaω.
– Hyperpriors can be placed on these coefficients (e.g., θ1(ω)∼N(μθ1,τθ12)\theta_1(\omega) \sim \mathcal{N}(\mu*{\theta*1}, \tau*{\theta_1}^2)θ1​(ω)∼N(μθ1​​,τθ1​2​)).

P(d∣ω)P(d \mid \omega)P(d∣ω) – Number of Traffic Incidents Given Weather (Discrete):
Traffic incidents might be more frequent in bad weather. We can model ddd with a Poisson distribution:
d∣ω∼Poisson(λ(ω)),d \mid \omega \sim \text{Poisson}(\lambda(\omega)),d∣ω∼Poisson(λ(ω)),
where λ(ω)\lambda(\omega)λ(ω) is the expected number of incidents given the weather type. Hyperpriors could also be placed on λ(ω)\lambda(\omega)λ(ω) if desired.

Perception

NUTS with continuous relaxation
NUTS can be applied without having to enumerate discrete sites.
Update prior to a posterior.

Defining likelihood functions
It specifies how probable the sensor observations are given the satet factors. So if we have a sensor that measures temperature with known noise charactresiitfcs we would model that using a normal likelihood. With a mean equal to the true tmeprateuer and a viarbance that reflects senors noise.

We need to check also what the sensors data mean. How can we define how much data are we going to pull from the senor. Here is where we can add statistical quality control and alerts. We are going to monitor the sensor dpedning on previous data to see the variance of the data.

Here is where we might use bayes factor to determine how hypotheses turn into posterieors when we dont have sensorial data.

---

In bayesian brain hypothesis, perception is applying and updating priors rather than creating them from scratch.

When brain receives sensory input, it doesnt start from zero. It applies pre existing expectations to the new information.

Combining with New Evidence:
In a Bayesian framework, this process is mathematically similar to multiplying the prior (what you expect) with the likelihood (what the new data suggests) to produce a posterior probability (an updated interpretation of the scene). Essentially, your brain is weighing the new evidence against what it already knows.

Sources of information
This is typically done by assigning each source its own uncertainty or variance.
Likelihood Functions with Uncertainty:; For each source, you model the likelihood function. A direct sensory observation might have a narrow likelihood (low variance), meaning it's very precise. In contrast, someone’s verbal report might have a wider likelihood (higher variance), indicating less certainty about the exact amount or occurrence of rain.
In Bayesian statistics, the reliability of a measurement is often expressed as its precision, which is the inverse of the variance.

Priors
We have layers of priors, our assumption is that llms can also express this.
So we have nested priors. Like we have a prior belief because we have a prior belief because we have a prior belief. These are not hyperpriors but rather hierarchical model.
BayesianGPT Brain
We assume here that LLM will utilize the knowledge it has to create the priors. Then we can see later if we break down a prior into other priors.
We are going to use abstraction of priors, then we can break these priors into derived priors.
Retrieval comes naturally from setting the problem. We set the problem by using sensoor context and uncertainty of sensor.
Feature extraction
Raw sensor data is processed an transformed into more meaningful features.
Onece features are extracted they can be integrated with prior knowledge.

Forming the likelihood

BayesianGPT Brain
Large language model has already accumulated knowledge about the internet, proxy of human informaion, proxy to a world model.
We can assume that all information is soteed somehow in neural nets where we use linguistics to articulate them.  
Forming the likelihood here might be creating the likelihood of the real data we have.
For example in the ads scenario, the real data its what we are trying to predict.

Beliefs Update

We are always processing data but for a specific situation we have to filter out noise. Since we are multi-purpose, our system is based on priorities. In our case we are creating first unipurpose agents. This is because we can easily engineer sensorial channels with uncertainty and context.

Sensorial Data
In the Bayesian brain hypothesis, sensorial data continuously flow in, whenever there is a change in environment, the system gathers the latest sensorial data and uses it to update its beliefs about the state of the world. This update is Bayesian inference and it happens before a decision is made.

Here is where we can introduce control charts. Factors will have random noise but when we have an alert, at that time there has to be an update of beliefs or an environment analysis.

Sensors
We can use SQC for unexpected variation in sensorial data. This allwows the bayesian brain to analyze the situation

OKay this is intereting the thign is that we need to define sensors in a way that can be used correclty so lets first define sensors. Sensors are the packests of information, coming in discrete, conitnuous or categorical way which have a level of trust. For example in human system, our eyes are trutsful because they are integrated to the brain. It is not third party information. But a story told by a stranger might not be atrustful source. Actually this is weird becasue we still are hearing it so it should be trsutful. sO INT THIS CAE THE WORDS ARTE TRUSTufl because we are hearing them but th einformation might not be. so basically sensors connected to our system are trsustuful but the infromation channel might not be. To make it easier for now we are just going to measure information trust because we are buidling the sensors so they should be trustuful connections.

So are we using like interfaces? So thta we define a sensor and then we could have many types of snesors. A sensor can get infromation. A sensor can send information a sensor has a reliability score. The get infroatino si from the sorufce. so FOR exmaple a wetahter sensro could eb an api. The get information is so that the bayes brain calls the sensor for informairon. thE SEND infromation is that the sensor is sedning infor to the brain. These are different because a brain might want to do something with infromarion that is "encountering" so it is seeking for informaiton or sensorial data migh tjust be flwoing to the brain.

Sensor Examples
Meta marketing campaign data.

Action space
Defining the action space.
In this step we set the actions available to the decision maker. The actions are determined by practical constraints and top-down considerations.

Why top down?
Top-down considerations
ACtion space is defined by our practical abilities and constraints, while our state might include that birds fly and they take less time transporting we should restrict our action space to car and walk because we are constrained/capped by human capability. Unless we have a alken or a bike. This is very interesting because top down considerations are created based on external factors, like tools. These restrictions are imposed externally by design.

Here also would be a perfect spot for policy making.

---

It can be generated bottom-up or top-down.

Action space can be predefined. Like in connections where the game has a predefined decision space. Or in our case we can determine beforehand what the options are and then we analyze towards what is the better choice.

Matrix A.

Choose the action with the highest expected utility.

BayesianGPT Brain
We can use predefined actions we have
Actions space can be pre-generated, generated purely with LLMs, or generated with isolated system data.
Like in physics we need to define what is part of the system and what isnt. Information thermodynamics.

Constraints and Feasible Action Space:
Extrinsic goals may also limit the action space. In a company, you might not have the option to pursue strategies that are too risky or deviate from the brand’s identity. This means the set of actions AAA is defined not only by what is physically or technically possible but also by organizational or regulatory constraints.
Utility Function
This quantifies the goodness or value of taking action given a particular state s. S can include factors like rain which is linear bayesian, temperature continuous and traffic discrete.

Specifying Utility Function
The utility function typically combines several components such as:
Base benefit: How desirable an action is independent of weather (e.g. convenience of driving)
Rain penalty: The negative impact of rain on the action (discomfort if you don't have an umbrella)
Temperature or other factors: Additional adjustments based on temperature comfort, etc.

Example:
4.2 Mock Values for Each Component
Assume the following for our four actions:
Base Benefit B(a,τ)B(a, \tau)B(a,τ)
Suppose that for a departure at 6 PM:
For Car:

With τ=Light\tau = \text{Light}τ=Light: B(Car, _)=90B(\text{Car, _}) = 90B(Car, \*)=90 if you have an umbrella, and 959595 if you don’t (perhaps because carrying an umbrella is slightly inconvenient when driving).

For Walk:

B(Walk, _)=80B(\text{Walk, _}) = 80B(Walk, \*)=80 regardless of traffic (walking is less affected by traffic conditions).

(Here “∗\*∗” means that for now, we assume traffic is fixed by our state estimate.)
Rain Penalty α(a,r)\alpha(a, r)α(a,r)
Define a linear penalty in rrr (which is predicted from our Bayesian linear regression):
For Car, Umbrella: α(Car, Umbrella,r)=0.2×r\alpha(\text{Car, Umbrella}, r) = 0.2 \times rα(Car, Umbrella,r)=0.2×r.

For Car, No Umbrella: α(Car, No Umbrella,r)=0.5×r\alpha(\text{Car, No Umbrella}, r) = 0.5 \times rα(Car, No Umbrella,r)=0.5×r.

For Walk, Umbrella: α(Walk, Umbrella,r)=0.3×r\alpha(\text{Walk, Umbrella}, r) = 0.3 \times rα(Walk, Umbrella,r)=0.3×r.

For Walk, No Umbrella: α(Walk, No Umbrella,r)=0.8×r\alpha(\text{Walk, No Umbrella}, r) = 0.8 \times rα(Walk, No Umbrella,r)=0.8×r.

Temperature Penalty β(a,T)\beta(a, T)β(a,T)
For simplicity, assume that temperature mainly affects walking:
For Car: β(Car, _,T)=0\beta(\text{Car, _}, T) = 0β(Car, \*,T)=0 (climate control in cars makes temperature irrelevant).

For Walk:

If T=25T = 25T=25°C: β(Walk, _,T)=2\beta(\text{Walk, _}, T) = 2β(Walk, \*,T)=2 (slightly warm, but tolerable).

If T=15T = 15T=15°C: β(Walk, _,T)=1\beta(\text{Walk, _}, T) = 1β(Walk, \*,T)=1 (cooler, slightly more comfortable).

4.3 Putting It Together
For example, consider state s = (r = 15\,\text{mm},\; T = 25\,^\circ\text{C},\; \tau = \text{Light}):
For a1=a_1 =a1​= (Car, Umbrella):
U(a1,s)=90−(0.2×15)−0=90−3=87.U(a_1, s) = 90 - (0.2 \times 15) - 0 = 90 - 3 = 87.U(a1​,s)=90−(0.2×15)−0=90−3=87.
For a2=a_2 =a2​= (Car, No Umbrella):
U(a2,s)=95−(0.5×15)−0=95−7.5=87.5.U(a_2, s) = 95 - (0.5 \times 15) - 0 = 95 - 7.5 = 87.5.U(a2​,s)=95−(0.5×15)−0=95−7.5=87.5.
For a3=a_3 =a3​= (Walk, Umbrella):
U(a3,s)=80−(0.3×15)−2=80−4.5−2=73.5.U(a_3, s) = 80 - (0.3 \times 15) - 2 = 80 - 4.5 - 2 = 73.5.U(a3​,s)=80−(0.3×15)−2=80−4.5−2=73.5.
For a4=a_4 =a4​= (Walk, No Umbrella):
U(a4,s)=80−(0.8×15)−2=80−12−2=66.U(a_4, s) = 80 - (0.8 \times 15) - 2 = 80 - 12 - 2 = 66.U(a4​,s)=80−(0.8×15)−2=80−12−2=66.
(These are example values for one particular state.)

IN bayesian brain hypothesis, te utility not typically and explicit symbolic function that is calculate das in fomral decision model. It is thought to emege form natural processes that assign value to outocems. For example brain regions like the orbitofrontal cortex and ventromedial prefrontal cortex are involved in evaluating rewards and costs and their activity reflects what we might think of as utility,

In our bayesbrainGPT system, the utility function could be inferred from the language model. But same as humans in workplaces, utility function could have constraints imposed by the environment or organization. Again we could also give the power for the llm to determine the constraints based on organization information. Or we can constraint it by defining a utility function. Fore example in marketing, the decision of which add to select is based purely on sales. so that is what the function include.

Computing Expected Utility for each action
We combine our updated beliefs about the state with th eutilyt function to compute the expected utility (EU) for each action.

So expected utility for action a is given by:
EU(a)=∫s​P(s∣Data)U(a,s)ds,

or in discrete approacimation over our state space s:
EU(a)=s∑​P(s∣Data)U(a,s).

For example, when you're working for a company, your actions might be constrained by corporate strategy, resource limits, or regulatory requirements. In such a case, your utility function would reflect extrinsic goals, such as maximizing company profits, adhering to brand guidelines, or meeting sales targets. This differs from a scenario where you’re free to choose your actions without those external constraints.

Shaping the Utility Function:
The utility function can explicitly include terms for external objectives. For example, a marketing analyst might define:
U(Ad)=Expected Revenue−Cost+λ⋅Brand Impact,U(\text{Ad}) = \text{Expected Revenue} - \text{Cost} + \lambda \cdot \text{Brand Impact},U(Ad)=Expected Revenue−Cost+λ⋅Brand Impact,
where λ\lambdaλ adjusts the weight of long-term brand value—a goal imposed by the company.

Decision Making
Bayesian Brain: Decisions emerge from neural computations that weigh rewards, costs, and uncertainties, often implicitly via dopamine and other reward signals.

Both approaches share the same principle: actions (or choices) are evaluated based on their expected outcomes, and the one with the highest “utility” (whether computed neurally or mathematically) is selected.

Actions
Problems
In bayesian updating ucnetainty is quantified by the variance or its inverse precision
the brain wighs predictionn errors by their precision
if certain sensory signals are noisy or correlated the precision adn the covariance matrix plays a central role in updating beliefs.
Hierarchical models, rater tgan assume independence the model captures the covariance structure between these factors. The brain might learn these covariance patterns through experience effectively tuning iits priors.

Conditional independence
IF we have two variables independent on another event c and they dont intersect based on c. Then they are independent. In physical words, if the elements are not interrelated within oir deficniont of the syste, , then tehy are independent. So independence depends on what the system encompass.

Multi-variate gaussian distributions
The brain wirghs prediction werrrs by theirreciioson. If certain sensory signals are noisy or correlated the precision plays a centra role un pudating beliefs

Approximate Inference:
The brain likely uses approximate methods (such as sampling or variational inference) to compute posterior distributions. These methods naturally incorporate the full covariance structure of the underlying distributions. For example, if the brain is “sampling” from a multivariate Gaussian that represents its belief about a scene, those samples will reflect the covariances among features.

multi-variate gaussian  
More elements

Reactions
Reactions come from sensorial data. In order to have a fully automatic agent, we need to define reactions and have a way to compute them. Here we will introduce statistical quality control and control charts. These are going to determine alerts and make the Aicon to recompute priors and posteriors to determine actions.

Continuous Relaxation (Discrete Vars)
Treat each hypothesis as an independent Bayesian update

How Ad performance fluctuated due to random effects.

LLM data into pseudo data.
Usually, Bayes Factors compare real empirical data. But here, we treat the LLM’s confidence or rhetorical strength as if it were “pseudo-evidence,” letting us quantify how persuasive the LLM is about
𝐻
H.

Practical, first we let the llm “continues the completion” then we feed that completion or thought to the tool bayes_factor_generation.

Extrinsic goals and constraints:
Shaping the Utility Function:
The utility function can explicitly include terms for external objectives. For example, a marketing analyst might define:

𝑈
(Ad) = Expected Revenue − Cost + 𝜆 ⋅ Brand Impact,
U(Ad)=Expected Revenue−Cost+λ⋅Brand Impact,
where 𝜆
λ adjusts the weight of long-term brand value—a goal imposed by the company.

Constraints and Feasible Action Space:
Extrinsic goals may also limit the action space. In a company, you might not have the option to pursue strategies that are too risky or deviate from the brand’s identity. This means the set of actions
𝐴
A is defined not only by what is physically or technically possible but also by organizational or regulatory constraints.

Higher-Level Priors:
In the Bayesian brain view, our brain’s expectations or higher-level goals can act like “hyperpriors” that bias lower-level decision processes. For example, if your company prioritizes sustainability, that extrinsic goal might bias your neural decision processes (via learned reward signals) toward choices that are more eco-friendly, even if those choices might seem suboptimal from a narrow profit perspective.

Statistical Quality Control

Continuous factors
In the application of the Bayesian brain hypothesis, we define a prediction of continuous factors.

Confidence in Monte Carlo Simulations
Confidence can be a parameter fro uncertainty of the system. MIT 6. MOnte carlos siualtion; typically our best guess is what weve seen. But we really should not have ery cmcuh confidence in that guess. Coula have been just an acciednt.

Variance is the important factor here. As variance grows we need larger samples to have the same amount of confidence.

Now applied to aicon.

Standar deviations ponders quite highly outliers. So an error which happens to humans as well is to undewrsand when an outlier should be included in the data. TH eporblem with outliers is that they also could be errors happening duing data capture. So outlier detection and diagnosis should also be a critical part of our aicon system.

ASSUMPTIONS:
Confidence intervals, empirical rule, work if there is nor bias and also if the distribution of the errors in estimates is normal.

pmf and pdf.

Using Tools
For the Aicon to complete the decisions we can make the tools in different ways. We can make
Examples
Break
For a marketing company, the first prototype of decision making is in digital ads campaigns in Meta. We started with meta as they define clearly what objectives mean in a campaign so that the data they track is towards understanding how different campaigns are doing based on the objective. They also have a great api so that we can give the Aicon the tool to post the ads.

The use case:
Right now we are using our bayesian brain so that it is integrated into an already ongoing campaign. This already have ads and historic data of ad performance. This way we could measure the optimization technique of the bayesbrain ads skill.

In order to determine how the aicon is going to be run, we need to see what action space is shaped.

What do we want the Aicon to decide from?
Allocate budget daily for each ads campaign.
The optima job of the Aicon is to take time as infinite projected time. We could decide this because the decision of taking some step S in the near future t + 1; could eb less optimal that taking step S in later future t + 100; when t is measured in days. But for simplicity of analysis and using a pragmatic point of view instead of a philosophical, we define also a time frame. (Disclaimer, everything should be able to be generated entirely from Aicon; but more on this later)
Tests
Metrics to understand if this system works properly.

References
https://www.youtube.com/watch?v=OgO1gpXSUzU&t=1988s

https://www.youtube.com/watch?v=8wVq5aGzSqY
