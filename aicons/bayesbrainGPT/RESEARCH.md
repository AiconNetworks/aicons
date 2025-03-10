Not so naive, gpt-enhanced bayes.
Generative bayesian inference process using LLMs as prior experience memory for statistically self-controlled decision making.

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
In the bayesian brain hypothesis, the brain is thought to have higher cortical areas which encode abstract, high-level expectations like “its stormy”,; it has lower cortical areas which process sensory details, kiek local rain intensity. And it has feedback looks that carry precitns form higher levels to lower levels and send back errors signals when prec iotns dont match sensory input.

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

Perception

NUTS with continuous relaxation
NUTS can be applied without having to enumerate discrete sites.
Update the prior to a posterior.
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
OKay this is intereting the thign is that we need to define sensors in a way that can be used correclty so lets first define sensors. Sensors are the packests of information, coming in discrete, conitnuous or categorical way which have a level of trust. For example in human system, our eyes are trutsful because they are integrated to the brain. It is not third party information. But a story told by a stranger might not be atrustful source. Actually this is weird becasue we still are hearing it so it should be trsutful. sO INT THIS CAE THE WORDS ARTE TRUSTufl because we are hearing them but th einformation might not be. so basically sensors connected to our system are trsustuful but the infromation channel might not be. To make it easier for now we are just going to measure information trust because we are buidling the sensors so they should be trustuful connections.

So are we using like interfaces? So thta we define a sensor and then we could have many types of snesors. A sensor can get infromation. A sensor can send information a sensor has a reliability score. The get infroatino si from the sorufce. so FOR exmaple a wetahter sensro could eb an api. The get information is so that the bayes brain calls the sensor for informairon. thE SEND infromation is that the sensor is sedning infor to the brain. These are different because a brain might want to do something with infromarion that is "encountering" so it is seeking for informaiton or sensorial data migh tjust be flwoing to the brain.

Action space
Defining the action space.
In this step we set the actions available to the decision maker. The actions are determined by practical constraints and top-down conisderations.

Why top down?
Top-down considerations
ACtion space is deinged by oir practical abilities and consterinats, while our state might inlude that birds fly and they take less time transportating we should restrict oruac tion space to car and walk because we are constaied/capped by human capability. Unless we have a alken or a bijjke. This is very interesting because top down considerations are created based on external factors, like tools. These restrictions are imposed externally by design.

Here also would be a perfect spot for palcoing policy making.

---

It can be generated bottom-up or top-down.

Action space can be predefined. Like in connections where the game has a predefined decision space. Or in our case we can determine beforehand what the options are and then we analyze towards what is the better choice.

Matrix A.

Choose the action with the highest expected utility.

BayesianGPT Brain
We can use predefined actions we have
Actions space can be pre-generated, generated purely with LLMs, or generated with isolated system data.
Like in physics we need to define what is part of the system and what isnt. Information thermodynamics.

Utility Function
This quantifies teh goodness or value of taking action a given a particular state s. S can include factors like rain which is linear bayesian, temperature continuous ad traffic discrete.

IN bayesian brain hypothesis, te utility not typically and explicit symbolic function that is calculate das in fomral decision model. It is thought to emege form natural processes that assign value to outocems. For example brain regions like the orbitrfonrtal cortex and ventromedial pfrontal corrtex are involde in evaluating rewards and costs and their activity refelcats what we might think of as utility,

In our bayesbrainGPT system, the utility function could be inferred from the language model. But same as humans in work places, utility function could have constraints imposed by the environment or organizatoin. Again we could also give hte power for teh llm to detemiend the constaintes based on organizatoin ifromaiton. Or we can constraint it by defining a utility function. Fore example in marketing, the decision of which add to select is bsed purely on sales. so that is twha tth efunciotn include.

Decision Making
There is some data that

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
In the application of bayesian brain hypothesis, we define a prediction of continuous factors.

Confidence in Monte Carlo Simulations
Confidence can be a parameter fro uncertainty of the system. MIT 6. MOnte carlos siualtion; typically our best guess is what weve seen. But we really should not have ery cmcuh confidence in that guess. Coula have been just an acciednt.

Variance is the important factor here. As variance grows we need larger samples to have the same amount of confidence.

Now applied to aicon.

Standar deviations ponders quite highly outliers. So an error which happens to humans as well is to undewrsand when an outlier should be included in the data. TH eporblem with outliers is that they also could be errors happening duing data capture. So outlier detection and diagnosis should also be a critical part of our aicon system.

ASSUMPTIONS:
Confidence intervals, empirical rule, work if there is nor bias and also if the distribution of the errors in estimates is normal.

pmf and pdf.
Examples
Break
For a marketing company, the first prototype of decision making is in digital ads campaigns in Meta. We started with meta as they define clearly what objectives mean in a campaign so that the data they track is towards understanding how different campaigns are doing based on the objective. They also have a great api so that we can give the Aicon the tool to post the ads.

The use case:
Right now we are using our bayesian brain so that it is integrated into an already ongoing campaign. This already have ads and historic data of ad performance. This way we could measure the optimization technique of the bayesbrain ads skill.

In order to determine how the aicon is going to be run, we need to see what action space is shaped.

What do we want the Aicon to decide from?
Allocate budget daily for each ads campaign.
The optima job of the Aicon is to take time as infinite projected time. We could decide this because the decision of taking some step S in the near future t + 1; could eb less optimal that taking step S in later future t + 100; when t is measured in days. But for simplicity of analysis and using a pragmatic point of view instead of a philosophical, we define also a time frame. (Disclaimer, everything should be able to be generated entirely from Aicon; but more on this later)
References
https://www.youtube.com/watch?v=OgO1gpXSUzU&t=1988s

https://www.youtube.com/watch?v=8wVq5aGzSqY
