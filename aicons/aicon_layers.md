# 🧠 Aicon Layers: A Taxonomy of Agent Intelligence

This document defines the cognitive architecture layers of an Aicon agent within the ADK (Agent Development Kit). These layers represent increasing levels of capability, from pure simulation to adaptive control and strategy. Developers and researchers can use these layers to design agents appropriate to their goals and complexity levels.

---

## 🔠 Layer 1: The Dreamer (Pure Divagation)

### Description:

The Dreamer is an isolated agent. It has no sensors and no ability to act. It operates solely on priors and internal simulation. It can imagine, hypothesize, and explore thought space using large language models and Monte Carlo sampling.

### Capabilities:

- Internal reasoning and simulation
- Generates scenarios, thoughts, or predictions
- Uses LLMs as experience priors

### Modules:

- Memory (LLM-based)
- Planning (Monte Carlo sampling)
- Utility model (internal, hypothetical)

### Use Cases:

- Creativity tools
- Hypothesis generation
- Fictional reasoning

---

## 👁️ Layer 2: The Observer (Passive Perception)

### Description:

The Observer can perceive the world via sensors but takes no actions. It processes external input and updates its beliefs about the world. It is a Bayesian learner, capable of forming probabilistic models.

### Capabilities:

- Receives sensor data
- Performs Bayesian inference
- Updates posterior beliefs based on likelihoods

### Modules:

- Perception (sensors + likelihood models)
- Memory (adaptive priors)
- Posterior sampler

### Use Cases:

- Market monitors
- Passive agents
- Forecasting

---

## 🤖 Layer 3: The Actor (Fixed Actor)

### Description:

The Actor has an action interface and executes decisions based on priors or rules, but does not adapt based on outcome. It may use simulation and planning, but lacks feedback to improve over time.

### Capabilities:

- Executes actions
- Uses planning to simulate consequences
- No sensory feedback or learning

### Modules:

- Planning
- Utility evaluation
- Action interface

### Use Cases:

- Scripted agents
- Tool-use prompt agents
- One-shot decision systems

---

## 🎯 Layer 4: The Learner (Perceptive Agent)

### Description:

The Learner is a full cognitive loop. It perceives, infers, acts, and adapts. It updates its priors based on prediction errors, using Bayesian inference. It chooses actions via expected utility.

### Capabilities:

- Perceives via sensors
- Updates beliefs with feedback
- Acts in the world
- Learns from outcome vs. expectation

### Modules:

- Sensors
- Perception (Bayesian inference)
- Planning (expected utility)
- Action interface
- Learning (posterior → prior update)

### Use Cases:

- Budget optimizers
- Adaptive decision agents
- Any self-improving intelligent system

---

## ⚙️ Layer 5: The Strategist (Meta-Agent)

### Description:

The Strategist is an advanced agent with meta-cognitive capacity. It can alter its own strategies, redefine goals, and manage other agents. It not only learns from outcomes, but learns how to learn.

### Capabilities:

- All abilities of the Learner
- Modifies utility functions
- Learns new planning strategies
- May coordinate sub-agents

### Modules:

- Meta-planner
- Policy learning
- Utility tuning
- Agent orchestration

### Use Cases:

- AI governance
- Self-improving agents
- High-level managers

---

## 📄 Summary Table

| Layer | Name       | Sensors | Actions | Learns | Meta-Learning | Use Case                        |
| ----- | ---------- | ------- | ------- | ------ | ------------- | ------------------------------- |
| 1     | Dreamer    | No      | No      | No     | No            | Simulation, ideation            |
| 2     | Observer   | Yes     | No      | Yes    | No            | Monitoring, world modeling      |
| 3     | Actor      | No      | Yes     | No     | No            | Scripted execution, automation  |
| 4     | Learner    | Yes     | Yes     | Yes    | No            | Adaptive, autonomous agents     |
| 5     | Strategist | Yes     | Yes     | Yes    | Yes           | Self-improvement, agent control |

---

Use these layers to guide the design and development of agents in your system. Start simple. Let your Aicons evolve.

Layer | Has Sensors? | Has Feedback? | What Triggers It?

1. Dreamer | ❌ No | ❌ No | Internal change (prior divergence, simulation noise)
2. Observer | ✅ Yes | ❌ No | Sensor input (event-based or time-based)
3. Actor | ❌ No | ❌ No | Manual or scheduled execution
4. Learner | ✅ Yes | ✅ Yes | Sensor events + feedback loop
5. Strategist | ✅ Yes | ✅ Yes + Meta | High-level reasoning, internal or external triggers

Thinking = default

Perceiving = optional input

Acting = conditional response

Learning = error-based correction

Strategizing = meta-adjustment
