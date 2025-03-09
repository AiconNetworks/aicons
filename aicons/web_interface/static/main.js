// State management
let currentState = {
    factors: {},
    sensors: {},
    decisions: {},
    isInitializing: false
};

// Start screen handling
async function startAicon() {
    const startScreen = document.getElementById('start-screen');
    const loadingScreen = document.getElementById('loading-screen');
    const aiconInterface = document.getElementById('aicon-interface');

    // Hide start screen, show loading
    startScreen.style.display = 'none';
    loadingScreen.style.display = 'flex';

    // Simulate initialization steps
    const steps = [
        'Initializing Bayesian network...',
        'Setting up priors...',
        'Calibrating sensors...',
        'Establishing decision space...',
        'Ready to begin'
    ];

    for (let step of steps) {
        await simulateLoadingStep(step);
    }

    // Hide loading, show main interface
    loadingScreen.style.display = 'none';
    aiconInterface.style.display = 'block';

    // Initialize the interface
    initializeInterface();
}

async function simulateLoadingStep(step) {
    const loadingStatus = document.getElementById('loading-status');
    loadingStatus.textContent = step;
    await new Promise(resolve => setTimeout(resolve, 1000));
}

// Interface initialization
async function initializeInterface() {
    await Promise.all([
        updateDecisionSpace(),
        updateSensors(),
        updatePriors(),
        updateQuestions()
    ]);
}

// Update display functions
async function updateDecisionSpace() {
    const decisionDisplay = document.getElementById('decision-display');
    try {
        const response = await fetch('/api/decisions');
        const decisions = await response.json();
        decisionDisplay.innerHTML = formatDecisions(decisions);
    } catch (error) {
        console.error('Error updating decision space:', error);
    }
}

async function updateSensors() {
    const sensorsList = document.getElementById('sensors-list');
    try {
        const response = await fetch('/api/sensors');
        const sensors = await response.json();
        sensorsList.innerHTML = formatSensors(sensors);
    } catch (error) {
        console.error('Error updating sensors:', error);
    }
}

async function updatePriors() {
    const priorsDisplay = document.getElementById('priors-display');
    try {
        const response = await fetch('/api/state');
        const state = await response.json();
        priorsDisplay.innerHTML = formatPriors(state.factors);
    } catch (error) {
        console.error('Error updating priors:', error);
    }
}

async function updateQuestions() {
    const questionsDisplay = document.getElementById('questions-display');
    try {
        const response = await fetch('/api/questions');
        const questions = await response.json();
        questionsDisplay.innerHTML = formatQuestions(questions);
    } catch (error) {
        console.error('Error updating questions:', error);
    }
}

// Formatting functions
function formatDecisions(decisions) {
    return `<div class="decisions-list">
        ${Object.entries(decisions || {}).map(([key, decision]) => `
            <div class="decision-item">
                <h3>${key}</h3>
                <p>${decision.description}</p>
                <div class="decision-value">${decision.value}</div>
            </div>
        `).join('')}
    </div>`;
}

function formatSensors(sensors) {
    return `<div class="sensors-list">
        ${Object.entries(sensors || {}).map(([key, sensor]) => `
            <div class="sensor-item">
                <h3>${key}</h3>
                <p>Status: ${sensor.status}</p>
                <p>Last Reading: ${sensor.lastReading}</p>
            </div>
        `).join('')}
    </div>`;
}

function formatPriors(factors) {
    return `<div class="priors-list">
        ${Object.entries(factors || {}).map(([name, factor]) => `
            <div class="prior-item">
                <h3>${name}</h3>
                <p>Type: ${factor.type}</p>
                <p>Value: ${factor.value}</p>
                ${factor.theta_prior ? `
                    <div class="prior-params">
                        <h4>Prior Parameters:</h4>
                        ${Object.entries(factor.theta_prior).map(([param, values]) => `
                            <p>${param}: μ=${values.mean}, σ²=${values.variance}</p>
                        `).join('')}
                    </div>
                ` : ''}
            </div>
        `).join('')}
    </div>`;
}

function formatQuestions(questions) {
    return `<div class="questions-list">
        ${(questions || []).map(question => `
            <div class="question-item">
                <p class="question-text">${question.text}</p>
                <p class="question-time">${new Date(question.timestamp).toLocaleString()}</p>
            </div>
        `).join('')}
    </div>`;
}

// Send message to AIcon
async function sendToAicon() {
    const messageInput = document.getElementById('user-input');
    const message = messageInput.value.trim();
    
    if (!message) return;
    
    try {
        // Add user message to chat
        addMessageToChat(message, 'user');
        
        // Send message to backend
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                current_state: currentState,
                is_initializing: currentState.isInitializing
            })
        });
        
        const data = await response.json();
        
        // Add AI response to chat
        addMessageToChat(data.response, 'ai');
        
        // Update state if it was modified
        if (data.new_state) {
            currentState = data.new_state;
            updateFactorsDisplay();
        }
        
        // Clear input
        messageInput.value = '';
    } catch (error) {
        console.error('Error:', error);
        addMessageToChat('Sorry, there was an error processing your message.', 'ai');
    }
}

// Initialize state
async function initializeState() {
    try {
        const response = await fetch('/api/state');
        currentState = await response.json();
        updateFactorsDisplay();
    } catch (error) {
        console.error('Error initializing state:', error);
    }
}

function updateFactorsDisplay() {
    const factorsDisplay = document.getElementById('factors-display');
    factorsDisplay.innerHTML = '';

    Object.entries(currentState.factors || {}).forEach(([name, factor]) => {
        const factorCard = document.createElement('div');
        factorCard.className = 'factor-card';
        
        let relationshipsHtml = '';
        if (factor.relationships && Object.keys(factor.relationships).length > 0) {
            relationshipsHtml = `
                <div class="relationships">
                    <strong>Relationships:</strong>
                    <ul>
                        ${Object.entries(factor.relationships).map(([key, value]) => 
                            `<li>${key}: ${JSON.stringify(value)}</li>`
                        ).join('')}
                    </ul>
                </div>
            `;
        }

        let priorInfo = '';
        if (factor.theta_prior) {
            priorInfo = `
                <div class="prior">
                    <strong>Priors:</strong>
                    <ul>
                        ${Object.entries(factor.theta_prior).map(([param, values]) => 
                            `<li>${param}: μ=${values.mean}, σ²=${values.variance}</li>`
                        ).join('')}
                    </ul>
                </div>
            `;
        }

        factorCard.innerHTML = `
            <h3>${name}</h3>
            <div class="type">Type: ${factor.type}</div>
            <div class="value">Value: ${factor.value || 'Not set'}</div>
            <div class="description">${factor.description || ''}</div>
            ${relationshipsHtml}
            ${priorInfo}
        `;
        
        factorsDisplay.appendChild(factorCard);
    });
}

// Fetch and update state
async function updateState() {
    const response = await fetch('/api/state');
    const state = await response.json();
    document.getElementById('state-display').textContent = JSON.stringify(state, null, 2);
}

// Submit new prompt
async function submitPrompt() {
    const promptInput = document.getElementById('prompt-input');
    const prompt = promptInput.value;
    
    try {
        const response = await fetch('/api/chain', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                prompt: prompt,
                response: "LLM response would go here",
                metadata: {
                    timestamp: new Date().toISOString()
                }
            })
        });
        
        await updateChainHistory();
        promptInput.value = '';
    } catch (error) {
        console.error('Error:', error);
    }
}

// Update chain history display
async function updateChainHistory() {
    const response = await fetch('/api/chain');
    const chain = await response.json();
    
    const historyDiv = document.getElementById('chain-history');
    historyDiv.innerHTML = chain.map(step => `
        <div class="chain-step">
            <h3>Step ${step.step_number}</h3>
            <p><strong>Prompt:</strong> ${step.prompt}</p>
            <p><strong>Response:</strong> ${step.response}</p>
            <p><small>Timestamp: ${step.metadata.timestamp}</small></p>
        </div>
    `).join('');
}

// Initial load
document.addEventListener('DOMContentLoaded', () => {
    initializeState();
    updateState();
    updateChainHistory();
    // Update state every 5 seconds
    setInterval(updateState, 5000);
});

function addMessageToChat(message, type) {
    const chatHistory = document.getElementById('chat-history');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    messageDiv.textContent = message;
    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('new-aicon-form').addEventListener('submit', createNewAicon);
    
    const messageInput = document.getElementById('user-input');
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendToAicon();
        }
    });
});

// Add at the beginning of the file
let currentStep = 1;
let totalSteps = 4;
let factorDefinitions = [];

function showCreateForm() {
    document.getElementById('create-modal').style.display = 'flex';
}

function hideModals() {
    document.getElementById('create-modal').style.display = 'none';
}

function createAIcon(event) {
    event.preventDefault();
    const name = document.getElementById('aicon-name').value;
    const type = document.getElementById('aicon-type').value;
    
    // Hide modal and start screen
    hideModals();
    document.getElementById('start-screen').style.display = 'none';
    
    // Show factor definition interface
    document.getElementById('factor-interface').style.display = 'block';
    currentState.isInitializing = true;
    
    // Add initial AI message
    addMessageToChat(
        `Hello! I'll help you define the factors for your ${type} AIcon named "${name}". ` +
        "Let's start with the first factor you want to define. For each factor, I need to know:\n\n" +
        "1. The factor name\n" +
        "2. Whether it's continuous or categorical\n" +
        "3. Any relationships with other factors\n" +
        "4. Prior beliefs about its behavior\n\n" +
        "What's the first factor you'd like to define?",
        'ai'
    );
}

function startAIcon() {
    // Hide factor interface
    document.getElementById('factor-interface').style.display = 'none';
    
    // Show main interface
    document.getElementById('main-interface').style.display = 'block';
    
    // Initialize the main interface
    initializeInterface();
}

function showFactorStep(step) {
    const form = document.getElementById('factor-init-form');
    currentStep = step;
    
    // Update progress indicator
    document.getElementById('current-step').textContent = `Step ${step}`;
    
    // Show/hide navigation buttons
    document.getElementById('prev-button').style.display = step > 1 ? 'block' : 'none';
    document.getElementById('next-button').style.display = step < totalSteps ? 'block' : 'none';
    document.getElementById('finish-button').style.display = step === totalSteps ? 'block' : 'none';

    // Different form content for each step
    switch(step) {
        case 1:
            form.innerHTML = `
                <h3>Define Basic Factors</h3>
                <p>Let's start with the basic continuous and categorical factors.</p>
                <div class="form-group">
                    <label>Factor Name:</label>
                    <input type="text" id="factor-name">
                </div>
                <div class="form-group">
                    <label>Factor Type:</label>
                    <select id="factor-type">
                        <option value="continuous">Continuous</option>
                        <option value="categorical">Categorical</option>
                    </select>
                </div>
                <button onclick="addFactor()">Add Factor</button>
            `;
            break;
        case 2:
            form.innerHTML = `
                <h3>Define Relationships</h3>
                <p>Now, let's define how these factors relate to each other.</p>
                <!-- Relationship definition form -->
            `;
            break;
        case 3:
            form.innerHTML = `
                <h3>Set Prior Distributions</h3>
                <p>Define the prior distributions for each factor.</p>
                <!-- Prior distribution form -->
            `;
            break;
        case 4:
            form.innerHTML = `
                <h3>Review and Confirm</h3>
                <p>Review all factor definitions before finalizing.</p>
                <!-- Summary view -->
            `;
            break;
    }
    
    updateFactorSummary();
}

function addFactor() {
    const name = document.getElementById('factor-name').value;
    const type = document.getElementById('factor-type').value;
    
    if (name) {
        factorDefinitions.push({ name, type });
        document.getElementById('factor-name').value = '';
        updateFactorSummary();
    }
}

function updateFactorSummary() {
    const summary = document.getElementById('factor-summary');
    summary.innerHTML = `
        <h3>Defined Factors</h3>
        ${factorDefinitions.map(factor => `
            <div class="factor-item">
                <strong>${factor.name}</strong>
                <span>${factor.type}</span>
            </div>
        `).join('')}
    `;
}

function previousStep() {
    if (currentStep > 1) {
        showFactorStep(currentStep - 1);
    }
}

function nextStep() {
    if (currentStep < totalSteps) {
        showFactorStep(currentStep + 1);
    }
}

async function finishInitialization() {
    // Show loading screen
    document.getElementById('factor-init-screen').style.display = 'none';
    document.getElementById('loading-screen').style.display = 'flex';
    
    // Simulate initialization steps
    const steps = [
        'Initializing Bayesian network...',
        'Setting up priors...',
        'Calibrating sensors...',
        'Establishing decision space...',
        'Ready to begin'
    ];

    for (let step of steps) {
        await simulateLoadingStep(step);
    }

    // Show main interface
    document.getElementById('loading-screen').style.display = 'none';
    document.getElementById('aicon-interface').style.display = 'block';
    
    // Initialize the interface
    initializeInterface();
}

async function startDashboard() {
    // Hide factor interface
    document.getElementById('factor-interface').style.display = 'none';
    
    // Show loading screen with cool messages
    document.getElementById('loading-screen').style.display = 'flex';
    
    const loadingSteps = [
        'Initializing Bayesian network...',
        'Configuring prior distributions...',
        'Calibrating sensor inputs...',
        'Establishing decision space...',
        'Setting up inference engine...',
        'Activating AIcon systems...'
    ];

    // Show each loading step with a delay
    for (const step of loadingSteps) {
        document.getElementById('loading-status').textContent = step;
        await new Promise(resolve => setTimeout(resolve, 800));
    }
    
    // Hide loading screen
    document.getElementById('loading-screen').style.display = 'none';
    
    // Show dashboard
    document.getElementById('dashboard').style.display = 'block';
    
    // Initialize dashboard with mock data for now
    initializeDashboard();
}

async function initializeDashboard() {
    // Mock data for initial dashboard state
    currentState = {
        factors: {
            'temperature': {
                type: 'continuous',
                value: 22.5,
                description: 'Ambient temperature',
                uncertainty: 0.5
            },
            'weather': {
                type: 'categorical',
                value: 'sunny',
                description: 'Current weather condition',
                possible_values: ['sunny', 'rainy', 'cloudy']
            }
        },
        sensors: {
            'temp_sensor': {
                status: 'active',
                lastReading: '22.5°C',
                reliability: 0.95
            },
            'weather_station': {
                status: 'active',
                lastReading: 'sunny',
                reliability: 0.98
            }
        },
        decisions: {
            'hvac_control': {
                description: 'HVAC system control decision',
                value: 'maintain',
                confidence: 0.89
            }
        }
    };

    // Update all dashboard panels
    updateDecisionSpace();
    updateSensors();
    updatePriors();
    updateQuestions();
} 