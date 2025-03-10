// State management
let currentState = {
    factors: {},
    sensors: {},
    decisions: {},
    isInitializing: false
};

let currentStep = 1;
let totalSteps = 4;
let factorDefinitions = [];
let selectedAiconType = '';
let factorCreationMethod = ''; // 'manual' or 'interactive'

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
    if (!factorCreationMethod === 'interactive') return;

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
                is_initializing: true,
                aicon_type: 'aiquon'
            })
        });
        
        const data = await response.json();
        
        // Add AI response to chat
        addMessageToChat(data.response, 'ai');
        
        // If factor was defined, add it to factorDefinitions
        if (data.new_factor) {
            factorDefinitions.push(data.new_factor);
            updateFactorSummary();
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

function showCreateForm() {
    document.getElementById('create-modal').style.display = 'flex';
    // Only show AiQuon type
    const typeSelect = document.getElementById('aicon-type');
    typeSelect.innerHTML = `
        <option value="">Select AIcon Type</option>
        <option value="aiquon">AiQuon</option>
    `;
}

function hideModals() {
    document.getElementById('create-modal').style.display = 'none';
}

function createAIcon(event) {
    event.preventDefault();
    const name = document.getElementById('aicon-name').value;
    selectedAiconType = document.getElementById('aicon-type').value;
    
    if (!name || !selectedAiconType) {
        alert('Please provide both name and type');
        return;
    }
    
    // Hide modal and start screen
    hideModals();
    document.getElementById('start-screen').style.display = 'none';
    
    // Show method selection screen with two big buttons
    const methodSelection = document.createElement('div');
    methodSelection.id = 'method-selection';
    methodSelection.className = 'method-selection-screen';
    methodSelection.innerHTML = `
        <div class="method-selection-container">
            <h2>How would you like to create your AiQuon?</h2>
            <div class="method-buttons">
                <div class="method-button" onclick="selectMethod('manual')">
                    <h3>Manual Creation</h3>
                    <p>Create your AiQuon step by step using forms</p>
                    <ul>
                        <li>Define factors one by one</li>
                        <li>Set up relationships</li>
                        <li>Configure prior distributions</li>
                    </ul>
                </div>
                <div class="method-button" onclick="selectMethod('interactive')">
                    <h3>Interactive Creation</h3>
                    <p>Create your AiQuon through natural conversation</p>
                    <ul>
                        <li>Chat with AI to define factors</li>
                        <li>Get guidance and suggestions</li>
                        <li>More flexible and intuitive</li>
                    </ul>
                </div>
            </div>
        </div>
    `;
    
    // Add the method selection screen to the body
    document.body.appendChild(methodSelection);
    
    // Add styles for the method selection screen
    const style = document.createElement('style');
    style.textContent = `
        .method-selection-screen {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: #f5f5f5;
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        
        .method-selection-container {
            max-width: 800px;
            padding: 2rem;
            text-align: center;
        }
        
        .method-selection-container h2 {
            margin-bottom: 2rem;
            color: #333;
            font-size: 2rem;
        }
        
        .method-buttons {
            display: flex;
            gap: 2rem;
            justify-content: center;
        }
        
        .method-button {
            background: white;
            border: 2px solid #007bff;
            border-radius: 10px;
            padding: 2rem;
            width: 300px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .method-button:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .method-button h3 {
            color: #007bff;
            margin-bottom: 1rem;
        }
        
        .method-button p {
            color: #666;
            margin-bottom: 1rem;
        }
        
        .method-button ul {
            text-align: left;
            color: #666;
            padding-left: 1.5rem;
        }
        
        .method-button li {
            margin: 0.5rem 0;
        }
    `;
    document.head.appendChild(style);
}

function selectMethod(method) {
    factorCreationMethod = method;
    
    // Remove the method selection screen
    const methodSelection = document.getElementById('method-selection');
    if (methodSelection) {
        methodSelection.remove();
    }
    
    if (method === 'manual') {
        // Show manual factor creation interface
        const factorInterface = document.createElement('div');
        factorInterface.id = 'factor-interface';
        factorInterface.className = 'factor-interface';
        factorInterface.innerHTML = `
            <div class="factor-interface-container">
                <h2>Create Your AiQuon Factors</h2>
                
                <!-- Left side: Factor Creation -->
                <div class="creation-panel">
                    <div class="factor-type-selector">
                        <h3>1. Choose Factor Type</h3>
                        <div class="type-buttons">
                            <button class="type-button" onclick="selectFactorType('continuous')">
                                <h4>Continuous Factor</h4>
                                <p>For numerical values (e.g., temperature, speed)</p>
                            </button>
                            <button class="type-button" onclick="selectFactorType('categorical')">
                                <h4>Categorical Factor</h4>
                                <p>For discrete categories (e.g., status, level)</p>
                            </button>
                        </div>
                    </div>

                    <div id="factor-form" class="factor-form">
                        <!-- Form will be populated based on selected type -->
                    </div>

                    <div id="relationship-form" class="relationship-form">
                        <h3>3. Add Relationships (Optional)</h3>
                        <!-- Will be enabled once there are 2+ factors -->
                    </div>
                </div>

                <!-- Right side: Factor Summary -->
                <div class="summary-panel">
                    <h3>Your Factors</h3>
                    <div id="factor-summary" class="factor-summary">
                        <!-- Will show live summary of created factors -->
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(factorInterface);
        
        // Add styles
        const style = document.createElement('style');
        style.textContent = `
            .factor-interface {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: #f5f5f5;
                display: flex;
                justify-content: center;
                align-items: flex-start;
                padding: 2rem;
                overflow-y: auto;
            }
            
            .factor-interface-container {
                max-width: 1200px;
                width: 100%;
                display: flex;
                gap: 2rem;
            }
            
            .creation-panel {
                flex: 2;
                background: white;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            
            .summary-panel {
                flex: 1;
                background: white;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                max-height: 80vh;
                overflow-y: auto;
            }
            
            .type-buttons {
                display: flex;
                gap: 1rem;
                margin-bottom: 2rem;
            }
            
            .type-button {
                flex: 1;
                padding: 1.5rem;
                border: 2px solid #007bff;
                border-radius: 8px;
                background: white;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            
            .type-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,123,255,0.2);
            }
            
            .type-button.selected {
                background: #007bff;
                color: white;
            }
            
            .type-button h4 {
                margin: 0 0 0.5rem 0;
                color: #007bff;
            }
            
            .type-button.selected h4 {
                color: white;
            }
            
            .type-button p {
                margin: 0;
                font-size: 0.9rem;
                color: #666;
            }
            
            .type-button.selected p {
                color: #fff;
            }
            
            .factor-form {
                margin-bottom: 2rem;
                padding-top: 1rem;
                border-top: 1px solid #eee;
            }
            
            .relationship-form {
                padding-top: 1rem;
                border-top: 1px solid #eee;
            }
            
            .factor-summary {
                display: flex;
                flex-direction: column;
                gap: 1rem;
            }
            
            .factor-item {
                padding: 1rem;
                border: 1px solid #ddd;
                border-radius: 6px;
                background: #f8f9fa;
            }
            
            .factor-item h4 {
                margin: 0 0 0.5rem 0;
                color: #007bff;
            }
            
            .form-group {
                margin-bottom: 1rem;
            }
            
            .form-group label {
                display: block;
                margin-bottom: 0.5rem;
                color: #333;
            }
            
            .form-group input,
            .form-group select {
                width: 100%;
                padding: 0.5rem;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            
            .relationship-item {
                display: flex;
                align-items: center;
                gap: 1rem;
                margin-bottom: 1rem;
            }
            
            .relationship-item select {
                flex: 1;
            }
        `;
        document.head.appendChild(style);
        
        // Initialize empty summary
        updateFactorSummary();
    } else {
        // Show interactive chat interface
        const chatInterface = document.createElement('div');
        chatInterface.id = 'chat-interface';
        chatInterface.className = 'chat-interface';
        chatInterface.innerHTML = `
            <div class="chat-container">
                <div id="chat-history" class="chat-history"></div>
                <div class="chat-input">
                    <input type="text" id="user-input" placeholder="Type your message...">
                    <button onclick="sendToAicon()">Send</button>
                </div>
            </div>
        `;
        document.body.appendChild(chatInterface);
        
        // Add styles for the chat interface
        const style = document.createElement('style');
        style.textContent = `
            .chat-interface {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: #f5f5f5;
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 1000;
            }
            
            .chat-container {
                max-width: 800px;
                width: 100%;
                height: 80vh;
                background: white;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                display: flex;
                flex-direction: column;
            }
            
            .chat-history {
                flex-grow: 1;
                overflow-y: auto;
                padding: 1rem;
            }
            
            .chat-input {
                display: flex;
                padding: 1rem;
                border-top: 1px solid #ddd;
            }
            
            .chat-input input {
                flex-grow: 1;
                padding: 0.5rem;
                border: 1px solid #ddd;
                border-radius: 4px;
                margin-right: 0.5rem;
            }
            
            .message {
                margin-bottom: 1rem;
                padding: 0.5rem 1rem;
                border-radius: 4px;
            }
            
            .user-message {
                background: #007bff;
                color: white;
                align-self: flex-end;
            }
            
            .ai-message {
                background: #f0f0f0;
                color: #333;
                align-self: flex-start;
            }
        `;
        document.head.appendChild(style);
        
        // Add initial AI message
        addMessageToChat(
            "Let's create your AiQuon together! I'll help you define the factors and their relationships. " +
            "What's the first factor you'd like to create?",
            'ai'
        );
    }
    
    currentState.isInitializing = true;
}

function selectFactorType(type) {
    // Update button styles
    document.querySelectorAll('.type-button').forEach(button => {
        button.classList.remove('selected');
        if (button.querySelector('h4').textContent.toLowerCase().includes(type)) {
            button.classList.add('selected');
        }
    });

    const form = document.getElementById('factor-form');
    if (type === 'continuous') {
        form.innerHTML = `
            <h3>2. Define Your Continuous Factor</h3>
            <div class="form-group">
                <label>Name:</label>
                <input type="text" id="factor-name" placeholder="e.g., rain_prediction">
            </div>
            <div class="form-group">
                <label>Description:</label>
                <input type="text" id="factor-description" placeholder="e.g., Rain prediction model based on humidity and pressure">
            </div>
            <div class="form-group">
                <label>Type:</label>
                <select id="factor-continuous-type">
                    <option value="continuous">Simple Continuous</option>
                    <option value="bayesian_linear">Bayesian Linear</option>
                </select>
            </div>
            <div id="bayesian-params" style="display: none;">
                <div class="form-group">
                    <label>Explanatory Variables (comma-separated name:value pairs):</label>
                    <div class="input-with-help">
                        <input type="text" id="explanatory-vars" placeholder="e.g., humidity:0.7, pressure:1013">
                        <div class="help-tooltip">
                            <span class="help-icon">?</span>
                            <span class="tooltip-text">
                                List variables that explain this factor, with their initial values.
                                Format: name:value, name:value
                            </span>
                        </div>
                    </div>
                </div>
                <div class="form-group">
                    <label>Prior Mean (comma-separated):</label>
                    <div class="input-with-help">
                        <input type="text" id="theta-prior-mean" placeholder="e.g., 0.0, 0.0">
                        <div class="help-tooltip">
                            <span class="help-icon">?</span>
                            <span class="tooltip-text">
                                Prior means for each explanatory variable.
                                Must match number of explanatory variables.
                            </span>
                        </div>
                    </div>
                </div>
                <div class="form-group">
                    <label>Prior Variance (comma-separated):</label>
                    <div class="input-with-help">
                        <input type="text" id="theta-prior-variance" placeholder="e.g., 1.0, 1.0">
                        <div class="help-tooltip">
                            <span class="help-icon">?</span>
                            <span class="tooltip-text">
                                Prior variances for each explanatory variable.
                                Must match number of explanatory variables.
                            </span>
                        </div>
                    </div>
                </div>
                <div class="form-group">
                    <label>Model Variance:</label>
                    <div class="input-with-help">
                        <input type="number" id="model-variance" step="0.1" value="1.0">
                        <div class="help-tooltip">
                            <span class="help-icon">?</span>
                            <span class="tooltip-text">
                                Overall variance of the model (e.g., 1.0)
                            </span>
                        </div>
                    </div>
                </div>
            </div>
            <div class="form-group">
                <label>Initial Value:</label>
                <div class="input-with-help">
                    <input type="number" id="factor-initial-value" step="0.1" value="0.0">
                    <div class="help-tooltip">
                        <span class="help-icon">?</span>
                        <span class="tooltip-text">
                            The current value for this factor
                        </span>
                    </div>
                </div>
            </div>
            <button onclick="addFactor('continuous')">Add Factor</button>
        `;

        // Add listener for type change
        document.getElementById('factor-continuous-type').addEventListener('change', function(e) {
            document.getElementById('bayesian-params').style.display = 
                e.target.value === 'bayesian_linear' ? 'block' : 'none';
        });
    } else {
        form.innerHTML = `
            <h3>2. Define Your Categorical Factor</h3>
            <div class="form-group">
                <label>Name:</label>
                <input type="text" id="factor-name" placeholder="e.g., weather">
            </div>
            <div class="form-group">
                <label>Description:</label>
                <input type="text" id="factor-description" placeholder="e.g., Current weather condition">
            </div>
            <div class="form-group">
                <label>Possible Values (comma-separated):</label>
                <div class="input-with-help">
                    <input type="text" id="factor-possible-values" placeholder="e.g., sunny, rainy, cloudy">
                    <div class="help-tooltip">
                        <span class="help-icon">?</span>
                        <span class="tooltip-text">
                            List all possible values this factor can take.
                            Separate them with commas.
                        </span>
                    </div>
                </div>
            </div>
            <div class="form-group">
                <label>Initial Value:</label>
                <div class="input-with-help">
                    <input type="text" id="factor-initial-value">
                    <div class="help-tooltip">
                        <span class="help-icon">?</span>
                        <span class="tooltip-text">
                            Must be one of the possible values listed above.
                        </span>
                    </div>
                </div>
                <small class="help-text" id="possible-values-help"></small>
            </div>
            <button onclick="addFactor('categorical')">Add Factor</button>
        `;
    }

    updateRelationshipForm();
}

function addFactor(type) {
    const name = document.getElementById('factor-name').value;
    const description = document.getElementById('factor-description').value;
    
    if (!name || !description) {
        alert('Please fill in all required fields');
        return;
    }
    
    if (type === 'continuous') {
        const continuousType = document.getElementById('factor-continuous-type').value;
        const value = parseFloat(document.getElementById('factor-initial-value').value) || 0.0;
        
        if (continuousType === 'bayesian_linear') {
            // Parse explanatory variables
            const expVarsStr = document.getElementById('explanatory-vars').value;
            const expVars = {};
            expVarsStr.split(',').forEach(pair => {
                const [name, value] = pair.trim().split(':');
                expVars[name] = parseFloat(value);
            });
            
            // Parse theta prior
            const priorMean = document.getElementById('theta-prior-mean').value
                .split(',').map(v => parseFloat(v.trim()));
            const priorVariance = document.getElementById('theta-prior-variance').value
                .split(',').map(v => parseFloat(v.trim()));
            
            // Create bayesian linear factor EXACTLY like in config.py
            factorDefinitions[name] = {
                type: 'bayesian_linear',
                value: value,
                description: description,
                explanatory_vars: expVars,
                theta_prior: {
                    mean: priorMean,
                    variance: priorVariance
                },
                variance: parseFloat(document.getElementById('model-variance').value),
                relationships: {
                    depends_on: [],
                    model: {}
                }
            };
        } else {
            // Create simple continuous factor
            factorDefinitions[name] = {
                type: 'continuous',
                value: value,
                description: description,
                relationships: {
                    depends_on: [],
                    model: {}
                }
            };
        }
    } else {
        const possibleValues = document.getElementById('factor-possible-values').value.split(',').map(v => v.trim());
        const initialValue = document.getElementById('factor-initial-value').value.trim();
        
        if (!possibleValues.includes(initialValue)) {
            alert('Initial value must be one of the possible values');
            return;
        }
        
        // Create categorical factor EXACTLY like in perception_example.py
        factorDefinitions[name] = {
            type: 'categorical',
            value: initialValue,
            description: description,
            possible_values: possibleValues,
            relationships: {
                depends_on: [],
                model: {}
            }
        };
    }
    
    // Clear form
    document.getElementById('factor-name').value = '';
    document.getElementById('factor-description').value = '';
    document.getElementById('factor-initial-value').value = '';
    if (type === 'continuous') {
        document.getElementById('factor-continuous-type').value = 'continuous';
        document.getElementById('bayesian-params').style.display = 'none';
        if (document.getElementById('explanatory-vars')) {
            document.getElementById('explanatory-vars').value = '';
            document.getElementById('theta-prior-mean').value = '';
            document.getElementById('theta-prior-variance').value = '';
            document.getElementById('model-variance').value = '1.0';
        }
    } else {
        document.getElementById('factor-possible-values').value = '';
    }
    
    updateFactorSummary();
    updateRelationshipForm();
}

function updateRelationshipForm() {
    const form = document.getElementById('relationship-form');
    if (factorDefinitions.length < 2) {
        form.innerHTML = `
            <p>Add at least two factors to define relationships</p>
        `;
        return;
    }

    form.innerHTML = `
        <div class="form-group">
            <label>Source Factor:</label>
            <select id="source-factor">
                ${Object.keys(factorDefinitions).map(name => `<option value="${name}">${name}</option>`).join('')}
            </select>
        </div>
        <div class="form-group">
            <label>Target Factor:</label>
            <select id="target-factor">
                ${Object.keys(factorDefinitions).map(name => `<option value="${name}">${name}</option>`).join('')}
            </select>
        </div>
        <div class="form-group">
            <label>Relationship Type:</label>
            <select id="relationship-type" onchange="updateRelationshipParams()">
                <option value="linear">Linear</option>
                <option value="categorical_effect">Categorical Effect</option>
            </select>
        </div>
        <div id="relationship-params"></div>
        <button onclick="addRelationship()">Add Relationship</button>
    `;
}

function updateRelationshipParams() {
    const type = document.getElementById('relationship-type').value;
    const sourceFactor = factorDefinitions[document.getElementById('source-factor').value];
    const paramsDiv = document.getElementById('relationship-params');
    
    if (type === 'linear') {
        paramsDiv.innerHTML = `
            <div class="form-group">
                <label>Base Value:</label>
                <input type="number" id="linear-base" step="0.1">
            </div>
            <div class="form-group">
                <label>Coefficient:</label>
                <input type="number" id="linear-coefficient" step="0.1">
            </div>
        `;
    } else if (type === 'categorical_effect') {
        if (sourceFactor.type !== 'categorical') {
            alert('Source factor must be categorical for categorical effects');
            return;
        }
        paramsDiv.innerHTML = `
            <div class="form-group">
                <label>Effects:</label>
                ${sourceFactor.possible_values.map(value => `
                    <div>
                        <label>${value}:</label>
                        <input type="number" id="effect-${value}" step="0.1">
                    </div>
                `).join('')}
            </div>
        `;
    }
}

function addRelationship() {
    const sourceName = document.getElementById('source-factor').value;
    const targetName = document.getElementById('target-factor').value;
    const type = document.getElementById('relationship-type').value;
    
    if (sourceName === targetName) {
        alert('Source and target factors must be different');
        return;
    }
    
    const targetFactor = factorDefinitions[targetName];
    const sourceFactor = factorDefinitions[sourceName];
    
    // Add source to depends_on if not already there
    if (!targetFactor.relationships.depends_on.includes(sourceName)) {
        targetFactor.relationships.depends_on.push(sourceName);
    }
    
    // Add relationship model EXACTLY like in perception_example.py
    if (type === 'linear') {
        const base = parseFloat(document.getElementById('linear-base').value);
        const coefficient = parseFloat(document.getElementById('linear-coefficient').value);
        
        targetFactor.relationships.model[sourceName] = {
            type: 'linear',
            base: base,  // Base value (e.g., 60.0 for base speed)
            coefficient: coefficient  // Effect coefficient (e.g., -5.0 for speed reduction)
        };
    } else if (type === 'categorical_effect') {
        if (sourceFactor.type !== 'categorical') {
            alert('Source factor must be categorical for categorical effects');
            return;
        }
        
        const effects = {};
        sourceFactor.possible_values.forEach(value => {
            effects[value] = parseFloat(document.getElementById(`effect-${value}`).value) || 0.0;
        });
        
        targetFactor.relationships.model[sourceName] = {
            type: 'categorical_effect',
            effects: effects  // e.g., {"sunny": 0.0, "rainy": 2.0, "cloudy": 1.0}
        };
    }
    
    updateFactorSummary();
}

function updateFactorSummary() {
    const summary = document.getElementById('factor-summary');
    if (!summary) return;
    
    summary.innerHTML = `
        <div class="factors-list">
            ${Object.entries(factorDefinitions).map(([name, factor]) => `
                <div class="factor-item">
                    <h4>${name}</h4>
                    <p><strong>Type:</strong> ${factor.type}</p>
                    <p><strong>Value:</strong> ${factor.value}</p>
                    <p><strong>Description:</strong> ${factor.description}</p>
                    ${factor.type === 'categorical' ? `
                        <p><strong>Possible Values:</strong> ${factor.possible_values.join(', ')}</p>
                    ` : ''}
                    ${factor.relationships.depends_on.length > 0 ? `
                        <div class="relationships">
                            <p><strong>Relationships:</strong></p>
                            <ul>
                                ${factor.relationships.depends_on.map(dep => {
                                    const rel = factor.relationships.model[dep];
                                    if (rel.type === 'linear') {
                                        return `<li>${dep}: Linear (base: ${rel.base}, coef: ${rel.coefficient})</li>`;
                                    } else {
                                        return `<li>${dep}: Categorical Effects (${Object.entries(rel.effects).map(([k,v]) => `${k}: ${v}`).join(', ')})</li>`;
                                    }
                                }).join('')}
                            </ul>
                        </div>
                    ` : ''}
                </div>
            `).join('')}
        </div>
    `;
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