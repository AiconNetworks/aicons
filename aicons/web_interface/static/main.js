// State management
let currentState = {};

// Initialize state
async function initializeState() {
    try {
        const response = await fetch('/api/state');
        currentState = await response.json();
    } catch (error) {
        console.error('Error initializing state:', error);
    }
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

async function sendMessage() {
    const messageInput = document.getElementById('message-input');
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
                current_state: currentState
            })
        });
        
        const data = await response.json();
        
        // Add AI response to chat
        addMessageToChat(data.response, 'ai');
        
        // Update state if it was modified
        if (data.new_state) {
            currentState = data.new_state;
        }
        
        // Clear input
        messageInput.value = '';
    } catch (error) {
        console.error('Error:', error);
        addMessageToChat('Sorry, there was an error processing your message.', 'ai');
    }
}

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
    const messageInput = document.getElementById('message-input');
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
}); 