"use strict";

/**
 * ZeroAIcon Chat Interface
 * 
 * This script handles the chat interface for interacting with ZeroAIcon,
 * including streaming responses and displaying the thinking process.
 */

(function() {
  // DOM Elements
  let messagesContainer;
  let messageInput;
  let sendButton;
  let typingIndicator;
  let contextWindowBtn;
  let contextWindowPanel;
  let configBtn;
  let configPanel;
  let closeBtns;
  let tabBtns;
  let tabPanes;
  
  // Form Elements
  let sensorForm;
  let stateFactorForm;
  let actionSpaceForm;
  let utilityForm;
  
  // State for dynamically showing/hiding form fields
  let factorTypeSelect;
  let spaceTypeSelect;
  let utilityTypeSelect;
  let sensorTypeSelect;
  
  // Current Configuration
  let currentSensors = [];
  let currentFactors = [];
  let currentActionSpace = null;
  let currentUtility = null;
  
  // Initialize Markdown parser
  const md = window.markdownit({
    html: false,
    linkify: true,
    typographer: true
  });

  // State
  let isProcessing = false;
  let lastTokenUsage = null;

  // Add these variables at the top with other state variables
  let currentAicon = "default";
  let aiconSelect;
  let createMarketingBtn;

  /**
   * Initialize the app when DOM is loaded
   */
  window.addEventListener("DOMContentLoaded", init);

  /**
   * Initialize the application and set up event listeners
   */
  function init() {
    // Get DOM elements
    messagesContainer = id("messages");
    messageInput = id("message-input");
    sendButton = id("send-button");
    typingIndicator = id("typing-indicator");
    contextWindowBtn = id("context-window-btn");
    contextWindowPanel = id("context-window-panel");
    configBtn = id("config-btn");
    configPanel = id("config-panel");
    createMarketingBtn = id("create-marketing-btn");
    
    // Get all close buttons
    closeBtns = document.querySelectorAll(".close-btn");
    
    // Get tabs elements
    tabBtns = document.querySelectorAll(".tab-btn");
    tabPanes = document.querySelectorAll(".tab-pane");
    
    // Get form elements
    sensorForm = id("sensor-form");
    stateFactorForm = id("state-factor-form");
    actionSpaceForm = id("action-space-form");
    utilityForm = id("utility-form");
    
    // Get dynamic form selects
    factorTypeSelect = id("factor-type");
    spaceTypeSelect = id("space-type");
    utilityTypeSelect = id("utility-type");
    sensorTypeSelect = id("sensor-type");
    
    // Get AIcon selector
    aiconSelect = id("aicon-select");
    
    // Set up event listeners
    sendButton.addEventListener("click", handleSendMessage);
    messageInput.addEventListener("keypress", handleKeyPress);
    contextWindowBtn.addEventListener("click", openContextWindow);
    configBtn.addEventListener("click", openConfigPanel);
    createMarketingBtn.addEventListener("click", handleCreateMarketing);
    
    // Setup close buttons
    closeBtns.forEach(btn => {
      btn.addEventListener("click", function() {
        const modal = findParentWithClass(btn, "modal");
        if (modal) {
          modal.style.display = "none";
        }
      });
    });
    
    // Setup tab buttons
    tabBtns.forEach(btn => {
      btn.addEventListener("click", function() {
        const tabName = btn.getAttribute("data-tab");
        switchTab(tabName);
      });
    });
    
    // Setup forms
    if (sensorForm) sensorForm.addEventListener("submit", handleSensorForm);
    if (stateFactorForm) stateFactorForm.addEventListener("submit", handleStateFactorForm);
    if (actionSpaceForm) actionSpaceForm.addEventListener("submit", handleActionSpaceForm);
    if (utilityForm) utilityForm.addEventListener("submit", handleUtilityForm);
    
    // Setup dynamic form fields
    if (factorTypeSelect) factorTypeSelect.addEventListener("change", toggleFactorFields);
    if (spaceTypeSelect) spaceTypeSelect.addEventListener("change", toggleSpaceFields);
    if (utilityTypeSelect) utilityTypeSelect.addEventListener("change", toggleUtilityFields);
    if (sensorTypeSelect) sensorTypeSelect.addEventListener("change", toggleSensorFields);
    
    // Setup window click for modals
    window.addEventListener("click", function(event) {
      if (event.target === contextWindowPanel) {
        contextWindowPanel.style.display = "none";
      }
      if (event.target === configPanel) {
        configPanel.style.display = "none";
      }
    });
    
    // Set up delegation for thinking toggles
    messagesContainer.addEventListener("click", handleThinkingToggleClick);
    
    // Load initial chat history
    loadChatHistory();
    
    // Load initial token usage
    loadTokenUsage();
    
    // Load initial configuration
    loadConfiguration();
    
    // Load AIcons
    loadAicons();
    
    // Add event listener for AIcon selection
    aiconSelect.addEventListener("change", handleAiconChange);
  }

  /**
   * Opens the context window modal
   */
  function openContextWindow() {
    // Always reload token usage when opening modal
    loadTokenUsage(true);
    contextWindowPanel.style.display = "block";
  }

  /**
   * Closes the context window modal
   */
  function closeContextWindow() {
    contextWindowPanel.style.display = "none";
  }

  /**
   * Fetches token usage data from the server
   */
  function loadTokenUsage(showLoadingState = false) {
    // Show loading state if requested
    if (showLoadingState) {
      id("total-usage-text").textContent = "Loading...";
      id("state-tokens").textContent = "Loading...";
      id("utility-tokens").textContent = "Loading...";
      id("action-tokens").textContent = "Loading...";
      id("inference-tokens").textContent = "Loading...";
    }
    
    fetch("/api/token-usage?" + new Date().getTime())  // Add timestamp to prevent caching
      .then(response => {
        if (!response.ok) {
          throw new Error("Failed to fetch token usage data");
        }
        return response.json();
      })
      .then(data => {
        console.log("Received token usage data:", data);
        updateTokenUsageDisplay(data);
        lastTokenUsage = data;
      })
      .catch(error => {
        console.error("Error loading token usage:", error);
        // Show error in modal
        id("total-usage-text").textContent = "Error loading data";
      });
  }

  /**
   * Updates the token usage display with the provided data
   */
  function updateTokenUsageDisplay(data) {
    if (!data || typeof data.total_used !== 'number') {
      console.error("Invalid token usage data:", data);
      id("total-usage-text").textContent = "Invalid data";
      return;
    }

    // Update total usage bar
    const totalUsed = data.total_used || 0;
    const totalAvailable = totalUsed + (data.remaining || 0);
    
    // Handle case where both values are 0 (no context used yet)
    const usagePercent = totalAvailable === 0 ? 0 : Math.round((totalUsed / totalAvailable) * 100);
    
    const totalBar = id("total-usage-bar");
    const totalText = id("total-usage-text");
    
    totalBar.style.width = `${usagePercent}%`;
    
    if (totalAvailable === 0) {
      totalText.textContent = "No context used yet";
    } else {
      totalText.textContent = `${usagePercent}% (${totalUsed.toLocaleString()} / ${totalAvailable.toLocaleString()} tokens)`;
    }
    
    // Set bar color based on usage
    if (usagePercent > 90) {
      totalBar.style.backgroundColor = "#dc3545"; // Red
    } else if (usagePercent > 70) {
      totalBar.style.backgroundColor = "#ffc107"; // Yellow
    } else {
      totalBar.style.backgroundColor = "#4a69bd"; // Blue
    }
    
    // Update component details
    updateComponentDisplay("state", data.state_representation);
    updateComponentDisplay("utility", data.utility_function);
    updateComponentDisplay("action", data.action_space);
    updateComponentDisplay("inference", data.inference);
  }

  /**
   * Updates a specific component's display
   */
  function updateComponentDisplay(prefix, componentData) {
    if (!componentData) {
      console.error(`Missing component data for ${prefix}`);
      return;
    }

    const tokensElement = id(`${prefix}-tokens`);
    const contentElement = id(`${prefix}-content`);
    
    // Display token count
    const tokens = componentData.tokens || 0;
    tokensElement.textContent = `${tokens.toLocaleString()} tokens`;
    
    // Format the content
    if (componentData.content && componentData.content.trim()) {
      contentElement.textContent = componentData.content;
    } else {
      contentElement.textContent = "No content available";
    }
  }

  /**
   * Handles clicks on thinking headers to expand/collapse
   */
  function handleThinkingToggleClick(event) {
    // Find if we clicked on a thinking header or its child elements
    const header = event.target.closest('.thinking-header');
    if (!header) return;
    
    // Find the associated content - it should be the next sibling
    const wrapper = header.closest('.thinking-content-wrapper');
    if (!wrapper) return;
    
    const content = wrapper.querySelector('.thinking-content');
    if (!content) return;
    
    // Find the indicator
    const indicator = header.querySelector('.toggle-indicator');
    if (!indicator) return;
    
    // Toggle collapsed state
    if (content.classList.contains('collapsed')) {
      // Expand
      content.classList.remove('collapsed');
      indicator.textContent = '[-]';
    } else {
      // Collapse
      content.classList.add('collapsed');
      indicator.textContent = '[+]';
    }
  }

  /**
   * Find parent element with a specific class
   */
  function findParentWithClass(element, className) {
    let current = element;
    while (current && !current.classList.contains(className)) {
      current = current.parentElement;
    }
    return current;
  }

  /**
   * Render text as Markdown HTML
   */
  function renderMarkdown(text) {
    return md.render(text);
  }

  /**
   * Handles the send button click
   */
  function handleSendMessage() {
    if (isProcessing) return;
    sendUserMessage();
  }

  /**
   * Handles Enter key press in the input field
   */
  function handleKeyPress(e) {
    if (e.key === "Enter" && !isProcessing) {
      sendUserMessage();
    }
  }

  /**
   * Loads the chat history from the server
   */
  function loadChatHistory() {
    fetch("/api/history")
      .then(response => response.json())
      .then(data => {
        data.history.forEach(msg => {
          appendMessage(msg.role, msg.content);
        });
        scrollToBottom();
      })
      .catch(error => {
        console.error("Error loading history:", error);
      });
  }

  /**
   * Sends the user message to the server and processes the response
   */
  function sendUserMessage() {
    const message = messageInput.value.trim();
    if (!message) return;
    
    // Set processing state
    isProcessing = true;
    
    // Clear input
    messageInput.value = "";
    
    // Append user message
    appendMessage("user", message);
    scrollToBottom();
    
    // Show typing indicator
    typingIndicator.style.display = "block";
    scrollToBottom();
    
    // Stream the response
    fetch("/api/stream-chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ 
        message: message, 
        show_thinking: true,
        aicon: currentAicon
      })
    })
    .then(response => {
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      
      // Set up event reader to stream response
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      // Create placeholder for thinking
      const thinkingDiv = document.createElement("div");
      thinkingDiv.className = "message thinking-message";
      const thinkingContentWrapper = document.createElement("div");
      thinkingContentWrapper.className = "message-content thinking-content-wrapper";
      
      // Create the thinking header
      const thinkingHeader = document.createElement("div");
      thinkingHeader.className = "thinking-header";
      thinkingHeader.innerHTML = `
        <strong>Thinking Process</strong>
        <span class="toggle-indicator">[-]</span>
      `;
      
      // Create the thinking content area
      const thinkingContent = document.createElement("div");
      thinkingContent.className = "thinking-content";
      thinkingContent.innerHTML = `<div class="typing-status">Processing...</div>`;
      
      // Assemble the thinking section
      thinkingContentWrapper.appendChild(thinkingHeader);
      thinkingContentWrapper.appendChild(thinkingContent);
      thinkingDiv.appendChild(thinkingContentWrapper);
      
      // Create placeholder for response
      const responseDiv = document.createElement("div");
      responseDiv.className = "message assistant-message";
      const responseContentDiv = document.createElement("div");
      responseContentDiv.className = "message-content markdown-content";
      responseContentDiv.innerHTML = '<div class="typing-status">Preparing response...</div>';
      responseDiv.appendChild(responseContentDiv);
      
      // Add both to the chat
      messagesContainer.insertBefore(thinkingDiv, typingIndicator);
      messagesContainer.insertBefore(responseDiv, typingIndicator);
      
      // Get references to thinking content for updating
      const thinkingContentEl = thinkingContent;
      
      // Keep track of accumulated thinking and response
      let accumulatedThinking = "";
      let accumulatedResponse = "";
      let hasDetectedThinkTag = false;
      let inThinkingSection = false;
      
      // Process the stream
      function processStream() {
        return reader.read().then(({ done, value }) => {
          if (done) {
            console.log("Stream complete");
            isProcessing = false;
            // Add a small delay before loading token usage to ensure server has updated
            setTimeout(() => loadTokenUsage(), 500);
            return;
          }
          
          // Decode the chunk
          const chunk = decoder.decode(value, { stream: true });
          
          // Process each event (data: {...}\n\n)
          const events = chunk.split("\n\n");
          for (const event of events) {
            if (event.startsWith("data: ")) {
              try {
                const jsonData = event.slice(6); // Remove "data: "
                const data = JSON.parse(jsonData);
                
                // Hide typing indicator
                typingIndicator.style.display = "none";
                
                // Process chunks
                if (data.chunk) {
                  const chunkText = data.chunk;
                  
                  // Check for thinking tags
                  if (chunkText.includes("<think>")) {
                    hasDetectedThinkTag = true;
                    inThinkingSection = true;
                    
                    // Extract what's before the tag for the response
                    const beforeThink = chunkText.split("<think>")[0];
                    if (beforeThink.trim()) {
                      accumulatedResponse += beforeThink;
                      responseContentDiv.innerHTML = renderMarkdown(accumulatedResponse);
                    }
                    
                    // Extract thinking content
                    const thinkContent = chunkText.split("<think>")[1];
                    accumulatedThinking += thinkContent;
                    thinkingContentEl.textContent = accumulatedThinking; // Keep thinking as plain text
                  } 
                  else if (hasDetectedThinkTag && chunkText.includes("</think>")) {
                    // End of thinking section
                    inThinkingSection = false;
                    
                    // Split the chunk at </think>
                    const parts = chunkText.split("</think>");
                    
                    // Add the first part to thinking
                    if (parts[0]) {
                      accumulatedThinking += parts[0];
                      thinkingContentEl.textContent = accumulatedThinking; // Keep thinking as plain text
                    }
                    
                    // Add the second part to response
                    if (parts[1]) {
                      accumulatedResponse += parts[1];
                      responseContentDiv.innerHTML = renderMarkdown(accumulatedResponse);
                    }
                  }
                  else if (inThinkingSection) {
                    // In thinking section, add to thinking
                    accumulatedThinking += chunkText;
                    thinkingContentEl.textContent = accumulatedThinking; // Keep thinking as plain text
                  }
                  else {
                    // Normal response content
                    accumulatedResponse += chunkText;
                    responseContentDiv.innerHTML = renderMarkdown(accumulatedResponse);
                  }
                }
                
                // Handle done signal
                if (data.done) {
                  // Finalize the thinking and response
                  if (accumulatedThinking.trim()) {
                    thinkingContentEl.textContent = accumulatedThinking; // Keep thinking as plain text
                  } else {
                    // No thinking content, remove the thinking bubble
                    thinkingDiv.remove();
                  }
                  
                  if (data.full_response) {
                    // Check if we need to handle a full response with thinking tags
                    if (data.full_response.includes("<think>") && data.full_response.includes("</think>")) {
                      const parts = data.full_response.split(/<think>|<\/think>/);
                      if (parts.length >= 3) {
                        // parts[0] is before <think>, parts[1] is inside <think>, parts[2] is after </think>
                        const finalResponse = (parts[0] + " " + (parts[2] || "")).trim();
                        responseContentDiv.innerHTML = renderMarkdown(finalResponse);
                        thinkingContentEl.textContent = parts[1].trim(); // Keep thinking as plain text
                      } else {
                        responseContentDiv.innerHTML = renderMarkdown(data.full_response);
                      }
                    } else {
                      responseContentDiv.innerHTML = renderMarkdown(data.full_response);
                    }
                  } else if (!accumulatedResponse.trim()) {
                    // No response content
                    responseDiv.remove();
                  }
                  
                  typingIndicator.style.display = "none"; // Ensure indicator is hidden
                  scrollToBottom();
                  isProcessing = false;
                  // Add a small delay before loading token usage to ensure server has updated
                  setTimeout(() => loadTokenUsage(), 500);
                  return;
                }
              } catch (error) {
                console.error("Error parsing stream data:", error, event);
              }
            }
          }
          
          scrollToBottom();
          
          // Continue reading
          return processStream();
        });
      }
      
      // Start processing
      return processStream();
    })
    .catch(error => {
      console.error("Fetch error:", error);
      displayError("Error: " + error.message);
      isProcessing = false;
    });
  }

  /**
   * Displays an error message
   */
  function displayError(errorMessage) {
    const responseDiv = document.createElement("div");
    responseDiv.className = "message assistant-message";
    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content";
    contentDiv.textContent = errorMessage;
    responseDiv.appendChild(contentDiv);
    messagesContainer.insertBefore(responseDiv, typingIndicator);
    
    typingIndicator.style.display = "none";
    scrollToBottom();
  }

  /**
   * Appends a message to the chat
   */
  function appendMessage(role, content) {
    // Check if content contains thinking tags
    if (role === "assistant" && content.includes("<think>") && content.includes("</think>")) {
      // Split the content into thinking and response parts
      const parts = content.split(/<think>|<\/think>/);
      if (parts.length >= 3) {
        // Create thinking bubble
        const thinkingDiv = document.createElement("div");
        thinkingDiv.className = "message thinking-message";
        thinkingDiv.innerHTML = `
          <div class="message-content thinking-content-wrapper">
            <div class="thinking-header">
              <strong>Thinking Process</strong>
              <span class="toggle-indicator">[-]</span>
            </div>
            <div class="thinking-content">${parts[1].trim()}</div>
          </div>
        `;
        
        // Create response bubble with markdown rendering
        const responseDiv = document.createElement("div");
        responseDiv.className = "message assistant-message";
        responseDiv.innerHTML = `
          <div class="message-content markdown-content">
            ${renderMarkdown((parts[0] + " " + (parts[2] || "")).trim())}
          </div>
        `;
        
        // Add both to chat
        messagesContainer.insertBefore(thinkingDiv, typingIndicator);
        messagesContainer.insertBefore(responseDiv, typingIndicator);
        return;
      }
    }
    
    // Regular message without thinking tags
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${role}-message`;
    
    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content";
    
    // Render markdown for assistant messages only
    if (role === "assistant") {
      contentDiv.className += " markdown-content"; 
      contentDiv.innerHTML = renderMarkdown(content);
    } else {
      contentDiv.textContent = content;
    }
    
    messageDiv.appendChild(contentDiv);
    messagesContainer.insertBefore(messageDiv, typingIndicator);
  }

  /**
   * Scrolls to the bottom of the messages container
   */
  function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  /**
   * Helper function to get element by ID
   * @param {string} elementId - ID of the element to retrieve
   * @returns {HTMLElement} The DOM element with the specified ID
   */
  function id(elementId) {
    return document.getElementById(elementId);
  }

  /**
   * Opens the configuration panel
   */
  function openConfigPanel() {
    loadConfiguration();
    configPanel.style.display = "block";
  }
  
  /**
   * Switches to the specified tab
   */
  function switchTab(tabName) {
    // Update button states
    tabBtns.forEach(btn => {
      if (btn.getAttribute("data-tab") === tabName) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    });
    
    // Update tab pane visibility
    tabPanes.forEach(pane => {
      if (pane.id === tabName + "-tab") {
        pane.style.display = "block";
      } else {
        pane.style.display = "none";
      }
    });
  }
  
  /**
   * Toggles factor fields based on factor type
   */
  function toggleFactorFields() {
    const factorType = factorTypeSelect.value;
    const continuousFields = id("continuous-fields");
    const categoricalFields = id("categorical-fields");
    const discreteFields = id("discrete-fields");
    
    // Hide all first
    continuousFields.style.display = "none";
    categoricalFields.style.display = "none";
    discreteFields.style.display = "none";
    
    // Show the selected one
    if (factorType === "continuous") {
      continuousFields.style.display = "block";
    } else if (factorType === "categorical") {
      categoricalFields.style.display = "block";
    } else if (factorType === "discrete") {
      discreteFields.style.display = "block";
    }
  }
  
  /**
   * Toggles space fields based on space type
   */
  function toggleSpaceFields() {
    // You can add more specific logic here as needed
    const spaceType = spaceTypeSelect.value;
    const budgetFields = id("budget-fields");
    
    if (spaceType === "budget_allocation" || spaceType === "marketing") {
      budgetFields.style.display = "block";
    } else {
      budgetFields.style.display = "none";
    }
  }
  
  /**
   * Toggles utility fields based on utility type
   */
  function toggleUtilityFields() {
    const utilityType = utilityTypeSelect.value;
    const marketingROIFields = id("marketing-roi-fields");
    
    if (utilityType === "marketing_roi") {
      marketingROIFields.style.display = "block";
    } else {
      marketingROIFields.style.display = "none";
    }
  }
  
  /**
   * Toggles sensor fields based on sensor type
   */
  function toggleSensorFields() {
    const sensorType = sensorTypeSelect.value;
    const metaAdsFields = id("meta-ads-fields");
    
    if (sensorType === "meta_ads") {
      metaAdsFields.style.display = "block";
    } else {
      metaAdsFields.style.display = "none";
    }
  }
  
  /**
   * Loads the current configuration from the server
   */
  function loadConfiguration() {
    fetch("/api/configuration")
      .then(response => response.json())
      .then(data => {
        // Update UI with configuration data
        updateConfigurationUI(data);
      })
      .catch(error => {
        console.error("Error loading configuration:", error);
      });
  }
  
  /**
   * Updates the configuration UI with the loaded data
   */
  function updateConfigurationUI(data) {
    // Store current configuration
    currentSensors = data.sensors || [];
    currentFactors = data.state_factors || [];
    currentActionSpace = data.action_space || null;
    currentUtility = data.utility_function || null;
    
    // Update review tab
    updateReviewTab();
    
    // Update current sensors display
    updateSensorsDisplay();
    
    // Update current state factors display
    updateFactorsDisplay();
    
    // Update current action space display
    updateActionSpaceDisplay();
    
    // Update current utility function display
    updateUtilityDisplay();
  }
  
  /**
   * Updates the review tab with all configurations
   */
  function updateReviewTab() {
    // Update sensors count
    const sensorsCount = id("sensors-count");
    sensorsCount.textContent = currentSensors.length;
    
    // Update factors count
    const factorsCount = id("factors-count");
    factorsCount.textContent = currentFactors.length;
    
    // Update sensors dashboard
    const sensorsDashboard = id("dashboard-sensors");
    if (!currentSensors || currentSensors.length === 0) {
      sensorsDashboard.innerHTML = '<div class="empty-message">No sensors configured yet</div>';
    } else {
      let html = '';
      currentSensors.forEach(sensor => {
        html += `
          <div class="dashboard-item">
            <div class="dashboard-item-header">
              <div class="dashboard-item-name">${sensor.name}</div>
              <div class="dashboard-item-type">${sensor.sensor_type}</div>
            </div>
            <div class="dashboard-item-details">
              Reliability: ${sensor.reliability || 'N/A'}
              <button class="toggle-details-btn" data-target="sensor-${sensor.name}">Show Details</button>
              <pre class="details-content" id="sensor-${sensor.name}" style="display: none;">${JSON.stringify(sensor, null, 2)}</pre>
            </div>
          </div>
        `;
      });
      sensorsDashboard.innerHTML = html;
    }
    
    // Update factors dashboard
    const factorsDashboard = id("dashboard-factors");
    if (!currentFactors || currentFactors.length === 0) {
      factorsDashboard.innerHTML = '<div class="empty-message">No state factors defined yet</div>';
    } else {
      let html = '';
      currentFactors.forEach(factor => {
        html += `
          <div class="dashboard-item">
            <div class="dashboard-item-header">
              <div class="dashboard-item-name">${factor.name}</div>
              <div class="dashboard-item-type">${factor.factor_type}</div>
            </div>
            <div class="dashboard-item-details">
              Value: ${factor.value}
              <button class="toggle-details-btn" data-target="factor-${factor.name}">Show Details</button>
              <pre class="details-content" id="factor-${factor.name}" style="display: none;">${JSON.stringify(factor, null, 2)}</pre>
            </div>
          </div>
        `;
      });
      factorsDashboard.innerHTML = html;
    }
    
    // Update action space dashboard
    const actionSpaceDashboard = id("dashboard-action-space");
    if (!currentActionSpace) {
      actionSpaceDashboard.innerHTML = '<div class="empty-message">No action space defined yet</div>';
    } else {
      let html = `
        <div class="dashboard-item action-space">
          <div class="dashboard-item-header">
            <div class="dashboard-item-name">${currentActionSpace.space_type || 'Custom'} Action Space</div>
            <div class="dashboard-item-type">Action Space</div>
          </div>
          <div class="dashboard-item-details">
            <span>${currentActionSpace.description || 'Dimensions: ' + (currentActionSpace.dimensions || 'N/A')}</span>
            <button class="toggle-details-btn" data-target="action-space-details">Show Details</button>
            <pre class="details-content" id="action-space-details" style="display: none;">${JSON.stringify(currentActionSpace, null, 2)}</pre>
          </div>
        </div>
      `;
      actionSpaceDashboard.innerHTML = html;
    }
    
    // Update utility dashboard
    const utilityDashboard = id("dashboard-utility");
    if (!currentUtility) {
      utilityDashboard.innerHTML = '<div class="empty-message">No utility function defined yet</div>';
    } else {
      let html = `
        <div class="dashboard-item utility">
          <div class="dashboard-item-header">
            <div class="dashboard-item-name">${currentUtility.utility_type || 'Custom'} Utility</div>
            <div class="dashboard-item-type">Utility Function</div>
          </div>
          <div class="dashboard-item-details">
            <span>${currentUtility.description || ''}</span>
            <button class="toggle-details-btn" data-target="utility-details">Show Details</button>
            <pre class="details-content" id="utility-details" style="display: none;">${JSON.stringify(currentUtility, null, 2)}</pre>
          </div>
        </div>
      `;
      utilityDashboard.innerHTML = html;
    }
    
    // Add event listeners to toggle detail buttons
    document.querySelectorAll('.toggle-details-btn').forEach(btn => {
      btn.addEventListener('click', function() {
        const targetId = this.getAttribute('data-target');
        const targetElem = id(targetId);
        if (targetElem.style.display === 'none') {
          targetElem.style.display = 'block';
          this.textContent = 'Hide Details';
        } else {
          targetElem.style.display = 'none';
          this.textContent = 'Show Details';
        }
      });
    });
  }
  
  /**
   * Updates the sensors display
   */
  function updateSensorsDisplay() {
    const container = id("current-sensors");
    
    if (!currentSensors || currentSensors.length === 0) {
      container.innerHTML = '<div class="no-items">No sensors configured yet.</div>';
      return;
    }
    
    let html = '';
    currentSensors.forEach(sensor => {
      html += `
        <div class="config-item">
          <div class="config-item-header">
            <h4>${sensor.name}</h4>
            <span class="config-item-type">${sensor.sensor_type}</span>
          </div>
          <div class="config-item-body">
            <pre>${JSON.stringify(sensor, null, 2)}</pre>
          </div>
        </div>
      `;
    });
    
    container.innerHTML = html;
  }
  
  /**
   * Updates the state factors display
   */
  function updateFactorsDisplay() {
    const container = id("current-factors");
    
    if (!currentFactors || currentFactors.length === 0) {
      container.innerHTML = '<div class="no-items">No state factors defined yet.</div>';
      return;
    }
    
    let html = '';
    currentFactors.forEach(factor => {
      html += `
        <div class="config-item">
          <div class="config-item-header">
            <h4>${factor.name}</h4>
            <span class="config-item-type">${factor.factor_type}</span>
          </div>
          <div class="config-item-body">
            <pre>${JSON.stringify(factor, null, 2)}</pre>
          </div>
        </div>
      `;
    });
    
    container.innerHTML = html;
  }
  
  /**
   * Updates the action space display
   */
  function updateActionSpaceDisplay() {
    const container = id("current-action-space");
    
    if (!currentActionSpace) {
      container.innerHTML = '<div class="no-items">No action space defined yet.</div>';
      return;
    }
    
    let html = `
      <div class="config-item">
        <div class="config-item-header">
          <h4>${currentActionSpace.space_type} Action Space</h4>
          <span class="config-item-type">Action Space</span>
        </div>
        <div class="config-item-body">
          <pre>${JSON.stringify(currentActionSpace, null, 2)}</pre>
        </div>
      </div>
    `;
    
    container.innerHTML = html;
  }
  
  /**
   * Updates the utility function display
   */
  function updateUtilityDisplay() {
    const container = id("current-utility");
    
    if (!currentUtility) {
      container.innerHTML = '<div class="no-items">No utility function defined yet.</div>';
      return;
    }
    
    let html = `
      <div class="config-item">
        <div class="config-item-header">
          <h4>${currentUtility.utility_type} Utility</h4>
          <span class="config-item-type">Utility Function</span>
        </div>
        <div class="config-item-body">
          <pre>${JSON.stringify(currentUtility, null, 2)}</pre>
        </div>
      </div>
    `;
    
    container.innerHTML = html;
  }

  /**
   * Handles the sensor form submission
   */
  function handleSensorForm(event) {
    event.preventDefault();
    
    const sensorType = id("sensor-type").value;
    const sensorName = id("sensor-name").value;
    
    let sensorData = {
      sensor_type: sensorType,
      name: sensorName
    };
    
    // Add sensor-specific fields
    if (sensorType === "meta_ads") {
      sensorData.access_token = id("access-token").value;
      sensorData.ad_account_id = id("ad-account-id").value;
      sensorData.campaign_id = id("campaign-id").value;
      sensorData.api_version = "v18.0";
      sensorData.time_granularity = "hour";
      sensorData.reliability = 0.9;
    }
    
    // Send to server
    fetch("/api/add-sensor", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(sensorData)
    })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        // Reset form
        sensorForm.reset();
        
        // Reload configuration
        loadConfiguration();
        
        // Show success message
        alert("Sensor added successfully!");
      } else {
        alert("Error adding sensor: " + data.error);
      }
    })
    .catch(error => {
      console.error("Error adding sensor:", error);
      alert("Error adding sensor. See console for details.");
    });
  }
  
  /**
   * Handles the state factor form submission
   */
  function handleStateFactorForm(event) {
    event.preventDefault();
    
    const factorName = id("factor-name").value;
    const factorType = id("factor-type").value;
    const factorValue = id("factor-value").value;
    const dependsOn = id("depends-on").value;
    
    // Parse the value based on type
    let parsedValue;
    if (factorType === "continuous") {
      parsedValue = parseFloat(factorValue);
    } else if (factorType === "discrete") {
      parsedValue = parseInt(factorValue);
    } else {
      parsedValue = factorValue;
    }
    
    // Prepare the factor data
    let factorData = {
      name: factorName,
      factor_type: factorType,
      value: parsedValue,
      params: {},
      relationships: {
        depends_on: dependsOn ? dependsOn.split(",").map(item => item.trim()) : []
      }
    };
    
    // Add type-specific parameters
    if (factorType === "continuous") {
      factorData.params.loc = parseFloat(id("loc").value);
      factorData.params.scale = parseFloat(id("scale").value);
      
      // Add constraints if provided
      const lowerBound = id("lower-bound").value;
      const upperBound = id("upper-bound").value;
      if (lowerBound || upperBound) {
        factorData.params.constraints = {};
        if (lowerBound) factorData.params.constraints.lower = parseFloat(lowerBound);
        if (upperBound) factorData.params.constraints.upper = parseFloat(upperBound);
      }
    } else if (factorType === "categorical") {
      const categories = id("categories").value.split(",").map(item => item.trim());
      const probabilities = id("probabilities").value.split(",").map(item => parseFloat(item.trim()));
      
      factorData.params.categories = categories;
      factorData.params.probs = probabilities;
    } else if (factorType === "discrete") {
      factorData.params.rate = parseFloat(id("rate").value);
    }
    
    // Send to server
    fetch("/api/add-state-factor", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(factorData)
    })
    .then(response => {
      if (!response.ok) {
        return response.json().then(data => {
          throw new Error(data.error || `Error: ${response.status} ${response.statusText}`);
        });
      }
      return response.json();
    })
    .then(data => {
      if (data.success) {
        // Reset form
        stateFactorForm.reset();
        
        // Reset factor fields
        toggleFactorFields();
        
        // Reload configuration
        loadConfiguration();
        
        // Show success message
        alert("State factor added successfully!");
      } else {
        alert("Error adding state factor: " + data.error);
      }
    })
    .catch(error => {
      console.error("Error adding state factor:", error);
      alert("Error adding state factor: " + error.message);
    });
  }
  
  /**
   * Handles the action space form submission
   */
  function handleActionSpaceForm(event) {
    event.preventDefault();
    
    const spaceType = id("space-type").value;
    
    // Prepare the action space data
    let spaceData = {
      space_type: spaceType
    };
    
    // Add type-specific parameters
    if (spaceType === "budget_allocation" || spaceType === "marketing") {
      spaceData.total_budget = parseFloat(id("total-budget").value);
      spaceData.items = id("items").value.split(",").map(item => item.trim());
      spaceData.budget_step = parseFloat(id("budget-step").value);
      spaceData.min_budget = parseFloat(id("min-budget").value);
    }
    
    // Send to server
    fetch("/api/define-action-space", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(spaceData)
    })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        // Reset form
        actionSpaceForm.reset();
        
        // Reset space fields
        toggleSpaceFields();
        
        // Reload configuration
        loadConfiguration();
        
        // Show success message
        alert("Action space defined successfully!");
      } else {
        alert("Error defining action space: " + data.error);
      }
    })
    .catch(error => {
      console.error("Error defining action space:", error);
      alert("Error defining action space. See console for details.");
    });
  }
  
  /**
   * Handles the utility function form submission
   */
  function handleUtilityForm(event) {
    event.preventDefault();
    
    const utilityType = id("utility-type").value;
    
    // Prepare the utility function data
    let utilityData = {
      utility_type: utilityType
    };
    
    // Add type-specific parameters
    if (utilityType === "marketing_roi") {
      utilityData.revenue_per_sale = parseFloat(id("revenue-per-sale").value);
      utilityData.num_days = parseInt(id("num-days").value);
      utilityData.ad_names = id("ad-names").value.split(",").map(item => item.trim());
    }
    
    // Send to server
    fetch("/api/define-utility-function", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(utilityData)
    })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        // Reset form
        utilityForm.reset();
        
        // Reset utility fields
        toggleUtilityFields();
        
        // Reload configuration
        loadConfiguration();
        
        // Show success message
        alert("Utility function defined successfully!");
      } else {
        alert("Error defining utility function: " + data.error);
      }
    })
    .catch(error => {
      console.error("Error defining utility function:", error);
      alert("Error defining utility function. See console for details.");
    });
  }

  /**
   * Handles the creation of a marketing AIcon
   */
  function handleCreateMarketing() {
    // Show loading state
    const originalText = createMarketingBtn.textContent;
    createMarketingBtn.textContent = "Creating Marketing AIcon...";
    
    // Create marketing AIcon
    fetch("/api/aicons/marketing", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      }
    })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        // Reload AIcons list
        loadAicons();
        // Show success message
        alert("Marketing AIcon created successfully!");
      } else {
        alert("Error creating Marketing AIcon: " + data.error);
      }
    })
    .catch(error => {
      console.error("Error creating Marketing AIcon:", error);
      alert("Error creating Marketing AIcon. See console for details.");
    })
    .finally(() => {
      // Restore original text
      createMarketingBtn.textContent = originalText;
    });
  }

  /**
   * Loads AIcons from the server
   */
  function loadAicons() {
    fetch("/api/aicons")
      .then(response => response.json())
      .then(data => {
        // Update the select dropdown
        aiconSelect.innerHTML = '';
        
        // Add existing AIcons
        data.aicons.forEach(aicon => {
          const option = document.createElement('option');
          option.value = aicon;
          option.textContent = aicon;
          if (aicon === data.current) {
            option.selected = true;
            currentAicon = aicon;
          }
          aiconSelect.appendChild(option);
        });
      })
      .catch(error => {
        console.error("Error loading AIcons:", error);
      });
  }

  /**
   * Handles AIcon selection change
   */
  function handleAiconChange() {
    const newAicon = aiconSelect.value;
    
    if (newAicon === currentAicon) return;
    
    // Set the new AIcon as current
    fetch("/api/aicons/current", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ name: newAicon })
    })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        currentAicon = newAicon;
        // Reload chat history for the new AIcon
        loadChatHistory();
        // Reload configuration
        loadConfiguration();
      } else {
        console.error("Error switching AIcon:", data.error);
      }
    })
    .catch(error => {
      console.error("Error switching AIcon:", error);
    });
  }
})(); 