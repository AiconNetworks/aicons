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
  let clearButton;
  let typingIndicator;
  
  // Initialize Markdown parser
  const md = window.markdownit({
    html: false,
    linkify: true,
    typographer: true
  });

  // State
  let isProcessing = false;
  let lastRawResponse = null;

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
    clearButton = id("clear-button");
    typingIndicator = id("typing-indicator");
    
    // Set up event listeners
    sendButton.addEventListener("click", handleSendMessage);
    messageInput.addEventListener("keypress", handleKeyPress);
    clearButton.addEventListener("click", handleClearChat);
    
    // Set up delegation for thinking toggles
    messagesContainer.addEventListener("click", handleThinkingToggleClick);
    
    // Load initial chat history
    loadChatHistory();
  }

  /**
   * Handles clicks on thinking headers to expand/collapse
   */
  function handleThinkingToggleClick(event) {
    const target = event.target;
    
    // Check if it's a thinking header or child of thinking header
    if (target.classList.contains("thinking-header") || 
        findParentWithClass(target, "thinking-header")) {
      
      const header = target.classList.contains("thinking-header") ? 
                    target : findParentWithClass(target, "thinking-header");
      
      const content = header.nextElementSibling;
      const indicator = header.querySelector(".toggle-indicator");
      
      if (content.classList.contains("collapsed")) {
        content.classList.remove("collapsed");
        indicator.textContent = "[-]";
      } else {
        content.classList.add("collapsed");
        indicator.textContent = "[+]";
      }
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
   * Handles clearing the chat history
   */
  function handleClearChat() {
    clearChat()
      .then(() => {
        resetChatUI();
      })
      .catch(error => {
        console.error("Error clearing chat:", error);
      });
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
        show_thinking: true 
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
      const thinkingContentDiv = document.createElement("div");
      thinkingContentDiv.className = "message-content thinking-content-wrapper";
      thinkingContentDiv.innerHTML = `
        <div class="thinking-header">
          <strong>Thinking Process</strong>
          <span class="toggle-indicator">[-]</span>
        </div>
        <div class="thinking-content">
          <div class="typing-status">Processing...</div>
        </div>
      `;
      thinkingDiv.appendChild(thinkingContentDiv);
      
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
      const thinkingContent = thinkingContentDiv.querySelector(".thinking-content");
      
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
                    thinkingContent.textContent = accumulatedThinking; // Keep thinking as plain text
                  } 
                  else if (hasDetectedThinkTag && chunkText.includes("</think>")) {
                    // End of thinking section
                    inThinkingSection = false;
                    
                    // Split the chunk at </think>
                    const parts = chunkText.split("</think>");
                    
                    // Add the first part to thinking
                    if (parts[0]) {
                      accumulatedThinking += parts[0];
                      thinkingContent.textContent = accumulatedThinking; // Keep thinking as plain text
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
                    thinkingContent.textContent = accumulatedThinking; // Keep thinking as plain text
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
                    thinkingContent.textContent = accumulatedThinking; // Keep thinking as plain text
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
                        thinkingContent.textContent = parts[1].trim(); // Keep thinking as plain text
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
                  
                  scrollToBottom();
                  isProcessing = false;
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
   * Clears the chat history on the server
   */
  function clearChat() {
    return fetch("/api/clear", { method: "POST" });
  }

  /**
   * Resets the chat UI after clearing
   */
  function resetChatUI() {
    messagesContainer.innerHTML = "";
    const initialMessage = document.createElement("div");
    initialMessage.className = "message assistant-message";
    initialMessage.innerHTML = '<div class="message-content">Hello! I\'m ZeroAIcon. How can I help you today?</div>';
    messagesContainer.appendChild(initialMessage);
    messagesContainer.appendChild(typingIndicator);
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
})(); 