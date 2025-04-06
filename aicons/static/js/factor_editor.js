/**
 * Factor Editor Module
 * 
 * Handles the integration between the Factor Graph visualization and the UI.
 * Manages the creation, editing, and deletion of hierarchical state factors.
 */

(function() {
  // Main objects
  let factorGraph = null;
  let currentFactors = []; // All factors (both local and server)
  let serverFactors = []; // Factors saved on the server
  let localFactors = []; // Factors only in local memory
  let editingFactor = null;
  
  // DOM elements
  let graphContainer;
  let editorPanel;
  let factorForm;
  let parentInfo;
  let saveButton;
  let cancelButton;
  let deleteButton;
  let saveAllButton;
  
  // Form fields
  let factorNameField;
  let factorTypeSelect;
  let factorValueField;
  let factorFields;
  
  /**
   * Initialize the factor editor when the DOM is loaded
   */
  document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing factor editor');
    
    // Run init with a slight delay to ensure DOM is fully prepared
    setTimeout(init, 100);
  });
  
  /**
   * Initialize the factor graph and editor
   */
  function init() {
    console.log('Initializing factor_editor.js');
    
    // Check if graph container exists 
    graphContainer = document.getElementById('factor-graph-container');
    
    // If container doesn't exist, the tab hasn't been rendered yet
    if (!graphContainer) {
      console.log('Factor graph container not found - waiting for config panel to open');
      
      // Wait for the config panel to be opened and tabs to be switched
      const configBtn = document.getElementById('config-btn');
      if (configBtn) {
        console.log('Config button found - adding click listener');
        
        // Add click listener if not already added
        if (!configBtn.dataset.hasListener) {
          configBtn.addEventListener('click', initAfterConfigOpen);
          configBtn.dataset.hasListener = 'true';
        }
      } else {
        console.warn('Config button not found');
      }
      
      return;
    }
    
    initFactorEditor();
  }
  
  /**
   * Initialize after config panel is opened
   */
  function initAfterConfigOpen() {
    console.log('Config panel opened, initializing factors tab');
    
    // Immediately fetch factors to have them ready
    fetch('/api/configuration')
      .then(response => response.json())
      .then(data => {
        if (data.state_factors && data.state_factors.length > 0) {
          console.log('Preloaded factors:', data.state_factors);
          currentFactors = data.state_factors;
          serverFactors = [...data.state_factors];
        }
      })
      .catch(error => {
        console.error('Error preloading factors:', error);
      });
    
    // Find and add click listener to factors tab
    const tabBtns = document.querySelectorAll('.tab-btn');
    const factorsTab = Array.from(tabBtns).find(btn => 
      btn.getAttribute('data-tab') === 'factors'
    );
    
    if (factorsTab) {
      console.log('Found factors tab - adding click listener');
      
      // Add click listener if not already added
      if (!factorsTab.dataset.hasGraphListener) {
        factorsTab.addEventListener('click', initFactorEditor);
        factorsTab.dataset.hasGraphListener = 'true';
        console.log('Added click listener to factors tab');
      }
    } else {
      console.warn('Factors tab not found');
    }
  }
  
  /**
   * Initialize the factor editor interface
   */
  function initFactorEditor() {
    // Initialize the graph editor UI
    console.log('Initializing factor graph editor UI');
    
    // Get the graph container
    const graphContainer = document.getElementById('factor-graph-container');
    if (!graphContainer) {
      console.warn('Could not find graph container after tab click');
      return;
    }
    
    // Force the container to be visible with a specific height
    graphContainer.style.display = 'block';
    graphContainer.style.height = '400px';
    graphContainer.style.width = '100%';
    graphContainer.style.border = '1px solid #e2e8f0';
    console.log('Set graph container height to 400px');
    
    // Create the factor graph
    if (!factorGraph) {
      try {
        console.log('Creating factor graph instance');
        factorGraph = new FactorGraph('factor-graph-container');
        // Store in window for direct access
        window.factorGraphInstance = factorGraph;
        console.log('Factor graph instance created successfully');
      } catch (e) {
        console.error('Error creating factor graph:', e);
      }
    }
    
    // Create editor panel if it doesn't exist
    if (!document.getElementById('factor-editor-panel')) {
      console.log('Creating factor editor panel');
      createEditorPanel();
    }
    
    // Create add factor form if it doesn't exist
    if (!document.getElementById('factor-form')) {
      console.log('Creating factor form');
      createFactorForm();
    }
    
    // Create save all button if it doesn't exist
    if (!saveAllButton) {
      console.log('Creating save all button');
      saveAllButton = document.createElement('button');
      saveAllButton.id = 'save-all-factors-button';
      saveAllButton.className = 'save-all-factors-button';
      saveAllButton.textContent = 'Save All Factors to Server';
      saveAllButton.style.display = 'none';
      saveAllButton.addEventListener('click', saveAllFactorsToServer);
      graphContainer.parentNode.insertBefore(saveAllButton, graphContainer);
    }
    
    // Add event listener to the add factor form
    const form = document.getElementById('factor-form');
    if (form) {
      form.removeEventListener('submit', handleFactorFormSubmit);
      form.addEventListener('submit', handleFactorFormSubmit);
    }
    
    // Add event listener to the factor type select
    const factorTypeSelect = document.getElementById('factor-type');
    if (factorTypeSelect) {
      factorTypeSelect.removeEventListener('change', handleFactorTypeChange);
      factorTypeSelect.addEventListener('change', handleFactorTypeChange);
    }
    
    // Load state factors from the server
    loadStateFactors();
  }
  
  // Make initFactorEditor available globally
  window.initFactorEditor = initFactorEditor;
  
  /**
   * Create the editor panel for factors
   */
  function createEditorPanel() {
    const graphContainer = document.getElementById('factor-graph-container');
    if (!graphContainer) return;
    
    const editorPanel = document.createElement('div');
    editorPanel.id = 'factor-editor-panel';
    editorPanel.className = 'factor-editor-panel';
    editorPanel.style.display = 'none';
    
    // Add editor after graph container
    graphContainer.parentNode.insertBefore(editorPanel, graphContainer.nextSibling);
    
    console.log('Editor panel created');
  }
  
  /**
   * Create the factor form
   */
  function createFactorForm() {
    const graphContainer = document.getElementById('factor-graph-container');
    if (!graphContainer) return;
    
    const formContainer = document.createElement('div');
    formContainer.id = 'factor-form-container';
    formContainer.className = 'factor-form-container';
    
    // Create the form HTML
    formContainer.innerHTML = `
      <form id="factor-form" class="factor-form">
        <h3>Add New Factor</h3>
        <div class="form-group">
          <label for="factor-name">Name:</label>
          <input type="text" id="factor-name" name="factor-name" required>
        </div>
        <div class="form-group">
          <label for="factor-type">Type:</label>
          <select id="factor-type" name="factor-type" required>
            <option value="continuous">Continuous</option>
            <option value="categorical">Categorical</option>
            <option value="binary">Binary</option>
          </select>
        </div>
        <div id="continuous-fields" class="factor-type-fields">
          <div class="form-group">
            <label for="factor-loc">Mean (μ):</label>
            <input type="number" id="factor-loc" name="factor-loc" value="0" step="0.1">
          </div>
          <div class="form-group">
            <label for="factor-scale">Standard Deviation (σ):</label>
            <input type="number" id="factor-scale" name="factor-scale" value="1" min="0.1" step="0.1">
          </div>
        </div>
        <div id="categorical-fields" class="factor-type-fields" style="display:none;">
          <div class="form-group">
            <label for="factor-categories">Categories (comma separated):</label>
            <input type="text" id="factor-categories" name="factor-categories" placeholder="cat,dog,bird">
          </div>
        </div>
        <div class="form-actions">
          <button type="submit">Add Factor</button>
        </div>
      </form>
    `;
    
    // Add to graph container parent
    graphContainer.parentNode.insertBefore(formContainer, graphContainer);
    
    console.log('Factor form created');
  }
  
  /**
   * Handle factor form submission
   */
  function handleFactorFormSubmit(event) {
    event.preventDefault();
    console.log('Factor form submitted');
    
    // Get form values
    const name = document.getElementById('factor-name').value;
    const type = document.getElementById('factor-type').value;
    
    // Create factor object
    const factor = {
      name: name,
      factor_type: type,
      value: 0,
      params: {},
      relationships: {
        depends_on: []
      }
    };
    
    // Add type-specific params
    if (type === 'continuous') {
      factor.params.loc = parseFloat(document.getElementById('factor-loc').value);
      factor.params.scale = parseFloat(document.getElementById('factor-scale').value);
    } else if (type === 'categorical') {
      const categories = document.getElementById('factor-categories').value.split(',').map(c => c.trim());
      factor.params.categories = categories;
      factor.params.probs = categories.map(() => 1 / categories.length);
    } else if (type === 'binary') {
      factor.params.prob = 0.5;
    }
    
    // If a node is selected, make this a child node
    if (factorGraph && factorGraph.selectedNode) {
      factor.relationships.depends_on = [factorGraph.selectedNode.id];
    }
    
    // Add factor to graph
    addFactor(factor);
    
    // Reset form
    event.target.reset();
  }
  
  /**
   * Handle factor type change
   */
  function handleFactorTypeChange(event) {
    const type = event.target.value;
    
    // Hide all fields
    const fieldSets = document.querySelectorAll('.factor-type-fields');
    fieldSets.forEach(fs => fs.style.display = 'none');
    
    // Show the selected type fields
    if (type === 'continuous') {
      document.getElementById('continuous-fields').style.display = 'block';
    } else if (type === 'categorical') {
      document.getElementById('categorical-fields').style.display = 'block';
    }
  }
  
  /**
   * Show or hide factor type specific fields
   */
  function toggleFactorFields() {
    const factorType = factorTypeSelect.value;
    
    // Hide all fields first
    Object.values(factorFields).forEach(field => {
      field.style.display = 'none';
    });
    
    // Show the selected type fields
    if (factorFields[factorType]) {
      factorFields[factorType].style.display = 'block';
    }
  }
  
  /**
   * Handle factor selection in the graph
   */
  function handleFactorSelect(node) {
    if (!node) {
      // Adding a new root factor
      showEditorPanel('Create Root Factor');
      editingFactor = { 
        isNew: true,
        factor: {
          relationships: { depends_on: [] }
        }
      };
      resetForm();
      parentInfo.style.display = 'none';
      deleteButton.style.display = 'none';
      return;
    }
    
    if (node.parent) {
      // Adding a child to an existing node
      showEditorPanel('Create Child Factor');
      editingFactor = { 
        isNew: true,
        parent: node.parent,
        factor: {
          relationships: { 
            depends_on: [node.parent.id] 
          }
        }
      };
      resetForm();
      
      // Show parent info
      parentInfo.style.display = 'block';
      document.getElementById('parent-factor-name').innerHTML = 
        `<span class="factor-parent-badge">${node.parent.label}</span>`;
      
      deleteButton.style.display = 'none';
      return;
    }
    
    // Editing an existing factor
    showEditorPanel('Edit Factor');
    editingFactor = {
      isNew: false,
      factor: node.factor
    };
    
    // Fill the form with current values
    fillFormWithFactorData(node.factor);
    
    // Show parent info if has parents
    if (node.factor.relationships && 
        node.factor.relationships.depends_on && 
        node.factor.relationships.depends_on.length > 0) {
      
      parentInfo.style.display = 'block';
      const parentNames = node.factor.relationships.depends_on
        .map(id => `<span class="factor-parent-badge">${id}</span>`)
        .join(' ');
        
      document.getElementById('parent-factor-name').innerHTML = parentNames;
    } else {
      parentInfo.style.display = 'none';
    }
    
    // Show delete button for existing factors
    deleteButton.style.display = 'inline-block';
  }
  
  /**
   * Show the editor panel with a specific title
   */
  function showEditorPanel(title) {
    editorPanel.style.display = 'block';
    document.querySelector('.factor-editor-title').textContent = title;
    
    // Scroll to editor panel
    editorPanel.scrollIntoView({ behavior: 'smooth' });
  }
  
  /**
   * Hide the editor panel
   */
  function hideEditorPanel() {
    editorPanel.style.display = 'none';
    editingFactor = null;
  }
  
  /**
   * Reset the form to default values
   */
  function resetForm() {
    factorForm.reset();
    
    // Set reasonable defaults
    factorNameField.value = '';
    factorTypeSelect.value = 'continuous';
    factorValueField.value = '';
    
    // Continuous defaults
    document.getElementById('loc').value = '0';
    document.getElementById('scale').value = '1';
    document.getElementById('lower-bound').value = '';
    document.getElementById('upper-bound').value = '';
    
    // Categorical defaults
    document.getElementById('categories').value = 'low, medium, high';
    document.getElementById('probabilities').value = '0.3, 0.4, 0.3';
    
    // Discrete defaults
    document.getElementById('rate').value = '5';
    
    // Show continuous fields by default
    toggleFactorFields();
  }
  
  /**
   * Fill the form with data from an existing factor
   */
  function fillFormWithFactorData(factor) {
    factorNameField.value = factor.name;
    factorTypeSelect.value = factor.factor_type;
    factorValueField.value = factor.value;
    
    // Set specific fields based on factor type
    if (factor.factor_type === 'continuous' && factor.params) {
      document.getElementById('loc').value = factor.params.loc || 0;
      document.getElementById('scale').value = factor.params.scale || 1;
      
      if (factor.params.constraints) {
        document.getElementById('lower-bound').value = 
          factor.params.constraints.lower !== undefined ? factor.params.constraints.lower : '';
        document.getElementById('upper-bound').value = 
          factor.params.constraints.upper !== undefined ? factor.params.constraints.upper : '';
      } else {
        document.getElementById('lower-bound').value = '';
        document.getElementById('upper-bound').value = '';
      }
    } else if (factor.factor_type === 'categorical' && factor.params) {
      document.getElementById('categories').value = 
        factor.params.categories ? factor.params.categories.join(', ') : '';
      document.getElementById('probabilities').value = 
        factor.params.probs ? factor.params.probs.join(', ') : '';
    } else if (factor.factor_type === 'discrete' && factor.params) {
      document.getElementById('rate').value = factor.params.rate || 5;
    }
    
    // Show relevant fields
    toggleFactorFields();
  }
  
  /**
   * Add a new state factor (locally only)
   */
  function addStateFactor(factorData) {
    // Add to graph immediately to show visual feedback
    const node = factorGraph.addFactor(factorData);
    
    // Add to current factors if not already there
    if (!currentFactors.find(f => f.name === factorData.name)) {
      currentFactors.push(factorData);
      
      // Also add to local factors
      if (!localFactors.find(f => f.name === factorData.name)) {
        localFactors.push(factorData);
        
        // Show save all button if we have local factors
        updateSaveAllButton();
      }
    }
    
    // Hide editor
    hideEditorPanel();
    
    // Show success message
    showToast('Factor added locally! Remember to save all factors to the server when done.');
    
    return node;
  }
  
  /**
   * Update an existing factor (locally only)
   */
  function updateStateFactor(factorData) {
    // Update in graph
    factorGraph.updateFactor(factorData);
    
    // Update in current factors
    const index = currentFactors.findIndex(f => f.name === editingFactor.factor.name);
    if (index !== -1) {
      currentFactors[index] = factorData;
      
      // Check if this was a server factor
      const serverIndex = serverFactors.findIndex(f => f.name === editingFactor.factor.name);
      if (serverIndex !== -1) {
        // Mark as local since we modified it
        serverFactors.splice(serverIndex, 1);
        
        // Add to local if not already there
        if (!localFactors.find(f => f.name === factorData.name)) {
          localFactors.push(factorData);
        }
        
        // Show save all button if we have local factors
        updateSaveAllButton();
      } else {
        // Update in local factors
        const localIndex = localFactors.findIndex(f => f.name === editingFactor.factor.name);
        if (localIndex !== -1) {
          localFactors[localIndex] = factorData;
        }
      }
    }
    
    // Hide editor
    hideEditorPanel();
    
    // Show success message
    showToast('Factor updated locally! Remember to save all factors to the server when done.');
  }
  
  /**
   * Handle deleting a factor (locally only)
   */
  function handleDeleteFactor() {
    if (!editingFactor || editingFactor.isNew) return;
    
    const factorName = editingFactor.factor.name;
    
    // Remove from graph
    factorGraph.removeFactor(factorName);
    
    // Remove from current factors
    const index = currentFactors.findIndex(f => f.name === factorName);
    if (index !== -1) {
      currentFactors.splice(index, 1);
    }
    
    // Remove from local or server factors
    const localIndex = localFactors.findIndex(f => f.name === factorName);
    if (localIndex !== -1) {
      localFactors.splice(localIndex, 1);
    } else {
      // If it was a server factor, add it to a delete list
      const serverIndex = serverFactors.findIndex(f => f.name === factorName);
      if (serverIndex !== -1) {
        serverFactors.splice(serverIndex, 1);
        
        // Mark as needing save
        updateSaveAllButton();
      }
    }
    
    // Hide editor
    hideEditorPanel();
    
    // Show success message
    showToast('Factor deleted locally! Remember to save changes to the server when done.');
  }
  
  /**
   * Save all factors to the server
   */
  function saveAllFactorsToServer() {
    showToast('Saving all factors to server...');
    
    // First, delete all factors on the server
    deleteAllServerFactors()
      .then(() => {
        // Then add all current factors
        return saveCurrentFactorsToServer();
      })
      .then(() => {
        // Update local tracking
        serverFactors = [...currentFactors];
        localFactors = [];
        
        // Update UI
        updateSaveAllButton();
        
        // Show success
        showToast('All factors saved to server successfully!');
      })
      .catch(error => {
        console.error('Error saving factors to server:', error);
        showError('Error saving factors: ' + error.message);
      });
  }
  
  /**
   * Delete all factors on the server
   */
  function deleteAllServerFactors() {
    // We'll use the configuration endpoint to get all factors
    return fetch('/api/configuration')
      .then(response => response.json())
      .then(data => {
        if (!data.state_factors || data.state_factors.length === 0) {
          return Promise.resolve(); // No factors to delete
        }
        
        // Create a promise chain to delete all factors
        let deletePromise = Promise.resolve();
        data.state_factors.forEach(factor => {
          deletePromise = deletePromise.then(() => {
            return fetch('/api/delete-state-factor', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json'
              },
              body: JSON.stringify({ name: factor.name })
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
              if (!data.success) {
                throw new Error(`Failed to delete factor ${factor.name}`);
              }
            });
          });
        });
        
        return deletePromise;
      });
  }
  
  /**
   * Save all current factors to the server
   */
  function saveCurrentFactorsToServer() {
    // Create a promise chain to add all factors
    let addPromise = Promise.resolve();
    
    // Process factors in topological order (roots first)
    const orderedFactors = factorGraph.generateHierarchicalRelationships();
    
    orderedFactors.forEach(factor => {
      addPromise = addPromise.then(() => {
        return fetch('/api/add-state-factor', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(factor)
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
          if (!data.success) {
            throw new Error(`Failed to add factor ${factor.name}`);
          }
        });
      });
    });
    
    return addPromise;
  }
  
  /**
   * Update the save all button visibility
   */
  function updateSaveAllButton() {
    if (localFactors.length > 0 || (serverFactors.length > 0 && currentFactors.length !== serverFactors.length)) {
      saveAllButton.style.display = 'block';
      saveAllButton.textContent = `Save ${currentFactors.length} Factors to Server`;
    } else {
      saveAllButton.style.display = 'none';
    }
  }
  
  /**
   * Load all state factors from the server
   */
  function loadStateFactors() {
    // If we already have factors loaded, use them immediately
    if (currentFactors && currentFactors.length > 0) {
      console.log('Using preloaded factors:', currentFactors);
      displayFactors(currentFactors);
      return;
    }
    
    fetch('/api/configuration')
      .then(response => response.json())
      .then(data => {
        if (data.state_factors && data.state_factors.length > 0) {
          currentFactors = data.state_factors;
          serverFactors = [...data.state_factors]; // Track server factors
          displayFactors(currentFactors);
        } else {
          console.log('No state factors found');
          if (factorGraph) {
            factorGraph.clear();
          }
        }
        
        // Update save all button
        updateSaveAllButton();
      })
      .catch(error => {
        console.error('Error loading state factors:', error);
        showError('Error loading state factors: ' + error.message);
      });
  }
  
  /**
   * Display the factors in the graph
   */
  function displayFactors(factors) {
    if (!factorGraph) return;
    
    console.log('Displaying factors:', factors);
    factorGraph.clear();
    factorGraph.setFactors(factors);
    factorGraph.autoLayout();
    
    // Update hidden div for compatibility with chat.js
    updateHiddenFactorsDiv(factors);
  }
  
  /**
   * Update the hidden current-factors div for compatibility
   */
  function updateHiddenFactorsDiv(factors) {
    const container = document.getElementById("current-factors");
    if (!container) return;
    
    if (!factors || factors.length === 0) {
      container.innerHTML = '<div class="no-items">No state factors defined yet.</div>';
      return;
    }
    
    let html = '';
    factors.forEach(factor => {
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
   * Show a toast message
   */
  function showToast(message) {
    // Disabled - no more toast messages
    console.log('Toast message (disabled):', message);
    return;
  }
  
  /**
   * Show an error message
   */
  function showError(message) {
    // Check if error container exists
    let errorContainer = document.getElementById('error-container');
    if (!errorContainer) {
      // Create error container
      errorContainer = document.createElement('div');
      errorContainer.id = 'error-container';
      errorContainer.className = 'error-container';
      document.body.appendChild(errorContainer);
    }
    
    // Create error
    const error = document.createElement('div');
    error.className = 'error';
    
    // Create error content
    const errorContent = document.createElement('div');
    errorContent.className = 'error-content';
    errorContent.textContent = message;
    
    // Create close button
    const closeButton = document.createElement('button');
    closeButton.className = 'error-close';
    closeButton.innerHTML = '&times;';
    closeButton.addEventListener('click', () => {
      error.remove();
    });
    
    // Assemble error
    error.appendChild(errorContent);
    error.appendChild(closeButton);
    
    // Add to container
    errorContainer.appendChild(error);
    
    // Remove after a while
    setTimeout(() => {
      error.classList.add('error-hide');
      setTimeout(() => {
        error.remove();
      }, 500);
    }, 10000);
  }
})(); 