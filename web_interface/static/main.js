"use strict";
(function () {
    // Application state
    const aiconState = {
        name: null,
        type: null,
        factors: {},
        sensors: {},
        actionSpace: null,
        utilityFunction: null
    };

    // Initialize when DOM is loaded
    window.addEventListener("load", init);

    function init() {
        console.log("AIcon Interface initialized");
        
        // Initialize tab navigation
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => handleTabClick(tab));
        });
        
        // Setup event listeners for form displays
        document.getElementById('discrete-distribution').addEventListener('change', updateDiscreteDist);
        document.getElementById('action-space-type').addEventListener('change', updateActionParams);
        document.getElementById('utility-function').addEventListener('change', updateUtilityParams);
        
        // Setup form submission handlers
        addClickHandler('create-aicon-btn', createAIcon);
        addClickHandler('add-continuous-btn', () => addFactor('continuous'));
        addClickHandler('add-categorical-btn', () => addFactor('categorical'));
        addClickHandler('add-discrete-btn', () => addFactor('discrete'));
        addClickHandler('add-hierarchical-btn', () => addFactor('hierarchical'));
        addClickHandler('set-action-space-btn', setActionSpace);
        addClickHandler('run-optimization-btn', runOptimization);
        
        // Setup predefined factor buttons
        addClickHandler('add-customer-ltv', addPredefinedCustomerLTV);
        addClickHandler('add-seasonality', addPredefinedSeasonality);
        addClickHandler('add-ad-fatigue', addPredefinedAdFatigue);
        addClickHandler('add-cross-campaign', addPredefinedCrossCampaign);
        addClickHandler('add-competitor', addPredefinedCompetitorIntensity);
        
        // Setup sensor toggle switches - now they just show/hide configuration
        document.getElementById('meta-ads-toggle').addEventListener('change', toggleMetaAdsConfig);
        
        // Setup sensor connect buttons
        addClickHandler('connect-meta-ads-btn', connectMetaAdsSensor);
        
        // Continue button after sensors
        addClickHandler('continue-after-sensors', continueAfterSensors);
    }

    function addClickHandler(id, handler) {
        const element = document.getElementById(id);
        if (element) {
            element.addEventListener('click', handler);
        } else {
            console.error(`Element with ID ${id} not found`);
        }
    }

    function handleTabClick(tab) {
        // Remove active class from all tabs and content
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        
        // Add active class to clicked tab and corresponding content
        tab.classList.add('active');
        const tabId = tab.getAttribute('data-tab');
        document.getElementById(`${tabId}-tab`).classList.add('active');
    }

    // Dynamic UI updates based on selection
    function updateDiscreteDist() {
        const distType = document.getElementById('discrete-distribution').value;
        toggleVisibility('poisson-params', distType === 'poisson');
        toggleVisibility('binomial-params', distType === 'binomial');
    }

    function updateActionParams() {
        const actionType = document.getElementById('action-space-type').value;
        toggleVisibility('budget-allocation-params', actionType === 'budget_allocation');
        toggleVisibility('bidding-params', actionType === 'bidding');
        toggleVisibility('custom-action-params', actionType === 'custom');
    }

    function updateUtilityParams() {
        const utilityType = document.getElementById('utility-function').value;
        toggleVisibility('custom-utility-params', utilityType === 'custom');
    }

    function toggleVisibility(id, isVisible) {
        const element = document.getElementById(id);
        if (element) {
            if (isVisible) {
                element.classList.remove('hidden');
            } else {
                element.classList.add('hidden');
            }
        }
    }

    // Form submission handlers
    async function createAIcon() {
        const name = document.getElementById('aicon-name').value.trim();
        const type = document.getElementById('aicon-type').value;
        
        if (!name) {
            showAlert('create-result', 'Please enter a name for your AIcon', 'error');
            return;
        }
        
        try {
            const response = await mockApiCall('/api/aicon/create', { name, type });
            
            // Update state
            aiconState.name = name;
            aiconState.type = type;
            
            // Update UI
            showAlert('create-result', `Successfully created AIcon "${name}" of type ${type}`, 'success');
            toggleVisibility('sensors-section', true);
            document.getElementById('status-indicator').textContent = `${name} (${type})`;
        } catch (error) {
            showAlert('create-result', `Error creating AIcon: ${error.message}`, 'error');
        }
    }

    async function addFactor(type) {
        if (!aiconState.name) {
            showAlert('prior-result', 'Please create an AIcon first', 'error');
            return;
        }
        
        try {
            const factorData = buildFactorData(type);
            const response = await mockApiCall('/api/factor/add', factorData);
            
            // Update state
            aiconState.factors[factorData.name] = factorData;
            
            // Update UI
            showAlert('prior-result', `Successfully added ${type} factor "${factorData.name}"`, 'success');
            updateFactorsList();
            clearFactorForm(type);
            
            // Ensure action section is visible
            toggleVisibility('action-section', true);
        } catch (error) {
            showAlert('prior-result', `Error adding factor: ${error.message}`, 'error');
        }
    }

    function buildFactorData(type) {
        let factorData = { type };
        
        if (type === 'continuous') {
            const name = getRequiredField('continuous-name', 'Factor name is required');
            factorData.name = name;
            factorData.initial_value = parseFloat(document.getElementById('continuous-initial').value) || 0.0;
            factorData.uncertainty = parseFloat(document.getElementById('continuous-uncertainty').value) || 1.0;
            
            // Optional constraints
            const lower = document.getElementById('continuous-lower').value;
            const upper = document.getElementById('continuous-upper').value;
            
            if (lower !== '') factorData.lower_bound = parseFloat(lower);
            if (upper !== '') factorData.upper_bound = parseFloat(upper);
        }
        else if (type === 'categorical') {
            const name = getRequiredField('categorical-name', 'Factor name is required');
            const valuesStr = getRequiredField('categorical-values', 'Possible values are required');
            
            const values = valuesStr.split(',').map(v => v.trim());
            const initialValue = document.getElementById('categorical-initial').value.trim();
            
            if (!values.includes(initialValue)) {
                throw new Error('Initial value must be one of the possible values');
            }
            
            factorData.name = name;
            factorData.possible_values = values;
            factorData.initial_value = initialValue;
            
            // Optional prior probabilities
            const probsStr = document.getElementById('categorical-probs').value.trim();
            if (probsStr) {
                const probs = probsStr.split(',').map(p => parseFloat(p.trim()));
                
                if (probs.length !== values.length) {
                    throw new Error('Number of probabilities must match number of possible values');
                }
                
                const sum = probs.reduce((a, b) => a + b, 0);
                if (Math.abs(sum - 1.0) > 0.001) {
                    throw new Error('Probabilities must sum to 1.0');
                }
                
                factorData.prior_probs = probs;
            }
        }
        else if (type === 'discrete') {
            const name = getRequiredField('discrete-name', 'Factor name is required');
            factorData.name = name;
            factorData.initial_value = parseInt(document.getElementById('discrete-initial').value) || 0;
            factorData.distribution = document.getElementById('discrete-distribution').value;
            
            if (factorData.distribution === 'poisson') {
                factorData.rate = parseFloat(document.getElementById('poisson-rate').value) || 1.0;
            } else if (factorData.distribution === 'binomial') {
                factorData.n = parseInt(document.getElementById('binomial-n').value) || 10;
                factorData.p = parseFloat(document.getElementById('binomial-p').value) || 0.5;
            }
        }
        else if (type === 'hierarchical') {
            const name = getRequiredField('hierarchical-name', 'Factor name is required');
            const parentsStr = getRequiredField('hierarchical-parents', 'Parent factors are required');
            
            const parents = parentsStr.split(',').map(p => p.trim());
            
            // Check if parent factors exist
            for (const parent of parents) {
                if (!aiconState.factors[parent]) {
                    throw new Error(`Parent factor "${parent}" does not exist`);
                }
            }
            
            const relation = document.getElementById('hierarchical-relation').value;
            
            let params;
            try {
                params = JSON.parse(document.getElementById('hierarchical-params').value);
            } catch (e) {
                throw new Error('Invalid JSON in parameters field');
            }
            
            factorData.name = name;
            factorData.parents = parents;
            factorData.relation_type = relation;
            factorData.parameters = params;
            factorData.initial_value = parseFloat(document.getElementById('hierarchical-initial').value) || 0.0;
            factorData.uncertainty = parseFloat(document.getElementById('hierarchical-uncertainty').value) || 1.0;
        }
        
        return factorData;
    }

    function getRequiredField(id, errorMessage) {
        const value = document.getElementById(id).value.trim();
        if (!value) {
            throw new Error(errorMessage);
        }
        return value;
    }

    function clearFactorForm(type) {
        if (type === 'continuous') {
            document.getElementById('continuous-name').value = '';
            document.getElementById('continuous-initial').value = '0.0';
            document.getElementById('continuous-lower').value = '';
            document.getElementById('continuous-upper').value = '';
        } else if (type === 'categorical') {
            document.getElementById('categorical-name').value = '';
            document.getElementById('categorical-values').value = '';
            document.getElementById('categorical-initial').value = '';
            document.getElementById('categorical-probs').value = '';
        } else if (type === 'discrete') {
            document.getElementById('discrete-name').value = '';
            document.getElementById('discrete-initial').value = '0';
        } else if (type === 'hierarchical') {
            document.getElementById('hierarchical-name').value = '';
            document.getElementById('hierarchical-parents').value = '';
            document.getElementById('hierarchical-params').value = '';
            document.getElementById('hierarchical-initial').value = '0.0';
        }
    }

    async function setActionSpace() {
        if (!aiconState.name) {
            showAlert('action-result', 'Please create an AIcon first', 'error');
            return;
        }
        
        try {
            const actionData = buildActionSpaceData();
            const response = await mockApiCall('/api/action/set', actionData);
            
            // Update state
            aiconState.actionSpace = actionData;
            
            // Update UI
            showAlert('action-result', `Successfully set action space of type "${actionData.type}" with utility "${actionData.utility}"`, 'success');
            toggleVisibility('run-section', true);
        } catch (error) {
            showAlert('action-result', `Error setting action space: ${error.message}`, 'error');
        }
    }

    function buildActionSpaceData() {
        const actionType = document.getElementById('action-space-type').value;
        const utilityType = document.getElementById('utility-function').value;
        
        let actionData = {
            type: actionType,
            utility: utilityType
        };
        
        if (actionType === 'budget_allocation') {
            const totalBudget = getRequiredField('budget-total', 'Total budget is required');
            const adIdsStr = getRequiredField('budget-ad-ids', 'Ad IDs are required');
            
            const adIds = adIdsStr.split(',').map(id => id.trim());
            
            actionData.total_budget = parseFloat(totalBudget);
            actionData.ad_ids = adIds;
            
            const minStep = document.getElementById('budget-min-step').value.trim();
            if (minStep) {
                actionData.min_step = parseFloat(minStep);
            }
        }
        else if (actionType === 'bidding') {
            const minBid = getRequiredField('bidding-min', 'Minimum bid is required');
            const maxBid = getRequiredField('bidding-max', 'Maximum bid is required');
            const keywordsStr = getRequiredField('bidding-keywords', 'Keywords are required');
            
            const keywords = keywordsStr.split(',').map(k => k.trim());
            
            actionData.min_bid = parseFloat(minBid);
            actionData.max_bid = parseFloat(maxBid);
            actionData.step_size = parseFloat(document.getElementById('bidding-step').value) || 0.1;
            actionData.keywords = keywords;
        }
        else if (actionType === 'custom') {
            const actionJson = getRequiredField('custom-action-json', 'Action space definition is required');
            
            try {
                const customDef = JSON.parse(actionJson);
                actionData.definition = customDef;
            } catch (e) {
                throw new Error('Invalid JSON in action space definition');
            }
        }
        
        // Handle utility function
        if (utilityType === 'custom') {
            const utilityCode = getRequiredField('custom-utility-code', 'Custom utility function code is required');
            actionData.utility_code = utilityCode;
        }
        
        return actionData;
    }

    async function runOptimization() {
        if (!aiconState.actionSpace) {
            showAlert('optimization-result', 'Please set an action space first', 'error');
            return;
        }
        
        try {
            // Update UI - show loading state
            const button = document.getElementById('run-optimization-btn');
            button.textContent = 'Running...';
            button.disabled = true;
            
            // Run optimization
            const numSamples = parseInt(document.getElementById('num-samples').value) || 500;
            const useGradient = document.getElementById('use-gradient').checked;
            
            const response = await mockApiCall('/api/optimize', {
                num_samples: numSamples,
                use_gradient: useGradient
            }, true);
            
            // Display results
            toggleVisibility('result-area', true);
            displayOptimizationResults(response, numSamples, useGradient);
            
            // Reset button state
            button.textContent = 'Find Best Action';
            button.disabled = false;
        } catch (error) {
            showAlert('optimization-result', `Error running optimization: ${error.message}`, 'error');
            
            // Reset button state
            document.getElementById('run-optimization-btn').textContent = 'Find Best Action';
            document.getElementById('run-optimization-btn').disabled = false;
        }
    }

    function displayOptimizationResults(response, numSamples, useGradient) {
        let resultHtml = '<div class="result-card">';
        
        if (aiconState.actionSpace.type === 'budget_allocation') {
            resultHtml += `<h4>Optimal Budget Allocation</h4>`;
            resultHtml += `<table class="result-table">
                <tr><th>Ad ID</th><th>Budget</th></tr>`;
            
            for (const [adId, budget] of Object.entries(response.best_action)) {
                resultHtml += `<tr><td>${adId}</td><td>$${budget.toFixed(2)}</td></tr>`;
            }
            
            resultHtml += `</table>`;
            resultHtml += `<p><strong>Expected Utility:</strong> ${response.expected_utility.toFixed(2)}</p>`;
            
            // Add visualization
            document.getElementById('visualization').innerHTML = `
                <h4>Budget Allocation Visualization</h4>
                <div class="chart-container">
                    <div class="chart-placeholder">
                        <svg width="400" height="200">
                            ${Object.entries(response.best_action).map(([adId, budget], i) => {
                                const width = (budget / aiconState.actionSpace.total_budget) * 380;
                                return `<rect x="10" y="${20 + i * 40}" width="${width}" height="30" fill="#3498db" />
                                       <text x="${width + 15}" y="${40 + i * 40}" fill="#333">${adId}: $${budget.toFixed(2)}</text>`;
                            }).join('')}
                        </svg>
                    </div>
                </div>
            `;
        } else if (aiconState.actionSpace.type === 'bidding') {
            resultHtml += `<h4>Optimal Bidding Strategy</h4>`;
            resultHtml += `<table class="result-table">
                <tr><th>Keyword</th><th>Bid Amount</th></tr>`;
            
            for (const [keyword, bid] of Object.entries(response.best_action)) {
                resultHtml += `<tr><td>${keyword}</td><td>$${bid.toFixed(2)}</td></tr>`;
            }
            
            resultHtml += `</table>`;
            resultHtml += `<p><strong>Expected Utility:</strong> ${response.expected_utility.toFixed(2)}</p>`;
        } else {
            resultHtml += `<h4>Optimal Action</h4>`;
            resultHtml += `<pre>${JSON.stringify(response.best_action, null, 2)}</pre>`;
            resultHtml += `<p><strong>Expected Utility:</strong> ${response.expected_utility.toFixed(2)}</p>`;
        }
        
        resultHtml += `<p><em>Optimization method: ${useGradient ? 'Gradient-based' : 'Random Sampling'}</em></p>`;
        resultHtml += `<p><em>Number of samples: ${numSamples}</em></p>`;
        resultHtml += '</div>';
        
        document.getElementById('optimization-result').innerHTML = resultHtml;
    }

    // Update the list of added factors
    function updateFactorsList() {
        const container = document.getElementById('factors-container');
        
        if (Object.keys(aiconState.factors).length === 0) {
            container.innerHTML = '<p>No factors added yet</p>';
            return;
        }
        
        let html = '';
        
        // Group factors by source
        const factorGroups = {
            'meta_ads': [],
            'google_ads': [],
            'web_analytics': [],
            'manual': []
        };
        
        // Sort factors into groups
        for (const [name, factor] of Object.entries(aiconState.factors)) {
            if (factor.auto_created && factor.source) {
                factorGroups[factor.source].push([name, factor]);
            } else {
                factorGroups['manual'].push([name, factor]);
            }
        }
        
        // First show manual factors
        if (factorGroups['manual'].length > 0) {
            html += `<h4 style="margin-top: 20px; border-bottom: 1px solid #ddd; padding-bottom: 8px;">Manually Added Factors</h4>`;
            
            for (const [name, factor] of factorGroups['manual']) {
                html += createFactorCard(name, factor);
            }
        }
        
        // Then show Meta Ads factors
        if (factorGroups['meta_ads'].length > 0) {
            html += `<h4 style="margin-top: 20px; border-bottom: 1px solid #ddd; padding-bottom: 8px;">Meta Ads Factors</h4>`;
            
            for (const [name, factor] of factorGroups['meta_ads']) {
                html += createFactorCard(name, factor);
            }
        }
        
        // Then show Google Ads factors
        if (factorGroups['google_ads'].length > 0) {
            html += `<h4 style="margin-top: 20px; border-bottom: 1px solid #ddd; padding-bottom: 8px;">Google Ads Factors</h4>`;
            
            for (const [name, factor] of factorGroups['google_ads']) {
                html += createFactorCard(name, factor);
            }
        }
        
        // Then show Web Analytics factors
        if (factorGroups['web_analytics'].length > 0) {
            html += `<h4 style="margin-top: 20px; border-bottom: 1px solid #ddd; padding-bottom: 8px;">Web Analytics Factors</h4>`;
            
            for (const [name, factor] of factorGroups['web_analytics']) {
                html += createFactorCard(name, factor);
            }
        }
        
        container.innerHTML = html;
    }
    
    function createFactorCard(name, factor) {
        let html = `<div class="card">`;
        html += `<h4 class="card-title">${name}`;
        
        // Add badge for auto-created factors
        if (factor.auto_created) {
            html += ` <span style="background-color: #e8f4ff; font-size: 12px; padding: 2px 6px; border-radius: 10px; color: #2980b9;">Sensor-created</span>`;
        }
        
        html += `</h4>`;
        html += `<p><strong>Type:</strong> ${factor.type}</p>`;
        
        if (factor.type === 'continuous') {
            html += `<p><strong>Initial Value:</strong> ${factor.initial_value}</p>`;
            html += `<p><strong>Uncertainty:</strong> ${factor.uncertainty}</p>`;
            
            if (factor.lower_bound !== undefined) {
                html += `<p><strong>Lower Bound:</strong> ${factor.lower_bound}</p>`;
            }
            
            if (factor.upper_bound !== undefined) {
                html += `<p><strong>Upper Bound:</strong> ${factor.upper_bound}</p>`;
            }
        }
        else if (factor.type === 'categorical') {
            html += `<p><strong>Initial Value:</strong> ${factor.initial_value}</p>`;
            html += `<p><strong>Possible Values:</strong> ${factor.possible_values.join(', ')}</p>`;
            
            if (factor.prior_probs) {
                html += `<p><strong>Prior Probabilities:</strong> ${factor.prior_probs.join(', ')}</p>`;
            }
        }
        else if (factor.type === 'discrete') {
            html += `<p><strong>Initial Value:</strong> ${factor.initial_value}</p>`;
            html += `<p><strong>Distribution:</strong> ${factor.distribution}</p>`;
            
            if (factor.distribution === 'poisson') {
                html += `<p><strong>Rate:</strong> ${factor.rate}</p>`;
            } else if (factor.distribution === 'binomial') {
                html += `<p><strong>n:</strong> ${factor.n}, <strong>p:</strong> ${factor.p}</p>`;
            }
        }
        else if (factor.type === 'hierarchical') {
            html += `<p><strong>Initial Value:</strong> ${factor.initial_value}</p>`;
            html += `<p><strong>Parents:</strong> ${factor.parents.join(', ')}</p>`;
            html += `<p><strong>Relation Type:</strong> ${factor.relation_type}</p>`;
        }
        
        html += `</div>`;
        return html;
    }

    // Update the list of added sensors
    function updateSensorsList() {
        const container = document.getElementById('sensors-container');
        
        if (Object.keys(aiconState.sensors).length === 0) {
            container.innerHTML = '<p>No data sources connected yet</p>';
            return;
        }
        
        let html = '';
        
        for (const [sensorType, sensor] of Object.entries(aiconState.sensors)) {
            html += `<div class="card">`;
            html += `<h4 class="card-title">${sensor.name}</h4>`;
            
            // Display some of the configuration info
            if (sensorType === 'meta_ads') {
                html += `<p><strong>Ad Account:</strong> ${sensor.config.ad_account_id}</p>`;
                html += `<p><strong>Campaign:</strong> ${sensor.config.campaign_id}</p>`;
            }
            
            // Count and list auto-created factors from this sensor
            const autoFactors = Object.entries(aiconState.factors)
                .filter(([_, factor]) => factor.auto_created && factor.source === sensorType)
                .map(([name, _]) => name);
            
            html += `<p><strong>Status:</strong> <span style="color: #28a745;">Connected</span></p>`;
            html += `<p><strong>Auto-created factors:</strong> ${autoFactors.length}</p>`;
            
            // Add a disconnect button if needed
            html += `<button class="disconnect-sensor-btn" data-sensor-type="${sensorType}" style="background-color: #dc3545; margin-top: 10px;">Disconnect</button>`;
            
            html += `</div>`;
        }
        
        container.innerHTML = html;
        
        // Add event listeners for disconnect buttons
        document.querySelectorAll('.disconnect-sensor-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const sensorType = e.target.getAttribute('data-sensor-type');
                disconnectSensor(sensorType);
            });
        });
    }

    // Show an alert message
    function showAlert(containerId, message, type = 'success') {
        const container = document.getElementById(containerId);
        container.textContent = message;
        container.className = `alert alert-${type}`;
        container.classList.remove('hidden');
        
        // Hide after 5 seconds
        setTimeout(() => {
            container.classList.add('hidden');
        }, 5000);
    }

    // Mock API call function
    async function mockApiCall(endpoint, data, delay = false) {
        console.log(`Mock API call to ${endpoint}:`, data);
        
        // Simulate API delay
        if (delay) {
            await new Promise(resolve => setTimeout(resolve, 2000));
        } else {
            await new Promise(resolve => setTimeout(resolve, 500));
        }
        
        // Simulate responses based on endpoint
        if (endpoint === '/api/aicon/create') {
            return { success: true, aicon_id: 'aicon_' + Date.now() };
        }
        else if (endpoint === '/api/factor/add') {
            return { success: true, factor_id: data.name };
        }
        else if (endpoint === '/api/optimize') {
            // Generate mock optimization results
            let bestAction = {};
            let expectedUtility = 0;
            
            if (aiconState.actionSpace.type === 'budget_allocation') {
                const adIds = aiconState.actionSpace.ad_ids;
                const totalBudget = aiconState.actionSpace.total_budget;
                
                // Create a random allocation that sums to totalBudget
                let remaining = totalBudget;
                
                for (let i = 0; i < adIds.length - 1; i++) {
                    const budget = Math.min(
                        remaining * Math.random() * 0.8, 
                        remaining - (adIds.length - i - 1)
                    );
                    bestAction[adIds[i]] = Math.round(budget * 100) / 100;
                    remaining -= bestAction[adIds[i]];
                }
                
                // Assign the remainder to the last ad
                bestAction[adIds[adIds.length - 1]] = Math.round(remaining * 100) / 100;
                
                // Generate a realistic expected utility (e.g., ROAS between 1.5 and 4.0)
                expectedUtility = totalBudget * (1.5 + Math.random() * 2.5);
            }
            else if (aiconState.actionSpace.type === 'bidding') {
                const keywords = aiconState.actionSpace.keywords;
                const minBid = aiconState.actionSpace.min_bid;
                const maxBid = aiconState.actionSpace.max_bid;
                
                // Create random bids within range
                for (const keyword of keywords) {
                    bestAction[keyword] = minBid + Math.random() * (maxBid - minBid);
                    bestAction[keyword] = Math.round(bestAction[keyword] * 100) / 100;
                }
                
                // Generate a realistic expected utility (e.g., clicks or conversions)
                expectedUtility = keywords.length * (10 + Math.random() * 90);
            }
            else {
                // For custom action spaces, generate a random vector
                const dimensions = aiconState.actionSpace.definition.dimensions || 2;
                
                for (let i = 0; i < dimensions; i++) {
                    bestAction[`dim_${i+1}`] = Math.random() * 10;
                }
                
                expectedUtility = 100 + Math.random() * 900;
            }
            
            return {
                success: true,
                best_action: bestAction,
                expected_utility: expectedUtility
            };
        }
        
        return { success: true };
    }

    // Predefined factor functions
    async function addPredefinedCustomerLTV() {
        if (!aiconState.name) {
            showAlert('prior-result', 'Please create an AIcon first', 'error');
            return;
        }
        
        try {
            const factorData = {
                type: 'continuous',
                name: 'customer_lifetime_value',
                initial_value: 250.0,  // $250 average LTV
                uncertainty: 50.0,
                lower_bound: 0.0
            };
            
            const response = await mockApiCall('/api/factor/add', factorData);
            
            // Update state
            aiconState.factors[factorData.name] = factorData;
            
            // Update UI
            showAlert('prior-result', `Successfully added Customer Lifetime Value factor`, 'success');
            updateFactorsList();
            
            // Ensure action section is visible 
            toggleVisibility('action-section', true);
        } catch (error) {
            showAlert('prior-result', `Error adding factor: ${error.message}`, 'error');
        }
    }
    
    async function addPredefinedSeasonality() {
        if (!aiconState.name) {
            showAlert('prior-result', 'Please create an AIcon first', 'error');
            return;
        }
        
        try {
            const factorData = {
                type: 'continuous',
                name: 'seasonality_multiplier',
                initial_value: 1.0,  // No effect by default
                uncertainty: 0.2,
                lower_bound: 0.5,
                upper_bound: 2.0
            };
            
            const response = await mockApiCall('/api/factor/add', factorData);
            
            // Update state
            aiconState.factors[factorData.name] = factorData;
            
            // Update UI
            showAlert('prior-result', `Successfully added Seasonality Effect factor`, 'success');
            updateFactorsList();
            
            // Ensure action section is visible
            toggleVisibility('action-section', true);
        } catch (error) {
            showAlert('prior-result', `Error adding factor: ${error.message}`, 'error');
        }
    }
    
    async function addPredefinedAdFatigue() {
        if (!aiconState.name) {
            showAlert('prior-result', 'Please create an AIcon first', 'error');
            return;
        }
        
        try {
            const factorData = {
                type: 'continuous',
                name: 'ad_fatigue_rate',
                initial_value: 0.05,  // 5% decrease per time period
                uncertainty: 0.02,
                lower_bound: 0.0,
                upper_bound: 1.0
            };
            
            const response = await mockApiCall('/api/factor/add', factorData);
            
            // Update state
            aiconState.factors[factorData.name] = factorData;
            
            // Update UI
            showAlert('prior-result', `Successfully added Ad Fatigue Rate factor`, 'success');
            updateFactorsList();
            
            // Ensure action section is visible
            toggleVisibility('action-section', true);
        } catch (error) {
            showAlert('prior-result', `Error adding factor: ${error.message}`, 'error');
        }
    }
    
    async function addPredefinedCrossCampaign() {
        if (!aiconState.name) {
            showAlert('prior-result', 'Please create an AIcon first', 'error');
            return;
        }
        
        try {
            const factorData = {
                type: 'continuous',
                name: 'cross_campaign_effect',
                initial_value: 0.1,  // 10% lift from cross-campaign effects
                uncertainty: 0.05,
                lower_bound: -0.5,
                upper_bound: 0.5
            };
            
            const response = await mockApiCall('/api/factor/add', factorData);
            
            // Update state
            aiconState.factors[factorData.name] = factorData;
            
            // Update UI
            showAlert('prior-result', `Successfully added Cross-Campaign Effect factor`, 'success');
            updateFactorsList();
            
            // Ensure action section is visible
            toggleVisibility('action-section', true);
        } catch (error) {
            showAlert('prior-result', `Error adding factor: ${error.message}`, 'error');
        }
    }
    
    async function addPredefinedCompetitorIntensity() {
        if (!aiconState.name) {
            showAlert('prior-result', 'Please create an AIcon first', 'error');
            return;
        }
        
        try {
            const factorData = {
                type: 'continuous',
                name: 'competitor_intensity',
                initial_value: 0.5,  // Medium competition
                uncertainty: 0.2,
                lower_bound: 0.0,
                upper_bound: 1.0
            };
            
            const response = await mockApiCall('/api/factor/add', factorData);
            
            // Update state
            aiconState.factors[factorData.name] = factorData;
            
            // Update UI
            showAlert('prior-result', `Successfully added Competitor Intensity factor`, 'success');
            updateFactorsList();
            
            // Ensure action section is visible
            toggleVisibility('action-section', true);
        } catch (error) {
            showAlert('prior-result', `Error adding factor: ${error.message}`, 'error');
        }
    }

    // Sensor functions - updated to show/hide config instead of creating sensors
    function toggleMetaAdsConfig(event) {
        const isChecked = event.target.checked;
        const configSection = document.getElementById('meta-ads-config');
        
        if (isChecked) {
            configSection.style.display = 'block';
        } else {
            configSection.style.display = 'none';
        }
    }
    
    // When the connect button is clicked, create the sensor with provided credentials
    async function connectMetaAdsSensor() {
        if (!aiconState.name) {
            showAlert('sensor-result', 'Please create an AIcon first', 'error');
            return;
        }
        
        // Get the configuration values
        const accessToken = document.getElementById('meta-access-token').value.trim();
        const adAccountId = document.getElementById('meta-ad-account').value.trim();
        const campaignId = document.getElementById('meta-campaign-id').value.trim();
        
        // Validate inputs
        if (!accessToken || !adAccountId || !campaignId) {
            showAlert('sensor-result', 'Please fill in all required fields for Meta Ads', 'error');
            return;
        }
        
        try {
            // Create a sensor configuration object
            const sensorConfig = {
                type: 'meta_ads',
                access_token: accessToken,
                ad_account_id: adAccountId,
                campaign_id: campaignId
            };
            
            // Make the API call to add the sensor
            const response = await mockApiCall('/api/sensor/add', sensorConfig);
            
            // Update state with the sensor
            addSensorToState('meta_ads', 'Meta Ads', sensorConfig);
            
            // Show success message
            showAlert('sensor-result', 'Meta Ads sensor connected! Performance factors have been auto-created based on your account data.', 'success');
            
            // Update UI
            updateSensorsList();
            updateFactorsList();
            
            // Disable the form inputs and connect button to prevent multiple connections
            document.getElementById('meta-access-token').disabled = true;
            document.getElementById('meta-ad-account').disabled = true;
            document.getElementById('meta-campaign-id').disabled = true;
            document.getElementById('connect-meta-ads-btn').disabled = true;
            document.getElementById('connect-meta-ads-btn').textContent = 'Connected';
            document.getElementById('connect-meta-ads-btn').style.backgroundColor = '#28a745';
            
        } catch (error) {
            showAlert('sensor-result', `Error connecting Meta Ads sensor: ${error.message}`, 'error');
        }
    }

    // Updated addSensorToState to include configuration
    function addSensorToState(sensorType, sensorDisplayName, config) {
        // Create the sensor in the state
        aiconState.sensors[sensorType] = {
            type: sensorType,
            name: sensorDisplayName,
            connected: true,
            config: config
        };
        
        // Create auto factors based on sensor type
        if (sensorType === 'meta_ads') {
            // In a real implementation, these would be fetched from the Meta Ads API
            // For now, we'll create some mock ads based on the campaign ID
            const campaignId = config.campaign_id;
            const adIds = [`${campaignId}_ad1`, `${campaignId}_ad2`, `${campaignId}_ad3`];
            
            for (const adId of adIds) {
                // Add conversion rate factors
                aiconState.factors[`conversion_rate_${adId}`] = {
                    type: 'continuous',
                    name: `conversion_rate_${adId}`,
                    initial_value: 0.02 + (Math.random() * 0.02),
                    uncertainty: 0.005,
                    lower_bound: 0.0,
                    upper_bound: 1.0,
                    auto_created: true,
                    source: 'meta_ads'
                };
                
                // Add CPC factors
                aiconState.factors[`cost_per_click_${adId}`] = {
                    type: 'continuous',
                    name: `cost_per_click_${adId}`,
                    initial_value: 1.0 + (Math.random() * 0.5),
                    uncertainty: 0.2,
                    lower_bound: 0.01,
                    auto_created: true,
                    source: 'meta_ads'
                };
                
                // Add CTR factors
                aiconState.factors[`click_through_rate_${adId}`] = {
                    type: 'continuous',
                    name: `click_through_rate_${adId}`,
                    initial_value: 0.01 + (Math.random() * 0.02),
                    uncertainty: 0.005,
                    lower_bound: 0.0,
                    upper_bound: 1.0,
                    auto_created: true,
                    source: 'meta_ads'
                };
            }
        }
        
        // We're hiding Google Ads and Analytics for now, so their code is omitted
    }
    
    // Function to disconnect a sensor
    function disconnectSensor(sensorType) {
        if (aiconState.sensors[sensorType]) {
            // Remove sensor from state
            delete aiconState.sensors[sensorType];
            
            // Remove auto-created factors
            Object.keys(aiconState.factors).forEach(key => {
                if (aiconState.factors[key].auto_created && aiconState.factors[key].source === sensorType) {
                    delete aiconState.factors[key];
                }
            });
            
            // Show success message
            showAlert('sensor-result', `${sensorType.charAt(0).toUpperCase() + sensorType.slice(1).replace('_', ' ')} sensor disconnected.`, 'success');
            
            // Reset the toggle and form
            if (sensorType === 'meta_ads') {
                document.getElementById('meta-ads-toggle').checked = false;
                document.getElementById('meta-ads-config').style.display = 'none';
                document.getElementById('meta-access-token').disabled = false;
                document.getElementById('meta-ad-account').disabled = false;
                document.getElementById('meta-campaign-id').disabled = false;
                document.getElementById('connect-meta-ads-btn').disabled = false;
                document.getElementById('connect-meta-ads-btn').textContent = 'Connect';
                document.getElementById('connect-meta-ads-btn').style.backgroundColor = '';
                
                // Clear input values
                document.getElementById('meta-access-token').value = '';
                document.getElementById('meta-ad-account').value = '';
                document.getElementById('meta-campaign-id').value = '';
            }
            
            // Update UI
            updateSensorsList();
            updateFactorsList();
        }
    }

    function continueAfterSensors() {
        // Show the factors section after configuring sensors
        toggleVisibility('priors-section', true);
        
        // If there are no sensors connected, show an information message
        if (Object.keys(aiconState.sensors).length === 0) {
            showAlert('prior-result', 'No data sources connected. You may want to go back and connect some sensors to auto-create relevant factors.', 'error');
        } else {
            showAlert('prior-result', `${Object.keys(aiconState.sensors).length} data source(s) connected. Auto-created factors are now available.`, 'success');
        }
        
        // Also show the action section 
        toggleVisibility('action-section', true);
    }
})(); 