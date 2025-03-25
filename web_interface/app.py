from flask import Flask, render_template, jsonify, request
import sys
import os
import uuid
import time
import random

# Simple path fix - just add the parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Correct import from latent_variables.py
from aicons.bayesbrainGPT.state_representation.latent_variables import (
    ContinuousLatentVariable, 
    CategoricalLatentVariable, 
    HierarchicalLatentVariable
)

app = Flask(__name__)
app.config['DEBUG'] = True

# Add CORS support
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

# Legacy state - kept for backward compatibility
current_state = {
    'factors': {
        'temperature': {
            'type': 'continuous',
            'value': 20.0,
            'description': 'Temperature in Celsius',
            'uncertainty': 1.0
        },
        'weather': {
            'type': 'categorical',
            'value': 'sunny',
            'description': 'Weather condition',
            'possible_values': ['sunny', 'rainy', 'cloudy']
        },
        'traffic_density': {
            'type': 'bayesian_linear',
            'value': None,
            'description': 'Traffic density prediction',
            'theta_prior': {
                'intercept': {'mean': 0.0, 'variance': 1.0},
                'weather_effect': {'mean': 0.5, 'variance': 0.1}
            },
            'relationships': {
                'depends_on': ['weather'],
                'model': {
                    'type': 'linear',
                    'parameters': {'base': 2.0, 'coefficient': 1.5}
                }
            }
        }
    }
}

# Application state for new interface
app_state = {
    'aicons': {},
    'current_aicon': None
}

@app.route('/')
def index():
    return render_template('index.html')

# New API endpoints for the updated interface
@app.route('/api/aicon/create', methods=['POST'])
def create_aicon():
    data = request.json
    aicon_id = f"aicon_{uuid.uuid4().hex[:8]}"
    
    aicon = {
        'id': aicon_id,
        'name': data.get('name', 'Unnamed AIcon'),
        'type': data.get('type', 'SimpleBadAIcon'),
        'factors': {},
        'sensors': {},
        'action_space': None,
        'created_at': time.time()
    }
    
    app_state['aicons'][aicon_id] = aicon
    app_state['current_aicon'] = aicon_id
    
    return jsonify({'success': True, 'aicon_id': aicon_id})

@app.route('/api/factor/add', methods=['POST'])
def add_factor():
    data = request.json
    
    if not app_state['current_aicon']:
        return jsonify({'success': False, 'error': 'No AIcon selected'}), 400
    
    aicon = app_state['aicons'][app_state['current_aicon']]
    factor_name = data.get('name')
    
    if not factor_name:
        return jsonify({'success': False, 'error': 'Factor name is required'}), 400
    
    if factor_name in aicon['factors']:
        return jsonify({'success': False, 'error': f'Factor {factor_name} already exists'}), 400
    
    # Store the factor data
    aicon['factors'][factor_name] = data
    
    return jsonify({'success': True, 'factor_id': factor_name})

@app.route('/api/sensor/add', methods=['POST'])
def add_sensor():
    data = request.json
    
    if not app_state['current_aicon']:
        return jsonify({'success': False, 'error': 'No AIcon selected'}), 400
    
    aicon = app_state['aicons'][app_state['current_aicon']]
    sensor_name = data.get('name')
    
    if not sensor_name:
        return jsonify({'success': False, 'error': 'Sensor name is required'}), 400
    
    if sensor_name in aicon['sensors']:
        return jsonify({'success': False, 'error': f'Sensor {sensor_name} already exists'}), 400
    
    # Store the sensor data
    aicon['sensors'][sensor_name] = data
    
    return jsonify({'success': True, 'sensor_id': sensor_name})

@app.route('/api/action/set', methods=['POST'])
def set_action_space():
    data = request.json
    
    if not app_state['current_aicon']:
        return jsonify({'success': False, 'error': 'No AIcon selected'}), 400
    
    aicon = app_state['aicons'][app_state['current_aicon']]
    
    # Store the action space data
    aicon['action_space'] = data
    
    return jsonify({'success': True})

@app.route('/api/optimize', methods=['POST'])
def optimize():
    data = request.json
    
    if not app_state['current_aicon']:
        return jsonify({'success': False, 'error': 'No AIcon selected'}), 400
    
    aicon = app_state['aicons'][app_state['current_aicon']]
    
    if not aicon['action_space']:
        return jsonify({'success': False, 'error': 'No action space defined'}), 400
    
    num_samples = data.get('num_samples', 500)
    use_gradient = data.get('use_gradient', True)
    
    # In a real implementation, this would use the AIcon to find the best action
    # For now, generate mock optimization results
    action_space = aicon['action_space']
    best_action = {}
    
    if action_space['type'] == 'budget_allocation':
        adIds = action_space['ad_ids']
        totalBudget = action_space['total_budget']
        
        # Create a random allocation that sums to totalBudget
        remaining = totalBudget
        
        for i in range(len(adIds) - 1):
            budget = min(
                remaining * random.random() * 0.8,
                remaining - (len(adIds) - i - 1)
            )
            best_action[adIds[i]] = round(budget * 100) / 100
            remaining -= best_action[adIds[i]]
        
        # Assign the remainder to the last ad
        best_action[adIds[-1]] = round(remaining * 100) / 100
        
        # Generate a realistic expected utility (e.g., ROAS between 1.5 and 4.0)
        expected_utility = totalBudget * (1.5 + random.random() * 2.5)
    
    elif action_space['type'] == 'bidding':
        keywords = action_space['keywords']
        minBid = action_space['min_bid']
        maxBid = action_space['max_bid']
        
        # Create random bids within range
        for keyword in keywords:
            best_action[keyword] = minBid + random.random() * (maxBid - minBid)
            best_action[keyword] = round(best_action[keyword] * 100) / 100
        
        # Generate a realistic expected utility (e.g., clicks or conversions)
        expected_utility = len(keywords) * (10 + random.random() * 90)
    
    else:
        # For custom action spaces, generate a random vector
        dimensions = action_space.get('definition', {}).get('dimensions', 2)
        
        for i in range(dimensions):
            best_action[f"dim_{i+1}"] = random.random() * 10
        
        expected_utility = 100 + random.random() * 900
    
    # If using gradient, slightly improve the result to show it's better
    if use_gradient:
        expected_utility *= 1.2
    
    return jsonify({
        'success': True,
        'best_action': best_action,
        'expected_utility': expected_utility
    })

# Legacy endpoints - kept for backward compatibility
@app.route('/api/state', methods=['GET'])
def get_state():
    return jsonify(current_state)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    
    # Mock response - in reality, this would process the message and update factors
    response = {
        'response': f"I understand you want to discuss the factors. Currently, we have defined temperature, weather, and traffic density. Would you like to modify any of these or define new factors?",
        'new_state': current_state
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True) 