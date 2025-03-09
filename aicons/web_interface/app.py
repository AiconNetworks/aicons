from flask import Flask, render_template, jsonify, request
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from aicons.bayesbrainGPT.state_representation.factors import (
    ContinuousFactor, 
    CategoricalFactor, 
    BayesianLinearFactor
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

# Example initial state with some mock factors
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

@app.route('/')
def index():
    return render_template('index.html')

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