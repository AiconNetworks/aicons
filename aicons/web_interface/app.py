from flask import Flask, jsonify, request, render_template
from bayesbrainGPT.state_representation.state import EnvironmentState
from typing import List, Dict, Any
import json

app = Flask(__name__)

class PromptChainManager:
    def __init__(self):
        self.state = EnvironmentState(use_llm=True)
        self.chain_steps: List[Dict[str, Any]] = []
    
    def add_step(self, prompt: str, response: str, metadata: Dict[str, Any] = None):
        step = {
            "prompt": prompt,
            "response": response,
            "metadata": metadata or {},
            "step_number": len(self.chain_steps) + 1
        }
        self.chain_steps.append(step)
        return step

    def get_chain_history(self):
        return self.chain_steps

chain_manager = PromptChainManager()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/state', methods=['GET'])
def get_state():
    return jsonify(chain_manager.state.get_state())

@app.route('/api/chain', methods=['POST'])
def add_chain_step():
    data = request.json
    step = chain_manager.add_step(
        prompt=data['prompt'],
        response=data['response'],
        metadata=data.get('metadata')
    )
    return jsonify(step)

@app.route('/api/chain', methods=['GET'])
def get_chain():
    return jsonify(chain_manager.get_chain_history())

if __name__ == '__main__':
    app.run(debug=True) 