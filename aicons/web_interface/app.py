from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

# In-memory storage for state and chat history
current_state = {}
chat_history = []

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
    
    # Here you would typically process the message with your AI system
    # For now, we'll just echo it back
    response = {
        'response': f"Received your message: {message}",
        'new_state': current_state
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000) 