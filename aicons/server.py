"""
Chat Server for ZeroAIcon

A simple Flask server that provides a chat interface for interacting with ZeroAIcon.
"""

from flask import Flask, request, jsonify, render_template, Response, url_for
import json
import asyncio
import os
import nest_asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv
import logging

# Import ZeroAIcon
from aicons.definitions.zero import ZeroAIcon

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Apply nest_asyncio to allow nested asyncio event loops
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Create Flask app with static folder
current_dir = os.path.dirname(os.path.abspath(__file__))
static_folder = os.path.join(current_dir, 'static')
template_folder = os.path.join(current_dir, 'templates')

app = Flask(__name__, 
            static_folder=static_folder,
            template_folder=template_folder)

# Create ZeroAIcon
aicon = ZeroAIcon(name="chat_aicon", description="Chat AIcon", model_name="deepseek-r1:7b")

# Store chat history
chat_history = []

@app.route('/')
def index():
    """Render the chat interface."""
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages and get responses from ZeroAIcon."""
    try:
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Add message to chat history
        chat_history.append({'role': 'user', 'content': message})
        
        # Get response from ZeroAIcon
        # We need to run the async function in the current event loop
        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(aicon.make_inference(message))
        
        # Add response to chat history
        chat_history.append({'role': 'assistant', 'content': response})
        
        return jsonify({
            'response': response,
            'history': chat_history
        })
    
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/stream-chat', methods=['POST'])
def stream_chat():
    """Stream chat responses from ZeroAIcon as they are generated."""
    try:
        data = request.json
        message = data.get('message', '')
        show_thinking = data.get('show_thinking', False)
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Add message to chat history
        chat_history.append({'role': 'user', 'content': message})
        
        def generate():
            # We need to run the async function in the current event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Create a custom async generator for streaming
                async def stream_inference():
                    full_response = ""
                    accumulated_chunk = ""
                    
                    # Get the context window
                    state_repr = aicon.get_state_representation()
                    context = {
                        "state": state_repr,
                        "utility_function": str(aicon.brain.utility_function) if aicon.brain.utility_function else None,
                        "action_space": str(aicon.brain.action_space),
                        "prompt": message
                    }
                    context_str = json.dumps(context)
                    
                    # Log the start of inference
                    logger.info(f"Streaming inference with prompt: {message[:50]}...")
                    yield f"data: {json.dumps({'chunk': '✓ Starting inference...', 'done': False})}\n\n"
                    
                    # Stream directly from the LLM
                    async for chunk in aicon.llm.generate(context_str):
                        full_response += chunk
                        accumulated_chunk += chunk
                        
                        # Print meaningful segments when we get newlines or periods
                        if '\n' in accumulated_chunk or len(accumulated_chunk) > 50 or '.' in accumulated_chunk:
                            logger.info(f"Message segment: {accumulated_chunk}")
                            # Stream to client
                            yield f"data: {json.dumps({'chunk': accumulated_chunk, 'done': False})}\n\n"
                            accumulated_chunk = ""
                        
                        # Log progress periodically
                        if len(full_response) % 200 == 0:
                            logger.info(f"Received {len(full_response)} characters so far")
                    
                    # Send any remaining accumulated chunk
                    if accumulated_chunk:
                        logger.info(f"Final segment: {accumulated_chunk}")
                        yield f"data: {json.dumps({'chunk': accumulated_chunk, 'done': False})}\n\n"
                    
                    # Process the response to remove the thinking part if not showing thinking
                    processed_response = full_response
                    
                    # Only remove thinking process if show_thinking is False
                    if not show_thinking:
                        # Remove the <think>...</think> section if present
                        think_start = processed_response.find("<think>")
                        think_end = processed_response.find("</think>")
                        
                        if think_start != -1 and think_end != -1 and think_end > think_start:
                            # Extract the content before <think> and after </think>
                            before_think = processed_response[:think_start].strip()
                            after_think = processed_response[think_end + 8:].strip()  # 8 is the length of "</think>"
                            
                            # Combine the parts, with space in between if both exist
                            if before_think and after_think:
                                processed_response = before_think + " " + after_think
                            else:
                                processed_response = before_think + after_think
                            
                            logger.info(f"Removed thinking process. Final response: {processed_response}")
                    else:
                        # If showing thinking, we keep the tags intact for client-side formatting
                        # Just log that we're preserving the thinking process
                        if "<think>" in processed_response and "</think>" in processed_response:
                            logger.info("Preserving thinking tags in response for client-side formatting")
                        else:
                            # If using the separator format, just replace the tag with a visual separator for readability
                            processed_response = processed_response.replace("<think>", "\n----- THINKING PROCESS -----\n")
                            processed_response = processed_response.replace("</think>", "\n----- END THINKING -----\n")
                            logger.info("Preserving thinking process in response with separators")
                    
                    # Add final response to chat history
                    chat_history.append({'role': 'assistant', 'content': processed_response.strip()})
                    
                    # Send completion message
                    yield f"data: {json.dumps({'chunk': '', 'done': True, 'full_response': processed_response.strip()})}\n\n"
                
                # Run the streaming
                async_gen = stream_inference()
                while True:
                    try:
                        chunk = loop.run_until_complete(async_gen.__anext__())
                        yield chunk
                    except StopAsyncIteration:
                        break
                
            except Exception as e:
                logger.error(f"Error in streaming: {str(e)}", exc_info=True)
                yield f"data: {json.dumps({'chunk': f'Error: {str(e)}', 'done': True})}\n\n"
            finally:
                loop.close()
        
        return Response(generate(), mimetype='text/event-stream')
        
    except Exception as e:
        logger.error(f"Error in stream chat: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get chat history."""
    return jsonify({'history': chat_history})

@app.route('/api/clear', methods=['POST'])
def clear_history():
    """Clear chat history."""
    global chat_history
    chat_history = []
    return jsonify({'status': 'success'})

def run_server(host='0.0.0.0', port=8000, debug=False):
    """Run the Flask server."""
    logger.info(f"Starting ZeroAIcon Chat Server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    # Simple command line arguments
    parser = argparse.ArgumentParser(description="ZeroAIcon Chat Server")
    parser.add_argument('--host', default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Run the server
    run_server(host=args.host, port=args.port, debug=args.debug) 