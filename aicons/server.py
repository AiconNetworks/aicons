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

# Import tools
from aicons.tools.speak_out_loud import SpeakOutLoudTool
from aicons.tools.ask_question import AskQuestionTool

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

# Store AIcon instances and their chat histories
aicons = {
    "default": {
        "instance": ZeroAIcon(name="default", description="Default AIcon", model_name="deepseek-r1:7b"),
        "chat_history": []
    }
}

# Current active AIcon - declare as global at the top level
current_aicon = "default"

@app.route('/')
def index():
    """Render the chat interface."""
    return render_template('chat.html')

@app.route('/api/aicons', methods=['GET'])
def get_aicons():
    """Get list of available AIcons."""
    return jsonify({
        "aicons": list(aicons.keys()),
        "current": current_aicon
    })

@app.route('/api/aicons', methods=['POST'])
def create_aicon():
    """Create a new AIcon instance."""
    try:
        data = request.json
        name = data.get('name')
        description = data.get('description', '')
        model_name = data.get('model_name', 'deepseek-r1:7b')
        
        if not name:
            return jsonify({'error': 'Name is required'}), 400
            
        if name in aicons:
            return jsonify({'error': 'AIcon with this name already exists'}), 400
            
        # Create the AIcon instance
        aicon = ZeroAIcon(name=name, description=description, model_name=model_name)
        
        # Add tools to the AIcon
        speak_tool = SpeakOutLoudTool()
        ask_tool = AskQuestionTool()
        aicon.add_tool(speak_tool)
        aicon.add_tool(ask_tool)
            
        aicons[name] = {
            "instance": aicon,
            "chat_history": []
        }
        
        return jsonify({'success': True, 'name': name})
        
    except Exception as e:
        logger.error(f"Error creating AIcon: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/aicons/<name>', methods=['DELETE'])
def delete_aicon(name):
    """Delete an AIcon instance."""
    try:
        if name not in aicons:
            return jsonify({'error': 'AIcon not found'}), 404
            
        if name == "default":
            return jsonify({'error': 'Cannot delete default AIcon'}), 400
            
        del aicons[name]
        
        # If we deleted the current AIcon, switch to default
        if name == current_aicon:
            current_aicon = "default"
            
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"Error deleting AIcon: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/aicons/current', methods=['GET'])
def get_current_aicon():
    """Get the current active AIcon."""
    return jsonify({'current': current_aicon})

@app.route('/api/aicons/current', methods=['POST'])
def set_current_aicon():
    """Set the current active AIcon."""
    try:
        data = request.json
        name = data.get('name')
        
        if not name:
            return jsonify({'error': 'Name is required'}), 400
            
        if name not in aicons:
            return jsonify({'error': 'AIcon not found'}), 404
            
        current_aicon = name
        
        return jsonify({'success': True, 'current': current_aicon})
        
    except Exception as e:
        logger.error(f"Error setting current AIcon: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages and get responses from ZeroAIcon."""
    try:
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Add message to chat history
        aicons[current_aicon]["chat_history"].append({'role': 'user', 'content': message})
        
        # Get response from ZeroAIcon
        # We need to run the async function in the current event loop
        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(aicons[current_aicon]["instance"].make_inference(message))
        
        # Add response to chat history
        aicons[current_aicon]["chat_history"].append({'role': 'assistant', 'content': response})
        
        return jsonify({
            'response': response,
            'history': aicons[current_aicon]["chat_history"]
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
        aicons[current_aicon]["chat_history"].append({'role': 'user', 'content': message})
        
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
                    state_repr = aicons[current_aicon]["instance"].get_state_representation()
                    context = {
                        "state": state_repr,
                        "utility_function": str(aicons[current_aicon]["instance"].brain.utility_function) if aicons[current_aicon]["instance"].brain.utility_function else None,
                        "action_space": str(aicons[current_aicon]["instance"].brain.action_space),
                        "prompt": message
                    }
                    context_str = json.dumps(context)
                    
                    # Log the start of inference
                    logger.info(f"Streaming inference with prompt: {message[:50]}...")
                    yield f"data: {json.dumps({'chunk': '✓ Starting inference...', 'done': False})}\n\n"
                    
                    # Stream directly from the LLM
                    async for chunk in aicons[current_aicon]["instance"].llm.generate(context_str):
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
                    aicons[current_aicon]["chat_history"].append({'role': 'assistant', 'content': processed_response.strip()})
                    
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
    return jsonify({'history': aicons[current_aicon]["chat_history"]})

@app.route('/api/clear', methods=['POST'])
def clear_history():
    """Clear chat history."""
    global aicons
    aicons[current_aicon]["chat_history"] = []
    return jsonify({'status': 'success'})

@app.route('/api/token-usage', methods=['GET'])
def get_token_usage():
    """Get token usage from ZeroAIcon."""
    try:
        # Get token usage report with updated data
        usage_report = aicons[current_aicon]["instance"].get_token_usage_report()
        
        # For debugging
        logger.info(f"Token usage report: total_used={usage_report['total_used']}, remaining={usage_report['remaining']}")
        
        # Return as JSON
        return jsonify(usage_report)
    
    except Exception as e:
        logger.error(f"Error getting token usage: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/configuration', methods=['GET'])
def get_configuration():
    """Get the current configuration of ZeroAIcon."""
    try:
        # Get sensors
        sensors = []
        aicon_instance = aicons[current_aicon]["instance"]
        
        # Log AIcon instance details
        logger.info(f"Getting configuration for AIcon: {current_aicon}")
        logger.info(f"AIcon instance: {aicon_instance}")
        logger.info(f"Has brain: {hasattr(aicon_instance, 'brain')}")
        
        # Check if the AIcon has sensors
        if hasattr(aicon_instance, 'brain') and hasattr(aicon_instance.brain, 'sensors'):
            logger.info(f"Has sensors: {aicon_instance.brain.sensors}")
            for sensor in aicon_instance.brain.sensors:
                logger.info(f"Found sensor: {sensor.__class__.__name__}")
                sensors.append({
                    'name': sensor.__class__.__name__,
                    'sensor_type': sensor.__class__.__name__,
                    'reliability': getattr(sensor, 'reliability', None)
                })
        
        # Log final sensors list
        logger.info(f"Final sensors list: {sensors}")
        
        # Get state factors with ALL properties
        state_factors = []
        if hasattr(aicon_instance.brain, 'state') and hasattr(aicon_instance.brain.state, 'get_state_factors'):
            factors = aicon_instance.brain.state.get_state_factors()
            for name, factor in factors.items():
                # Get factor type
                factor_type = factor.__class__.__name__.replace('LatentVariable', '').lower()
                
                # Create the basic factor data with complete information
                factor_data = {
                    'name': name,
                    'factor_type': factor_type,
                    'value': factor.value if hasattr(factor, 'value') else factor.get('value', 0),
                    'params': {},
                    'relationships': {'depends_on': []}
                }
                
                # Add type-specific parameters - ensure all required params are included
                if factor_type == 'continuous':
                    factor_data['params']['loc'] = factor.loc if hasattr(factor, 'loc') else 0
                    factor_data['params']['scale'] = factor.scale if hasattr(factor, 'scale') else 1
                    
                    # Add constraints if they exist
                    if hasattr(factor, 'constraints') and factor.constraints:
                        factor_data['params']['constraints'] = factor.constraints
                
                elif factor_type == 'categorical':
                    factor_data['params']['categories'] = factor.categories if hasattr(factor, 'categories') else []
                    factor_data['params']['probs'] = factor.probs if hasattr(factor, 'probs') else []
                
                elif factor_type == 'discrete':
                    factor_data['params']['rate'] = factor.rate if hasattr(factor, 'rate') else 5
                
                # Get relationships
                if hasattr(factor, 'parents') and factor.parents:
                    factor_data['relationships']['depends_on'] = [parent.name for parent in factor.parents]
                
                state_factors.append(factor_data)
        
        # Get action space
        action_space = None
        if aicon_instance.brain.action_space:
            action_space = {
                'space_type': 'custom',  # Default
                'dimensions': len(aicon_instance.brain.action_space.dimensions) if hasattr(aicon_instance.brain.action_space, 'dimensions') else 0,
                'description': str(aicon_instance.brain.action_space)
            }
        
        # Get utility function
        utility_function = None
        if aicon_instance.brain.utility_function:
            utility_function = {
                'utility_type': 'custom',  # Default
                'description': str(aicon_instance.brain.utility_function)
            }
        
        return jsonify({
            'sensors': sensors,
            'state_factors': state_factors,
            'action_space': action_space,
            'utility_function': utility_function
        })
    
    except Exception as e:
        logger.error(f"Error getting configuration: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/add-sensor', methods=['POST'])
def add_sensor():
    """Add a sensor to ZeroAIcon."""
    try:
        data = request.json
        sensor_type = data.get('sensor_type')
        name = data.get('name')
        
        if not sensor_type or not name:
            return jsonify({'success': False, 'error': 'Sensor type and name are required'}), 400
        
        # Create the appropriate sensor based on type
        if sensor_type == 'meta_ads':
            from aicons.bayesbrainGPT.sensors.meta_s.meta_ads_sales_sensor import MetaAdsSalesSensor
            
            # Get required parameters
            access_token = data.get('access_token')
            ad_account_id = data.get('ad_account_id')
            campaign_id = data.get('campaign_id')
            api_version = data.get('api_version', 'v18.0')
            time_granularity = data.get('time_granularity', 'hour')
            reliability = data.get('reliability', 0.9)
            
            if not access_token or not ad_account_id or not campaign_id:
                return jsonify({
                    'success': False, 
                    'error': 'Meta Ads sensor requires access_token, ad_account_id, and campaign_id'
                }), 400
            
            # Create the sensor
            sensor = MetaAdsSalesSensor(
                name=name,
                reliability=reliability,
                access_token=access_token,
                ad_account_id=ad_account_id,
                campaign_id=campaign_id,
                api_version=api_version,
                time_granularity=time_granularity
            )
            
            # Add the sensor to AIcon
            aicons[current_aicon]["instance"].add_sensor(name, sensor)
            
            return jsonify({'success': True})
        
        else:
            return jsonify({'success': False, 'error': f'Unsupported sensor type: {sensor_type}'}), 400
    
    except Exception as e:
        logger.error(f"Error adding sensor: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/add-state-factor', methods=['POST'])
def add_state_factor():
    """Add a state factor to ZeroAIcon."""
    try:
        data = request.json
        name = data.get('name')
        factor_type = data.get('factor_type')
        value = data.get('value')
        params = data.get('params', {})
        relationships = data.get('relationships', {'depends_on': []})
        
        if not name or not factor_type or value is None:
            return jsonify({
                'success': False, 
                'error': 'Name, factor_type, and value are required'
            }), 400
        
        logger.info(f"Adding state factor: {name}, type: {factor_type}, value: {value}")
        logger.info(f"Parameters: {params}")
        logger.info(f"Relationships: {relationships}")
        
        # Add the state factor
        try:
            factor = aicons[current_aicon]["instance"].add_state_factor(
                name=name,
                factor_type=factor_type,
                value=value,
                params=params,
                relationships=relationships
            )
            
            return jsonify({'success': True})
            
        except TypeError as e:
            logger.error(f"Type error when adding state factor: {str(e)}", exc_info=True)
            return jsonify({
                'success': False, 
                'error': f"Parameter type error: {str(e)}"
            }), 400
        except ValueError as e:
            logger.error(f"Value error when adding state factor: {str(e)}", exc_info=True)
            return jsonify({
                'success': False, 
                'error': f"Invalid value: {str(e)}"
            }), 400
        except AttributeError as e:
            logger.error(f"Attribute error when adding state factor: {str(e)}", exc_info=True)
            return jsonify({
                'success': False, 
                'error': f"Missing attribute: {str(e)}"
            }), 400
        except Exception as e:
            logger.error(f"Unexpected error adding state factor: {str(e)}", exc_info=True)
            return jsonify({
                'success': False, 
                'error': f"Failed to add state factor: {str(e)}"
            }), 500
    
    except Exception as e:
        logger.error(f"Error adding state factor: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/delete-state-factor', methods=['POST'])
def delete_state_factor():
    """Delete a state factor from ZeroAIcon."""
    try:
        data = request.json
        name = data.get('name')
        
        if not name:
            return jsonify({
                'success': False, 
                'error': 'Factor name is required'
            }), 400
        
        logger.info(f"Deleting state factor: {name}")
        
        # Check if the factor exists
        if not hasattr(aicons[current_aicon]["instance"].brain, 'state') or not hasattr(aicons[current_aicon]["instance"].brain.state, 'get_state_factors'):
            return jsonify({
                'success': False, 
                'error': 'State representation system not initialized'
            }), 500
            
        factors = aicons[current_aicon]["instance"].brain.state.get_state_factors()
        if name not in factors:
            return jsonify({
                'success': False, 
                'error': f'State factor "{name}" does not exist'
            }), 404
        
        # Delete the factor
        try:
            # The current ZeroAIcon API doesn't have a direct delete method,
            # so we need to recreate all factors except the one to delete
            current_factors = []
            for factor_name, factor in factors.items():
                if factor_name != name:
                    factor_type = factor.__class__.__name__.replace('LatentVariable', '').lower()
                    
                    # Extract factor details for reconstruction
                    factor_data = {
                        'name': factor_name,
                        'factor_type': factor_type,
                        'value': factor.value if hasattr(factor, 'value') else factor.get('value', 0),
                        'params': {},
                        'relationships': {'depends_on': []}
                    }
                    
                    # If categorical, get categories and probs
                    if factor_type == 'categorical' and hasattr(factor, 'categories'):
                        factor_data['params']['categories'] = factor.categories
                        factor_data['params']['probs'] = factor.probs
                    
                    # If continuous, get loc and scale
                    elif factor_type == 'continuous' and hasattr(factor, 'loc') and hasattr(factor, 'scale'):
                        factor_data['params']['loc'] = factor.loc
                        factor_data['params']['scale'] = factor.scale
                        
                        # Check for constraints
                        if hasattr(factor, 'constraints'):
                            factor_data['params']['constraints'] = factor.constraints
                    
                    # If discrete, get rate
                    elif factor_type == 'discrete' and hasattr(factor, 'rate'):
                        factor_data['params']['rate'] = factor.rate
                    
                    current_factors.append(factor_data)
            
            # Reset the state factors
            aicons[current_aicon]["instance"].brain.state.reset()
            
            # Recreate all factors except the deleted one
            for factor_data in current_factors:
                aicons[current_aicon]["instance"].add_state_factor(
                    name=factor_data['name'],
                    factor_type=factor_data['factor_type'],
                    value=factor_data['value'],
                    params=factor_data['params'],
                    relationships=factor_data['relationships']
                )
            
            return jsonify({'success': True})
            
        except Exception as e:
            logger.error(f"Error deleting state factor: {str(e)}", exc_info=True)
            return jsonify({
                'success': False, 
                'error': f"Failed to delete state factor: {str(e)}"
            }), 500
    
    except Exception as e:
        logger.error(f"Error in delete state factor endpoint: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/define-action-space', methods=['POST'])
def define_action_space():
    """Define an action space for ZeroAIcon."""
    try:
        data = request.json
        space_type = data.get('space_type')
        
        if not space_type:
            return jsonify({'success': False, 'error': 'Space type is required'}), 400
        
        # Define the action space based on type
        if space_type == 'budget_allocation':
            total_budget = data.get('total_budget', 1000.0)
            items = data.get('items', [])
            budget_step = data.get('budget_step', 100.0)
            min_budget = data.get('min_budget', 0.0)
            
            # Validate parameters
            if not items:
                return jsonify({'success': False, 'error': 'Items list is required'}), 400
            
            # Define the action space
            aicons[current_aicon]["instance"].define_action_space(
                space_type="budget_allocation",
                total_budget=total_budget,
                items=items,
                budget_step=budget_step,
                min_budget=min_budget
            )
            
            return jsonify({'success': True})
        
        else:
            return jsonify({'success': False, 'error': f'Unsupported action space type: {space_type}'}), 400
    
    except Exception as e:
        logger.error(f"Error defining action space: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/define-utility-function', methods=['POST'])
def define_utility_function():
    """Define a utility function for ZeroAIcon."""
    try:
        data = request.json
        utility_type = data.get('utility_type')
        
        if not utility_type:
            return jsonify({'success': False, 'error': 'Utility type is required'}), 400
        
        # Ensure we have an action space first
        if not aicons[current_aicon]["instance"].brain.action_space:
            return jsonify({
                'success': False, 
                'error': 'Action space must be defined before defining a utility function'
            }), 400
        
        # Define the utility function based on type
        if utility_type == 'marketing_roi':
            revenue_per_sale = data.get('revenue_per_sale', 50.0)
            num_days = data.get('num_days', 1)
            ad_names = data.get('ad_names', [])
            
            # Validate parameters
            if not ad_names:
                return jsonify({'success': False, 'error': 'Ad names list is required'}), 400
            
            # Define the utility function
            aicons[current_aicon]["instance"].define_utility_function(
                utility_type="marketing_roi",
                revenue_per_sale=revenue_per_sale,
                num_days=num_days,
                ad_names=ad_names
            )
            
            return jsonify({'success': True})
        
        else:
            return jsonify({'success': False, 'error': f'Unsupported utility function type: {utility_type}'}), 400
    
    except Exception as e:
        logger.error(f"Error defining utility function: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/aicons/marketing', methods=['POST'])
def create_marketing_aicon():
    """Create a new marketing-focused AIcon with predefined configuration."""
    try:
        # Create the marketing AIcon
        name = "marketing_aicon"
        if name in aicons:
            return jsonify({'error': 'Marketing AIcon already exists'}), 400
            
        # Create the AIcon instance
        marketing_aicon = ZeroAIcon(
            name=name,
            description="Marketing-focused AIcon for ad optimization",
            model_name="deepseek-r1:7b"
        )
        
        # Add tools to the marketing AIcon
        speak_tool = SpeakOutLoudTool()
        ask_tool = AskQuestionTool()
        marketing_aicon.add_tool(speak_tool)
        marketing_aicon.add_tool(ask_tool)
        
        # Add Meta Ads sensor
        from aicons.bayesbrainGPT.sensors.meta_s.meta_ads_sales_sensor import MetaAdsSalesSensor
        
        # Hardcoded Meta Ads credentials
        access_token = "EAAZAn8wmq1IEBOZCz8oyDZBBgiazAgnQKIoAr4mFTbkV7jxi6t3APzOSxFybXNIkBgwQACdagbs5lFE8tpnNOBOOpWtS3KjZAdf9MNAlySpwEaDrX32oQwUTNmOZAaSXjT5Os5Q8YqRo57tXOUukB7QtcO8nQ8JuqrnnshCr7A0giynZBnJKfuPakrZBWoZD"
        ad_account_id = "act_252267674525035"
        campaign_id = "120218631288730217"
        
        sensor = MetaAdsSalesSensor(
            name="meta_ads",
            reliability=0.9,
            access_token=access_token,
            ad_account_id=ad_account_id,
            campaign_id=campaign_id,
            api_version="v18.0",
            time_granularity="hour"
        )
        
        marketing_aicon.add_sensor("meta_ads", sensor)
        
        # Get active ads and their IDs
        active_ads = sensor.get_active_ads()
        ad_ids = [ad['ad_id'] for ad in active_ads]
        
        # Define action space
        marketing_aicon.define_action_space(
            space_type="budget_allocation",
            total_budget=1000.0,
            items=ad_ids,
            budget_step=100.0,
            min_budget=0.0
        )
        
        # Define utility function
        marketing_aicon.define_utility_function(
            utility_type="marketing_roi",
            revenue_per_sale=50.0,
            num_days=1,
            ad_names=ad_ids
        )
        
        # Store the AIcon
        aicons[name] = {
            "instance": marketing_aicon,
            "chat_history": []
        }
        
        # Set as current AIcon
        global current_aicon
        current_aicon = name
        
        return jsonify({'success': True, 'name': name})
        
    except Exception as e:
        logger.error(f"Error creating marketing AIcon: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

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