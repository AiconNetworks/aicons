#!/usr/bin/env python
"""
ZeroAIcon Chat Server launcher script
"""

import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        # Import and run the server directly
        from aicons.server import run_server
        import argparse
        
        # Command line arguments
        parser = argparse.ArgumentParser(description="ZeroAIcon Chat Server")
        parser.add_argument('--host', default='0.0.0.0', help='Host to run the server on')
        parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
        parser.add_argument('--debug', action='store_true', help='Run in debug mode')
        
        args = parser.parse_args()
        
        # Run the server
        run_server(host=args.host, port=args.port, debug=args.debug)
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Make sure you are running this script with Poetry: 'poetry run python run_chat.py'")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error running server: {e}", exc_info=True)
        sys.exit(1) 