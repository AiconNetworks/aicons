"""
AIcon Persistence Module

This module provides functionality for saving and loading the state of an AIcon,
including its brain, factors, and other components to and from a PostgreSQL database.
"""

import os
import json
import pickle
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import psycopg2
import psycopg2.extras
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIconPersistence:
    """
    Manages persistence for AIcon objects, saving and loading their state
    to and from a PostgreSQL database.
    """
    
    def __init__(self, db_connection_string: str = None, schema_name: str = "aicon"):
        """
        Initialize the persistence manager with a database connection string.
        
        Args:
            db_connection_string: PostgreSQL connection string
            schema_name: Name of the schema to use for AIcon tables
        """
        self.db_connection_string = db_connection_string or os.getenv("AICON_DB_URL")
        self.schema_name = schema_name
        
        # When no connection string is provided, store a warning but don't fail
        # This allows the class to be instantiated without a database during development
        if not self.db_connection_string:
            logger.warning("No database connection string provided. "
                          "Persistence will not be available.")
    
    def _get_connection(self):
        """Get a database connection"""
        if not self.db_connection_string:
            raise ValueError("No database connection string available")
            
        try:
            conn = psycopg2.connect(self.db_connection_string)
            conn.autocommit = False
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def initialize_schema(self):
        """
        Initialize the database schema if it doesn't exist.
        Creates tables for AIcon state persistence.
        """
        if not self.db_connection_string:
            logger.warning("No database connection, skipping schema initialization")
            return False
            
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Create schema if it doesn't exist
            cursor.execute(f"""
                CREATE SCHEMA IF NOT EXISTS {self.schema_name};
            """)
            
            # Create the aicons table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema_name}.aicons (
                    id UUID PRIMARY KEY,
                    name VARCHAR NOT NULL,
                    type VARCHAR NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    state JSONB,
                    config JSONB
                );
            """)
            
            # Create the state_factors table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema_name}.state_factors (
                    id UUID PRIMARY KEY,
                    aicon_id UUID REFERENCES {self.schema_name}.aicons(id) ON DELETE CASCADE,
                    name VARCHAR NOT NULL,
                    type VARCHAR CHECK (type IN ('discrete', 'continuous', 'categorical')),
                    parameters JSONB,
                    prior JSONB,
                    current_belief JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)
            
            # Create the binary_data table for storing pickle data
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema_name}.binary_data (
                    id UUID PRIMARY KEY,
                    aicon_id UUID REFERENCES {self.schema_name}.aicons(id) ON DELETE CASCADE,
                    name VARCHAR NOT NULL,
                    data BYTEA NOT NULL,
                    format VARCHAR NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)
            
            # Create indexes
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_state_factors_aicon_id 
                ON {self.schema_name}.state_factors(aicon_id);
                
                CREATE INDEX IF NOT EXISTS idx_binary_data_aicon_id 
                ON {self.schema_name}.binary_data(aicon_id);
            """)
            
            conn.commit()
            logger.info("Database schema initialized successfully")
            return True
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to initialize schema: {e}")
            raise
        finally:
            conn.close()
    
    def save_aicon(self, aicon, save_pickle: bool = True) -> str:
        """
        Save an AIcon to the database.
        
        Args:
            aicon: The AIcon object to save
            save_pickle: Whether to also save a pickle representation for complex objects
            
        Returns:
            The AIcon ID
        """
        if not self.db_connection_string:
            logger.warning("No database connection, cannot save AIcon")
            return None
            
        # Convert AIcon to dictionary format
        aicon_data = self._aicon_to_dict(aicon)
        aicon_id = aicon_data['id']
        
        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            
            # Check if AIcon exists
            cursor.execute(f"""
                SELECT id FROM {self.schema_name}.aicons WHERE id = %s
            """, (aicon_id,))
            aicon_exists = cursor.fetchone() is not None
            
            if aicon_exists:
                # Update existing AIcon
                cursor.execute(f"""
                    UPDATE {self.schema_name}.aicons
                    SET name = %s, type = %s, state = %s, config = %s, last_updated = %s
                    WHERE id = %s
                    RETURNING id
                """, (
                    aicon_data['name'],
                    aicon_data['type'],
                    json.dumps(aicon_data.get('state', {})),
                    json.dumps(aicon_data.get('config', {})),
                    datetime.now(),
                    aicon_id
                ))
            else:
                # Create new AIcon
                cursor.execute(f"""
                    INSERT INTO {self.schema_name}.aicons 
                    (id, name, type, state, config)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    aicon_id,
                    aicon_data['name'],
                    aicon_data['type'],
                    json.dumps(aicon_data.get('state', {})),
                    json.dumps(aicon_data.get('config', {}))
                ))
            
            # Save state factors
            if 'state_factors' in aicon_data:
                self._save_state_factors(cursor, aicon_id, aicon_data['state_factors'])
            
            # Save pickle representation if needed
            if save_pickle and hasattr(aicon, 'brain'):
                self._save_brain_pickle(cursor, aicon_id, aicon.brain)
            
            conn.commit()
            logger.info(f"AIcon {aicon_data['name']} saved successfully with ID: {aicon_id}")
            return aicon_id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to save AIcon: {e}")
            raise
        finally:
            conn.close()
    
    def load_aicon(self, aicon_id: str, aicon_class=None, load_pickle: bool = True) -> Any:
        """
        Load an AIcon from the database
        
        Args:
            aicon_id: ID of the AIcon to load
            aicon_class: Optional class to instantiate (if None, returns data dict)
            load_pickle: Whether to load pickle data for complex objects
            
        Returns:
            The loaded AIcon object or data dictionary
        """
        if not self.db_connection_string:
            logger.warning("No database connection, cannot load AIcon")
            return None
            
        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            
            # Get AIcon data
            cursor.execute(f"""
                SELECT id, name, type, state, config, created_at, last_updated
                FROM {self.schema_name}.aicons WHERE id = %s
            """, (aicon_id,))
            
            aicon_row = cursor.fetchone()
            if not aicon_row:
                logger.warning(f"AIcon with ID {aicon_id} not found")
                return None
            
            aicon_data = dict(aicon_row)
            aicon_data['state'] = json.loads(aicon_data['state']) if aicon_data['state'] else {}
            aicon_data['config'] = json.loads(aicon_data['config']) if aicon_data['config'] else {}
            
            # Get state factors
            aicon_data['state_factors'] = self._load_state_factors(cursor, aicon_id)
            
            # Load pickle data if requested
            if load_pickle:
                brain_data = self._load_brain_pickle(cursor, aicon_id)
                if brain_data:
                    aicon_data['brain_pickle'] = brain_data
            
            # If a class is provided, instantiate it
            if aicon_class:
                return self._dict_to_aicon(aicon_data, aicon_class)
                
            return aicon_data
            
        except Exception as e:
            logger.error(f"Failed to load AIcon: {e}")
            raise
        finally:
            conn.close()
    
    def delete_aicon(self, aicon_id: str) -> bool:
        """
        Delete an AIcon from the database
        
        Args:
            aicon_id: ID of the AIcon to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.db_connection_string:
            logger.warning("No database connection, cannot delete AIcon")
            return False
            
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            cursor.execute(f"""
                DELETE FROM {self.schema_name}.aicons WHERE id = %s
            """, (aicon_id,))
            
            deleted = cursor.rowcount > 0
            conn.commit()
            
            if deleted:
                logger.info(f"AIcon with ID {aicon_id} deleted successfully")
            else:
                logger.warning(f"AIcon with ID {aicon_id} not found for deletion")
                
            return deleted
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to delete AIcon: {e}")
            return False
        finally:
            conn.close()
    
    def _save_state_factors(self, cursor, aicon_id: str, factors: List[Dict[str, Any]]) -> None:
        """Save state factors for the AIcon"""
        # Clear existing factors for this AIcon
        cursor.execute(f"""
            DELETE FROM {self.schema_name}.state_factors WHERE aicon_id = %s
        """, (aicon_id,))
        
        # Insert new factors
        for factor in factors:
            factor_id = factor.get('id', str(uuid.uuid4()))
            
            cursor.execute(f"""
                INSERT INTO {self.schema_name}.state_factors 
                (id, aicon_id, name, type, parameters, prior, current_belief)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                factor_id,
                aicon_id,
                factor['name'],
                factor['type'],
                json.dumps(factor.get('parameters', {})),
                json.dumps(factor.get('prior', {})),
                json.dumps(factor.get('current_belief', {}))
            ))
    
    def _load_state_factors(self, cursor, aicon_id: str) -> List[Dict[str, Any]]:
        """Load state factors for an AIcon"""
        cursor.execute(f"""
            SELECT id, name, type, parameters, prior, current_belief
            FROM {self.schema_name}.state_factors
            WHERE aicon_id = %s
        """, (aicon_id,))
        
        factors = []
        for row in cursor.fetchall():
            factor = dict(row)
            factor['parameters'] = json.loads(factor['parameters']) if factor['parameters'] else {}
            factor['prior'] = json.loads(factor['prior']) if factor['prior'] else {}
            factor['current_belief'] = json.loads(factor['current_belief']) if factor['current_belief'] else {}
            factors.append(factor)
        
        return factors
    
    def _save_brain_pickle(self, cursor, aicon_id: str, brain: Any) -> None:
        """Save a pickled representation of the brain"""
        try:
            # Pickle the brain object
            brain_pickle = pickle.dumps(brain)
            
            # Check if a pickle already exists
            cursor.execute(f"""
                SELECT id FROM {self.schema_name}.binary_data 
                WHERE aicon_id = %s AND name = 'brain_pickle'
            """, (aicon_id,))
            
            pickle_exists = cursor.fetchone() is not None
            
            if pickle_exists:
                cursor.execute(f"""
                    UPDATE {self.schema_name}.binary_data
                    SET data = %s, last_updated = %s
                    WHERE aicon_id = %s AND name = 'brain_pickle'
                """, (
                    psycopg2.Binary(brain_pickle),
                    datetime.now(),
                    aicon_id
                ))
            else:
                cursor.execute(f"""
                    INSERT INTO {self.schema_name}.binary_data
                    (id, aicon_id, name, data, format)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    str(uuid.uuid4()),
                    aicon_id,
                    'brain_pickle',
                    psycopg2.Binary(brain_pickle),
                    'pickle'
                ))
                
        except Exception as e:
            logger.error(f"Failed to save brain pickle: {e}")
            # Continue without the pickle - we still have the JSON representation
    
    def _load_brain_pickle(self, cursor, aicon_id: str) -> Any:
        """Load the pickled brain representation"""
        try:
            cursor.execute(f"""
                SELECT data FROM {self.schema_name}.binary_data
                WHERE aicon_id = %s AND name = 'brain_pickle'
            """, (aicon_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
                
            # Unpickle the brain object
            return pickle.loads(row[0].tobytes())
                
        except Exception as e:
            logger.error(f"Failed to load brain pickle: {e}")
            return None
    
    def _aicon_to_dict(self, aicon: Any) -> Dict[str, Any]:
        """Convert an AIcon instance to a dictionary format"""
        # Ensure the AIcon has an ID
        aicon_id = getattr(aicon, 'id', None)
        if not aicon_id:
            aicon_id = str(uuid.uuid4())
            # Try to set the ID if possible
            try:
                setattr(aicon, 'id', aicon_id)
            except:
                pass
        
        # Basic AIcon data
        aicon_data = {
            'id': aicon_id,
            'name': getattr(aicon, 'name', 'Unnamed AIcon'),
            'type': aicon.__class__.__name__,
            'state': {},
            'config': {}
        }
        
        # Brain state
        if hasattr(aicon, 'brain') and aicon.brain:
            aicon_data['state']['brain'] = self._brain_to_dict(aicon.brain)
        
        # Extract state factors
        aicon_data['state_factors'] = self._extract_factors(aicon)
        
        # Other state and config attributes
        aicon_data['state'].update(self._extract_state(aicon))
        aicon_data['config'].update(self._extract_config(aicon))
        
        return aicon_data
    
    def _brain_to_dict(self, brain: Any) -> Dict[str, Any]:
        """Extract brain state in serializable format"""
        brain_state = {}
        
        # State factors
        if hasattr(brain, 'state_factors'):
            brain_state['state_factors'] = {}
            for name, factor in brain.state_factors.items():
                # For TensorFlow distributions, save the parameters
                if 'tf_distribution' in factor:
                    factor_copy = factor.copy()
                    del factor_copy['tf_distribution']  # Remove non-serializable TF object
                    brain_state['state_factors'][name] = factor_copy
                else:
                    brain_state['state_factors'][name] = factor
        
        # Posterior samples - convert numpy arrays to lists
        if hasattr(brain, 'posterior_samples') and brain.posterior_samples:
            brain_state['posterior_samples'] = {}
            for k, v in brain.posterior_samples.items():
                if hasattr(v, 'tolist'):
                    brain_state['posterior_samples'][k] = v.tolist()
                else:
                    brain_state['posterior_samples'][k] = v
        
        # Decision parameters
        if hasattr(brain, 'decision_params'):
            brain_state['decision_params'] = brain.decision_params
            
        # Action space - extract name and parameters
        if hasattr(brain, 'action_space') and brain.action_space:
            action_space = brain.action_space
            brain_state['action_space'] = {
                'type': action_space.__class__.__name__,
                'dimensions': getattr(action_space, 'dimensions', {}),
                'bounds': getattr(action_space, 'bounds', {})
            }
            
        # Utility function - extract name and parameters
        if hasattr(brain, 'utility_function') and brain.utility_function:
            utility_fn = brain.utility_function
            brain_state['utility_function'] = {
                'type': utility_fn.__class__.__name__,
                'parameters': getattr(utility_fn, 'parameters', {})
            }
            
        return brain_state
    
    def _extract_factors(self, aicon: Any) -> List[Dict[str, Any]]:
        """Extract state factors from an AIcon"""
        factors = []
        
        # Get state factors directly from the brain
        if hasattr(aicon, 'brain') and hasattr(aicon.brain, 'state_factors'):
            brain_factors = aicon.brain.state_factors
            for name, factor in brain_factors.items():
                factor_data = {
                    'name': name,
                    'type': factor.get('type', 'unknown')
                }
                
                # Extract parameters
                if 'params' in factor:
                    factor_data['parameters'] = factor['params']
                    
                # Extract current value/belief
                if 'value' in factor:
                    factor_data['current_belief'] = {'value': factor['value']}
                
                # Remove TensorFlow distribution if present
                if 'tf_distribution' in factor:
                    # Don't include in the saved data
                    pass
                    
                factors.append(factor_data)
                
        # Also check for factors attribute in AIcon
        elif hasattr(aicon, 'factors'):
            aicon_factors = aicon.factors
            for name, factor in aicon_factors.items():
                factor_data = {
                    'name': name,
                    'type': factor.get('type', 'unknown'),
                    'parameters': factor.get('parameters', {}),
                    'prior': factor.get('prior', {}),
                    'current_belief': factor.get('current_belief', {})
                }
                factors.append(factor_data)
        
        return factors
    
    def _extract_state(self, aicon: Any) -> Dict[str, Any]:
        """Extract serializable state from an AIcon instance"""
        state = {}
        
        # Extract run stats if present
        if hasattr(aicon, 'run_stats'):
            state['run_stats'] = aicon.run_stats
            
        # Extract campaigns if present
        if hasattr(aicon, 'campaigns'):
            state['campaigns'] = aicon.campaigns
            
        # Extract is_running flag
        if hasattr(aicon, 'is_running'):
            state['is_running'] = aicon.is_running
        
        return state
    
    def _extract_config(self, aicon: Any) -> Dict[str, Any]:
        """Extract configuration from an AIcon instance"""
        config = {}
        
        # Extract capabilities
        if hasattr(aicon, 'capabilities'):
            config['capabilities'] = aicon.capabilities
            
        # Extract type
        if hasattr(aicon, 'type'):
            config['type'] = aicon.type
        
        return config
    
    def _dict_to_aicon(self, aicon_data: Dict[str, Any], aicon_class) -> Any:
        """Reconstruct an AIcon instance from dictionary data"""
        # Create a new instance
        aicon = aicon_class(aicon_data['name'])
        
        # Set the ID
        if hasattr(aicon, 'id'):
            setattr(aicon, 'id', aicon_data['id'])
        
        # Restore brain from pickle if available
        if 'brain_pickle' in aicon_data and hasattr(aicon, 'brain'):
            aicon.brain = aicon_data['brain_pickle']
        # Otherwise restore brain from JSON data
        elif 'state' in aicon_data and 'brain' in aicon_data['state'] and hasattr(aicon, 'brain'):
            brain_data = aicon_data['state']['brain']
            self._restore_brain_state(aicon.brain, brain_data)
        
        # Restore capabilities
        if 'config' in aicon_data and 'capabilities' in aicon_data['config'] and hasattr(aicon, 'capabilities'):
            aicon.capabilities = aicon_data['config']['capabilities']
            
        # Restore type
        if 'config' in aicon_data and 'type' in aicon_data['config'] and hasattr(aicon, 'type'):
            aicon.type = aicon_data['config']['type']
            
        # Restore state factors if not restored via brain pickle
        if 'brain_pickle' not in aicon_data and 'state_factors' in aicon_data:
            for factor in aicon_data['state_factors']:
                # Call the appropriate factor creation method if available
                if factor['type'] == 'continuous' and hasattr(aicon, 'add_factor_continuous'):
                    value = factor.get('current_belief', {}).get('value', 0.0)
                    if isinstance(value, list) and len(value) > 0:
                        value = value[0]  # Take first element if it's a list
                    aicon.add_factor_continuous(factor['name'], value)
                    
                elif factor['type'] == 'categorical' and hasattr(aicon, 'add_factor_categorical'):
                    value = factor.get('current_belief', {}).get('value', '')
                    categories = factor.get('parameters', {}).get('categories', [])
                    aicon.add_factor_categorical(factor['name'], value, categories)
                    
                elif factor['type'] == 'discrete' and hasattr(aicon, 'add_factor_discrete'):
                    value = factor.get('current_belief', {}).get('value', 0)
                    aicon.add_factor_discrete(factor['name'], value)
        
        # Restore run stats
        if 'state' in aicon_data and 'run_stats' in aicon_data['state'] and hasattr(aicon, 'run_stats'):
            aicon.run_stats = aicon_data['state']['run_stats']
            
        # Restore is_running
        if 'state' in aicon_data and 'is_running' in aicon_data['state'] and hasattr(aicon, 'is_running'):
            aicon.is_running = aicon_data['state']['is_running']
            
        # Restore campaigns
        if 'state' in aicon_data and 'campaigns' in aicon_data['state'] and hasattr(aicon, 'campaigns'):
            aicon.campaigns = aicon_data['state']['campaigns']
            
        return aicon
    
    def _restore_brain_state(self, brain: Any, brain_data: Dict[str, Any]) -> None:
        """Restore brain state from data"""
        # Restore state factors
        if 'state_factors' in brain_data and hasattr(brain, 'set_state_factors'):
            brain.set_state_factors(brain_data['state_factors'])
            
        # Restore posterior samples
        if 'posterior_samples' in brain_data and hasattr(brain, 'set_posterior_samples'):
            # Convert lists back to numpy arrays
            posterior_samples = {}
            for k, v in brain_data['posterior_samples'].items():
                if isinstance(v, list):
                    posterior_samples[k] = np.array(v)
                else:
                    posterior_samples[k] = v
            brain.set_posterior_samples(posterior_samples)
            
        # Restore decision parameters
        if 'decision_params' in brain_data and hasattr(brain, 'set_decision_params'):
            brain.set_decision_params(brain_data['decision_params']) 