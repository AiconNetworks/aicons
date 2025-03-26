import os
import json
import pickle
import psycopg2
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

class AIconPersistence:
    """
    Handles database persistence operations for AIcon instances.
    
    Provides functionality to:
    - Connect to PostgreSQL database
    - Create required schema and tables
    - Perform CRUD operations on AIcon states
    - Handle binary data efficiently
    """
    
    def __init__(self, connection_string: str = None, schema_name: str = "aicon"):
        """
        Initialize the persistence manager.
        
        Args:
            connection_string: PostgreSQL connection string
            schema_name: Database schema to use
        """
        self.connection_string = connection_string
        self.schema_name = schema_name
        self._check_and_create_schema()
    
    def _get_connection(self):
        """Get a database connection."""
        if not self.connection_string:
            raise ValueError("No database connection string provided")
        
        return psycopg2.connect(self.connection_string)
    
    def _check_and_create_schema(self):
        """Create the necessary schema and tables if they don't exist."""
        if not self.connection_string:
            # Skip if no connection string (e.g., during testing)
            return
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # Create schema if it doesn't exist
                    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema_name}")
                    
                    # Create AIcon table
                    cursor.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self.schema_name}.aicons (
                            id TEXT PRIMARY KEY,
                            name TEXT NOT NULL,
                            aicon_type TEXT NOT NULL,
                            created_at TIMESTAMP NOT NULL,
                            last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            config JSONB DEFAULT '{{}}'::jsonb,
                            state JSONB DEFAULT '{{}}'::jsonb,
                            metadata JSONB DEFAULT '{{}}'::jsonb
                        )
                    """)
                    
                    # Create binary data table for storing pickled objects
                    cursor.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self.schema_name}.binary_data (
                            id TEXT PRIMARY KEY,
                            aicon_id TEXT NOT NULL REFERENCES {self.schema_name}.aicons(id) ON DELETE CASCADE,
                            name TEXT NOT NULL,
                            data BYTEA NOT NULL,
                            format TEXT NOT NULL,
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # Create factors table for direct querying
                    cursor.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self.schema_name}.factors (
                            id TEXT PRIMARY KEY,
                            aicon_id TEXT NOT NULL REFERENCES {self.schema_name}.aicons(id) ON DELETE CASCADE,
                            name TEXT NOT NULL,
                            factor_type TEXT NOT NULL,
                            parameters JSONB DEFAULT '{{}}'::jsonb,
                            current_belief JSONB DEFAULT '{{}}'::jsonb,
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(aicon_id, name)
                        )
                    """)
                    
                    # Create indices for better performance
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_aicons_name ON {self.schema_name}.aicons(name)")
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_aicons_type ON {self.schema_name}.aicons(aicon_type)")
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_binary_data_aicon_id ON {self.schema_name}.binary_data(aicon_id)")
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_factors_aicon_id ON {self.schema_name}.factors(aicon_id)")
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_factors_name ON {self.schema_name}.factors(name)")
                
                conn.commit()
        except Exception as e:
            print(f"Error creating database schema: {e}")
    
    def save_aicon(self, aicon_id: str, name: str, aicon_type: str, 
                   created_at: str, config: Dict = None, state: Dict = None, 
                   metadata: Dict = None) -> bool:
        """
        Save or update an AIcon in the database.
        
        Args:
            aicon_id: Unique identifier for the AIcon
            name: Name of the AIcon
            aicon_type: Type of the AIcon
            created_at: Creation timestamp
            config: Configuration data
            state: State data
            metadata: Additional metadata
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # Check if AIcon already exists
                    cursor.execute(f"""
                        SELECT 1 FROM {self.schema_name}.aicons WHERE id = %s
                    """, (aicon_id,))
                    
                    aicon_exists = cursor.fetchone() is not None
                    
                    if aicon_exists:
                        # Update existing AIcon
                        cursor.execute(f"""
                            UPDATE {self.schema_name}.aicons
                            SET name = %s,
                                aicon_type = %s,
                                last_updated = %s,
                                config = %s,
                                state = %s,
                                metadata = %s
                            WHERE id = %s
                        """, (
                            name,
                            aicon_type,
                            datetime.now(),
                            json.dumps(config or {}),
                            json.dumps(state or {}),
                            json.dumps(metadata or {}),
                            aicon_id
                        ))
                    else:
                        # Insert new AIcon
                        cursor.execute(f"""
                            INSERT INTO {self.schema_name}.aicons
                            (id, name, aicon_type, created_at, last_updated, config, state, metadata)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            aicon_id,
                            name,
                            aicon_type,
                            datetime.fromisoformat(created_at),
                            datetime.now(),
                            json.dumps(config or {}),
                            json.dumps(state or {}),
                            json.dumps(metadata or {})
                        ))
                
                conn.commit()
                return True
        except Exception as e:
            print(f"Error saving AIcon to database: {e}")
            return False
    
    def load_aicon(self, aicon_id: str) -> Dict[str, Any]:
        """
        Load an AIcon from the database.
        
        Args:
            aicon_id: ID of the AIcon to load
            
        Returns:
            Dict containing AIcon data or None if not found
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                        SELECT id, name, aicon_type, created_at, last_updated, 
                               config, state, metadata
                        FROM {self.schema_name}.aicons
                        WHERE id = %s
                    """, (aicon_id,))
                    
                    row = cursor.fetchone()
                    
                    if not row:
                        return None
                    
                    return {
                        "id": row[0],
                        "name": row[1],
                        "aicon_type": row[2],
                        "created_at": row[3].isoformat(),
                        "last_updated": row[4].isoformat(),
                        "config": json.loads(row[5]),
                        "state": json.loads(row[6]),
                        "metadata": json.loads(row[7])
                    }
        except Exception as e:
            print(f"Error loading AIcon from database: {e}")
            return None
    
    def save_binary_data(self, aicon_id: str, name: str, data, format: str = "pickle") -> bool:
        """
        Save binary data associated with an AIcon.
        
        Args:
            aicon_id: ID of the associated AIcon
            name: Name/identifier for this binary data
            data: The binary data object to save
            format: Format identifier (e.g., "pickle", "brain", etc.)
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Serialize the data
            if format == "pickle":
                binary_data = pickle.dumps(data)
            else:
                binary_data = data
            
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # Generate a unique ID for this binary data
                    import uuid
                    binary_id = str(uuid.uuid4())
                    
                    # Check if data with this name already exists for this AIcon
                    cursor.execute(f"""
                        SELECT id FROM {self.schema_name}.binary_data
                        WHERE aicon_id = %s AND name = %s
                    """, (aicon_id, name))
                    
                    existing_id = cursor.fetchone()
                    
                    if existing_id:
                        # Update existing binary data
                        cursor.execute(f"""
                            UPDATE {self.schema_name}.binary_data
                            SET data = %s,
                                format = %s,
                                last_updated = %s
                            WHERE id = %s
                        """, (
                            binary_data,
                            format,
                            datetime.now(),
                            existing_id[0]
                        ))
                    else:
                        # Insert new binary data
                        cursor.execute(f"""
                            INSERT INTO {self.schema_name}.binary_data
                            (id, aicon_id, name, data, format, created_at, last_updated)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, (
                            binary_id,
                            aicon_id,
                            name,
                            binary_data,
                            format,
                            datetime.now(),
                            datetime.now()
                        ))
                
                conn.commit()
                return True
        except Exception as e:
            print(f"Error saving binary data to database: {e}")
            return False
    
    def load_binary_data(self, aicon_id: str, name: str) -> Tuple[Any, str]:
        """
        Load binary data associated with an AIcon.
        
        Args:
            aicon_id: ID of the associated AIcon
            name: Name/identifier for the binary data
            
        Returns:
            Tuple of (data_object, format) or (None, None) if not found
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                        SELECT data, format
                        FROM {self.schema_name}.binary_data
                        WHERE aicon_id = %s AND name = %s
                    """, (aicon_id, name))
                    
                    row = cursor.fetchone()
                    
                    if not row:
                        return None, None
                    
                    binary_data, format = row
                    
                    # Deserialize based on format
                    if format == "pickle":
                        return pickle.loads(binary_data), format
                    else:
                        return binary_data, format
        except Exception as e:
            print(f"Error loading binary data from database: {e}")
            return None, None
    
    def save_factor(self, aicon_id: str, factor_name: str, factor_type: str,
                    parameters: Dict = None, current_belief: Dict = None) -> bool:
        """
        Save or update a state factor.
        
        Args:
            aicon_id: ID of the associated AIcon
            factor_name: Name of the factor
            factor_type: Type of the factor (e.g., "continuous", "categorical", "discrete")
            parameters: Factor parameters
            current_belief: Current belief state
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # Generate a unique ID for this factor
                    import uuid
                    factor_id = str(uuid.uuid4())
                    
                    # Check if factor already exists
                    cursor.execute(f"""
                        SELECT id FROM {self.schema_name}.factors
                        WHERE aicon_id = %s AND name = %s
                    """, (aicon_id, factor_name))
                    
                    existing_id = cursor.fetchone()
                    
                    if existing_id:
                        # Update existing factor
                        cursor.execute(f"""
                            UPDATE {self.schema_name}.factors
                            SET factor_type = %s,
                                parameters = %s,
                                current_belief = %s,
                                last_updated = %s
                            WHERE id = %s
                        """, (
                            factor_type,
                            json.dumps(parameters or {}),
                            json.dumps(current_belief or {}),
                            datetime.now(),
                            existing_id[0]
                        ))
                    else:
                        # Insert new factor
                        cursor.execute(f"""
                            INSERT INTO {self.schema_name}.factors
                            (id, aicon_id, name, factor_type, parameters, current_belief, created_at, last_updated)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            factor_id,
                            aicon_id,
                            factor_name,
                            factor_type,
                            json.dumps(parameters or {}),
                            json.dumps(current_belief or {}),
                            datetime.now(),
                            datetime.now()
                        ))
                
                conn.commit()
                return True
        except Exception as e:
            print(f"Error saving factor to database: {e}")
            return False
    
    def load_factors(self, aicon_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Load all factors for an AIcon.
        
        Args:
            aicon_id: ID of the associated AIcon
            
        Returns:
            Dict mapping factor names to their data
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                        SELECT name, factor_type, parameters, current_belief
                        FROM {self.schema_name}.factors
                        WHERE aicon_id = %s
                    """, (aicon_id,))
                    
                    factors = {}
                    for row in cursor.fetchall():
                        name, factor_type, parameters, current_belief = row
                        factors[name] = {
                            "type": factor_type,
                            "parameters": json.loads(parameters),
                            "current_belief": json.loads(current_belief)
                        }
                    
                    return factors
        except Exception as e:
            print(f"Error loading factors from database: {e}")
            return {}
    
    def delete_aicon(self, aicon_id: str) -> bool:
        """
        Delete an AIcon and all associated data.
        
        Args:
            aicon_id: ID of the AIcon to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # Due to foreign key constraints with CASCADE, 
                    # deleting the AIcon will also delete associated binary data and factors
                    cursor.execute(f"""
                        DELETE FROM {self.schema_name}.aicons
                        WHERE id = %s
                    """, (aicon_id,))
                
                conn.commit()
                return True
        except Exception as e:
            print(f"Error deleting AIcon from database: {e}")
            return False
    
    def list_aicons(self, aicon_type: str = None) -> List[Dict[str, Any]]:
        """
        List all AIcon instances, optionally filtered by type.
        
        Args:
            aicon_type: Filter by AIcon type
            
        Returns:
            List of AIcon metadata dictionaries
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    if aicon_type:
                        cursor.execute(f"""
                            SELECT id, name, aicon_type, created_at, last_updated
                            FROM {self.schema_name}.aicons
                            WHERE aicon_type = %s
                            ORDER BY last_updated DESC
                        """, (aicon_type,))
                    else:
                        cursor.execute(f"""
                            SELECT id, name, aicon_type, created_at, last_updated
                            FROM {self.schema_name}.aicons
                            ORDER BY last_updated DESC
                        """)
                    
                    aicons = []
                    for row in cursor.fetchall():
                        aicons.append({
                            "id": row[0],
                            "name": row[1],
                            "type": row[2],
                            "created_at": row[3].isoformat(),
                            "last_updated": row[4].isoformat()
                        })
                    
                    return aicons
        except Exception as e:
            print(f"Error listing AIcon instances: {e}")
            return [] 