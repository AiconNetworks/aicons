import os
import pickle
import json
from typing import Any, Dict, Optional

class CloudStorageManager:
    """
    Handles persistence of large objects to Google Cloud Storage.
    
    Used for storing objects that are too large for efficient database storage,
    such as complex brain models with TensorFlow components.
    """
    
    def __init__(self, bucket_name: str = "aicon-brain-storage", project_id: Optional[str] = None):
        """
        Initialize the cloud storage manager.
        
        Args:
            bucket_name: GCS bucket name
            project_id: Google Cloud project ID (optional, defaults to current environment)
        """
        self.bucket_name = bucket_name
        self.project_id = project_id
        self._client = None
    
    def _get_client(self):
        """Get or create a Google Cloud Storage client."""
        if self._client is None:
            try:
                from google.cloud import storage
                self._client = storage.Client(project=self.project_id)
            except ImportError:
                raise ImportError(
                    "Google Cloud Storage library not installed. "
                    "Please install it with 'pip install google-cloud-storage'."
                )
        return self._client
    
    def _get_bucket(self):
        """Get the configured GCS bucket."""
        client = self._get_client()
        return client.bucket(self.bucket_name)
    
    def save_large_object(self, aicon_id: str, object_name: str, data: Any, 
                          format: str = "pickle") -> str:
        """
        Save a large object to Google Cloud Storage.
        
        Args:
            aicon_id: AIcon ID to associate with the object
            object_name: Name/identifier for the object
            data: Object to save
            format: Format to use (default: "pickle")
            
        Returns:
            str: GCS path where the object is stored
        """
        bucket = self._get_bucket()
        
        # Create a path with AIcon ID for organization
        path = f"aicons/{aicon_id}/{object_name}.{format}"
        blob = bucket.blob(path)
        
        # Prepare data based on format
        if format == "pickle":
            # Serialize with pickle
            serialized_data = pickle.dumps(data)
            blob.upload_from_string(serialized_data)
        elif format == "json":
            # Serialize with JSON
            serialized_data = json.dumps(data)
            blob.upload_from_string(serialized_data)
        else:
            # Assume data is already in appropriate format
            if isinstance(data, bytes):
                blob.upload_from_string(data)
            else:
                blob.upload_from_string(str(data))
        
        # Set metadata to track the content
        blob.metadata = {
            "aicon_id": aicon_id,
            "object_name": object_name,
            "format": format,
            "timestamp": str(int(import_time().time()))
        }
        blob.patch()
        
        return path
    
    def load_large_object(self, path: str, format: Optional[str] = None) -> Any:
        """
        Load a large object from Google Cloud Storage.
        
        Args:
            path: GCS path to the object
            format: Format hint for deserialization (optional, inferred from path if None)
            
        Returns:
            The loaded object
        """
        bucket = self._get_bucket()
        blob = bucket.blob(path)
        
        # Download the data
        data = blob.download_as_bytes()
        
        # Determine format if not provided
        if format is None:
            format = path.split('.')[-1]
        
        # Deserialize based on format
        if format == "pickle":
            return pickle.loads(data)
        elif format == "json":
            return json.loads(data.decode('utf-8'))
        else:
            return data
    
    def delete_large_object(self, path: str) -> bool:
        """
        Delete a large object from Google Cloud Storage.
        
        Args:
            path: GCS path to the object
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            bucket = self._get_bucket()
            blob = bucket.blob(path)
            blob.delete()
            return True
        except Exception as e:
            print(f"Error deleting object from GCS: {e}")
            return False
    
    def save_aicon_brain(self, aicon_id: str, brain_object: Any) -> str:
        """
        Specialized method for saving AIcon brain objects.
        
        Args:
            aicon_id: ID of the AIcon
            brain_object: The brain object to save
            
        Returns:
            str: GCS path where the brain is stored
        """
        return self.save_large_object(
            aicon_id=aicon_id,
            object_name="brain",
            data=brain_object,
            format="pickle"
        )
    
    def load_aicon_brain(self, aicon_id: str, path: Optional[str] = None) -> Any:
        """
        Specialized method for loading AIcon brain objects.
        
        Args:
            aicon_id: ID of the AIcon
            path: Custom path (optional, uses default brain path if None)
            
        Returns:
            The loaded brain object
        """
        if path is None:
            path = f"aicons/{aicon_id}/brain.pickle"
        
        return self.load_large_object(path, format="pickle")
    
    def list_aicon_objects(self, aicon_id: str) -> Dict[str, str]:
        """
        List all objects associated with an AIcon.
        
        Args:
            aicon_id: ID of the AIcon
            
        Returns:
            Dict mapping object names to their GCS paths
        """
        bucket = self._get_bucket()
        prefix = f"aicons/{aicon_id}/"
        blobs = bucket.list_blobs(prefix=prefix)
        
        result = {}
        for blob in blobs:
            # Extract object name from path
            name = blob.name.split('/')[-1].split('.')[0]
            result[name] = blob.name
        
        return result
    
    def ensure_bucket_exists(self) -> bool:
        """
        Ensure the configured bucket exists, creating it if necessary.
        
        Returns:
            bool: True if bucket exists or was created successfully
        """
        client = self._get_client()
        try:
            bucket = client.bucket(self.bucket_name)
            if not bucket.exists():
                bucket = client.create_bucket(self.bucket_name)
                
                # Set a lifecycle policy to delete old versions after 30 days
                lifecycle_rules = {
                    "rule": [
                        {
                            "action": {"type": "Delete"},
                            "condition": {
                                "age": 30,
                                "isLive": False
                            }
                        }
                    ]
                }
                bucket.lifecycle_rules = lifecycle_rules
                bucket.patch()
                
            return True
        except Exception as e:
            print(f"Error ensuring bucket exists: {e}")
            return False

# Utility to import time module only when needed
def import_time():
    import time
    return time 