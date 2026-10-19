"""
MultiSense API: MongoDB Database Client & Telemetry Persistence Manager
Handles connection pooling to MongoDB Atlas / Local MongoDB,
graceful fallback buffering when DB is unconfigured, and schema serialization.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import uuid

# Load environment variables from .env file
from dotenv import load_dotenv

# Search for .env in project root and current directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
env_candidates = [
    PROJECT_ROOT / ".env",
    Path(__file__).resolve().parent.parent / ".env",
    Path.cwd() / ".env"
]
for env_path in env_candidates:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        break

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False

try:
    import certifi
    CA_FILE = certifi.where()
except Exception:
    CA_FILE = None

# Configuration Defaults
DEFAULT_DB_NAME = "multisense_db"
DEFAULT_COLLECTION = "sensor_telemetry"

# Circular in-memory fallback buffer (holds last 150 readings if DB not connected)
MEMORY_TELEMETRY_BUFFER: deque = deque(maxlen=150)


class DatabaseManager:
    """Manages MongoDB Atlas connection and telemetry persistence."""

    def __init__(self):
        self.client: Optional[Any] = None
        self.db = None
        self.collection = None
        self._is_connected = False
        self._last_error = ""
        self._last_connect_attempt = 0.0

    def get_uri(self) -> str:
        return os.environ.get("MONGODB_URI", "").strip()

    def get_db_name(self) -> str:
        return os.environ.get("MONGODB_DB_NAME", DEFAULT_DB_NAME).strip() or DEFAULT_DB_NAME

    def get_collection_name(self) -> str:
        return os.environ.get("MONGODB_COLLECTION", DEFAULT_COLLECTION).strip() or DEFAULT_COLLECTION

    def connect(self) -> Tuple[bool, str]:
        """Attempts to establish connection to MongoDB if URI is configured."""
        import time
        self._last_connect_attempt = time.time()

        if not PYMONGO_AVAILABLE:
            self._is_connected = False
            self._last_error = "pymongo package is not installed."
            return False, self._last_error

        uri = self.get_uri()
        if not uri:
            self._is_connected = False
            self._last_error = "MONGODB_URI is not set. Telemetry is being buffered locally."
            return False, self._last_error

        try:
            # Connect with certifi CA bundle for secure TLS on macOS / Linux
            client_kwargs = {
                "serverSelectionTimeoutMS": 2500,
                "connectTimeoutMS": 2500
            }
            if CA_FILE:
                client_kwargs["tlsCAFile"] = CA_FILE

            self.client = MongoClient(uri, **client_kwargs)
            # Verify connection with ping
            self.client.admin.command('ping')
            
            db_name = self.get_db_name()
            coll_name = self.get_collection_name()
            self.db = self.client[db_name]
            self.collection = self.db[coll_name]
            self._is_connected = True
            self._last_error = ""
            print(f"[+] Successfully connected to MongoDB: database='{db_name}', collection='{coll_name}'")
            return True, "Connected"
        except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as e:
            self._is_connected = False
            err_str = str(e)
            if "TLSV1_ALERT_INTERNAL_ERROR" in err_str or "SSL handshake failed" in err_str:
                self._last_error = "ATLAS_IP_WHITELIST_REQUIRED: Please add 0.0.0.0/0 in MongoDB Atlas -> Network Access."
            else:
                self._last_error = err_str
            print(f"[!] MongoDB connection error: {self._last_error}")
            return False, self._last_error

    def get_collection(self):
        """Returns the collection if connected, attempting reconnect with 30s cooldown."""
        import time
        if not self._is_connected or self.collection is None:
            if self.get_uri() and (time.time() - self._last_connect_attempt > 30.0):
                self.connect()
        return self.collection if self._is_connected else None

    def get_status(self) -> Dict[str, Any]:
        """Returns current MongoDB connection diagnostic status."""
        uri = self.get_uri()
        has_uri = bool(uri)
        # Mask credentials in URI for display
        masked_uri = "Not configured"
        if has_uri:
            if "@" in uri:
                prefix = uri.split("@")[0].split("//")[0] + "//"
                host_part = uri.split("@")[-1]
                masked_uri = f"{prefix}***:***@{host_part}"
            else:
                masked_uri = "mongodb://configured"

        coll = self.get_collection()
        total_records = len(MEMORY_TELEMETRY_BUFFER)
        db_records = 0
        if coll is not None:
            try:
                db_records = coll.count_documents({})
                total_records = db_records
            except Exception:
                pass

        return {
            "has_uri": has_uri,
            "connected": self._is_connected,
            "masked_uri": masked_uri,
            "database": self.get_db_name(),
            "collection": self.get_collection_name(),
            "total_records": total_records,
            "database_records": db_records,
            "buffered_in_memory": len(MEMORY_TELEMETRY_BUFFER),
            "status_message": "CONNECTED" if self._is_connected else (
                "READY_FOR_MONGODB_URI" if not has_uri else f"CONNECTION_ERROR: {self._last_error}"
            )
        }

    def save_record(
        self,
        raw_input: Dict[str, Any],
        computed_features: Dict[str, Any],
        ml_prediction: Dict[str, Any],
        clinical_alert: Dict[str, Any],
        device_id: str = "sensor_node_01"
    ) -> Dict[str, Any]:
        """
        Saves sensor reading + ML prediction to MongoDB.
        If MongoDB is offline/unconfigured, appends to the memory buffer.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        record_id = str(uuid.uuid4())

        doc = {
            "_id": record_id,
            "timestamp": timestamp,
            "device_id": device_id,
            "input_sensors": raw_input,
            "computed_features": computed_features,
            "ml_output": {
                "predicted_class": ml_prediction.get("predicted_class"),
                "class_id": ml_prediction.get("class_id"),
                "confidence": ml_prediction.get("confidence"),
                "probabilities": ml_prediction.get("probabilities", {}),
                "model_used": ml_prediction.get("model_used")
            },
            "clinical_alert": clinical_alert,
            "storage_status": "PENDING"
        }

        # Attempt to save to MongoDB
        coll = self.get_collection()
        if coll is not None:
            try:
                # Insert copy into Mongo
                mongo_doc = dict(doc)
                coll.insert_one(mongo_doc)
                doc["storage_status"] = "SAVED_TO_MONGODB"
            except Exception as e:
                print(f"[!] Failed to insert document into MongoDB: {e}")
                self._is_connected = False
                self._last_error = str(e)
                doc["storage_status"] = "BUFFERED_LOCALLY"
        else:
            doc["storage_status"] = "BUFFERED_LOCALLY"

        # Always save to memory buffer as quick local cache
        MEMORY_TELEMETRY_BUFFER.append(doc)
        return doc

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent sensor telemetry records."""
        coll = self.get_collection()
        if coll is not None:
            try:
                cursor = coll.find({}).sort("timestamp", -1).limit(limit)
                records = []
                for item in cursor:
                    item["_id"] = str(item.get("_id"))
                    records.append(item)
                return records
            except Exception as e:
                print(f"[!] Failed to read from MongoDB: {e}")

        # Return from local memory buffer (newest first)
        return list(reversed(list(MEMORY_TELEMETRY_BUFFER)))[:limit]


# Global database manager instance
db_manager = DatabaseManager()

# Try initial connection if env is already provided
if db_manager.get_uri():
    db_manager.connect()
