"""
MongoDB Connection Manager
Handles connection to MongoDB and provides database access.
"""

import logging
import os
from typing import Optional
from urllib.parse import urlparse, urlunparse

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from app_settings import get_env_or_config

logger = logging.getLogger("mongo_db")

DEFAULT_MONGODB_URI = "mongodb://mongodb:27017/ocr_app"
DEFAULT_MONGODB_DB_NAME = "ocr_app"


class MongoDBManager:
    """MongoDB connection manager with runtime configuration support."""

    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.is_connected = False
        self.env = os.getenv("ENVIRONMENT", "development")
        logger.info("MongoDBManager initialized for environment: %s", self.env)

    def get_connection_uri(self) -> str:
        """
        Resolve MongoDB URI.
        Priority:
        1) MONGODB_URI
        2) local default URI
        """
        mongodb_uri = str(
            get_env_or_config("MONGODB_URI", ("database", "uri"), "")
        ).strip()
        if mongodb_uri:
            logger.info("Using MongoDB URI from MONGODB_URI environment variable")
            return mongodb_uri

        logger.warning(
            "MONGODB_URI is not set. Falling back to default local URI: %s",
            self._sanitize_uri_for_log(DEFAULT_MONGODB_URI),
        )
        return DEFAULT_MONGODB_URI

    def _resolve_db_name(self, uri: str) -> str:
        """
        Resolve MongoDB database name.
        Priority:
        1) MONGODB_DB_NAME
        2) URI database path
        3) my-next-app
        """
        db_name_override = str(
            get_env_or_config("MONGODB_DB_NAME", ("database", "name"), "")
        ).strip()
        if db_name_override:
            logger.info("Using MongoDB database from MONGODB_DB_NAME: %s", db_name_override)
            return db_name_override

        try:
            parsed = urlparse(uri)
            path_value = (parsed.path or "").lstrip("/").strip()
            if path_value:
                db_name_from_uri = path_value.split("/", 1)[0]
                if db_name_from_uri:
                    logger.info("Using MongoDB database parsed from URI: %s", db_name_from_uri)
                    return db_name_from_uri
        except Exception as exc:
            logger.warning("Could not parse MongoDB URI for database name: %s", exc)

        logger.warning(
            "MONGODB_DB_NAME is not set and URI has no database path. Using default database: %s",
            DEFAULT_MONGODB_DB_NAME,
        )
        return DEFAULT_MONGODB_DB_NAME

    def _sanitize_uri_for_log(self, uri: str) -> str:
        """Redact credentials for safe URI logging."""
        try:
            parsed = urlparse(uri)
            netloc = parsed.netloc
            if "@" in netloc:
                host = netloc.rsplit("@", 1)[1]
                netloc = f"***:***@{host}"
            redacted = parsed._replace(netloc=netloc, query="", fragment="")
            return urlunparse(redacted)
        except Exception:
            return "<unparseable-mongodb-uri>"

    def _get_timeout_ms(self, env_name: str, default_value: int) -> int:
        """Read timeout env var safely as positive integer milliseconds."""
        raw_value = (os.getenv(env_name) or "").strip()
        if not raw_value:
            return default_value

        try:
            parsed_value = int(raw_value)
            if parsed_value <= 0:
                raise ValueError("timeout must be greater than 0")
            return parsed_value
        except ValueError:
            logger.warning(
                "Invalid %s=%r. Using default %d ms.",
                env_name,
                raw_value,
                default_value,
            )
            return default_value

    async def connect(self) -> bool:
        """Connect to MongoDB and initialize database handle."""
        if self.is_connected:
            logger.info("Already connected to MongoDB")
            return True

        try:
            uri = self.get_connection_uri()
            db_name = self._resolve_db_name(uri)

            server_selection_timeout = self._get_timeout_ms(
                "MONGODB_SERVER_SELECTION_TIMEOUT_MS",
                60000,
            )
            connect_timeout = self._get_timeout_ms("MONGODB_CONNECT_TIMEOUT_MS", 30000)
            socket_timeout = self._get_timeout_ms("MONGODB_SOCKET_TIMEOUT_MS", 30000)

            logger.info("Connecting to MongoDB...")
            logger.info("  URI: %s", self._sanitize_uri_for_log(uri))
            logger.info("  Database: %s", db_name)
            logger.info(
                "  Timeouts(ms): serverSelection=%d connect=%d socket=%d",
                server_selection_timeout,
                connect_timeout,
                socket_timeout,
            )

            self.client = AsyncIOMotorClient(
                uri,
                serverSelectionTimeoutMS=server_selection_timeout,
                connectTimeoutMS=connect_timeout,
                socketTimeoutMS=socket_timeout,
            )

            await self.client.admin.command("ping")
            self.db = self.client[db_name]
            self.is_connected = True

            logger.info("Successfully connected to MongoDB")
            logger.info("  Database ready: %s", db_name)
            return True

        except ServerSelectionTimeoutError as exc:
            logger.error("MongoDB connection timeout: %s", exc)
            self.is_connected = False
            return False

        except ConnectionFailure as exc:
            logger.error("MongoDB connection failed: %s", exc)
            self.is_connected = False
            return False

        except Exception as exc:
            logger.error("Unexpected error connecting to MongoDB: %s", exc)
            logger.exception("Full traceback:")
            self.is_connected = False
            return False

    async def disconnect(self):
        """Disconnect from MongoDB."""
        if self.client:
            logger.info("Disconnecting from MongoDB...")
            self.client.close()
            self.is_connected = False
            logger.info("Disconnected from MongoDB")

    def get_db(self) -> Optional[AsyncIOMotorDatabase]:
        """Get database instance if connected."""
        if not self.is_connected:
            logger.warning("Not connected to MongoDB")
            return None
        return self.db

    async def get_collection(self, collection_name: str):
        """Get collection by name if connected."""
        db = self.get_db()
        if db is None:
            return None
        return db[collection_name]


_mongo_manager: Optional[MongoDBManager] = None


def get_mongo_manager() -> MongoDBManager:
    """Get singleton MongoDB manager instance."""
    global _mongo_manager
    if _mongo_manager is None:
        _mongo_manager = MongoDBManager()
    return _mongo_manager


async def connect_to_mongodb() -> bool:
    """Connect to MongoDB (convenience function)."""
    manager = get_mongo_manager()
    return await manager.connect()


async def disconnect_from_mongodb():
    """Disconnect from MongoDB (convenience function)."""
    manager = get_mongo_manager()
    await manager.disconnect()


def get_database():
    """Get database instance (convenience function)."""
    manager = get_mongo_manager()
    return manager.get_db()
