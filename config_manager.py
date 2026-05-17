"""
Configuration Manager
Loads frontend connectivity settings from MongoDB and environment variables.
"""

import ipaddress
import logging
import os
from typing import Dict, Optional
from urllib.parse import urlparse, urlunparse

from app_settings import get_env_or_config, get_int_env_or_config
from mongo_db import get_mongo_manager

logger = logging.getLogger("config_manager")

DEFAULT_FRONTEND_SCHEME = str(
    get_env_or_config("FRONTEND_SCHEME", ("frontend", "scheme"), "http")
)
DEFAULT_FRONTEND_PORT = get_int_env_or_config(
    "FRONTEND_PORT", ("frontend", "port"), 3000
)
DEFAULT_FALLBACK_BASE_URL = str(
    get_env_or_config(
        "BASE_URL_FALLBACK",
        ("frontend", "fallback_base_url"),
        "http://localhost:3000",
    )
)


class ConfigManager:
    """Configuration manager that loads settings from MongoDB."""

    def __init__(self):
        self.base_url: Optional[str] = None
        self.secondary_ip: Optional[str] = None
        self.settings: Dict = {}
        self.base_url_source: str = "uninitialized"
        self.env = os.getenv("ENVIRONMENT", "development")
        logger.info("ConfigManager initialized for environment: %s", self.env)

    async def load_settings_from_db(self) -> bool:
        """
        Load runtime config from MongoDB settings collection.
        Reads only the secondary IP document where type='secondary'.

        Returns:
            True if a valid secondary IP was loaded from MongoDB.
        """
        self.secondary_ip = None
        self.settings = {}
        loaded_secondary_ip = False

        try:
            mongo_manager = get_mongo_manager()
            if not mongo_manager.is_connected:
                logger.warning("MongoDB is not connected; cannot fetch secondary IP from settings")
            else:
                settings_collection = await mongo_manager.get_collection("settings")
                if settings_collection is None:
                    logger.warning("Could not access settings collection")
                else:
                    logger.info("Loading MongoDB settings document where type='secondary'...")
                    secondary_doc = await settings_collection.find_one(
                        {"type": "secondary"},
                        {"_id": 1, "type": 1, "ip": 1},
                    )

                    if not secondary_doc:
                        logger.warning("No settings document found for type='secondary'")
                    else:
                        self.settings = {"secondary": secondary_doc}
                        self.secondary_ip = self._normalize_secondary_ip(
                            secondary_doc.get("ip")
                        )

                        if self.secondary_ip:
                            loaded_secondary_ip = True
                            logger.info(
                                "Loaded secondary IP from MongoDB settings: %s",
                                self.secondary_ip,
                            )
                        else:
                            logger.warning(
                                "Found type='secondary' in settings but IP is missing/invalid"
                            )

        except Exception as exc:
            logger.error("Error loading settings from database: %s", exc)
            logger.exception("Full traceback:")

        # Always build a base URL, even if DB loading failed.
        self._build_base_url()
        logger.info(
            "Resolved base URL: %s (source=%s)",
            self.base_url,
            self.base_url_source,
        )

        return loaded_secondary_ip

    def _normalize_secondary_ip(self, raw_ip: Optional[str]) -> Optional[str]:
        """Validate and normalize secondary IP value."""
        if raw_ip is None:
            logger.warning("Secondary IP field is missing in MongoDB document")
            return None

        ip_value = str(raw_ip).strip()
        if not ip_value:
            logger.warning("Secondary IP in MongoDB is empty")
            return None

        try:
            return str(ipaddress.ip_address(ip_value))
        except ValueError:
            logger.warning("Secondary IP is not a valid IPv4/IPv6 value: %r", ip_value)
            return None

    def _get_frontend_scheme(self) -> str:
        """Get frontend scheme for secondary-IP base URL construction."""
        scheme = (os.getenv("FRONTEND_SCHEME") or DEFAULT_FRONTEND_SCHEME).strip().lower()
        if scheme not in {"http", "https"}:
            logger.warning(
                "Invalid FRONTEND_SCHEME=%r. Using default scheme '%s'.",
                scheme,
                DEFAULT_FRONTEND_SCHEME,
            )
            return DEFAULT_FRONTEND_SCHEME
        return scheme

    def _get_frontend_port(self) -> int:
        """Get frontend port for secondary-IP base URL construction."""
        raw_port = (os.getenv("FRONTEND_PORT") or str(DEFAULT_FRONTEND_PORT)).strip()
        try:
            port = int(raw_port)
            if port < 1 or port > 65535:
                raise ValueError("Port out of valid range")
            return port
        except ValueError:
            logger.warning(
                "Invalid FRONTEND_PORT=%r. Using default port %d.",
                raw_port,
                DEFAULT_FRONTEND_PORT,
            )
            return DEFAULT_FRONTEND_PORT

    def _normalize_base_url(self, raw_base_url: str) -> Optional[str]:
        """Normalize and validate base URL from environment override."""
        value = (raw_base_url or "").strip()
        if not value:
            return None

        if not value.startswith(("http://", "https://")):
            logger.warning("BASE_URL has no scheme; assuming http://")
            value = f"http://{value}"

        parsed = urlparse(value)
        if not parsed.scheme or not parsed.netloc:
            logger.warning("BASE_URL is invalid and will be ignored: %r", raw_base_url)
            return None

        normalized = parsed._replace(query="", fragment="")
        return urlunparse(normalized).rstrip("/")

    def _build_base_url(self):
        """Build base URL using env override -> secondary IP -> localhost fallback."""
        explicit_base_url = self._normalize_base_url(os.getenv("BASE_URL", ""))
        if explicit_base_url:
            self.base_url = explicit_base_url
            self.base_url_source = "env.BASE_URL"
            logger.info("Using BASE_URL from environment override: %s", self.base_url)
            return

        if self.secondary_ip:
            scheme = self._get_frontend_scheme()
            port = self._get_frontend_port()
            self.base_url = f"{scheme}://{self.secondary_ip}:{port}"
            self.base_url_source = "mongodb.settings.secondary.ip"
            logger.info(
                "Built base URL from MongoDB secondary IP: %s", self.base_url
            )
            return

        self.base_url = DEFAULT_FALLBACK_BASE_URL
        self.base_url_source = "fallback.localhost"
        logger.warning(
            "Secondary IP unavailable. Falling back to base URL: %s",
            self.base_url,
        )

    def get_base_url(self) -> Optional[str]:
        """Get resolved base URL for frontend resource access."""
        if not self.base_url:
            logger.warning("Base URL not configured")
        return self.base_url

    def get_base_url_source(self) -> str:
        """Get source used to resolve base URL."""
        return self.base_url_source

    def get_secondary_ip(self) -> Optional[str]:
        """Get validated secondary IP from MongoDB settings."""
        return self.secondary_ip

    def get_full_image_url(self, relative_path: str) -> str:
        """
        Convert relative image path to full URL.
        Example: api/templates/serve-image/abc.png -> http://host:3000/api/templates/serve-image/abc.png
        """
        if relative_path.startswith(("http://", "https://")):
            return relative_path

        if not self.base_url:
            logger.warning(
                "No base URL configured, returning original relative path: %s",
                relative_path,
            )
            return relative_path

        normalized_relative = relative_path.lstrip("/")
        base = self.base_url.rstrip("/")
        return f"{base}/{normalized_relative}"

    def get_settings(self) -> Dict:
        """Get loaded settings payload."""
        return self.settings


_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get singleton config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


async def initialize_config() -> bool:
    """Initialize configuration from MongoDB."""
    config_manager = get_config_manager()
    return await config_manager.load_settings_from_db()
