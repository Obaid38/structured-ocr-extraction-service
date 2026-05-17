"""
Startup Initialization
Handles application startup sequence:
1. Connect to MongoDB
2. Load configuration from settings
3. Load active templates into memory
"""

import asyncio
import logging

from config_manager import get_config_manager, initialize_config
from mongo_db import connect_to_mongodb, get_mongo_manager
from template_loader import get_template_loader, load_templates_from_database

logger = logging.getLogger("startup")


async def initialize_application() -> bool:
    """
    Initialize application on startup.

    Returns:
        True if initialization successful, False otherwise.
    """
    logger.info("\n" + "=" * 80)
    logger.info("APPLICATION STARTUP INITIALIZATION")
    logger.info("=" * 80)

    try:
        logger.info("\nSTEP 1: Connecting to MongoDB...")
        logger.info("-" * 80)

        mongo_connected = await connect_to_mongodb()
        if not mongo_connected:
            logger.error("Failed to connect to MongoDB")
            logger.error("Application will run with limited functionality")
            logger.error("Template matching will not work without templates")
            return False

        logger.info("MongoDB connection successful")

        logger.info("\nSTEP 2: Loading runtime configuration...")
        logger.info("-" * 80)

        config_loaded = await initialize_config()
        config_manager = get_config_manager()

        if not config_loaded:
            logger.warning("Could not load valid secondary IP from MongoDB settings")
            logger.warning("Using fallback/override base URL configuration")

        logger.info("Configuration status:")
        logger.info("  Secondary IP: %s", config_manager.get_secondary_ip() or "Not configured")
        logger.info("  Base URL: %s", config_manager.get_base_url() or "Not configured")
        logger.info("  Base URL source: %s", config_manager.get_base_url_source())

        logger.info("\nSTEP 3: Loading active templates from database...")
        logger.info("-" * 80)

        templates_loaded = await load_templates_from_database()
        if not templates_loaded:
            logger.error("Failed to load templates from database")
            logger.error("Template matching will not work without templates")
            return False

        logger.info("Templates loaded successfully")

        logger.info("\n" + "=" * 80)
        logger.info("APPLICATION INITIALIZATION COMPLETE")
        logger.info("=" * 80)

        mongo_manager = get_mongo_manager()
        template_loader = get_template_loader()

        logger.info("  MongoDB: %s", "Connected" if mongo_manager.is_connected else "Disconnected")
        logger.info("  Secondary IP: %s", config_manager.get_secondary_ip() or "Not configured")
        logger.info("  Base URL: %s", config_manager.get_base_url() or "Not configured")
        logger.info("  Base URL source: %s", config_manager.get_base_url_source())
        logger.info("  Templates loaded: %s", template_loader.loaded_count)
        logger.info("  Templates updated: %s", template_loader.updated_count)

        if template_loader.failed_count > 0:
            logger.warning("  Templates failed: %s", template_loader.failed_count)

        logger.info("=" * 80 + "\n")
        return True

    except Exception as exc:
        logger.error("Critical error during initialization: %s", exc)
        logger.exception("Full traceback:")
        logger.error("\n" + "=" * 80)
        logger.error("APPLICATION INITIALIZATION FAILED")
        logger.error("=" * 80 + "\n")
        return False


async def shutdown_application():
    """Graceful shutdown - disconnect from MongoDB."""
    logger.info("\n" + "=" * 80)
    logger.info("APPLICATION SHUTDOWN")
    logger.info("=" * 80)

    try:
        from mongo_db import disconnect_from_mongodb

        await disconnect_from_mongodb()
        logger.info("Disconnected from MongoDB")
    except Exception as exc:
        logger.error("Error during shutdown: %s", exc)

    logger.info("=" * 80 + "\n")


def run_initialization():
    """Run initialization synchronously (for non-async contexts)."""
    return asyncio.run(initialize_application())


def run_shutdown():
    """Run shutdown synchronously (for non-async contexts)."""
    return asyncio.run(shutdown_application())
