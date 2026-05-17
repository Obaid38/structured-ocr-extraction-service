"""
Template Loader
Loads active templates from MongoDB and stores them in memory
Handles image URL prefixing with base URL
"""

import logging
from typing import List, Dict, Optional
from mongo_db import get_mongo_manager
from config_manager import get_config_manager
from template_engine import get_template_store

logger = logging.getLogger("template_loader")


class TemplateLoader:
    """
    Loads templates from MongoDB and manages template lifecycle
    """
    
    def __init__(self):
        self.loaded_count = 0
        self.updated_count = 0
        self.failed_count = 0
        logger.info("🔧 TemplateLoader initialized")
    
    async def load_active_templates_from_db(self) -> bool:
        """
        Load all active templates from MongoDB and store in memory
        
        Returns:
            True if templates loaded successfully
        """
        try:
            mongo_manager = get_mongo_manager()
            
            if not mongo_manager.is_connected:
                logger.error("❌ MongoDB not connected, cannot load templates")
                return False
            
            logger.info("\n" + "=" * 80)
            logger.info("📥 LOADING ACTIVE TEMPLATES FROM DATABASE")
            logger.info("=" * 80)
            
            # Get templates collection
            templates_collection = await mongo_manager.get_collection("templates")
            
            if templates_collection is None:
                logger.error("❌ Could not access templates collection")
                return False
            
            # Find all active templates
            cursor = templates_collection.find({"status": "active"})
            templates = await cursor.to_list(length=None)
            
            logger.info(f"✅ Found {len(templates)} active template(s) in database")
            
            if len(templates) == 0:
                logger.warning("⚠️  No active templates found in database")
                return True  # Not an error, just no templates
            
            # Get template store
            template_store = get_template_store()
            
            # Reset counters
            self.loaded_count = 0
            self.updated_count = 0
            self.failed_count = 0
            
            # Load each template
            for idx, template_doc in enumerate(templates, 1):
                logger.info(f"\n📋 Processing template {idx}/{len(templates)}")
                await self._load_single_template(template_doc, template_store)
            
            logger.info("\n" + "=" * 80)
            logger.info("📊 TEMPLATE LOADING SUMMARY")
            logger.info("=" * 80)
            logger.info(f"  ✅ Loaded: {self.loaded_count}")
            logger.info(f"  🔄 Updated: {self.updated_count}")
            logger.info(f"  ❌ Failed: {self.failed_count}")
            logger.info(f"  📊 Total in memory: {len(template_store._templates)}")
            logger.info("=" * 80 + "\n")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading templates from database: {e}")
            logger.exception("Full traceback:")
            return False
    
    async def _load_single_template(self, template_doc: Dict, template_store) -> bool:
        """
        Load a single template into memory
        
        Args:
            template_doc: Template document from MongoDB
            template_store: Template store instance
        
        Returns:
            True if successful
        """
        try:
            # ✅ NEW: Convert ObjectId to string
            if "_id" in template_doc:
                from bson import ObjectId
                if isinstance(template_doc["_id"], ObjectId):
                    template_doc["_id"] = str(template_doc["_id"])
                    logger.info(f"  Converted MongoDB ObjectId to string: {template_doc['_id']}")

            template_id = template_doc.get("template_id", "Unknown")
            template_name = template_doc.get("template_name", "Unknown")
            
            logger.info(f"  Template ID: {template_id}")
            logger.info(f"  Template Name: {template_name}")
            
            # Fix image URLs
            template_doc = self._fix_image_urls(template_doc)
            
            # Check if template already exists in memory
            existing = template_store.get_template(template_id)
            
            if existing:
                logger.info(f"  ℹ️  Template already in memory, updating...")
                # Update existing template
                success = template_store.add_template(template_doc)
                if success:
                    logger.info(f"  ✅ Updated template: {template_id}")
                    self.updated_count += 1
                else:
                    logger.error(f"  ❌ Failed to update template: {template_id}")
                    self.failed_count += 1
            else:
                # Add new template
                success = template_store.add_template(template_doc)
                if success:
                    logger.info(f"  ✅ Loaded new template: {template_id}")
                    self.loaded_count += 1
                else:
                    logger.error(f"  ❌ Failed to load template: {template_id}")
                    self.failed_count += 1
            
            return success
            
        except Exception as e:
            logger.error(f"  ❌ Error loading template: {e}")
            self.failed_count += 1
            return False
    
    def _fix_image_urls(self, template_doc: Dict) -> Dict:
        """
        Fix reference image URLs by prepending base URL
        
        Handles TWO structures:
        1. MongoDB: identification.reference_images[{file_path: "..."}]
        2. Frontend: reference_images["...", "..."]
        
        Args:
            template_doc: Template document
        
        Returns:
            Template document with fixed URLs
        """
        config_manager = get_config_manager()
        base_url = config_manager.get_base_url()
        
        if not base_url:
            logger.warning("  ⚠️  No base URL configured, skipping image URL fix")
            return template_doc
        
        logger.info(f"  🔗 Base URL: {base_url}")
        
        # ========== STRUCTURE 1: MongoDB format (nested in identification) ==========
        identification = template_doc.get("identification", {})
        ref_images = identification.get("reference_images", [])
        
        if ref_images and isinstance(ref_images, list) and len(ref_images) > 0:
            # Check if it's array of objects with file_path
            if isinstance(ref_images[0], dict) and "file_path" in ref_images[0]:
                logger.info(f"  📁 Fixing {len(ref_images)} reference image(s) - MongoDB format")
                
                for ref_img in ref_images:
                    if "file_path" in ref_img:
                        original_path = ref_img["file_path"]
                        
                        # Skip if already full URL
                        if original_path.startswith(("http://", "https://")):
                            logger.info(f"    ✓ Already full URL: {original_path[:60]}...")
                            continue
                        
                        # Build full URL
                        full_url = config_manager.get_full_image_url(original_path)
                        ref_img["file_path"] = full_url
                        
                        logger.info(f"    ✓ Fixed: {original_path[:50]}...")
                        logger.info(f"      → {full_url[:70]}...")
        
        # ========== STRUCTURE 2: Frontend format (direct array of strings) ==========
        # This is what test_template API sends!
        if "reference_images" in template_doc:
            ref_images_direct = template_doc.get("reference_images", [])
            
            if ref_images_direct and isinstance(ref_images_direct, list) and len(ref_images_direct) > 0:
                # Check if it's array of strings
                if isinstance(ref_images_direct[0], str):
                    logger.info(f"  📁 Fixing {len(ref_images_direct)} reference image(s) - Frontend format")
                    
                    fixed_images = []
                    for img_url in ref_images_direct:
                        # Skip if already full URL
                        if img_url.startswith(("http://", "https://")):
                            fixed_images.append(img_url)
                            logger.info(f"    ✓ Already full URL: {img_url[:60]}...")
                            continue
                        
                        # Build full URL
                        # If starts with /, use get_full_image_url
                        if img_url.startswith("/"):
                            full_url = config_manager.get_full_image_url(img_url)
                        else:
                            full_url = f"{base_url}/{img_url}"
                        
                        fixed_images.append(full_url)
                        logger.info(f"    ✓ Fixed: {img_url[:50]}...")
                        logger.info(f"      → {full_url[:70]}...")
                    
                    # Update with fixed URLs
                    template_doc["reference_images"] = fixed_images
                    logger.info(f"  ✅ Fixed all {len(fixed_images)} reference image URL(s)")
        
        return template_doc

# Global template loader instance
_template_loader: Optional[TemplateLoader] = None


def get_template_loader() -> TemplateLoader:
    """
    Get global template loader instance (singleton)
    
    Returns:
        TemplateLoader instance
    """
    global _template_loader
    
    if _template_loader is None:
        _template_loader = TemplateLoader()
    
    return _template_loader


async def load_templates_from_database() -> bool:
    """
    Load templates from database (convenience function)
    
    Returns:
        True if successful
    """
    loader = get_template_loader()
    return await loader.load_active_templates_from_db()