"""
In-Memory Template Storage
Stores active templates received from Backend via /api/templates/sync
"""

from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger("template_engine.memory_store")


class TemplateMemoryStore:
    """
    Singleton class to store templates in memory
    Templates are organized by category for faster lookup
    """
    
    _instance = None
    _templates: Dict[str, Dict] = {}  # {template_id: template_data}
    _templates_by_category: Dict[str, List[str]] = {}  # {category: [template_ids]}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._templates = {}
            cls._instance._templates_by_category = {}
            logger.info("=" * 70)
            logger.info("🎯 TemplateMemoryStore initialized (Singleton)")
            logger.info("=" * 70)
        return cls._instance
    
    def add_template(self, template: Dict) -> bool:
        """
        Add or update template in memory
        
        Args:
            template: Template dict with all configuration
            
        Returns:
            bool: Success status
        """
        logger.info("\n" + "=" * 70)
        logger.info("📥 ADD_TEMPLATE: Starting template addition/update")
        logger.info("=" * 70)
        
        try:
            template_id = template.get("template_id")
            # category = template.get("category", "").lower()
            category = "bol"
            template_name = template.get("template_name", "Unknown")
            
            logger.info(f"  Template ID: {template_id}")
            logger.info(f"  Template Name: {template_name}")
            logger.info(f"  Category: {category}")
            
            if not template_id:
                logger.error("  ❌ Template missing template_id!")
                return False
            
            # Check if updating existing template
            is_update = template_id in self._templates
            action = "UPDATE" if is_update else "ADD"
            
            logger.info(f"  Action: {action}")
            
            # Store template
            self._templates[template_id] = template
            logger.info(f"  ✅ Template stored in memory")
            
            # ✅ FIX: Clean up category index first if updating
            if category and category in self._templates_by_category:
                # Remove any existing entries for this template_id (prevent duplicates)
                self._templates_by_category[category] = [
                    tid for tid in self._templates_by_category[category] 
                    if tid != template_id
                ]
            
            # Index by category
            if category:
                if category not in self._templates_by_category:
                    self._templates_by_category[category] = []
                    logger.info(f"  📁 Created new category index: {category}")
                
                if template_id not in self._templates_by_category[category]:
                    self._templates_by_category[category].append(template_id)
                    logger.info(f"  🔗 Linked template to category: {category}")
            
            # Log template details
            regions = template.get("region_config", {}).get("yolo_config", {}).get("classes", [])
            logger.info(f"  Regions configured: {len(regions)}")
            
            prompts = template.get("prompts", {})
            logger.info(f"  Prompts configured: {list(prompts.keys())}")
            
            field_mappings = template.get("field_mapping", {})
            logger.info(f"  Field mappings: {len(field_mappings)}")
            
            logger.info("=" * 70)
            logger.info(f"✅ Template {template_id} successfully {action}ED to memory")
            logger.info("=" * 70 + "\n")
            
            return True
            
        except Exception as e:
            logger.error("=" * 70)
            logger.error(f"❌ Error adding template: {e}")
            logger.exception("Full traceback:")
            logger.error("=" * 70 + "\n")
            return False
    
    def remove_template(self, template_id: str) -> bool:
        """
        Remove template from memory
        Supports both template_id string and MongoDB _id
        """
        logger.info("\n" + "=" * 70)
        logger.info(f"🗑️  REMOVE_TEMPLATE: Removing template {template_id}")
        logger.info("=" * 70)
        
        try:
            # Strategy 1: Direct lookup by template_id key
            if template_id in self._templates:
                category = self._templates[template_id].get("category", "").lower()
                logger.info(f"  Category: {category}")
                
                # Remove from main storage
                del self._templates[template_id]
                logger.info(f"  ✅ Removed from main storage")
                
                # Remove from category index using template_id
                if category and category in self._templates_by_category:
                    if template_id in self._templates_by_category[category]:
                        self._templates_by_category[category].remove(template_id)
                        logger.info(f"  ✅ Removed from category index")
                        
                    if not self._templates_by_category[category]:
                        del self._templates_by_category[category]
                        logger.info(f"  🧹 Cleaned up empty category: {category}")
                
                logger.info("=" * 70)
                logger.info(f"✅ Template {template_id} successfully removed")
                logger.info("=" * 70 + "\n")
                return True
            
            # Strategy 2: Search by MongoDB _id
            for key, stored_template in list(self._templates.items()):
                if stored_template.get("_id") == template_id:
                    category = stored_template.get("category", "").lower()
                    logger.info(f"  Found by MongoDB _id, actual key: {key}")
                    logger.info(f"  Category: {category}")
                    
                    # Remove from main storage using actual key
                    del self._templates[key]
                    logger.info(f"  ✅ Removed from main storage")
                    
                    # Remove from category index using actual key (NOT template_id!)
                    if category and category in self._templates_by_category:
                        if key in self._templates_by_category[category]:
                            self._templates_by_category[category].remove(key)
                            logger.info(f"  ✅ Removed from category index")
                            
                        if not self._templates_by_category[category]:
                            del self._templates_by_category[category]
                            logger.info(f"  🧹 Cleaned up empty category: {category}")
                    
                    logger.info("=" * 70)
                    logger.info(f"✅ Template {template_id} successfully removed")
                    logger.info("=" * 70 + "\n")
                    return True
            
            # Not found
            logger.warning(f"  ⚠️  Template {template_id} not found in memory")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error removing template: {e}")
            logger.exception("Full traceback:")
            return False

    def get_template(self, template_id: str) -> Optional[Dict]:
        """
        Get template by template_id string OR MongoDB _id
        
        Args:
            template_id: Can be either:
                - Template ID string (e.g., "STAMP_STANDARD_V555")
                - MongoDB _id (e.g., "694916bf20ced4b680db6f3f")
        
        Returns:
            Template dict or None
        """
        # First, try direct lookup (template_id string is the key)
        template = self._templates.get(template_id)
        if template:
            return template
        
        # If not found, search by MongoDB _id
        for stored_template in self._templates.values():
            if stored_template.get("_id") == template_id:
                return stored_template
        
        # Not found
        return None
    
    def get_templates_by_category(self, category: str) -> List[Dict]:
        """Get all templates for a category"""
        logger.info(f"\n🔍 GET_TEMPLATES_BY_CATEGORY: {category}")
        category = category.lower()
        template_ids = self._templates_by_category.get(category, [])
        templates = [self._templates[tid] for tid in template_ids if tid in self._templates]
        
        template_info = " | ".join(
            f"({t.get('template_id')}, {t.get('_id')})"
            for t in templates
        )

        logger.info(
            f"Found {len(templates)} template(s) in category '{category}'"
            + (f" | {template_info}" if template_info else "")
        )
        
        return templates
    
    def get_all_templates(self) -> List[Dict]:
        """Get all templates in memory"""
        return list(self._templates.values())
    
    def clear_all(self) -> None:
        """Clear all templates (use with caution)"""
        self._templates.clear()
        self._templates_by_category.clear()
        logger.warning("⚠️  ALL TEMPLATES CLEARED FROM MEMORY!")
    
    def get_stats(self) -> Dict:
        """Get memory store statistics with validation"""
        
        # Count valid templates in category lists
        validated_by_category = {}
        for cat, tids in self._templates_by_category.items():
            # Count only IDs that actually exist in _templates
            valid_count = sum(1 for tid in tids if tid in self._templates)
            validated_by_category[cat] = valid_count
            
            # Log if there's a mismatch
            if len(tids) != valid_count:
                logger.warning(
                    f"⚠️  Category '{cat}' has {len(tids)} entries "
                    f"but only {valid_count} exist in _templates!"
                )
                logger.warning(f"  Category list: {tids}")
                logger.warning(f"  Valid templates: {list(self._templates.keys())}")
                
                # Log the invalid entries
                invalid = [tid for tid in tids if tid not in self._templates]
                logger.warning(f"  Invalid entries in category: {invalid}")
        
        stats = {
            "total_templates": len(self._templates),
            "by_category": validated_by_category  # Use validated counts
        }
        
        logger.info("\n📊 MEMORY STORE STATS:")
        logger.info(f"  Total templates: {stats['total_templates']}")
        logger.info(f"  Categories: {list(stats['by_category'].keys())}")
        for cat, count in stats['by_category'].items():
            raw_count = len(self._templates_by_category.get(cat, []))
            if raw_count != count:
                logger.info(f"    - {cat}: {count} valid (raw: {raw_count}) ⚠️")
            else:
                logger.info(f"    - {cat}: {count} template(s)")
        
        return stats


# Singleton instance
_template_store = None


def get_template_store() -> TemplateMemoryStore:
    """Get singleton instance of TemplateMemoryStore"""
    global _template_store
    if _template_store is None:
        _template_store = TemplateMemoryStore()
    return _template_store