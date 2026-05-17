"""
Post-Processing Engine
Applies validation, transformation, and business logic rules
"""

from typing import Dict, List, Any
import logging
import re
from datetime import datetime

logger = logging.getLogger("template_engine.post_processor")


class PostProcessor:
    """
    Applies post-processing rules from template
    """
    
    def __init__(self):
        logger.info("🔧 PostProcessor initialized")
    
    def apply_field_mapping(
        self, 
        extracted_data: Dict[str, Any], 
        template: Dict
    ) -> Dict[str, Any]:
        """
        Map extracted fields to final output format
        
        Args:
            extracted_data: Raw OCR output {region.field: value}
            template: Template with field_mapping config
            
        Returns:
            Mapped data dict
        """
        logger.info("\n" + "=" * 70)
        logger.info("🗺️  FIELD_MAPPING: Mapping extracted fields")
        logger.info("=" * 70)
        logger.info(f"  Extracted fields: {len(extracted_data)}")
        
        field_mapping = template.get("field_mapping", {})
        mapped_data = {}
        
        for target_field, mapping_config in field_mapping.items():
            source_field = mapping_config.get("source_field")
            fallback_fields = mapping_config.get("fallback_fields", [])
            default_value = mapping_config.get("default_value")
            
            logger.debug(f"  📌 Mapping: {target_field} <- {source_field}")
            
            # Try source field
            value = extracted_data.get(source_field)
            
            # Try fallback fields
            if value is None:
                for fallback in fallback_fields:
                    value = extracted_data.get(fallback)
                    if value is not None:
                        logger.debug(f"    ✓ Used fallback: {fallback}")
                        break
            
            # Use default if still None
            if value is None:
                value = default_value
                logger.debug(f"    ✓ Used default: {default_value}")
            
            mapped_data[target_field] = value
        
        logger.info(f"✅ FIELD_MAPPING: Mapped {len(mapped_data)} fields")
        logger.info("=" * 70 + "\n")
        
        return mapped_data
    
    def apply_rules(
        self, 
        data: Dict[str, Any], 
        template: Dict
    ) -> Dict[str, Any]:
        """
        Apply validation, transformation, and business logic rules
        
        Args:
            data: Mapped data
            template: Template with post_processing_rules
            
        Returns:
            Processed data
        """
        logger.info("\n" + "=" * 70)
        logger.info("🔧 POST_PROCESSING: Applying rules")
        logger.info("=" * 70)
        
        rules = template.get("post_processing_rules", {})
        
        # Apply validation rules
        validation_rules = rules.get("validation_rules", [])
        logger.info(f"  Validation rules: {len(validation_rules)}")
        data = self._apply_validation_rules(data, validation_rules)
        
        # Apply transformation rules
        transformation_rules = rules.get("transformation_rules", [])
        logger.info(f"  Transformation rules: {len(transformation_rules)}")
        data = self._apply_transformation_rules(data, transformation_rules)
        
        # Apply business logic
        business_logic = rules.get("business_logic", [])
        logger.info(f"  Business logic rules: {len(business_logic)}")
        data = self._apply_business_logic(data, business_logic)
        
        logger.info("✅ POST_PROCESSING: Rules applied")
        logger.info("=" * 70 + "\n")
        
        return data
    
    def _apply_validation_rules(
        self, 
        data: Dict[str, Any], 
        rules: List[Dict]
    ) -> Dict[str, Any]:
        """Apply validation rules"""
        for rule in rules:
            rule_name = rule.get("rule_name")
            field = rule.get("field")
            action = rule.get("action")
            
            logger.debug(f"    Validating: {rule_name}")
            
            # TODO: Implement actual validation logic
            # For now, placeholder
        
        return data
    
    def _apply_transformation_rules(
        self, 
        data: Dict[str, Any], 
        rules: List[Dict]
    ) -> Dict[str, Any]:
        """Apply transformation rules"""
        for rule in rules:
            rule_name = rule.get("rule_name")
            field = rule.get("field")
            action = rule.get("action")
            
            logger.debug(f"    Transforming: {rule_name}")
            
            # TODO: Implement actual transformation logic
        
        return data
    
    def _apply_business_logic(
        self, 
        data: Dict[str, Any], 
        rules: List[Dict]
    ) -> Dict[str, Any]:
        """Apply business logic rules"""
        for rule in rules:
            rule_name = rule.get("rule_name")
            condition = rule.get("condition")
            action = rule.get("action")
            
            logger.debug(f"    Business logic: {rule_name}")
            
            # TODO: Implement actual business logic
        
        return data