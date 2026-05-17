"""
Dynamic Prompt Loading
Loads prompts from template configuration
"""

from typing import Dict, List
import logging
import numpy as np

logger = logging.getLogger("template_engine.prompt_loader")


class PromptLoader:
    """
    Builds OCR prompts from template configuration
    """
    
    def __init__(self):
        logger.info("📝 PromptLoader initialized")
    
    def build_batch_prompts(
        self, 
        template: Dict, 
        detected_regions: Dict[str, np.ndarray]
    ) -> Dict[str, str]:
        """
        Build prompts for each detected region
        
        Args:
            template: Template configuration
            detected_regions: Dict of {region_name: region_image}
            
        Returns:
            Dict of {region_name: prompt_text}
        """
        logger.info("\n" + "=" * 70)
        logger.info("📝 PROMPT_LOADING: Building prompts for regions")
        logger.info("=" * 70)
        logger.info(f"  Regions to process: {len(detected_regions)}")
        
        prompts_config = template.get("prompts", {})
        batch_prompts = {}
        
        for region_name in detected_regions.keys():
            logger.info(f"\n  📋 Processing region: {region_name}")
            
            if region_name in prompts_config:
                prompt_config = prompts_config[region_name]
                prompt_text = prompt_config.get("prompt_text", "")
                
                logger.info(f"    ✓ Prompt found (length: {len(prompt_text)} chars)")
                logger.info(f"    Expected schema: {list(prompt_config.get('expected_output_schema', {}).keys())}")
                
                batch_prompts[region_name] = prompt_text
            else:
                logger.warning(f"    ⚠️  No prompt configured for region: {region_name}")
        
        logger.info("\n" + "=" * 70)
        logger.info(f"✅ PROMPT_LOADING: Built {len(batch_prompts)} prompt(s)")
        logger.info("=" * 70 + "\n")
        
        return batch_prompts