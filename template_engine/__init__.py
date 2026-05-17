"""
Template Engine Module
Handles in-memory template storage, matching, and processing
"""

from .memory_store import TemplateMemoryStore, get_template_store
from .matcher import TemplateMatcher
from .region_detector import RegionDetector
from .prompt_loader import PromptLoader
from .post_processor import PostProcessor

__all__ = [
    'TemplateMemoryStore',
    'get_template_store',
    'TemplateMatcher',
    'RegionDetector',
    'PromptLoader',
    'PostProcessor'
]