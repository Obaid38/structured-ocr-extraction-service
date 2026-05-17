"""
Dynamic Region Detection
Detects regions using YOLO, coordinates, or hybrid approach
FIXED: Handles both PIL Image and numpy array inputs
"""

from typing import Dict, List, Optional, Union
import logging
import numpy as np
from PIL import Image
from ultralytics import YOLO
from app_settings import get_env_or_config

logger = logging.getLogger("template_engine.region_detector")


class RegionDetector:
    """
    Detects document regions using template configuration
    Returns both image data AND bounding box metadata
    """
    
    def __init__(self):
        logger.info("🔍 RegionDetector initialized")
        self.yolo_models = {}  # Cache for loaded YOLO models
    
    def detect_regions(
        self, 
        image: Union[Image.Image, np.ndarray],
        template: Dict,
        category: str
    ) -> Dict[str, Dict]:  # Returns Dict[str, Dict] with image+bbox
        """
        Detect regions in document image
        
        Args:
            image: Document image as PIL Image or numpy array
            template: Template configuration
            category: Document category (BOL, Receipt, etc.)
            
        Returns:
            Dict of {region_name: {"image": np.ndarray, "bbox": tuple, "confidence": float}}
        """
        logger.info("\n" + "=" * 70)
        logger.info("🔍 REGION_DETECTION: Starting region detection")
        logger.info("=" * 70)
        
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            logger.info(f"  Input type: PIL Image (size: {image.size})")
            image_np = np.array(image)
            logger.info(f"  Converted to numpy array (shape: {image_np.shape})")
        else:
            logger.info(f"  Input type: numpy array (shape: {image.shape})")
            image_np = image
        
        logger.info(f"  Category: {category}")
        
        if template is None:
            logger.info(f"  Template: None (using default config for {category})")
            # Default configuration when no template provided
            region_config = {
                "detection_method": "yolo",
                "yolo_config": {
                    "model_path": get_env_or_config(
                        "REGION_MODEL_PATH",
                        ("models", "region_detector"),
                        "Models/bol_regions_best.pt",
                    ),
                    "confidence_threshold": 0.6,
                    "classes": []  # Will detect all regions
                }
            }
        else:
            region_config = template.get("region_config", {})

        detection_method = region_config.get("detection_method", "yolo")
        
        logger.info(f"  Detection method: {detection_method}")
        
        detected_regions = {}
        
        try:
            if detection_method == "yolo":
                detected_regions = self._detect_with_yolo(image_np, region_config, category)
            
            elif detection_method == "coordinates":
                detected_regions = self._detect_with_coordinates(image_np, region_config)
            
            elif detection_method == "hybrid":
                # Try YOLO first, fallback to coordinates
                detected_regions = self._detect_with_yolo(image_np, region_config, category)
                
                if not detected_regions:
                    logger.info("  ⚠️  YOLO failed, falling back to coordinates")
                    detected_regions = self._detect_with_coordinates(image_np, region_config)
            
            logger.info("\n" + "=" * 70)
            logger.info(f"✅ REGION_DETECTION: Detected {len(detected_regions)} region(s)")
            for region_name, region_data in detected_regions.items():
                bbox = region_data.get('bbox', 'N/A')
                conf = region_data.get('confidence', 0)
                logger.info(f"    - {region_name}: bbox={bbox}, conf={conf:.2f}")
            logger.info("=" * 70 + "\n")
            
            return detected_regions
            
        except Exception as e:
            logger.error(f"❌ Error in region detection: {e}")
            logger.exception("Full traceback:")
            return {}
    
    def _detect_with_yolo(
        self, 
        image: np.ndarray, 
        region_config: Dict,
        category: str
    ) -> Dict[str, Dict]:
        """Detect regions using YOLO model"""
        logger.info("  🤖 Using YOLO detection")
        
        try:
            yolo_config = region_config.get("yolo_config", {})
            model_path = yolo_config.get("model_path", "")
            conf_threshold = yolo_config.get("confidence_threshold", 0.6)
            
            logger.info(f"    Model path: {model_path}")
            logger.info(f"    Confidence threshold: {conf_threshold}")
            
            # CACHE: Check if model already loaded
            if model_path not in self.yolo_models:
                logger.info(f"    🔄 Loading YOLO model: {model_path}")
                self.yolo_models[model_path] = YOLO(model_path)
                logger.info(f"    ✅ Model loaded and cached")
            else:
                logger.debug(f"    ✓ Using cached YOLO model: {model_path}")

            model = self.yolo_models[model_path]
            
            # Run detection
            results = model.predict(image, conf=conf_threshold, verbose=False)
            logger.info(f"    Detections: {len(results[0].boxes) if results else 0}")
            
            # Extract regions
            detected_regions = {}
            classes = yolo_config.get("classes", [])

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Find region name for this class_id
                    region_name = None
                    
                    # ✅ FIX: If no classes defined, use YOLO's class names
                    if not classes:
                        # Use model's class names directly
                        if hasattr(result, 'names') and class_id in result.names:
                            region_name = result.names[class_id]
                        else:
                            region_name = f"region_{class_id}"  # Fallback
                    else:
                        # Use template-defined mapping
                        for class_config in classes:
                            if str(class_config.get("class_id")) == str(class_id):
                                region_name = class_config.get("region_name")
                                break
                    
                    if region_name:
                        # Extract region
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        region_image = image[int(y1):int(y2), int(x1):int(x2)]
                        
                        # Return dict with metadata
                        detected_regions[region_name] = {
                            "image": region_image,
                            "bbox": (int(x1), int(y1), int(x2), int(y2)),
                            "confidence": confidence
                        }
                        logger.info(f"    ✓ Detected: {region_name} (conf: {confidence:.2f}, bbox: ({int(x1)},{int(y1)},{int(x2)},{int(y2)}))")
            
            return detected_regions
            
        except Exception as e:
            logger.error(f"    ❌ YOLO detection failed: {e}")
            return {}
    
    def _detect_with_coordinates(
        self, 
        image: np.ndarray, 
        region_config: Dict
    ) -> Dict[str, Dict]:
        """Detect regions using fixed coordinates"""
        logger.info("  📐 Using coordinate-based detection")
        
        try:
            coord_regions = region_config.get("coordinate_regions", [])
            logger.info(f"    Configured regions: {len(coord_regions)}")
            
            detected_regions = {}
            h, w = image.shape[:2]
            
            for coord_config in coord_regions:
                region_name = coord_config.get("region_name")
                x1_ratio = coord_config.get("x1_ratio", 0)
                y1_ratio = coord_config.get("y1_ratio", 0)
                x2_ratio = coord_config.get("x2_ratio", 1)
                y2_ratio = coord_config.get("y2_ratio", 1)
                
                # Convert ratios to pixels
                x1 = int(x1_ratio * w)
                y1 = int(y1_ratio * h)
                x2 = int(x2_ratio * w)
                y2 = int(y2_ratio * h)
                
                # Extract region
                region_image = image[y1:y2, x1:x2]
                
                # Return dict with metadata
                detected_regions[region_name] = {
                    "image": region_image,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": 1.0  # Coordinate-based has 100% confidence
                }
                
                logger.info(f"    ✓ Extracted: {region_name} at ({x1},{y1})-({x2},{y2})")
            
            return detected_regions
            
        except Exception as e:
            logger.error(f"    ❌ Coordinate detection failed: {e}")
            return {}
