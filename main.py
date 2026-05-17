"""
Optimized Main OCR Processing Module
Handles BOL and Receipt document processing with YOLO detection and VLM OCR
"""
import os
import json
import re
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter
from pathlib import Path
from io import BytesIO
import base64
import copy

# Third-party imports
import numpy as np
import cv2
import torch
import httpx
from PIL import Image
from pydantic import BaseModel
from fastapi import HTTPException
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils.plotting import save_one_box
from doctr.io import DocumentFile
from doctr.models import page_orientation_predictor, mobilenet_v3_small_page_orientation
from langsmith import traceable
from dotenv import load_dotenv

# Local imports
from utils import (
    dynamic_preprocess,
    build_transform,
    convert_image_to_base64,
    save_image_to_tmp_dir,
    make_image_square,
    base64topil,
    preprocess_date,
    process_customer_numbers,
    adjust_date,
    resize_large_image,
    safe_int_conversion,
    safe_str_conversion,
    safe_numeric_operation,
    normalize_ocr_data,
    heuristic_stamp_crop_base64
)
import models
import prompts
from app_settings import get_env_or_config

# ============================================================================
# CONFIGURATION & INITIALIZATION
# ============================================================================

load_dotenv()
OCR_PORT = 3203

# Configure logging
logging.basicConfig(
    level=get_env_or_config("LOG_LEVEL", ("runtime", "log_level"), "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("main")

# Ensure tmp directory exists
Path("./tmp").mkdir(exist_ok=True)

def is_image_file(file_path: str) -> bool:
    """Check if file is an image"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    return any(file_path.lower().endswith(ext) for ext in image_extensions)


class DetectionConfig:
    """Centralized detection configuration"""
    RECEIPT_CONF = 0.45
    BOL_CONF = 0.60
    PAGE_CONF = 0.60
    STAMP_CONF = 0.75
    IOU_THRESHOLD = 0.40
    MIN_REGION_SIZE = (80, 80)
    BLUR_THRESHOLD = 100


class ModelRegistry:
    """Centralized model loading and management"""
    
    def __init__(self):
        self.bol_region_detector = None
        self.page_classifier = None
        self.receipt_region_detector = None
        self.stamp_detector = None
        self.page_orient_predictor = None
        self._load_models()
    
    def _load_models(self):
        """Load all ML models once at initialization"""
        logger.info("🔄 Loading ML models...")
        
        # YOLO models
        self.bol_region_detector = YOLO(
            get_env_or_config(
                "REGION_MODEL_PATH",
                ("models", "region_detector"),
                "Models/bol_regions_best.pt",
            )
        ).to("cpu")
        self.page_classifier = YOLO(
            get_env_or_config(
                "DOC_CLASSIFIER_MODEL_PATH",
                ("models", "document_classifier"),
                "Models/doc_classify_best.pt",
            )
        ).to("cpu")
        self.receipt_region_detector = YOLO(
            get_env_or_config(
                "RECEIPT_REGION_MODEL_PATH",
                ("models", "receipt_region_detector"),
                "Models/receipts_regions_best.pt",
            )
        ).to("cpu")
        self.stamp_detector = YOLO(
            get_env_or_config(
                "STAMP_DETECTOR_MODEL_PATH",
                ("models", "stamp_detector"),
                "Models/stamp_existence.pt",
            )
        ).to("cpu")
        
        # Page orientation model
        page_det_model = mobilenet_v3_small_page_orientation(
            pretrained=False, 
            pretrained_backbone=False
        )
        page_det_params = torch.load(
            get_env_or_config(
                "ORIENTATION_MODEL_PATH",
                ("models", "orientation_detector"),
                "Models/page_orientation.pt",
            ),
            map_location="cpu",
        )
        page_det_model.load_state_dict(page_det_params)
        self.page_orient_predictor = page_orientation_predictor(
            arch=page_det_model, 
            pretrained=False
        )
        
        logger.info("✅ All models loaded successfully")


# Initialize models globally
MODEL_REGISTRY = ModelRegistry()

# Region to Model mappings
REGION_TO_MODEL = {
    "carrier": models.CarrierInfo,
    "stamp": models.Stamp,
    "customer_order_info": models.CustomerOrderInfo,
    "signatures": models.Signatures,
    "bill_of_lading": models.BillOfLadings,
    "date": models.DeliveryReceipt,
}

# Region to Prompt mappings
REGION_TO_PROMPT = {
    "stamp": prompts.STAMP_LATEST_PROMPT,
    "customer_order_info": prompts.CUSTOMER_ORDER_INFO_PROMPT,
    "signatures": prompts.SIGNATURE_PROMPT,
    "bill_of_lading": prompts.BILL_OF_LADING_PROMPT,
    "receipt": prompts.DELIVERY_RECEIPT_PROMPT,
    "total_received": prompts.RECEIPT_TOTAL_RECEIVED_PROMPT,
    "damage": prompts.RECEIPT_DAMAGE_PROMPT,
    "refused": prompts.RECEIPT_REFUSED_PROMPT,
    "customer_order_num": prompts.RECEIPT_CUSTOMER_ORDER_NUMBER_PROMPT,
    "pod_date": prompts.RECEIPT_DATE_PROMPT,
    "pod_sign": prompts.RECEIPT_SIGNATURE_PROMPT,
}

# BOL class names
BOL_CLASS_NAMES = {
    0: "stamp",
    1: "bill_of_lading",
    2: "customer_order_info",
    3: "signatures",
}

# Receipt class names
RECEIPT_CLASS_NAMES = {
    0: "customer_order_num",
    1: "damage",
    2: "pod_date",
    3: "total_received",
    4: "refused",
    6: "pod_sign",
}

# Page classification names
PAGE_CLASS_NAMES = {
    0: "BOL",
    1: "others",
    2: "receipt"
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_image_quality(image: Image.Image, min_size: Tuple[int, int] = (100, 100)) -> bool:
    """
    Validate if image meets minimum quality requirements
    
    Args:
        image: PIL Image to validate
        min_size: Minimum (width, height) tuple
    
    Returns:
        bool: True if image passes quality checks
    """
    if image.width < min_size[0] or image.height < min_size[1]:
        logger.warning(f"Image too small: {image.size}")
        return False
    
    # Check blur using Laplacian variance
    img_array = np.array(image.convert('L'))
    laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
    
    if laplacian_var < DetectionConfig.BLUR_THRESHOLD:
        logger.warning(f"Image too blurry: {laplacian_var:.2f}")
        return False
    
    return True


def log_detection_stats(results: List[Results], detection_type: str) -> None:
    """Log detection statistics for debugging"""
    if results and len(results) > 0 and results[0].boxes:
        confidences = results[0].boxes.conf.tolist()
        if confidences:
            logger.info(
                f"{detection_type} - Detections: {len(confidences)}, "
                f"Avg Conf: {np.mean(confidences):.2f}, "
                f"Min: {min(confidences):.2f}, Max: {max(confidences):.2f}"
            )


def log_extraction_summary(ocr_response: dict, region_name: str) -> None:
    """Log summary of extracted data for debugging"""
    if region_name == "stamp":
        logger.info(f"""
╔═══════════════════════════════════════════════════════╗
║ 📋 STAMP EXTRACTION SUMMARY                           ║
╠═══════════════════════════════════════════════════════╣
║ total_received: {str(ocr_response.get('total_received', 'N/A')):<35} ║
║ pod_date:       {str(ocr_response.get('pod_date', 'N/A')):<35} ║
║ pod_sign:       {str(ocr_response.get('pod_sign', 'N/A')):<35} ║
║ damage:         {str(ocr_response.get('damage', 'N/A')):<35} ║
║ short:          {str(ocr_response.get('short', 'N/A')):<35} ║
║ over:           {str(ocr_response.get('over', 'N/A')):<35} ║
║ refused:        {str(ocr_response.get('refused', 'N/A')):<35} ║
║ stamp_exist:    {str(ocr_response.get('stamp_exist', 'N/A')):<35} ║
║ seal_intact:    {str(ocr_response.get('seal_intact', 'N/A')):<35} ║
║ notation_exist: {str(ocr_response.get('notation_exist', 'N/A')):<35} ║
╚═══════════════════════════════════════════════════════╝
        """)


# ============================================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================================

@traceable(run_type="tool", name="straighten-img", dangerously_allow_filesystem=True)
async def straighten_img(img: str) -> dict:
    """
    Straighten an image using document orientation detection
    
    Args:
        img: Base64 encoded image string
    
    Returns:
        dict: Contains straightened image, rotation angle, and confidence
    """
    pil_img = base64topil(img)
    docs = DocumentFile.from_images(save_image_to_tmp_dir(pil_img))
    
    _, classes, probs = zip(MODEL_REGISTRY.page_orient_predictor(docs))
    page_orientations = [(cls, prob) for cls, prob in zip(classes[0], probs[0])]
    angle, confidence = page_orientations[0]
    
    rotated_img = pil_img.rotate(angle, expand=True, resample=Image.BICUBIC)
    
    return {
        "img": convert_image_to_base64(rotated_img),
        "angle": angle,
        "confidence": confidence,
    }


async def pdf_to_images(
    pdf_path_or_bytes,
    max_pages: int = 100,
    dpi: int = 200
) -> List[Image.Image]:
    """
    Convert PDF/Image to list of PIL Images
    Handles: local files, URLs, and bytes

    Primary: PyMuPDF (fitz) at 200 DPI — tested on 32-doc set, 100% YOLO accuracy.
    Fallback 1: pypdfium2
    Fallback 2: pdf2image (Poppler)
    """
    import gc
    import httpx
    from io import BytesIO

    try:
        # ✅ STEP 1: Download if URL
        if isinstance(pdf_path_or_bytes, str) and pdf_path_or_bytes.startswith(('http://', 'https://')):
            logger.info(f"Downloading from URL: {pdf_path_or_bytes}")

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(pdf_path_or_bytes)
                response.raise_for_status()
                pdf_path_or_bytes = response.content

            logger.info(f"Downloaded {len(pdf_path_or_bytes)} bytes from URL")

        # ✅ STEP 2: Check if it's an image file
        is_image = False

        if isinstance(pdf_path_or_bytes, str):
            is_image = is_image_file(pdf_path_or_bytes)
        elif isinstance(pdf_path_or_bytes, bytes):
            try:
                from PIL import Image
                test_img = Image.open(BytesIO(pdf_path_or_bytes))
                is_image = True
                test_img.close()
            except:
                is_image = False

        if is_image:
            logger.info("Detected image file (not PDF)")
            from PIL import Image

            if isinstance(pdf_path_or_bytes, str):
                img = Image.open(pdf_path_or_bytes).convert('RGB')
            else:
                img = Image.open(BytesIO(pdf_path_or_bytes)).convert('RGB')

            logger.info(f"Image loaded successfully: {img.size}")
            return [img]

        # ✅ STEP 3: Process as PDF with PyMuPDF (fitz)
        import fitz

        if isinstance(pdf_path_or_bytes, bytes):
            doc = fitz.open(stream=pdf_path_or_bytes, filetype="pdf")
        else:
            doc = fitz.open(pdf_path_or_bytes)

        page_count = len(doc)
        if page_count > max_pages:
            raise ValueError(f"PDF has too many pages: {page_count}. Maximum allowed: {max_pages}")

        logger.info(f"Processing PDF with PyMuPDF: {page_count} pages at {dpi} DPI")

        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        images = []

        for i in range(page_count):
            pix = doc[i].get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            resized_image = resize_large_image(img)
            images.append(resized_image)

            if page_count > 20 and (i + 1) % 10 == 0:
                logger.info(f"Rendered {i + 1}/{page_count} pages")
                gc.collect()

        doc.close()
        logger.info(f"PyMuPDF processing complete: {len(images)} pages rendered")
        return images

    except Exception as fitz_error:
        logger.warning(f"PyMuPDF failed: {str(fitz_error)}")
        logger.info("Attempting fallback to pypdfium2...")

        try:
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(pdf_path_or_bytes)
            page_count = len(pdf)

            if page_count > max_pages:
                raise ValueError(f"PDF has too many pages: {page_count}. Maximum allowed: {max_pages}")

            scale = 2 if page_count > 50 else (3 if page_count > 20 else 4)
            logger.info(f"Processing PDF with pypdfium2: {page_count} pages at scale={scale}")

            images = []
            for i in range(page_count):
                rendered_image = pdf[i].render(scale=scale).to_pil()
                images.append(resize_large_image(rendered_image))

            logger.info(f"pypdfium2 fallback successful: {len(images)} pages rendered")
            return images

        except Exception as pdfium_error:
            logger.warning(f"pypdfium2 also failed: {str(pdfium_error)}")
            logger.info("Attempting fallback to pdf2image (Poppler)...")

            try:
                from pdf2image import convert_from_bytes, convert_from_path
                from pathlib import Path

                if isinstance(pdf_path_or_bytes, (str, Path)):
                    if isinstance(pdf_path_or_bytes, str) and pdf_path_or_bytes.startswith(('http://', 'https://')):
                        raise ValueError("URL should have been downloaded already")

                    if os.path.isfile(str(pdf_path_or_bytes)):
                        pages = convert_from_path(pdf_path_or_bytes, dpi=200, fmt='jpeg', thread_count=2)
                    else:
                        raise FileNotFoundError(f"File not found: {pdf_path_or_bytes}")
                else:
                    pages = convert_from_bytes(pdf_path_or_bytes, dpi=200, fmt='jpeg', thread_count=2)

                if len(pages) > max_pages:
                    pages = pages[:max_pages]

                resized_pages = [resize_large_image(page) for page in pages]
                logger.info(f"pdf2image fallback successful: {len(resized_pages)} pages rendered")
                return resized_pages

            except ImportError:
                raise ValueError(
                    f"All PDF parsers failed. PyMuPDF: {str(fitz_error)}, "
                    f"pypdfium2: {str(pdfium_error)}, pdf2image: not installed"
                )
            except Exception as fallback_error:
                raise ValueError(
                    f"All PDF parsers failed. PyMuPDF: {str(fitz_error)}, "
                    f"pypdfium2: {str(pdfium_error)}, pdf2image: {str(fallback_error)}"
                )

# ============================================================================
# REGION DETECTION FUNCTIONS
# ============================================================================

@traceable(run_type="tool", name="detect-regions-receipt", dangerously_allow_filesystem=True)
def receipt_detect_regions(image: str) -> models.ReceiptRegions:
    """
    Detect and extract regions from receipt images
    
    Args:
        image: Base64 encoded image
    
    Returns:
        ReceiptRegions object with detected regions
    """
    results: List[Results] = MODEL_REGISTRY.receipt_region_detector.predict(
        base64topil(image), 
        conf=DetectionConfig.RECEIPT_CONF,
        iou=DetectionConfig.IOU_THRESHOLD
    )
    
    log_detection_stats(results, "Receipt Region Detection")
    
    boxes = {}
    region_scores = {}
    
    for result in results:
        for class_id, box, conf in zip(
            result.boxes.cls.tolist(), 
            result.cpu().boxes.xyxy,
            result.boxes.conf.tolist()
        ):
            if class_id in RECEIPT_CLASS_NAMES:
                region_name = RECEIPT_CLASS_NAMES[class_id]
                
                # Keep highest confidence detection per region
                if region_name not in region_scores or conf > region_scores[region_name]:
                    cropped = Image.fromarray(
                        save_one_box(box, im=result.orig_img, save=False)
                    )
                    
                    if validate_image_quality(cropped, min_size=DetectionConfig.MIN_REGION_SIZE):
                        boxes[region_name] = convert_image_to_base64(make_image_square(cropped))
                        region_scores[region_name] = conf
                        logger.info(f"Detected {region_name} with confidence: {conf:.2f}")
                    else:
                        logger.warning(f"Rejected {region_name} due to poor quality")
    
    return models.ReceiptRegions(**boxes)


@traceable(run_type="tool", name="detect-regions-bol", dangerously_allow_filesystem=True)
def bol_detect_regions(image: str) -> models.Regions:
    """
    Detect and extract regions from BOL images
    
    Args:
        image: Base64 encoded image
    
    Returns:
        Regions object with detected regions
    """
    results: List[Results] = MODEL_REGISTRY.bol_region_detector.predict(
        base64topil(image), 
        conf=DetectionConfig.BOL_CONF,
        iou=DetectionConfig.IOU_THRESHOLD
    )
    
    log_detection_stats(results, "BOL Region Detection")
    
    # Log detected regions
    logger.info(f"""
        ╔═══════════════════════════════════════════════════════════╗
        ║ 🎯 BOL REGION DETECTION START                             ║
        ╚═══════════════════════════════════════════════════════════╝
            """)
    detected_regions = []
    for result in results:
        if result.boxes:
            for class_id, conf in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
                if class_id in BOL_CLASS_NAMES:
                    detected_regions.append(f"{BOL_CLASS_NAMES[class_id]} (conf: {conf:.2f})")
    
    logger.info(f"\n🎯 BOL REGIONS DETECTED: {detected_regions}\n")
    
    boxes = {}
    region_scores = {}
    
    for result in results:
        orig_img = result.orig_img
        h, w, _ = orig_img.shape
        
        for class_id, box, conf in zip(
            result.boxes.cls.tolist(), 
            result.cpu().boxes.xyxy,
            result.boxes.conf.tolist()
        ):
            region_name = BOL_CLASS_NAMES[class_id]
            
            # Keep highest confidence detection per region
            if region_name not in region_scores or conf > region_scores[region_name]:
                x1, y1, x2, y2 = map(int, box)
                
                # Special handling for bill_of_lading - extend to corners
                if region_name == "bill_of_lading":
                    x1, y1 = 0, 0
                    x2 = min(x2, w)
                    y2 = min(y2, h)
                
                box_tensor = torch.tensor([x1, y1, x2, y2])
                cropped_img = save_one_box(box_tensor, im=orig_img, save=False)
                cropped_pil = Image.fromarray(cropped_img)
                
                if validate_image_quality(cropped_pil, min_size=DetectionConfig.MIN_REGION_SIZE):
                    boxes[region_name] = convert_image_to_base64(cropped_pil)
                    region_scores[region_name] = conf
                    logger.info(f"Detected {region_name} with confidence: {conf:.2f}")
                else:
                    logger.warning(f"Rejected {region_name} due to poor quality")

    
    logger.info(f"""
        ╔═══════════════════════════════════════════════════════════╗
        ║ 🎯 BOL REGION DETECTION COMPLETE                          ║
        ╠═══════════════════════════════════════════════════════════╣
        ║ Stamp:             {'✓ Detected' if 'stamp' in boxes else '✗ Not Found':<36} ║
        ║ Bill of Lading:    {'✓ Detected' if 'bill_of_lading' in boxes else '✗ Not Found':<36} ║
        ║ Customer Order:    {'✓ Detected' if 'customer_order_info' in boxes else '✗ Not Found':<36} ║
        ║ Signatures:        {'✓ Detected' if 'signatures' in boxes else '✗ Not Found':<36} ║
        ╚═══════════════════════════════════════════════════════════╝
            """)
    
    return models.Regions(**boxes)


async def bol_stamp_exist(image: str) -> str:
    """
    Detect if stamp exists in BOL image
    
    Args:
        image: Base64 encoded image
    
    Returns:
        'yes' or 'no' indicating stamp presence
    """
    class_names = {0: "no", 1: "yes"}
    results = MODEL_REGISTRY.stamp_detector.predict(
        base64topil(image), 
        conf=DetectionConfig.STAMP_CONF
    )
    
    log_detection_stats(results, "Stamp Detection")
    
    for result in results:
        classes = result.boxes.cls.tolist()
        if classes:
            class_idx = safe_int_conversion(classes[0], 0)
            return class_names[class_idx]
    
    return "no"


def classify_documents(images: List[str]) -> Dict[str, List[str]]:
    """
    Classify document types in images
    
    Args:
        images: List of base64 encoded images
    
    Returns:
        Dictionary of classified regions by type
    """
    logger.info(f"""
        ╔═══════════════════════════════════════════════════════════╗
        ║ 📄 DOCUMENT CLASSIFICATION START                          ║
        ╠═══════════════════════════════════════════════════════════╣
        ║ Total Images: {len(images):<42} ║
        ║ Timestamp:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<42} ║
        ╚═══════════════════════════════════════════════════════════╝
            """)
    
    processed_images = []
    for image in images:
        try:
            pil_image = base64topil(image)
            if pil_image is not None:
                pil_image = resize_large_image(pil_image)
                processed_images.append(pil_image)
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            continue
    
    if not processed_images:
        raise HTTPException(
            status_code=400,
            detail="No valid images to process after resizing"
        )
    
    results: List[Results] = MODEL_REGISTRY.page_classifier.predict(
        processed_images, 
        conf=DetectionConfig.PAGE_CONF
    )
    
    log_detection_stats(results, "Page Classification")
    
    boxes = defaultdict(list)
    for result in results:
        if result.boxes:
            for class_id, box in zip(result.boxes.cls.tolist(), result.cpu().boxes.xyxy):
                if class_id in PAGE_CLASS_NAMES:
                    boxes[PAGE_CLASS_NAMES[class_id]].append(
                        convert_image_to_base64(
                            Image.fromarray(save_one_box(box, im=result.orig_img, save=False))
                        )
                    )
    logger.info(f"""
        ╔═══════════════════════════════════════════════════════════╗
        ║ 📄 CLASSIFICATION COMPLETE                                ║
        ╠═══════════════════════════════════════════════════════════╣
        ║ BOL Pages:     {len(boxes.get('BOL', [])):<42} ║
        ║ Receipt Pages: {len(boxes.get('receipt', [])):<42} ║
        ║ Other Pages:   {len(boxes.get('others', [])):<42} ║
        ╚═══════════════════════════════════════════════════════════╝
            """)
    
    return dict(boxes)


# ============================================================================
# DATA VALIDATION & CLEANING
# ============================================================================

def validate_and_clean_stamp_data(ocr_response: dict, region_name: str) -> dict:
    """
    Validate and clean stamp OCR responses
    
    Args:
        ocr_response: Raw OCR response
        region_name: Name of the region
    
    Returns:
        Cleaned OCR response
    """
    if region_name != "stamp":
        return ocr_response
    
    cleaned = ocr_response.copy()
    
    # Validate total_received
    total_received = cleaned.get("total_received")
    
    # Parse patterns like "208 cases, 9=4, 70 ctns"
    if total_received and isinstance(total_received, str):
        match = re.match(r'^(\d+)', total_received.strip())
        if match:
            total_received = safe_int_conversion(match.group(1), "null")
            cleaned["total_received"] = total_received
    
    # Filter suspicious values (likely false positives)
    suspicious_values = [226, 3511, 2028, 2098, 1504, 1430, 1424, 412113, 287868]
    
    if isinstance(total_received, (int, str, float)):
        try:
            # Handle all numeric types safely
            total_rec_int = safe_int_conversion(total_received, None)
            
            if total_rec_int is not None:
                if total_rec_int in suspicious_values:
                    logger.warning(f"⚠️ Suspicious total_received: {total_received}")
                    cleaned["total_received"] = "null"
                elif len(str(total_rec_int)) == 4 and total_rec_int > 1000:
                    logger.warning(f"⚠️ 4-digit number {total_received} likely NOT cartons")
                    cleaned["total_received"] = "null"
                elif total_rec_int > 99999:
                    logger.warning(f"⚠️ Number {total_received} too large for carton count")
                    cleaned["total_received"] = "null"
        except Exception as e:
            logger.warning(f"⚠️ Error validating total_received: {e}")
            pass
    
    # Validate pod_date
    pod_date = cleaned.get("pod_date")
    example_dates = ["10/23/24", "10-23-2024", "OCT 23 2024", "OCT 23, 2024", 
                     "OCT - 23, 2024", "OCT / 23, 2024", "10/22/24", "9/16/25"]
    
    if pod_date in example_dates:
        logger.warning(f"⚠️ pod_date matches prompt example: {pod_date}")
        cleaned["pod_date"] = "empty"
    
    if pod_date and pod_date not in ["null", "empty", None]:
        try:
            date_str = str(pod_date).replace("-", "/")
            parts = date_str.split("/")
            
            if len(parts) >= 2:
                first_num = safe_int_conversion(parts[0], 0)
                second_num = safe_int_conversion(parts[1], 0)
                
                if first_num > 31 or second_num > 31:
                    logger.warning(f"⚠️ Invalid date values: {pod_date}")
                    cleaned["pod_date"] = "null"
                elif first_num > 12 and second_num > 12:
                    logger.warning(f"⚠️ Invalid date: {pod_date}")
                    cleaned["pod_date"] = "null"
        except Exception as e:
            logger.warning(f"⚠️ Error validating pod_date: {e}")
            pass
    
    # Validate damage/short/over/refused
    for field in ["damage", "short", "over", "refused", "roc_damaged", "damaged_kept"]:
        value = cleaned.get(field)
        
        if value in ["D", "S", "O", "R", "d", "s", "o", "r"]:
            logger.warning(f"⚠️ {field} is just a letter")
            cleaned[field] = "empty"
        
        if isinstance(value, (int, str, float)):
            try:
                val_int = safe_int_conversion(value, None)
                if val_int is not None:
                    if val_int < 0:
                        logger.warning(f"⚠️ {field} value {val_int} is negative")
                        cleaned[field] = "null"
                    elif val_int > 10000:
                        logger.warning(f"⚠️ {field} value {val_int} unreasonably high")
                        cleaned[field] = "null"
            except Exception as e:
                logger.warning(f"⚠️ Error validating {field}: {e}")
                pass
    
    # Validate seal_intact
    # Annotation-only documents can legitimately have stamp_exist="no" while still
    # carrying a seal status. Only null it out when no stamp evidence exists and the
    # document does not appear to be an annotation-style confirmation.
    seal_intact = cleaned.get("seal_intact")
    if seal_intact in ["yes", "no"] and cleaned.get("stamp_exist") == "no" and cleaned.get("notation_exist") != "yes":
        logger.warning("⚠️ seal_intact has value but no stamp exists")
        cleaned["seal_intact"] = "null"
    
    # Log annotation-style detection
    stamp_exist = cleaned.get("stamp_exist")
    if stamp_exist == "no":
        has_data = any([
            cleaned.get("pod_date") not in ["null", "empty", None],
            cleaned.get("pod_sign") not in ["null", "empty", None],
            cleaned.get("total_received") not in ["null", "empty", None]
        ])
        if has_data:
            logger.info("✓ Annotation-style region detected (stamp_exist='no' but data present)")
    
    return cleaned


# ============================================================================
# BOL DATA MERGING
# ============================================================================

def calculate_data_quality_score(bol: dict) -> int:
    """
    Calculate quality score for a BOL entry
    
    Args:
        bol: BOL data dictionary
    
    Returns:
        Quality score (higher is better)
    """
    score = 0
    stamp = bol.get("stamp", {})
    
    # Critical fields
    if stamp.get("pod_date") not in ["null", "empty", "", None]:
        pod_date = str(stamp["pod_date"])
        if "/" in pod_date and len(pod_date) >= 8:
            score += 50
        elif pod_date.isdigit() and len(pod_date) <= 2:
            score += 5
        else:
            score += 20
    
    if stamp.get("total_received") not in ["null", "empty", "", 0, None]:
        score += 40
    
    # Important fields
    if stamp.get("pod_sign") not in ["null", "empty", "", None]:
        score += 15
    
    if stamp.get("stamp_exist") == "yes":
        score += 10
    
    # Discrepancy fields
    for field in ["damage", "short", "over", "refused"]:
        if stamp.get(field) not in ["null", "empty", "", None]:
            score += 5
    
    # Other fields
    if stamp.get("seal_intact") not in ["null", "empty", "", None]:
        score += 5
    
    if stamp.get("notation_exist") not in ["null", "empty", "", None]:
        score += 5
    
    return score


def merge_bol_data(data: List[dict]) -> List[dict]:
    """
    Merge multiple BOL entries for the same bill number
    
    Args:
        data: List of BOL data entries
    
    Returns:
        Merged BOL data with best quality fields
    """
    merged_data = []
    
    def resolve_conflict(values):
        """Resolve conflicting values"""
        values_counter = Counter(values)
        return "null" if values_counter["null"] > values_counter["empty"] else "empty"
    
    # Group by bill number
    grouped = {}
    for bol in data:
        bill_no = bol["bill_of_lading"]["bill_no"]
        if bill_no not in grouped:
            grouped[bill_no] = []
        grouped[bill_no].append(bol)
    
    # Merge each group
    for bill_no, bol_list in grouped.items():
        if len(bol_list) == 1:
            merged_data.append(bol_list[0])
            logger.info(f"Single BOL page for bill {bill_no}")
            continue
        
        # Score and sort pages
        scored_bols = [(bol, calculate_data_quality_score(bol)) for bol in bol_list]
        scored_bols.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"\n🔍 BOL MERGE QUALITY SCORES for {bill_no}:")
        for idx, (bol, score) in enumerate(scored_bols):
            logger.info(
                f"  Page {idx}: Score={score}, pod_date={bol['stamp'].get('pod_date')}, "
                f"total_received={bol['stamp'].get('total_received')}"
            )
        
        # Use highest quality BOL as base
        best_bol = scored_bols[0][0]
        merged = copy.deepcopy(best_bol)
        logger.info(f"✅ Using page with score {scored_bols[0][1]} as base for {bill_no}")
        
        # Merge better values from other pages
        for key in merged["stamp"].keys():
            values = [bol["stamp"][key] for bol in bol_list]
            
            if all(value == values[0] for value in values):
                continue
            
            non_null_values = [
                bol["stamp"][key] for bol in bol_list 
                if bol["stamp"][key] not in ["null", "empty", "", None]
            ]
            
            if non_null_values:
                if key == "pod_date":
                    # Prefer longer/more complete dates
                    sorted_dates = sorted(non_null_values, key=lambda x: len(str(x)), reverse=True)
                    merged["stamp"][key] = sorted_dates[0]
                    if len(non_null_values) > 1:
                        logger.info(f"  Merged {key}: chose '{sorted_dates[0]}' over {non_null_values[1:]}")
                
                elif key in ["total_received", "damage", "short", "over", "refused"]:
                    # Prefer non-zero values
                    numeric_values = [v for v in non_null_values if v != 0]
                    merged["stamp"][key] = numeric_values[0] if numeric_values else non_null_values[0]
            else:
                merged["stamp"][key] = resolve_conflict(values)
        
        merged_data.append(merged)
        logger.info(
            f"✅ Final merged data for {bill_no}: pod_date={merged['stamp']['pod_date']}, "
            f"total_received={merged['stamp']['total_received']}\n"
        )
    
    return merged_data


# ============================================================================
# OCR PROCESSING
# ============================================================================

@traceable(run_type="llm", name="OCR-Batch", dangerously_allow_filesystem=True)
async def batch_ocr(
    ocr_batches: List[models.OCRBatch], 
    port: Optional[int] = None
) -> List[models.OCRBatchResponse]:
    """
    Process multiple OCR requests as a batch
    
    Args:
        ocr_batches: List of OCR batch requests
        port: Optional OCR service port
    
    Returns:
        List of OCR batch responses
    """
    global OCR_PORT
    url = f"http://localhost:{port or OCR_PORT}/batch-pixtral-inference"
    
    pixtral_requests = []
    metadata_list = []
    
    for ocr_batch in ocr_batches:
        # Remove data URL prefix if present
        image_data = ocr_batch.image.split(",")[1] if "," in ocr_batch.image else ocr_batch.image
        
        pixtral_requests.append(models.InterVL2Request(
            image_url_or_path_or_base64=image_data,
            prompt=ocr_batch.prompt
        ))
        
        metadata_list.append({
            "page_type": ocr_batch.page_type,
            "region_name": ocr_batch.region_name,
            "stamp_exist": ocr_batch.stamp_exist
        })
    
    if not pixtral_requests:
        return []
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=[req.model_dump() for req in pixtral_requests])
            response.raise_for_status()
        
        pixtral_batch_responses = [models.InterVL2Response(**item) for item in response.json()]
        
        final_ocr_responses = []
        for i, pixtral_resp in enumerate(pixtral_batch_responses):
            metadata = metadata_list[i]
            json_to_parse = pixtral_resp.response

            # Extract JSON from markdown - enhanced to handle all code block types
            # Matches ```json, ```java, ```javascript, or plain ``` with optional whitespace
            pattern = r'```(?:json|java|javascript)?\s*(.*?)\s*```'
            match = re.search(pattern, json_to_parse, re.DOTALL)
            if match:
                json_to_parse = match.group(1).strip()
            else:
                # Fallback: strip any remaining code fences
                json_to_parse = json_to_parse.strip()
                if json_to_parse.startswith('```'):
                    # Remove first line if it's a code fence
                    lines = json_to_parse.split('\n', 1)
                    if len(lines) > 1:
                        json_to_parse = lines[1]
                if json_to_parse.endswith('```'):
                    json_to_parse = json_to_parse.rsplit('```', 1)[0]
                json_to_parse = json_to_parse.strip()

            # Clean and parse JSON
            json_content = json_to_parse.replace("'null'", "null")

            try:
                parsed_data = json.loads(json_content)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed for {metadata['region_name']}: {e}")
                logger.error(f"Raw response (first 500 chars): {pixtral_resp.response[:500]}")
                logger.error(f"Cleaned JSON attempt: {json_content[:500]}")

                # Attempt 1: extract a bare JSON object {...} embedded anywhere in the response
                raw_text = pixtral_resp.response
                json_obj_match = re.search(r'\{[^{}]*\}', raw_text, re.DOTALL)
                if json_obj_match:
                    try:
                        parsed_data = json.loads(json_obj_match.group(0))
                        logger.warning(f"Recovered JSON object from plain-text response for {metadata['region_name']}")
                    except json.JSONDecodeError:
                        parsed_data = {}
                else:
                    parsed_data = {}

                # Attempt 2: for customer_order_info specifically, extract integer from prose
                # e.g. "The ISSUED QUANTITY for the customer order is 230."
                # e.g. "The total_order_quantity is 490."
                if not parsed_data and metadata['region_name'] == 'customer_order_info':
                    qty_match = re.search(
                        r'(?:issued[\s_]+quantity|total[\s_]+order[\s_]+quantity|total_order_quantity|total[\s_]+quantity|grand[\s_]+total|qty)\D{0,30}?(\d{1,5})\b',
                        raw_text,
                        re.IGNORECASE
                    )
                    if qty_match:
                        candidate = int(qty_match.group(1))
                        if 1 <= candidate <= 9999:
                            parsed_data = {"total_order_quantity": candidate}
                            logger.warning(
                                f"Recovered total_order_quantity={candidate} via prose fallback "
                                f"for {metadata['region_name']}"
                            )
            
            # Validate and clean data
            validated_data = validate_and_clean_stamp_data(parsed_data, metadata["region_name"])
            log_extraction_summary(validated_data, metadata["region_name"])
            
            final_ocr_responses.append(
                models.OCRBatchResponse(
                    page_type=metadata["page_type"],
                    region_name=metadata["region_name"],
                    ocr_response=validated_data,
                    stamp_exist=metadata["stamp_exist"]
                )
            )
        
        return final_ocr_responses
    
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error during batch OCR: {e.response.status_code} - {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Batch OCR service error: {e.response.text}"
        )
    except httpx.RequestError as e:
        logger.error(f"Network error during batch OCR: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Batch OCR service unreachable: {e}"
        )
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Invalid JSON response from batch OCR service: {e}"
        )
    except Exception as e:
        logger.exception("Unexpected error in batch_ocr")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


def group_by_page_type(data: List[models.OCRBatchResponse]) -> Dict[str, List[models.OCRBatchResponse]]:
    """Group OCR responses by page type"""
    grouped_data = defaultdict(list)
    for item in data:
        grouped_data[item.page_type].append(item)
    return dict(grouped_data)


async def batch_list(data: List, batch_size: int) -> List[List]:
    """Split data into batches"""
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


# ============================================================================
# POST-PROCESSING FUNCTIONS
# ============================================================================

def initialize_pod_fields() -> dict:
    """Initialize POD fields with default values"""
    return {
        "B/L Number": "null",
        "Stamp Exists": "null",
        "Seal Intact": "null",
        "POD Date": "null",
        "Signature Exists": "null",
        "Issued Qty": "null",
        "Received Qty": "null",
        "Damage Qty": "null",
        "Short Qty": "null",
        "Over Qty": "null",
        "Refused Qty": "null",
        "Customer Order Num": "null",
    }


def process_quantities(
    bol_data: dict, 
    stamp_total_received: int, 
    stamp_total_received_str: str, 
    customer_order_qty: int,
    POD_FIELDS: dict,
    customer_order_num: str,
) -> dict:
    """
    Process quantity fields based on stamp and order data
    
    Args:
        bol_data: BOL data dictionary
        stamp_total_received: Total received as integer
        stamp_total_received_str: Total received as string
        customer_order_qty: Customer order quantity
        POD_FIELDS: POD fields dictionary to update
    
    Returns:
        Updated POD_FIELDS
    """
    is_notation = bol_data["stamp"].get("notation_exist") in ["yes", True, "True"]
    not_stamp_exist = bol_data["stamp"]["stamp_exist"] != "yes"
    not_receipt = customer_order_num == "null"

    logger.info(f"\n✅ customer_order_num: {customer_order_num}\n")

    # ========== CHANGE 1: NOTATION CASE HANDLING - MOVED TO TOP ==========
    # When notation exists, return specific values immediately
    if is_notation and not_stamp_exist and not_receipt:
        logger.info("✅ NOTATION CASE DETECTED - Applying special logic")
        
        # Received qty: check if has value
        if stamp_total_received_str not in ["null", "empty", "0"]:
            POD_FIELDS["Received Qty"] = stamp_total_received
        else:
            POD_FIELDS["Received Qty"] = "empty"
        
        # Damage: check if has value, else "empty"
        damage_value = bol_data["stamp"].get("damage")
        if damage_value not in ["null", "empty", "", None, 0]:
            POD_FIELDS["Damage Qty"] = safe_int_conversion(damage_value, 0)
        else:
            POD_FIELDS["Damage Qty"] = "empty"
        
        # Short: check if has value, else "empty"
        short_value = bol_data["stamp"].get("short")
        if short_value not in ["null", "empty", "", None, 0]:
            POD_FIELDS["Short Qty"] = safe_int_conversion(short_value, 0)
        else:
            POD_FIELDS["Short Qty"] = "empty"
        
        # Over: check if has value, else "empty"
        over_value = bol_data["stamp"].get("over")
        if over_value not in ["null", "empty", "", None, 0]:
            POD_FIELDS["Over Qty"] = safe_int_conversion(over_value, 0)
        else:
            POD_FIELDS["Over Qty"] = "empty"
        
        # Refused: ALWAYS "null" (label not present)
        POD_FIELDS["Refused Qty"] = "null"
        
        return POD_FIELDS
    # ========== END OF CHANGE 1 ==========

    # Normal case (not notation)
    if stamp_total_received_str not in ["null", "empty", "0"]:
        if customer_order_qty == stamp_total_received:
            # All quantities match
            POD_FIELDS["Refused Qty"] = 0
            POD_FIELDS["Short Qty"] = 0
            POD_FIELDS["Damage Qty"] = 0
            POD_FIELDS["Over Qty"] = 0
            POD_FIELDS["Received Qty"] = stamp_total_received
        else:
            # Has discrepancies - using safe dictionary access
            POD_FIELDS["Refused Qty"] = safe_int_conversion(bol_data["stamp"].get("refused", 0), 0)
            POD_FIELDS["Short Qty"] = safe_int_conversion(bol_data["stamp"].get("short", 0), 0)
            POD_FIELDS["Damage Qty"] = safe_int_conversion(bol_data["stamp"].get("damage", 0), 0)
            POD_FIELDS["Over Qty"] = safe_int_conversion(bol_data["stamp"].get("over", 0), 0)
            POD_FIELDS["Received Qty"] = stamp_total_received
    else:
        # No total received - using safe dictionary access
        POD_FIELDS["Refused Qty"] = safe_int_conversion(bol_data["stamp"].get("refused", 0), 0)
        POD_FIELDS["Short Qty"] = safe_int_conversion(bol_data["stamp"].get("short", 0), 0)
        POD_FIELDS["Damage Qty"] = safe_int_conversion(bol_data["stamp"].get("damage", 0), 0)
        POD_FIELDS["Over Qty"] = safe_int_conversion(bol_data["stamp"].get("over", 0), 0)
        POD_FIELDS["Received Qty"] = safe_str_conversion(bol_data["stamp"].get("total_received", "null"), "null")
    
    return POD_FIELDS

def validate_received_qty(POD_FIELDS: dict, bol_data: dict) -> dict:
    """
    Validate and correct received quantity fields
    
    Args:
        POD_FIELDS: POD fields dictionary
        bol_data: BOL data dictionary
    
    Returns:
        Updated POD_FIELDS
    """
    received_qty = str(POD_FIELDS['Received Qty']).lower()
    issued_qty = str(POD_FIELDS['Issued Qty']).lower()
    is_notation = bol_data["stamp"].get("notation_exist") in ["yes", True, "True"]
    
    if (
        received_qty in ["null", "empty", "", "n/a", "0"]
        and issued_qty not in ["null", "empty", "", "n/a", "0"]
        and not is_notation
    ):
        if bol_data["stamp"]["stamp_exist"] == "yes" or bol_data["stamp"]["notation_exist"] == "yes":
            stamp_total_received = bol_data["stamp"].get("total_received", "null")
            
            if stamp_total_received in ["null", "empty", "", "N/A", 0]:
                POD_FIELDS['Received Qty'] = "null"
                POD_FIELDS['Damage Qty'] = "null"
                POD_FIELDS['Short Qty'] = "null"
                POD_FIELDS['Over Qty'] = "null"
                POD_FIELDS['Refused Qty'] = "null"
                POD_FIELDS["Status"] = "incomplete_data"
            else:
                POD_FIELDS['Received Qty'] = "null"
                POD_FIELDS['Damage Qty'] = 0
                POD_FIELDS['Short Qty'] = 0
                POD_FIELDS['Over Qty'] = 0
                POD_FIELDS['Refused Qty'] = 0
                POD_FIELDS["Status"] = "valid"
        else:
            # Invalid stamp
            POD_FIELDS['Received Qty'] = "null"
            POD_FIELDS['Damage Qty'] = "null"
            POD_FIELDS['Short Qty'] = "null"
            POD_FIELDS['Over Qty'] = "null"
            POD_FIELDS['Refused Qty'] = "null"
            POD_FIELDS["POD Date"] = "null"
            POD_FIELDS["Signature Exists"] = "null"
            POD_FIELDS["Stamp Exists"] = "null"
            POD_FIELDS["Status"] = "invalid"
    else:
        POD_FIELDS["Status"] = "valid"
    
    return POD_FIELDS


# ============================================================================
# MAIN OCR PROCESSING
# ============================================================================

@traceable(run_type="chain", name="run-ocr", dangerously_allow_filesystem=True)
async def run_ocr(
    file_url_or_path: str,
    batch_size: int = 4,
    port: Optional[int] = None
) -> List[models.OCRResponse]:
    """
    Main OCR processing pipeline

    Args:
        file_url_or_path: File path or URL
        batch_size: Batch size for processing
        port: Optional OCR service port

    Returns:
        List of POD fields
    """
    # File size limits
    MAX_FILE_SIZE_MB = 50
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    MAX_PDF_PAGES = 100

    try:
        logger.info(f"\n🚀 STARTING OCR PROCESSING\nFile: {file_url_or_path}\n")

        # Load document
        if os.path.isfile(str(file_url_or_path)):
            # Validate file size
            file_size = os.path.getsize(file_url_or_path)
            if file_size > MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large: {file_size / 1024 / 1024:.2f}MB. Maximum allowed: {MAX_FILE_SIZE_MB}MB"
                )

            logger.info(f"Processing local file: {file_url_or_path} ({file_size / 1024 / 1024:.2f}MB)")

            ext = os.path.splitext(file_url_or_path)[1].lower()

            if ext == ".pdf":
                pages = await pdf_to_images(file_url_or_path, max_pages=MAX_PDF_PAGES)
            elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
                with Image.open(file_url_or_path) as img:
                    with BytesIO() as output:
                        img.convert("RGB").save(output, format="JPEG")
                        output.seek(0)
                        pages = [Image.open(output).copy()]
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
        else:
            # Fetch from URL with streaming and size limits
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=300.0,  # 5 minutes for large files
                    write=60.0,
                    pool=10.0
                )
            ) as client:
                # Stream download with size validation
                async with client.stream('GET', file_url_or_path) as response:
                    if response.status_code != 200:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Failed to fetch file from URL: {file_url_or_path} (HTTP {response.status_code})"
                        )

                    # Check Content-Length header
                    content_length = response.headers.get('content-length')
                    if content_length:
                        size_mb = int(content_length) / 1024 / 1024
                        if int(content_length) > MAX_FILE_SIZE_BYTES:
                            raise HTTPException(
                                status_code=413,
                                detail=f"File too large: {size_mb:.2f}MB. Maximum allowed: {MAX_FILE_SIZE_MB}MB"
                            )
                        logger.info(f"Downloading file from URL: {file_url_or_path} ({size_mb:.2f}MB)")

                    # Stream download with size limit enforcement
                    chunks = []
                    total_size = 0
                    async for chunk in response.aiter_bytes(chunk_size=64 * 1024):  # 64KB chunks
                        chunks.append(chunk)
                        total_size += len(chunk)
                        if total_size > MAX_FILE_SIZE_BYTES:
                            raise HTTPException(
                                status_code=413,
                                detail=f"File exceeded {MAX_FILE_SIZE_MB}MB during download"
                            )

                    content = b''.join(chunks)
                    logger.info(f"Downloaded {total_size / 1024 / 1024:.2f}MB from URL")

            content_type = response.headers.get("Content-Type", "").lower()

            if "pdf" in content_type:
                pages = await pdf_to_images(content, max_pages=MAX_PDF_PAGES)
            elif any(fmt in content_type for fmt in ["jpeg", "png", "bmp", "tiff", "webp"]):
                img = Image.open(BytesIO(content))
                with BytesIO() as output:
                    img.convert("RGB").save(output, format="JPEG")
                    output.seek(0)
                    pages = [Image.open(output).copy()]
            else:
                raise HTTPException(status_code=400, detail="Unsupported content type in URL")
        
        # Classify documents
        logger.info("\n📄 DOCUMENT CLASSIFICATION START\n")
        classified_pages = classify_documents([convert_image_to_base64(page) for page in pages])
        logger.info(
            f"\n📄 CLASSIFICATION COMPLETE\n"
            f"Pages: {len(classified_pages)}\n"
            f"Types: {list(classified_pages.keys())}\n"
        )
        
        # Prepare OCR requests
        ocr_output_pdf = defaultdict(list)
        all_requests = []
        
        for cls, cls_pages in classified_pages.items():
            if cls == "BOL":
                for page_idx, page in enumerate(cls_pages):
                    # Straighten image
                    straighten_result = await straighten_img(page)
                    page = straighten_result["img"]
                    
                    # Detect stamp
                    stamp_exist = await bol_stamp_exist(page)
                    
                    # Detect regions
                    bill_of_ladings_regions = bol_detect_regions(page).model_dump()
                    logger.info(
                        f"\n🎯 BOL REGIONS\n"
                        f"Page: {page_idx}\n"
                        f"Regions: {list(bill_of_ladings_regions.keys())}\n"
                        f"Stamp: {stamp_exist}\n"
                    )
                    
                    if bill_of_ladings_regions:
                        for region_name, image_base64 in bill_of_ladings_regions.items():
                            region: BaseModel = REGION_TO_MODEL[region_name]
                            
                            if not image_base64:
                                logger.warning(f"⚠️ Missing {region_name}, using full page")
                                image_base64 = page
                            
                            prompt = REGION_TO_PROMPT[region_name]
                            
                            all_requests.append(
                                models.OCRBatch(
                                    page_type=f"BOL_{page_idx}",
                                    region_name=region_name,
                                    prompt=prompt,
                                    image=image_base64,
                                    stamp_exist=stamp_exist,
                                )
                            )
                    else:
                        # Fallback to heuristic stamp crop
                        logger.warning("⚠️ No BOL regions detected, using heuristic fallback")
                        fallback_stamp = heuristic_stamp_crop_base64(page)
                        
                        if fallback_stamp:
                            all_requests.append(
                                models.OCRBatch(
                                    page_type=f"BOL_{page_idx}",
                                    region_name="stamp",
                                    prompt=REGION_TO_PROMPT["stamp"],
                                    image=fallback_stamp,
                                    stamp_exist=stamp_exist,
                                )
                            )
            
            elif cls == "receipt":
                # Straighten receipt
                receipt_page = await straighten_img(cls_pages[0])
                receipt_page = receipt_page["img"]
                
                # Detect regions
                receipt_regions = receipt_detect_regions(receipt_page).model_dump()
                logger.info(f"\n🎯 RECEIPT REGIONS\nRegions: {list(receipt_regions.keys())}\n")
                
                if receipt_regions:
                    for region_name, image_base64 in receipt_regions.items():
                        if not image_base64:
                            logger.warning(f"⚠️ Missing {region_name}, using full page")
                            image_base64 = receipt_page
                        
                        prompt = REGION_TO_PROMPT[region_name]
                        
                        all_requests.append(
                            models.OCRBatch(
                                page_type="receipt",
                                region_name=region_name,
                                prompt=prompt,
                                image=image_base64,
                            )
                        )
                else:
                    # Fallback to critical regions
                    logger.warning("⚠️ No receipt regions detected, using critical regions fallback")
                    critical_regions = ["pod_date", "pod_sign", "total_received"]
                    
                    for region_name in critical_regions:
                        prompt = REGION_TO_PROMPT[region_name]
                        all_requests.append(
                            models.OCRBatch(
                                page_type="receipt",
                                region_name=region_name,
                                prompt=prompt,
                                image=receipt_page,
                            )
                        )
        
        logger.info(f"\n🔍 REGION DETECTION COMPLETE\nTotal requests: {len(all_requests)}\n")
        
        # Process OCR
        logger.info(f"\n🤖 VLM OCR START\nBatch size: {len(all_requests)}\n")
        batch_responses = await batch_ocr(all_requests)
        logger.info(f"\n🤖 VLM OCR COMPLETE\nResponses: {len(batch_responses)}\n")
        
        # Group responses
        grouped_batch_responses = group_by_page_type(batch_responses)
        logger.info(f"\n⚙️ POST-PROCESSING START\nGroups: {list(grouped_batch_responses.keys())}\n")
        
        # Process responses
        for page_type, responses in grouped_batch_responses.items():
            ocr_output_image = {}
            
            if "BOL" in page_type:
                for response in responses:
                    logger.info(
                        f"\n📊 OCR RESPONSE\n"
                        f"Region: {response.region_name}\n"
                        f"Page: {response.page_type}\n"
                    )
                    
                    # Normalize stamp data
                    if response.region_name == "stamp":
                        stamp_field_types = {
                            "damage": int,
                            "short": int,
                            "refused": int,
                            "over": int,
                            "total_received": int,
                            "stamp_exist": str,
                            "notation_exist": str,
                            "pod_sign": str,
                            "pod_date": str,
                            "seal_intact": str
                        }
                        response.ocr_response = normalize_ocr_data(
                            response.ocr_response, 
                            stamp_field_types
                        )
                        logger.info(f"✅ Stamp data normalized")
                    
                    elif response.region_name == "customer_order_info":
                        customer_order_field_types = {"total_order_quantity": int}
                        response.ocr_response = normalize_ocr_data(
                            response.ocr_response, 
                            customer_order_field_types
                        )
                        logger.info(f"✅ Customer order data normalized")
                    
                    # Parse response
                    region: BaseModel = REGION_TO_MODEL[response.region_name]
                    response_parsed = region(**response.ocr_response)
                    logger.info(f"✅ === response.region_name, {response.region_name}")
                    if response.region_name == "stamp":
                        response_parsed.stamp_exist = response.stamp_exist
                        response_parsed.seal_intact = response.ocr_response.get("seal_intact", "null")
                        
                        # Parse total_received if string with numbers
                        if response_parsed.total_received not in ["empty", "null", "", "N/A"]:
                            if isinstance(response_parsed.total_received, str):
                                total_rec = re.findall(r"-?\d+", response_parsed.total_received)
                                if isinstance(total_rec, list) and len(total_rec) > 1:
                                    response_parsed.total_received = safe_int_conversion(total_rec[0], "null")
                        
                        # Signature fallback
                        if (response.ocr_response.get("pod_sign") in ["null", "empty", "", None] and
                            "signatures" in ocr_output_image and 
                            ocr_output_image["signatures"].get("receiver_signature")):
                            response_parsed.pod_sign = ocr_output_image["signatures"]["receiver_signature"]
                            logger.info("Using signatures region fallback")
                    
                    ocr_output_image.update({response.region_name: response_parsed})
                logger.info(f"✅ === before any message :)")
                try:
                    logger.info(f"✅ === after any message :)")
                    ocr_output_pdf["BOL"].append(
                        models.OCRResponse(**ocr_output_image).model_dump()
                    )
                except Exception as e:
                    logger.error(f"Validation failed for OCRResponse: {e}")
                    raise HTTPException(status_code=500, detail=f"Invalid response structure: {e}")
            
            elif page_type == "receipt":
                for response in responses:
                    # Normalize receipt data
                    if response.region_name in ["total_received", "damage", "refused"]:
                        if isinstance(response.ocr_response.get(response.region_name), list):
                            items = response.ocr_response[response.region_name]
                            response.ocr_response[response.region_name] = [
                                safe_int_conversion(item, 0) for item in items
                            ]
                    
                    # Sum lists - using safe dictionary access to prevent KeyError
                    if response.region_name == "total_received":
                        total_received_val = response.ocr_response.get("total_received")
                        if total_received_val not in ["empty", "null", "N/A", "", None]:
                            if isinstance(total_received_val, list):
                                response.ocr_response["total_received"] = sum(total_received_val)
                    elif response.region_name == "damage":
                        damage_val = response.ocr_response.get("damage")
                        if damage_val not in ["empty", "null", "N/A", "", None]:
                            if isinstance(damage_val, list):
                                response.ocr_response["damage"] = sum(damage_val)
                    elif response.region_name == "refused":
                        refused_val = response.ocr_response.get("refused")
                        if refused_val not in ["empty", "null", "N/A", "", None]:
                            if isinstance(refused_val, list):
                                response.ocr_response["refused"] = sum(refused_val)
                    
                    ocr_output_image.update(response.ocr_response)
                
                try:
                    ocr_output_pdf["receipt"].append(
                        models.DeliveryReceipt(**ocr_output_image).model_dump()
                    )
                except Exception as e:
                    logger.error(f"Validation failed for DeliveryReceipt: {e}")
                    raise HTTPException(status_code=500, detail=f"Invalid response structure: {e}")
        
        # Merge BOL data
        ocr_output_pdf["BOL"] = merge_bol_data(ocr_output_pdf["BOL"])
        
        logger.info("\n⚙️ POST-PROCESSING COMPLETE\n")
        
        # Generate POD fields
        POD_FIELDS = initialize_pod_fields()
        POD_FIELDS_LIST = []
        
        # Case 1: Receipt + single BOL
        if len(ocr_output_pdf.get("receipt", [])) == 1 and len(ocr_output_pdf.get("BOL", [])) == 1:
            delivery_receipt_data = ocr_output_pdf["receipt"][0]
            bol_data = ocr_output_pdf["BOL"][0]
            
            POD_FIELDS["POD Date"] = preprocess_date(delivery_receipt_data["pod_date"])
            POD_FIELDS["Signature Exists"] = delivery_receipt_data["pod_sign"]
            POD_FIELDS["Customer Order Num"] = process_customer_numbers(
                delivery_receipt_data["customer_order_num"]
            )
            POD_FIELDS["Issued Qty"] = safe_int_conversion(
                bol_data["customer_order_info"]["total_order_quantity"], 0
            )
            POD_FIELDS["B/L Number"] = bol_data["bill_of_lading"]["bill_no"]
            POD_FIELDS["Stamp Exists"] = bol_data["stamp"]["stamp_exist"]
            POD_FIELDS["Received Qty"] = safe_int_conversion(delivery_receipt_data["total_received"], "null")
            
            # Handle invalid stamp + no receipt + null issue date
            if (bol_data["stamp"]["stamp_exist"] != "yes" and 
                bol_data["bill_of_lading"].get("date", "null") in ["null", "empty", "", None]):
                yesterday = datetime.now() - timedelta(days=1)
                POD_FIELDS["POD Date"] = f"{yesterday.month:02d}/{yesterday.day:02d}/{yesterday.year % 100:02d}"
            
            # Process quantities
            if (delivery_receipt_data["total_received"] not in ["null", "empty", "0", "", None] and 
                POD_FIELDS["Issued Qty"] not in ["null", "empty", "N/A", "", "-", "0"]):
                POD_FIELDS["Received Qty"] = safe_int_conversion(POD_FIELDS["Issued Qty"], "null")
                
                issued_qty_int = safe_int_conversion(POD_FIELDS["Issued Qty"], 0)
                received_qty_int = safe_int_conversion(POD_FIELDS["Issued Qty"], 0)
                
                POD_FIELDS["Refused Qty"] = "null"
                POD_FIELDS["Short Qty"] = "null"
                POD_FIELDS["Damage Qty"] = "null"
                POD_FIELDS["Over Qty"] = "null"
            else:
                # Validate received quantity
                POD_FIELDS = validate_received_qty(POD_FIELDS, bol_data)
            
            POD_FIELDS_LIST = [POD_FIELDS]
        
        # Case 2: Multiple BOLs + receipt
        elif len(ocr_output_pdf.get("BOL", [])) > 1 and len(ocr_output_pdf.get("receipt", [])) == 1:
            POD_FIELDS["Status"] = "failed"
            POD_FIELDS_LIST = [POD_FIELDS]
        
        # Case 3: Multiple BOLs without receipt
        else:
            for bol_data in ocr_output_pdf.get("BOL", []):
                POD_FIELDS = initialize_pod_fields()
                
                bill_date_str = bol_data["bill_of_lading"].get("date", None)
                
                # Check for invalid stamp + no receipt + null issue date
                if ((bol_data["stamp"].get("stamp_exist") != "yes" and 
                     bol_data["stamp"].get("notation_exist") != "yes") and
                    len(ocr_output_pdf.get("receipt", [])) == 0 and
                    (bill_date_str is None or bill_date_str in ["null", "empty", ""])):
                    
                    yesterday = datetime.now() - timedelta(days=1)
                    POD_FIELDS["POD Date"] = f"{yesterday.month:02d}/{yesterday.day:02d}/{yesterday.year % 100:02d}"
                    POD_FIELDS["B/L Number"] = bol_data["bill_of_lading"]["bill_no"]
                    POD_FIELDS["Issued Qty"] = safe_int_conversion(
                        bol_data["customer_order_info"]["total_order_quantity"], 0
                    )
                    POD_FIELDS["Stamp Exists"] = bol_data["stamp"]["stamp_exist"]
                    POD_FIELDS["Seal Intact"] = bol_data["stamp"]["seal_intact"]
                    POD_FIELDS["Signature Exists"] = bol_data["stamp"]["pod_sign"]
                    
                    # Signature fallback
                    if (bol_data["stamp"]["pod_sign"] in ["null", "empty", "", None] and
                        "signatures" in bol_data and bol_data["signatures"].get("receiver_signature")):
                        POD_FIELDS["Signature Exists"] = bol_data["signatures"]["receiver_signature"]
                    
                    # Set quantities to null
                    for field in ["Received Qty", "Damage Qty", "Short Qty", "Over Qty", "Refused Qty"]:
                        POD_FIELDS[field] = "null"
                    POD_FIELDS["POD Date"] = "null"
                    POD_FIELDS["Signature Exists"] = "null"
                    POD_FIELDS["Stamp Exists"] = "null"
                    POD_FIELDS["Status"] = "invalid"
                    
                    logger.info("Using current date-1 due to invalid stamp + no receipt + null issue date")
                else:
                    # Parse dates
                    pod_date = preprocess_date(bol_data["stamp"]["pod_date"])
                    pod_month = None
                    
                    if pod_date not in ["null", "empty", "Error Processing Date"]:
                        try:
                            pod_parts = pod_date.split('/')
                            if len(pod_parts) >= 2:
                                pod_month = safe_int_conversion(pod_parts[0], None)
                        except:
                            pod_month = None
                    
                    processed_bill_date = preprocess_date(bill_date_str)
                    extracted_year = None
                    month_bill = None
                    day_bill = None
                    
                    if processed_bill_date not in ["null", "empty", "Error Processing Date"]:
                        try:
                            parts = processed_bill_date.split('/')
                            if len(parts) >= 2:
                                first_part = safe_int_conversion(parts[0], 0)
                                
                                # Use POD month if bill date is missing month
                                if pod_month is not None and first_part > 12:
                                    month_bill = pod_month
                                    day_bill = first_part
                                else:
                                    month_bill = first_part
                                    day_bill = safe_int_conversion(parts[1], 0)
                                
                                if len(parts) >= 3:
                                    year_str = parts[2]
                                    year_int = safe_int_conversion(year_str, None)
                                    if year_int is not None:
                                        extracted_year = 2000 + year_int if len(year_str) == 2 else year_int
                                    else:
                                        extracted_year = datetime.now().year
                                else:
                                    extracted_year = datetime.now().year
                        except Exception as e:
                            logger.error(f"Error parsing bill_of_lading date: {e}")
                    
                    # Check stamp validity
                    if ((bol_data["stamp"].get("stamp_exist") in ["yes", True, "True"] or
                         bol_data["stamp"].get("notation_exist") in ["yes", True, "True"]) and
                        bol_data["stamp"].get("pod_sign") in ["yes", "no", True, False, "True", "False"]):
                        
                        # Valid stamp - prefer POD date, then bill date, then yesterday
                        if pod_date not in ["null", "empty", "Error Processing Date"]:
                            POD_FIELDS["POD Date"] = pod_date
                            logger.info(f"✅ Using POD date: {POD_FIELDS['POD Date']}")
                        elif processed_bill_date not in ["null", "empty", "Error Processing Date"]:
                            POD_FIELDS["POD Date"] = processed_bill_date
                            logger.info(f"Using bill date: {POD_FIELDS['POD Date']}")
                        else:
                            POD_FIELDS["POD Date"] = pod_date
                            logger.info(f"Using yesterday date: {POD_FIELDS['POD Date']}")
                    else:
                        # Invalid stamp
                        POD_FIELDS["POD Date"] = "null"
                        POD_FIELDS["Signature Exists"] = "null"
                        POD_FIELDS["Stamp Exists"] = "null"
                        POD_FIELDS["Status"] = "invalid"
                        logger.info("Invalid stamp - setting POD date to null")
                    
                    # Set basic fields
                    POD_FIELDS["B/L Number"] = bol_data["bill_of_lading"]["bill_no"]
                    POD_FIELDS["Issued Qty"] = safe_int_conversion(
                        bol_data["customer_order_info"]["total_order_quantity"], 0
                    )
                    POD_FIELDS["Stamp Exists"] = bol_data["stamp"]["stamp_exist"]
                    POD_FIELDS["Seal Intact"] = bol_data["stamp"]["seal_intact"]
                    POD_FIELDS["Signature Exists"] = bol_data["stamp"]["pod_sign"]
                    
                    # Signature fallback
                    if (bol_data["stamp"]["pod_sign"] in ["null", "empty", "", None] and
                        "signatures" in bol_data and bol_data["signatures"].get("receiver_signature")):
                        POD_FIELDS["Signature Exists"] = bol_data["signatures"]["receiver_signature"]
                        logger.info("Using signatures region fallback")
                    
                    # Process quantities
                    stamp_total_received = safe_int_conversion(bol_data["stamp"]["total_received"], 0)
                    stamp_total_received_str = safe_str_conversion(bol_data["stamp"]["total_received"], "null")
                    customer_order_qty = safe_int_conversion(
                        bol_data["customer_order_info"]["total_order_quantity"], 0
                    )
                    
                    POD_FIELDS = process_quantities(
                        bol_data,
                        stamp_total_received,
                        stamp_total_received_str,
                        customer_order_qty,
                        POD_FIELDS,
                        POD_FIELDS.get("Customer_Order_Num","null")
                    )
                    
                    logger.info(f"End of post_process: {datetime.now()}")
                    
                    # Validate received quantity
                    POD_FIELDS = validate_received_qty(POD_FIELDS, bol_data)
                
                POD_FIELDS_LIST.append(POD_FIELDS)
        
        return POD_FIELDS_LIST
    
    except Exception as e:
        logger.exception("Error during OCR analysis")
        raise HTTPException(status_code=500, detail=f"Error during analysis: {str(e)}")
