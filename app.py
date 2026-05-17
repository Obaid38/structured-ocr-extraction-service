from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from typing import Optional, List, Dict, Any, Tuple
from langsmith import traceable
from dotenv import load_dotenv
import os
import asyncio
from ultralytics import YOLO
import concurrent.futures
import logging
from app_settings import get_env_or_config, get_int_env_or_config
from utils import convert_image_to_base64
from pydantic import BaseModel, Field, ValidationError
import time
from openai import AsyncOpenAI

import models
from models import (
    OCRRequest, 
    PODResponse,
    TemplateSyncRequest,
    TemplateSyncResponse,
    EnhancedOCRRequest,
    EnhancedOCRResponse,
    TemplateTestRequest,
    TemplateTestResponse,
    ClassificationDetails,
    SuggestedTemplate,
    MemoryStoreStats
)
import time
import numpy as np
from main import run_ocr

from template_engine import get_template_store, TemplateMatcher, RegionDetector, PromptLoader, PostProcessor

from startup import initialize_application, shutdown_application

try:
    from sticker_pipeline import StickerProcessor
except ImportError:
    StickerProcessor = None

try:
    from receipt_pipeline.receipt_processor import ReceiptProcessor
except ImportError:
    ReceiptProcessor = None

# Receipt case → template ID mapping
RECEIPT_TEMPLATE_MAP = {
    "R1": "GENERIC_RECEIPT_LAYOUT_1",
    "R2": "GENERIC_RECEIPT_LAYOUT_2",
    "R3": "GENERIC_RECEIPT_LAYOUT_3",
    "R4": "GENERIC_RECEIPT_LAYOUT_4",
    "R5": "GENERIC_RECEIPT_LAYOUT_5",
}

# Global sticker processor (initialized at startup)
sticker_processor = None

# Global receipt processor (initialized at startup)
receipt_processor = None

load_dotenv()

app = FastAPI(default_response_class=ORJSONResponse)

# Basic structured logging for API
logging.basicConfig(
    level=get_env_or_config("LOG_LEVEL", ("runtime", "log_level"), "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("app")

# Create shared AsyncOpenAI client
lmdeploy_async_client = AsyncOpenAI(
    api_key="EMPTY",
    base_url=get_env_or_config(
        "LMDEPLOY_URL",
        ("services", "vlm_base_url"),
        "http://127.0.0.1:23333/v1",
    ),
)

# Template engine components
template_store = get_template_store()
memory_store = template_store
template_matcher = TemplateMatcher(lmdeploy_client=lmdeploy_async_client)
region_detector = RegionDetector()
prompt_loader = PromptLoader()
post_processor = PostProcessor()

logger.info("Template engine components initialized")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== GLOBALS ==========
# Note: Ensure your YOLO model loading is thread-safe if using multiple workers.
# For Uvicorn with workers=1, this is fine. For >1, look into proper model management.
YOLO_MODEL = YOLO(
    get_env_or_config(
        "DOC_CLASSIFIER_MODEL_PATH",
        ("models", "document_classifier"),
        "Models/doc_classify_best.pt",
    )
)
YOLO_QUEUE = asyncio.Queue()

# ========== GPU Inference Background Worker ==========
async def gpu_worker():
    while True:
        job = await YOLO_QUEUE.get()
        try:
            images = job["images"]
            conf = job.get("conf", 0.55)
            batch = job.get("batch", len(images))
            result = YOLO_MODEL.predict(images, conf=conf, batch=batch)
            job["future"].set_result(result)
        except Exception as e:
            job["future"].set_exception(e)
        finally:
            YOLO_QUEUE.task_done()
 
# Launch the GPU worker(s) at startup
@app.on_event("startup")
async def startup_event():
    """
    Application startup handler
    1. Initialize MongoDB connection
    2. Load configuration
    3. Load templates into memory
    4. Start GPU workers
    5. Initialize sticker processor
    """
    global sticker_processor, receipt_processor

    logger.info("🚀 Starting application...")

    # Initialize MongoDB and load templates
    try:
        success = await initialize_application()
        if not success:
            logger.error("⚠️  Application started with errors - check logs above")
    except Exception as e:
        logger.error(f"❌ Critical startup error: {e}")
        logger.exception("Full traceback:")

    # Start GPU workers
    num_workers = get_int_env_or_config("GPU_WORKERS", ("runtime", "gpu_workers"), 2)
    logger.info(f"🔧 Starting {num_workers} GPU worker(s)...")
    for _ in range(num_workers):
        asyncio.create_task(gpu_worker())

    # Initialize sticker processor (SK1/SK2/SK3 detection)
    try:
        sticker_model_path = os.getenv("STICKER_MODEL_PATH", "Models/best_sticker_ob.pt")
        orientation_model_path = os.getenv("ORIENTATION_MODEL_PATH", "Models/page_orientation.pt")
        sticker_processor = StickerProcessor(
            model_path=sticker_model_path,
            orientation_model_path=orientation_model_path,
            vlm_base_url=os.getenv("LMDEPLOY_URL", "http://127.0.0.1:23333/v1")
        )
        logger.info(f"✅ Sticker processor initialized (model: {sticker_model_path}, orientation: {orientation_model_path})")
    except Exception as e:
        logger.warning(f"⚠️  Sticker processor not initialized: {e}")
        sticker_processor = None

    # Initialize receipt processor (R1-R5 detection)
    # Uses v3 model for R1-R4 (better accuracy) and v2 model for R5 (100% R5 accuracy)
    try:
        receipt_model_path = os.getenv("RECEIPT_MODEL_PATH", "Models/best_receipt_ob_v3.pt")
        r5_model_path = os.getenv("R5_MODEL_PATH", "Models/best_receipt_ob_v2.pt")
        receipt_processor = ReceiptProcessor(
            model_path=receipt_model_path,
            r5_model_path=r5_model_path,
            orientation_model_path=orientation_model_path,
            vlm_base_url=os.getenv("LMDEPLOY_URL", "http://127.0.0.1:23333/v1")
        )
        logger.info(f"✅ Receipt processor initialized (model: {receipt_model_path}, R5 model: {r5_model_path})")
    except Exception as e:
        logger.warning(f"⚠️  Receipt processor not initialized: {e}")
        receipt_processor = None

    logger.info("✅ Application startup complete")

# Shutdown handler to clean up resources
@app.on_event("shutdown")
async def shutdown_event():
    """
    Application shutdown handler
    1. Disconnect from MongoDB
    2. Shutdown thread pool
    3. Close template matcher HTTP client
    """
    logger.info("🛑 Shutting down application...")
    
    # Disconnect from MongoDB
    try:
        await shutdown_application()
    except Exception as e:
        logger.error(f"❌ Error during MongoDB shutdown: {e}")
    
    # Shutdown thread pool
    logger.info("Shutting down ThreadPoolExecutor...")
    CPU_POOL.shutdown(wait=True, cancel_futures=False)
    logger.info("ThreadPoolExecutor shut down complete")
    
    # Close template matcher HTTP client
    logger.info("Closing template matcher HTTP client...")
    try:
        await template_matcher.close()
        logger.info("Template matcher HTTP client closed")
    except Exception as e:
        logger.error(f"Error closing template matcher: {e}")
    
    logger.info("✅ Application shutdown complete")

# This section with commented-out code seems unused in your final endpoint.
# You can remove it for clarity if it's no longer needed.
# # ========== CLASSIFY via YOLO QUEUE ==========
# async def classify_with_gpu_queue(images, conf=0.55, batch=5):
#     ...
# # ========== OCR Endpoint (Upload + Classify + OCR) ==========
# @app.post("/ocr")
# async def handle_user_ocr(file: UploadFile = File(...)):
#     ...
# @app.on_event("startup")
# async def load_models():
#     ...

# ====================================================================
#  START: MODIFIED SECTION FOR CONCURRENT PROCESSING
# ====================================================================

CPU_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count())

def _cap_image_resolution(images: list, max_long_edge: int = 2000) -> list:
    """
    Resize any page whose longest side exceeds max_long_edge.

    Why: large-format PDFs (engineering drawings, tabloid scans) can render
    to 9000x7000+ pixels at 200 DPI. PIL/numpy ops on 71-MP images block the
    asyncio event loop for 60-90 seconds, causing all concurrent requests to
    timeout. 2000px is more than sufficient for YOLO detection and VLM OCR.
    """
    from PIL import Image
    capped = []
    for img in images:
        w, h = img.size
        long_edge = max(w, h)
        if long_edge > max_long_edge:
            scale = max_long_edge / long_edge
            new_w, new_h = int(w * scale), int(h * scale)
            logger.warning(
                f"Oversized page detected: {w}×{h} px ({w*h//1_000_000}MP) — "
                f"resizing to {new_w}×{new_h} to prevent event-loop blocking"
            )
            img = img.resize((new_w, new_h), Image.LANCZOS)
        capped.append(img)
    return capped

def clear_gpu_cache():
    """Release unused GPU memory back to the CUDA allocator."""
    try:
        import torch
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        logger.debug(f"GPU cache clear skipped: {e}")

def check_gpu_memory() -> bool:
    """
    Check if GPU memory is low
    Returns True if memory is low, False otherwise
    """
    try:
        import torch
        if torch.cuda.is_available():
            # Get memory info for GPU 0
            mem_free = torch.cuda.mem_get_info(0)[0]
            mem_total = torch.cuda.mem_get_info(0)[1]
            usage_percent = ((mem_total - mem_free) / mem_total) * 100
            
            logger.info(f"GPU Memory Usage: {usage_percent:.1f}%")
            
            # Threshold at 97%: LMDeploy pre-allocates InternVL2-8B model weights
            # + KV cache at startup, permanently consuming ~89% of VRAM on an A40.
            # This is expected standing usage — not garbage that can be freed.
            # The old 85% threshold caused sequential mode to be ALWAYS ON, doubling
            # batch processing time and pushing requests past upstream request timeouts.
            # 97% leaves ~1.4 GB headroom, enough for concurrent YOLO inference (~500 MB each).
            return usage_percent > 97
        else:
            logger.warning("CUDA not available, assuming sufficient memory")
            return False
    except Exception as e:
        logger.error(f"Error checking GPU memory: {e}")
        return False

def _run_ocr_sync(path: str):
    """sync wrapper: execute the async pipeline in this worker thread"""
    return asyncio.run(run_ocr(path))

async def run_ocr_async(path: str):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(CPU_POOL, _run_ocr_sync, path)

async def process_single_ocr_request(ocr_request: OCRRequest) -> List[PODResponse]:
    """
    Handles the complete processing for a single OCR request.
    This function contains the logic from your original endpoint.
    """
    try:
        # 1. Run the core OCR and analysis logic for one file
        logger.info(f"\n\n🎯 STARTING SINGLE OCR REQUEST\nRequest ID: {ocr_request.id}\nFile path: {ocr_request.file_url_or_path}")
        result_items = await run_ocr_async(ocr_request.file_url_or_path)
        logger.info(f"\n\n✅ COMPLETED SINGLE OCR REQUEST\nRequest ID: {ocr_request.id}\nItems generated: {len(result_items)}\n\n")

        # 2. Map the results to the PODResponse model, adding the specific id
        responses = []
        for item in result_items:
            logger.info(f"\n\n📋 FINAL POD FIELDS FOR REQUEST {ocr_request.id}\nB/L Number: {item.get('B/L Number')}\nStamp Exists: {item.get('Stamp Exists')}\nPOD Date: {item.get('POD Date')}\nReceived Qty: {item.get('Received Qty')}\nDamage Qty: {item.get('Damage Qty')}\nShort Qty: {item.get('Short Qty')}\nOver Qty: {item.get('Over Qty')}\nRefused Qty: {item.get('Refused Qty')}\nStatus: {item.get('Status')}\nNotation Exists: {item.get('Notation Exists')}\n\n")
            response_dict = {
                "B_L_Number": item.get("B/L Number"),
                "Stamp_Exists": item.get("Stamp Exists"),
                "Seal_Intact": item.get("Seal Intact"),
                "POD_Date": item.get("POD Date"),
                "Signature_Exists": item.get("Signature Exists"),
                "Issued_Qty": item.get("Issued Qty"),
                "Received_Qty": item.get("Received Qty"),
                "Damage_Qty": item.get("Damage Qty"),
                "Short_Qty": item.get("Short Qty"),
                "Over_Qty": item.get("Over Qty"),
                "Refused_Qty": item.get("Refused Qty"),
                "Customer_Order_Num": item.get("Customer Order Num"),
                "Notation_Exists": item.get("Notation Exists"),
                "Status": item.get("Status"),
                "id": ocr_request.id  # Attach the ID from the original request
            }
            responses.append(PODResponse.model_validate(response_dict))
        
        return responses

    except Exception as e:
        # Log the error with context and re-raise it so asyncio.gather can catch it.
        # This prevents one failed file from crashing the entire batch.
        logger.exception(f"Error processing request id={ocr_request.id}: {e}")
        # Re-raising the exception is important for the gathering logic
        raise e


@traceable(run_type="chain", name="api_batch_ocr")

# ============================================================================
# 🆕 STEP 3: ENDPOINT - Template Sync
# ============================================================================

@app.post("/api/templates/sync", response_model=TemplateSyncResponse)
async def sync_template(request: TemplateSyncRequest):
    """
    Sync template with AI memory store
    Called by Backend when template status changes
    
    Actions:
    - 'add': Add new active template to memory
    - 'remove': Remove inactive/deprecated template from memory
    - 'update': Update existing template in memory
    """
    try:
        action = request.action.lower()
        
        if action == "add" or action == "update":
            if not request.template:
                return TemplateSyncResponse(
                    success=False,
                    message="Template data required for add/update action"
                )

            # Convert ObjectId to string if present
            if "_id" in request.template:
                from bson import ObjectId
                if isinstance(request.template["_id"], ObjectId):
                    request.template["_id"] = str(request.template["_id"])
            
            # update image URLs before adding to memory
            from template_loader import get_template_loader
            loader = get_template_loader()
            fixed_template = loader._fix_image_urls(request.template)
            
            # Add/update with fixed template
            success = template_store.add_template(fixed_template)
            template_id = fixed_template.get("template_id")
            
            if success:
                stats = template_store.get_stats()
                
                # Log the action
                logger.info(f"✅ Template sync successful: {action.upper()}")
                logger.info(f"   Template ID: {template_id}")
                logger.info(f"   MongoDB _id: {fixed_template.get('_id', 'N/A')}")
                logger.info(f"   Template Name: {fixed_template.get('template_name', 'Unknown')}")
                
                return TemplateSyncResponse(
                    success=True,
                    message=f"Template {template_id} {'added' if action == 'add' else 'updated'} successfully",
                    template_id=template_id,
                    stats=stats
                )
            else:
                return TemplateSyncResponse(
                    success=False,
                    message=f"Failed to {action} template"
                )
        
        elif action == "remove":
            if not request.template_id:
                return TemplateSyncResponse(
                    success=False,
                    message="Template ID required for remove action"
                )
            
            logger.info(f"→ Attempting to remove template: {request.template_id}")
            logger.info(f"→ Current templates in memory: {list(template_store._templates.keys())}")
            
            # Try to remove template (supports both MongoDB _id and template_id string)
            success = template_store.remove_template(request.template_id)
            
            if success:
                stats = template_store.get_stats()
                
                logger.info(f"✅ Template removed successfully: {request.template_id}")
                logger.info(f"→ Remaining templates in memory: {len(template_store._templates)}")
                
                return TemplateSyncResponse(
                    success=True,
                    message=f"Template {request.template_id} removed successfully",
                    template_id=request.template_id,
                    stats=stats
                )
            else:
                logger.error(f"❌ Template {request.template_id} not found in memory")
                logger.error(f"→ Available template IDs: {list(template_store._templates.keys())}")
                
                # Get more details about what's in memory
                if template_store._templates:
                    logger.error("→ Template details in memory:")
                    for t_id, t_data in template_store._templates.items():
                        logger.error(f"   - Key: {t_id}")
                        logger.error(f"     MongoDB _id: {t_data.get('_id', 'N/A')}")
                        logger.error(f"     template_id: {t_data.get('template_id', 'N/A')}")
                        logger.error(f"     template_name: {t_data.get('template_name', 'N/A')}")
                
                return TemplateSyncResponse(
                    success=False,
                    message=f"Template {request.template_id} not found in memory. Available templates: {list(template_store._templates.keys())}"
                )
        
        else:
            return TemplateSyncResponse(
                success=False,
                message=f"Unknown action: {action}. Use 'add', 'remove', or 'update'"
            )
    
    except Exception as e:
        logger.error(f"❌ Error in template sync: {e}")
        logger.exception("Full traceback:")
        return TemplateSyncResponse(
            success=False,
            message=f"Error: {str(e)}"
        )
# ============================================================================
# 🆕 STEP 4: NEW ENDPOINT - Template Testing
# ============================================================================

@app.post("/api/templates/test", response_model=TemplateTestResponse)
async def test_template(request: TemplateTestRequest):
    """
    Test a template that is NOT active/NOT in memory
    Used by Frontend for testing templates before activation
    
    Frontend sends complete template JSON in request body
    AI processes document with provided template (NO MATCHING)
    """
    start_time = time.time()
    
    try:
        # ========== STEP 1: DOWNLOAD AND CLASSIFY PDF ==========
        logger.info("→ Step 1: Downloading and classifying PDF...")
        
        from main import pdf_to_images, classify_documents
        from PIL import Image
        from utils import convert_image_to_base64
        import numpy as np
        
        images = await pdf_to_images(request.file_url)
        if not images:
            raise Exception("Failed to process document")
        
        logger.info(f"→ PDF has {len(images)} page(s), classifying all pages...")
        
        # Convert all images to base64
        images_base64 = [convert_image_to_base64(img) for img in images]
        
        # Classify all pages to find BOL
        classified_pages = classify_documents(images_base64)
        
        logger.info(f"  Classification results:")
        for doc_type, pages in classified_pages.items():
            logger.info(f"    - {doc_type}: {len(pages)} page(s)")
        
        # Find BOL page (or use first page as fallback)
        bol_pages = classified_pages.get("BOL", [])
        if bol_pages:
            image_base64 = bol_pages[0]
            classification_category = "BOL"
            primary_confidence = 0.95
            logger.info(f"  ✅ Found BOL page (out of {len(images)} total pages)")
        else:
            receipt_pages = classified_pages.get("receipt", [])
            if receipt_pages:
                image_base64 = receipt_pages[0]
                classification_category = "receipt"
                primary_confidence = 0.90
                logger.info(f"  ℹ️  No BOL found, using receipt page")
            else:
                image_base64 = images_base64[0]
                classification_category = "others"
                primary_confidence = 0.50
                logger.info(f"  ⚠️  No BOL/receipt found, using first page")
        
        # Convert base64 back to PIL Image
        from utils import base64topil
        image = base64topil(image_base64)
        
        logger.info(f"✓ Step 1: Classification complete - {classification_category} (confidence: {primary_confidence:.2f})")
        
        # ========== STEP 2: FIX TEMPLATE IMAGE URLS ==========
        logger.info("→ Step 2: Fixing template image URLs...")
        from template_loader import get_template_loader
        loader = get_template_loader()
        fixed_template = loader._fix_image_urls(request.template)
        logger.info("✓ Step 2: Template image URLs fixed")
        
        # ========== STEP 3: GET TEMPLATE CATEGORY ==========
        template_category = fixed_template.get("category", "BOL")
        logger.info(f"→ Step 3: Template category: {template_category}")
        
        # Ensure image is PIL Image
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
        
        # ========== STEP 4: PRE-DETECT REGIONS ONCE ==========
        logger.info("→ Step 4: Pre-detecting regions (for confidence calculation)...")
        
        detected_regions_cache = region_detector.detect_regions(
            image=image_pil,
            template=fixed_template,
            category=template_category
        )
        
        logger.info(f"✓ Step 4: Detected {len(detected_regions_cache)} regions (cached)")
        for region_name in detected_regions_cache.keys():
            logger.info(f"  - {region_name}")
        
        # ========== STEP 5: PROCESS DOCUMENT WITH PROVIDED TEMPLATE ==========
        # ✅ NO MATCHING - Use provided template directly
        logger.info("→ Step 5: Processing document with PROVIDED template (no matching)...")
        
        result = await process_document_with_template(
            file_url=request.file_url,
            template=fixed_template,
            user_id=request.user_id,
            image=image_pil,
            skip_matching=True
        )
        
        logger.info("✓ Step 5: Document processed successfully")
        
        # ========== STEP 6: CALCULATE CONFIDENCE (for testing purposes) ==========
        logger.info("→ Step 6: Calculating template confidence (for testing metrics)...")
        
        _, template_confidence, _ = await template_matcher.match_templates(
            image=image_pil,
            templates=[fixed_template],
            cached_regions=detected_regions_cache
        )
        
        logger.info(f"✓ Step 6: Confidence calculated: {template_confidence:.4f} ({template_confidence*100:.2f}%)")
        
        # ========== STEP 7: BUILD REGION METADATA ==========
        logger.info("→ Step 7: Building region metadata...")
        
        regions_metadata = []
        for region_name, region_data in detected_regions_cache.items():
            if isinstance(region_data, dict) and 'bbox' in region_data:
                x1, y1, x2, y2 = region_data['bbox']
                regions_metadata.append({
                    "region_name": region_name,
                    "bbox": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2)
                    },
                    "confidence": region_data.get('confidence', 0.0)
                })
        
        logger.info(f"✓ Step 7: Built metadata for {len(regions_metadata)} regions")
        
        # ========== STEP 8: ADD METADATA TO RESULT ==========
        processing_time = int((time.time() - start_time) * 1000)
        
        result.template_id = request.template.get("_id")
        result.template_identifier = request.template.get("template_id")
        result.confidence = template_confidence
        result.processing_time = processing_time
        result.classification_details = ClassificationDetails(
            primary_model_prediction=classification_category,
            primary_confidence=primary_confidence
        )
        result.suggested_templates = []
        
        logger.info(f"""
        ╔═══════════════════════════════════════════════════════════╗
        ║ ✅ TEMPLATE TEST COMPLETE                                 ║
        ╠═══════════════════════════════════════════════════════════╣
        ║ Template:         {request.template.get('template_name', 'Unknown'):<36} ║
        ║ Confidence:       {template_confidence*100:.2f}%{' '*(37-len(f'{template_confidence*100:.2f}%'))}║
        ║ Processing Time:  {processing_time} ms{' '*(37-len(f'{processing_time} ms'))}║
        ║ Status:           {'PASS (≥75%)' if template_confidence >= 0.75 else 'FAIL (<75%)':<36} ║
        ╚═══════════════════════════════════════════════════════════╝
                """)
        
        return TemplateTestResponse(
            success=True,
            extracted_data=result,
            processing_time=processing_time,
            regions_metadata=regions_metadata
        )
    
    except Exception as e:
        logger.error(f"❌ Error in template testing: {e}")
        logger.exception("Full traceback:")
        processing_time = int((time.time() - start_time) * 1000)
        
        return TemplateTestResponse(
            success=False,
            error=str(e),
            processing_time=processing_time
        )

# ============================================================================
# 🆕 STEP 5: NEW ENDPOINT - Memory Store Stats
# ============================================================================

@app.get("/api/templates/stats", response_model=MemoryStoreStats)
async def get_memory_stats():
    """Get memory store statistics"""
    # Debug: Check if memory_store exists
    logger.info(f"memory_store type: {type(memory_store)}")
    logger.info(f"memory_store exists: {memory_store is not None}")
    
    try:
        stats = memory_store.get_stats()
        logger.info(f"Stats retrieved: {stats}")
        
        return MemoryStoreStats(
            total_templates=stats.get('total_templates', 0),
            categories=list(stats.get('by_category', {}).keys()),
            templates_by_category=stats.get('by_category', {})
        )
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run-ocr")
async def analyze_images_concurrently(ocr_requests: List[EnhancedOCRRequest]):
    """
    MODIFIED: Enhanced OCR endpoint with template support
    
    Handles 3 cases:
    1. Normal flow: Match template, process if score >= 0.75
    2. Unregistered doc: template_id in request (skip matching)
    3. No match: Return suggested templates
    """
    results = []
    
    # Check GPU memory
    is_low_memory = check_gpu_memory()
    
    if is_low_memory:
        logger.warning("Low GPU memory detected, processing sequentially")
        for request in ocr_requests:
            try:
                result = await process_single_document_enhanced(request)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing document: {e}")
                results.append(create_error_response(str(e)))
    else:
        # Concurrent processing
        tasks = [process_single_document_enhanced(req) for req in ocr_requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        results = [
            res if not isinstance(res, Exception) else create_error_response(str(res))
            for res in results
        ]
    
    return results

# reprocess un-registered docs
@app.post("/api/ocr/reprocess", response_model=EnhancedOCRResponse)
async def reprocess_unregistered_document(request: EnhancedOCRRequest):
    """
    🔄 REPROCESS UNREGISTERED DOCUMENT WITH ASSIGNED TEMPLATE
    
    Flow:
    1. Document was previously unregistered (no match)
    2. Admin reviewed suggestions and assigned a template
    3. Frontend sends document + template_id in request body
    4. AI fetches template from memory (NO MATCHING), processes directly
    
    ✅ CORRECT: template_id is in request.template_id → Fetch from memory, skip matching
    
    Request Body:
    {
        "file_url": "https://example.com/document.pdf",
        "_id": "12345",
        "template_id": "693a48d6527f2aeee281245a"  ← REQUIRED
    }
    """
    start_time = time.time()
    
    logger.info(f"""
        ╔═══════════════════════════════════════════════════════════╗
        ║ 🔄 REPROCESSING API - UNREGISTERED DOCUMENT               ║
        ╠═══════════════════════════════════════════════════════════╣
        ║ File URL:     {request.file_url:<40} ║
        ║ _id:          {request.id or 'N/A':<40} ║
        ║ Template ID:  {request.template_id or 'N/A':<40} ║
        ╚═══════════════════════════════════════════════════════════╝
            """)
    
    try:
        # ========== STEP 1: VALIDATE TEMPLATE_ID ==========
        if not request.template_id:
            logger.error("❌ Template ID missing in reprocess request")
            response = create_error_response("Template ID is required for reprocessing")
            response.id = request.id
            return response
        
        logger.info(f"✓ Step 1/7: Template ID validated: {request.template_id}")
        
        # ========== STEP 2: GET TEMPLATE FROM MEMORY ==========
        template = template_store.get_template(request.template_id)
                
        if not template:
            logger.error(f"❌ Template {request.template_id} not found in memory")
            logger.error(f"❌ Total templates in memory: {len(template_store._templates)}")
            
            if len(template_store._templates) == 0:
                error_msg = "No templates loaded in memory. Please sync templates first."
            else:
                available = list(template_store._templates.keys())
                logger.info(f"Available templates: {available}")
                error_msg = f"Template '{request.template_id}' not found in memory. Available: {len(available)} template(s)"
            
            response = EnhancedOCRResponse(
                B_L_Number="",
                Stamp_Exists="",
                Seal_Intact="null",
                POD_Date="",
                Signature_Exists="",
                Issued_Qty=0,
                Received_Qty=0,
                Damage_Qty="0",
                Short_Qty="0",
                Over_Qty="0",
                Refused_Qty="null",
                Customer_Order_Num="null",
                id=request.id,
                template_id=None,
                template_identifier=None,
                confidence=0.0,
                processing_time=int((time.time() - start_time) * 1000),
                classification_details=None,
                suggested_templates=[],
                error=error_msg
            )
            return response
        
        logger.info(f"""
        ✓ Step 2/7: Template retrieved from memory
        Template Name: {template.get('template_name', 'Unknown')}
        Category: {template.get('category', 'Unknown')}
                """)
        
        # ========== STEP 3: DOWNLOAD AND CLASSIFY ALL PAGES ==========
        logger.info(f"→ Step 3/7: Downloading PDF from {request.file_url}")
        
        from main import pdf_to_images, classify_documents
        from utils import convert_image_to_base64
        
        images = await pdf_to_images(request.file_url)
        
        if not images:
            logger.error("❌ Failed to process PDF")
            response = create_unregistered_response("Failed to process PDF", start_time)
            response.id = request.id
            return response
        
        logger.info(f"→ PDF has {len(images)} page(s), classifying all pages...")
        
        # Convert all images to base64
        images_base64 = [convert_image_to_base64(img) for img in images]
        
        # Classify all pages to find BOL
        classified_pages = classify_documents(images_base64)
        
        logger.info(f"  Classification results:")
        for doc_type, pages in classified_pages.items():
            logger.info(f"    - {doc_type}: {len(pages)} page(s)")
        
        # Find BOL page (or use first page as fallback)
        bol_pages = classified_pages.get("BOL", [])
        if bol_pages:
            image_base64 = bol_pages[0]
            classification_category = "BOL"
            logger.info(f"  ✅ Found BOL page (out of {len(images)} total pages)")
        else:
            receipt_pages = classified_pages.get("receipt", [])
            if receipt_pages:
                image_base64 = receipt_pages[0]
                classification_category = "receipt"
                logger.info(f"  ℹ️  No BOL found, using receipt page")
            else:
                image_base64 = images_base64[0]
                classification_category = "others"
                logger.info(f"  ⚠️  No BOL/receipt found, using first page")
        
        # Convert base64 back to PIL Image
        from utils import base64topil
        image = base64topil(image_base64)
        
        logger.info(f"✓ Step 3/7: PDF processed - {len(images)} page(s), using page classified as {classification_category}")
        
        # ========== STEP 4: GET TEMPLATE CATEGORY ==========
        template_category = template.get("category", "BOL")
        logger.info(f"→ Step 4/7: Template category: {template_category}")
        
        # Ensure PIL Image format
        from PIL import Image as PILImage
        import numpy as np
        if isinstance(image, np.ndarray):
            image_pil = PILImage.fromarray(image)
        else:
            image_pil = image
        
        # ========== STEP 5: PRE-DETECT REGIONS ONCE ==========
        logger.info("→ Step 5/7: Pre-detecting regions (for confidence calculation)...")
        
        detected_regions_cache = region_detector.detect_regions(
            image=image_pil,
            template=template,
            category=template_category
        )
        
        logger.info(f"✓ Step 5/7: Detected {len(detected_regions_cache)} regions (cached)")
        for region_name in detected_regions_cache.keys():
            logger.info(f"  - {region_name}")
        
        # ========== STEP 6: PROCESS WITH ASSIGNED TEMPLATE ==========
        # ✅ NO MATCHING - Use fetched template directly
        logger.info(f"→ Step 6/7: Processing with ASSIGNED template (no matching)...")
        
        result = await process_document_with_template(
            file_url=request.file_url,
            template=template,
            image=image_pil
        )
        
        logger.info(f"✓ Step 6/7: Document processed successfully")
        
        # ========== STEP 7: CALCULATE CONFIDENCE (for metrics) ==========
        logger.info(f"→ Step 7/7: Calculating template confidence (for metrics)...")
        
        _, template_confidence, _ = await template_matcher.match_templates(
            image=image_pil,
            templates=[template],
            cached_regions=detected_regions_cache
        )
        
        logger.info(f"✓ Step 7/7: Confidence calculated: {template_confidence:.4f} ({template_confidence*100:.2f}%)")
        
        # ========== STEP 8: ADD METADATA ==========
        processing_time = int((time.time() - start_time) * 1000)
        
        # Get primary classification confidence
        primary_confidence = 0.95 if classification_category == "BOL" else 0.90 if classification_category == "receipt" else 0.50
        
        result.id = request.id
        result.template_id = template.get("_id")
        result.template_identifier = template.get("template_id")
        result.confidence = template_confidence
        result.processing_time = processing_time
        result.classification_details = ClassificationDetails(
            primary_model_prediction=classification_category,
            primary_confidence=primary_confidence
        )
        result.suggested_templates = []
        
        logger.info(f"""
        ╔═══════════════════════════════════════════════════════════╗
        ║ ✅ REPROCESSING COMPLETE                                  ║
        ╠═══════════════════════════════════════════════════════════╣
        ║ Template Used:    {template.get('template_name', 'Unknown'):<36} ║
        ║ Confidence:       {template_confidence*100:.2f}%{' '*(37-len(f'{template_confidence*100:.2f}%'))}║
        ║ Processing Time:  {processing_time} ms{' '*(37-len(f'{processing_time} ms'))}║
        ╚═══════════════════════════════════════════════════════════╝
                """)
        
        return result
    
    except Exception as e:
        logger.error(f"❌ Error in reprocessing: {e}")
        logger.exception("Full traceback:")
        response = create_error_response(str(e))
        response.id = request.id
        return response

@app.post("/api/templates/cleanup")
async def cleanup_category_index():
    """
    Clean up category index by removing invalid/ghost entries
    Fixes the mismatch between total_templates and templates_by_category
    """
    logger.info("=" * 70)
    logger.info("🧹 CLEANUP: Starting category index cleanup")
    logger.info("=" * 70)
    
    try:
        # Show current state
        logger.info(f"Before cleanup:")
        logger.info(f"  Total templates in memory: {len(template_store._templates)}")
        logger.info(f"  Template keys: {list(template_store._templates.keys())}")
        
        cleaned_count = 0
        for category, template_ids in template_store._templates_by_category.items():
            logger.info(f"  Category '{category}' has {len(template_ids)} entries: {template_ids}")
            
            # Find invalid entries
            invalid_ids = [tid for tid in template_ids if tid not in template_store._templates]
            if invalid_ids:
                logger.warning(f"  ⚠️  Found {len(invalid_ids)} invalid entries: {invalid_ids}")
                cleaned_count += len(invalid_ids)
        
        # Clean up by keeping only valid template IDs
        cleaned = {}
        for category, template_ids in template_store._templates_by_category.items():
            valid_ids = [tid for tid in template_ids if tid in template_store._templates]
            cleaned[category] = valid_ids
        
        # Update the category index
        template_store._templates_by_category = cleaned
        
        logger.info(f"After cleanup:")
        logger.info(f"  Removed {cleaned_count} invalid entries")
        for category, template_ids in template_store._templates_by_category.items():
            logger.info(f"  Category '{category}' now has {len(template_ids)} entries: {template_ids}")
        
        # Get new stats
        stats = template_store.get_stats()
        
        logger.info("=" * 70)
        logger.info("✅ CLEANUP: Complete")
        logger.info("=" * 70)
        
        return {
            "success": True,
            "message": f"Category index cleaned up. Removed {cleaned_count} invalid entries.",
            "stats": stats
        }
    
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        logger.exception("Full traceback:")
        return {
            "success": False,
            "message": str(e)
        }

async def process_single_document_enhanced(request: EnhancedOCRRequest) -> EnhancedOCRResponse:
    """
    🚀 PROCESS SINGLE DOCUMENT - ENHANCED WITH TEMPLATE SUPPORT
    
    Flow:
    1. Download & process PDF
    2. Primary classification (YOLO)
    3. Template matching OR use provided template_id
    4. Region detection
    5. Prompt loading
    6. OCR extraction
    7. Field mapping & post-processing
    
    Args:
        request: EnhancedOCRRequest with file_url, user_id, optional template_id
    
    Returns:
        EnhancedOCRResponse with extracted data + template info
    """
    start_time = time.time()
    
    logger.info(f"""
        ╔═══════════════════════════════════════════════════════════╗
        ║ 🚀 ENHANCED OCR PROCESSING START                          ║
        ╠═══════════════════════════════════════════════════════════╣
        ║ File URL:     {request.file_url[:50]:<40}... ║
        ║ User ID:      {request.user_id or 'N/A':<42} ║
        ║ id:           {request.id or 'N/A':<42} ║
        ║ Template ID:  {request.template_id or 'Auto-Match':<42} ║
        ║ Mode:         {'Manual' if request.template_id else 'Auto-Match':<42} ║
        ╚═══════════════════════════════════════════════════════════╝
            """)
    
    try:
        # Download and process document
        from main import pdf_to_images, classify_documents
        from utils import convert_image_to_base64

        images = await pdf_to_images(request.file_url)
        images = await asyncio.to_thread(_cap_image_resolution, images)

        if not images:
            response = create_unregistered_response("Failed to process PDF", start_time)
            response.id = request.id
            return response

        logger.info(f"✓ Step 1/8: PDF downloaded and processed ({len(images)} pages)")

        # ========== NEW STEP: STICKER DETECTION (SK1/SK2/SK3) ==========
        sticker_template = None
        sticker_result = None
        receipt_result = None
        receipt_template = None

        if sticker_processor is not None:
            logger.info("→ Step 2/8: Running sticker detection...")
            try:
                sticker_result = await sticker_processor.process(images, lmdeploy_async_client)

                if sticker_result.sticker_case == "SK1_SK2" and sticker_result.is_sticker_valid:
                    # Valid SK1/SK2 case - use sticker template
                    logger.info(f"✅ SK1/SK2 sticker detected (FR-11 IoU: {sticker_result.fr11_iou_score:.2f})")
                    sticker_template = template_store.get_template("STICKER_SK1_SK2_V001")
                    if not sticker_template:
                        logger.warning("⚠️  STICKER_SK1_SK2_V001 template not found - falling back to normal flow")

                elif sticker_result.sticker_case == "SK3" and sticker_result.is_sticker_valid:
                    # Valid SK3 case - use sticker template
                    logger.info(f"✅ SK3 sticker detected (blank page validated)")
                    sticker_template = template_store.get_template("STICKER_SK3_V001")
                    if not sticker_template:
                        logger.warning("⚠️  STICKER_SK3_V001 template not found - falling back to normal flow")

                elif sticker_result.sticker_case == "EXCLUDED":
                    logger.warning(f"⚠️  Sticker EXCLUDED: {sticker_result.exclusion_reason}")
                    logger.info("Continuing with normal template flow...")

                else:
                    logger.info("No stickers detected - proceeding with normal flow")

            except Exception as e:
                logger.error(f"❌ Sticker detection error: {e}")
                logger.info("Continuing with normal template flow...")
        else:
            logger.info("→ Step 2/8: Sticker processor not available - skipping sticker detection")

        # ========== NEW STEP 2b: RECEIPT DETECTION (R1-R5) ==========
        # Only runs if no valid sticker was found
        if sticker_template is None and receipt_processor is not None:
            logger.info("→ Step 2b/8: Running receipt detection (no valid sticker found)...")
            try:
                receipt_result = await receipt_processor.process(images, lmdeploy_async_client)

                if receipt_result.receipt_detected and receipt_result.is_valid:
                    logger.info(f"✅ Receipt detected: {receipt_result.receipt_case}")
                    logger.info(f"  Receipt data: {receipt_result.extracted_data}")

                    # Map receipt case to template (like sticker pipeline)
                    template_id = RECEIPT_TEMPLATE_MAP.get(receipt_result.receipt_case)
                    if template_id:
                        receipt_template = template_store.get_template(template_id)
                        if receipt_template:
                            logger.info(f"✅ Using receipt template: {template_id}")
                        else:
                            logger.warning(f"⚠️  {template_id} template not found in memory - falling back to normal flow")
                    else:
                        logger.warning(f"⚠️  No template mapping for receipt case: {receipt_result.receipt_case}")

                elif receipt_result.receipt_detected and not receipt_result.is_valid:
                    logger.warning(f"⚠️  Receipt detected but invalid: {receipt_result.exclusion_reason}")
                    receipt_result = None
                else:
                    logger.info("No receipts detected - proceeding with normal flow")
                    receipt_result = None

            except Exception as e:
                logger.error(f"❌ Receipt detection error: {e}")
                logger.info("Continuing with normal template flow...")
                receipt_result = None
        elif sticker_template is None:
            logger.info("→ Step 2b/8: Receipt processor not available - skipping receipt detection")

        # ========== CLASSIFY ALL PAGES, FIND BOL PAGE ==========
        logger.info(f"→ Step 3/8: Classifying all {len(images)} page(s)...")
        
        # Convert all images to base64
        images_base64 = [convert_image_to_base64(img) for img in images]
        
        # Classify all pages
        classified_pages = classify_documents(images_base64)
        
        logger.info(f"  Classification results:")
        for doc_type, pages in classified_pages.items():
            logger.info(f"    - {doc_type}: {len(pages)} page(s)")
        
        # Find BOL page (or first classified page)
        bol_pages = classified_pages.get("BOL", [])
        if bol_pages:
            # Use first BOL page
            image_base64 = bol_pages[0]
            category = "BOL"
            primary_confidence = 0.95
            logger.info(f"  ✅ Found BOL page (out of {len(images)} total pages)")
        else:
            # No BOL found, try receipt or use first page
            receipt_pages = classified_pages.get("receipt", [])
            if receipt_pages:
                image_base64 = receipt_pages[0]
                category = "receipt"
                primary_confidence = 0.90
                logger.info(f"  ℹ️  No BOL found, using receipt page")
            else:
                # Use first page as fallback
                image_base64 = images_base64[0]
                category = "others"
                primary_confidence = 0.50
                logger.info(f"  ⚠️  No BOL/receipt found, using first page")
        
        # Convert base64 back to PIL Image for template matching
        from utils import base64topil
        image = base64topil(image_base64)
        
        logger.info(f"✓ Step 3/8: Primary classification: {category} (confidence: {primary_confidence:.2f})")

        # ========== Pre-detect regions ONCE (shared across all templates) ==========
        logger.info("→ Step 4/8: Pre-detecting regions (shared across templates)...")

        # Detect regions BEFORE template matching
        detected_regions_cache = region_detector.detect_regions(
            image=image,  # PIL Image from Step 3
            template=None,  # Will use category default
            category=category
        )

        logger.info(f"✓ Detected {len(detected_regions_cache)} regions (cached for all templates)")
        for region_name in detected_regions_cache.keys():
            logger.info(f"  - {region_name}")

        # ========== TEMPLATE HANDLING ==========
        # Priority: 1. Sticker template (if detected) 2. Receipt template (if detected) 3. Manual template_id 4. Auto-match
        if sticker_template is not None:
            # ========== STICKER MODE (SK1/SK2/SK3 detected) ==========
            logger.info(f"✓ Step 5/8: Using STICKER template: {sticker_template.get('template_id')} (Sticker mode)")
            template = sticker_template
            template_confidence = 0.95  # High confidence for sticker detection
            suggested_templates = []

            # Add sticker-specific data to be merged later
            if sticker_result and sticker_result.sticker_data:
                logger.info(f"  Sticker data: {sticker_result.sticker_data}")

        elif receipt_template is not None:
            # ========== RECEIPT MODE (R1-R5 detected) ==========
            logger.info(f"✓ Step 5/8: Using RECEIPT template: {receipt_template.get('template_id')} (Receipt mode - {receipt_result.receipt_case})")
            template = receipt_template
            template_confidence = 0.95  # High confidence for receipt detection
            suggested_templates = []

            if receipt_result and receipt_result.extracted_data:
                logger.info(f"  Receipt data: {receipt_result.extracted_data}")

        elif request.template_id:
            # ========== MANUAL MODE (Unregistered doc with provided template_id) ==========
            logger.info(f"✓ Step 5/8: Using provided template: {request.template_id} (Manual mode)")
            template = template_store.get_template(request.template_id)

            if not template:
                response = create_unregistered_response(
                    f"Template {request.template_id} not found in memory",
                    start_time
                )
                response.id = request.id
                return response

            template_confidence = 1.0  # Direct assignment
            suggested_templates = []  # No suggestions in manual mode

        else:
            # ========== AUTO-MATCH MODE ==========
            logger.info(f"→ Step 5/8: Matching templates for category: {category}")
            
            templates = template_store.get_templates_by_category(category)
            
            if not templates:
                logger.error(f"❌ No templates found for category: {category}")

                # If receipt detected, return receipt data even without BOL templates
                if receipt_result is not None and receipt_result.is_valid:
                    logger.info(f"📋 Receipt detected ({receipt_result.receipt_case}) - returning receipt data without BOL template")
                    response = create_receipt_response(receipt_result, start_time, category, primary_confidence)
                    response.id = request.id
                    return response

                logger.error(f"❌ Cannot process document without templates")

                # Return error response with clear message
                response = EnhancedOCRResponse(
                    B_L_Number="",
                    Stamp_Exists="",
                    Seal_Intact="null",
                    POD_Date="",
                    Signature_Exists="",
                    Issued_Qty=0,
                    Received_Qty=0,
                    Damage_Qty="0",
                    Short_Qty="0",
                    Over_Qty="0",
                    Refused_Qty="null",
                    Customer_Order_Num="null",
                    id=request.id,
                    template_id=None,
                    template_identifier=None,
                    confidence=0.0,
                    processing_time=int((time.time() - start_time) * 1000),
                    classification_details=ClassificationDetails(
                        primary_model_prediction=category,
                        primary_confidence=primary_confidence
                    ),
                    suggested_templates=[],
                    error=f"No templates available for category '{category}'. Please add templates first."  # Add error field
                )
                return response
            
            # Ensure image is PIL Image
            from PIL import Image
            if isinstance(image, np.ndarray):
                image_pil = Image.fromarray(image)
            else:
                image_pil = image
            
            match_start = time.time()

            # Match templates in parallel WITH CACHED REGIONS
            template, template_confidence, suggested_templates = await template_matcher.match_templates(
                image=image_pil,
                templates=templates,
                cached_regions=detected_regions_cache  # ← NEW PARAMETER
            )

            match_time = time.time() - match_start
            logger.info(f"✅ Template matching complete: {match_time:.2f}s")
            
            # Check confidence threshold (0.75)
            if template_confidence < 0.75:
                # If receipt detected, return receipt data even with low BOL confidence
                if receipt_result is not None and receipt_result.is_valid:
                    logger.info(f"📋 Receipt detected ({receipt_result.receipt_case}) + low BOL confidence ({template_confidence:.2f}) - returning receipt data")
                    response = create_receipt_response(receipt_result, start_time, category, primary_confidence)
                    response.id = request.id
                    return response

                # Confidence too low - return unregistered with suggestions
                logger.warning(f"❌ Best match confidence {template_confidence:.2f} < 0.75 threshold")
                logger.info(f"📋 Returning {len(suggested_templates)} suggested templates")

                response = create_unregistered_response(
                    f"No template matched with confidence >= 0.75 (best: {template_confidence:.2f})",
                    start_time,
                    category,
                    primary_confidence,
                    suggested_templates  # Include suggestions when confidence < 0.75
                )
                response.id = request.id
                return response
            
            # Confidence >= 0.75 - Safe to process
            logger.info(f"✓ Step 5/8: Template matched: {template.get('template_id')} (confidence: {template_confidence:.2f})")

        # ========== STEP 6: PROCESS WITH TEMPLATE ==========
        logger.info(f"→ Step 6/8: Processing with template: {template.get('template_id')}")
        result = await process_document_with_template(
            file_url=request.file_url,
            template=template,
            user_id=request.user_id,
            image=image
        )

        # ========== STEP 7: MERGE STICKER DATA (if applicable) ==========
        if sticker_result is not None and sticker_result.is_sticker_valid:
            logger.info("→ Step 7/8: Merging sticker data into result...")

            # For SK1/SK2: Override Received_Qty with B: value from sticker
            if sticker_result.sticker_case == "SK1_SK2" and sticker_result.sticker_data:
                boxes = sticker_result.sticker_data.get("boxes")
                if boxes is not None and boxes != "null":
                    result.Received_Qty = int(boxes) if isinstance(boxes, (int, str)) and str(boxes).isdigit() else 0
                    logger.info(f"  ✅ Received_Qty set from sticker B: value: {result.Received_Qty}")

                # Set sticker-specific fields
                result.Stamp_Exists = "no"  # Sticker cases have no stamp

                sticker_date = sticker_result.sticker_data.get("sticker_date")
                if sticker_date and sticker_date != "null":
                    result.POD_Date = sticker_date
                    logger.info(f"  ✅ POD_Date set from sticker: {result.POD_Date}")

            # For SK3: Use stamp's total_received as Received_Qty (do NOT force it to equal Issued_Qty)
            elif sticker_result.sticker_case == "SK3":

                # Merge SK3 sticker-specific fields (Equip_ID, Equip_Arrival_Date, Carrier)
                if sticker_result.sticker_data:
                    equip_id = sticker_result.sticker_data.get("equip_id")
                    equip_arrival = sticker_result.sticker_data.get("equip_arrival_date")
                    carrier = sticker_result.sticker_data.get("carrier")

                    if equip_id and equip_id != "null":
                        result.Equip_ID = equip_id
                    if equip_arrival and equip_arrival != "null":
                        result.Equip_Arrival_Date = equip_arrival
                    if carrier and carrier != "null":
                        result.Carrier = carrier

                    logger.info(f"  ✅ SK3 sticker fields merged: Equip_ID={result.Equip_ID}, Equip_Arrival_Date={result.Equip_Arrival_Date}, Carrier={result.Carrier}")

            logger.info("✓ Step 7/8: Sticker data merged")
        elif receipt_result is not None and receipt_result.is_valid:
            logger.info(f"→ Step 7/8: Merging receipt data into result ({receipt_result.receipt_case})...")

            receipt_data = receipt_result.extracted_data or {}

            # POD_Date from receipt pipeline (all cases)
            pod_date = receipt_data.get("POD_Date")
            if pod_date and pod_date != "null":
                result.POD_Date = pod_date
                logger.info(f"  ✅ POD_Date set from receipt ({receipt_result.receipt_case}): {result.POD_Date}")

            # Customer_Order_Num from receipt (R1, R2, R3 — NOT R4)
            customer_order_nums = receipt_data.get("Customer_Order_Num", [])
            if customer_order_nums and receipt_result.receipt_case != "R4":
                if isinstance(customer_order_nums, list):
                    result.Customer_Order_Num = ", ".join(str(v) for v in customer_order_nums)
                else:
                    result.Customer_Order_Num = str(customer_order_nums)
                logger.info(f"  ✅ Customer_Order_Num set from receipt: {result.Customer_Order_Num}")

            # appointment_id ONLY for R4
            if receipt_result.receipt_case == "R4":
                appt_id = receipt_data.get("appointment_id")
                if appt_id and appt_id != "null":
                    result.appointment_id = str(appt_id)
                    logger.info(f"  ✅ appointment_id set from R4: {result.appointment_id}")

            # delivery_confirmed_by for R5
            if receipt_result.receipt_case == "R5":
                dcb = receipt_data.get("delivery_confirmed_by")
                if dcb and dcb != "null":
                    result.delivery_confirmed_by = str(dcb)
                    logger.info(f"  ✅ delivery_confirmed_by set from R5: {result.delivery_confirmed_by}")

            # Set receipt/sticker flags
            result.receipt_exist = "yes"
            result.sticker_exist = "null"
            result.Signature_Exists = "null"

            logger.info("✓ Step 7/8: Receipt data merged")
        else:
            logger.info("→ Step 7/8: No sticker or receipt data to merge")

        # ========== STEP 8: ADD METADATA ==========
        processing_time = int((time.time() - start_time) * 1000)

        # Return template's id in template_id field
        result.id = request.id
        result.template_id = template.get("_id")  # MongoDB id
        result.template_identifier = template.get("template_id")  # Original template_id string
        result.confidence = template_confidence
        result.processing_time = processing_time
        result.classification_details = ClassificationDetails(
            primary_model_prediction=category,
            primary_confidence=primary_confidence
        )

        # Only include suggestions if confidence < 0.75
        if template_confidence < 0.75:
            result.suggested_templates = suggested_templates
        else:
            result.suggested_templates = []  # Empty when confident

        # Add sticker_exist / receipt_exist fields for safety
        if sticker_result is not None and sticker_result.is_sticker_valid:
            if hasattr(result, 'sticker_exist'):
                result.sticker_exist = "yes"
        elif receipt_result is not None and receipt_result.is_valid:
            if hasattr(result, 'receipt_exist'):
                result.receipt_exist = "yes"

        logger.info(f"✅ Processing complete - template: {template.get('template_id')}, confidence: {template_confidence:.2f}")

        clear_gpu_cache()
        return result

    except Exception as e:
        logger.error(f"Error in enhanced processing: {e}")
        logger.exception("Full traceback:")
        response = create_error_response(str(e))
        response.id = request.id  # Always include id even on error
        clear_gpu_cache()
        return response

async def process_document_with_template(
    file_url: str,
    template: Dict,
    user_id: Optional[str] = None,
    image: Optional[np.ndarray] = None,
    skip_matching: bool = False
) -> EnhancedOCRResponse:
    """
    Process document using specific template
    Used for both matched templates and template testing
    
    FIXED: Correctly uses batch_ocr(ocr_batches: List[models.OCRBatch])
    """
    logger.info("\n" + "=" * 70)
    logger.info("🔧 PROCESS_WITH_TEMPLATE: Starting template-driven processing")
    logger.info("=" * 70)
    logger.info(f"  File URL: {file_url}")
    logger.info(f"  Template ID: {template.get('template_id')}")
    logger.info(f"  Template Name: {template.get('template_name')}")
    logger.info(f"  Skip Matching: {skip_matching}")
    
    try:
        # ========== STEP 1: GET IMAGE ==========
        if image is None:
            logger.info("\n  📥 STEP 1: Downloading PDF...")
            from main import pdf_to_images
            images = await pdf_to_images(file_url)
            if not images:
                raise Exception("Failed to process PDF")
            image = images[0]
            logger.info(f"  ✅ PDF processed: {len(images)} page(s)")
        else:
            logger.info("\n  ✅ STEP 1: Using provided image")
        
        # ========== STEP 1.5: STRAIGHTEN IMAGE (ORIENTATION CORRECTION) ==========
        logger.info("\n  🔄 STEP 1.5: Correcting image orientation...")
        from main import straighten_img
        from utils import convert_image_to_base64
        from PIL import Image
        
        # Convert to base64 for straighten_img
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
            image_base64 = convert_image_to_base64(image_pil)
        else:
            image_base64 = convert_image_to_base64(image)
        
        # Straighten image
        straighten_result = await straighten_img(image_base64)
        image_base64 = straighten_result["img"]
        rotation_angle = straighten_result["angle"]
        
        logger.info(f"  ✅ Image orientation corrected (rotated: {rotation_angle}°)")
        
        # Convert back to PIL Image for region detection
        from utils import base64topil
        image = base64topil(image_base64)
        
        # Get category
        category = template.get("category", "BOL")
        logger.info(f"  📁 Category: {category}")
        
        # ========== STEP 2: REGION DETECTION ==========
        logger.info("\n  🔍 STEP 2: Region Detection")
        detected_regions = region_detector.detect_regions(image, template, category)
        
        if not detected_regions:
            logger.warning("  ⚠️  No regions detected, using full image")
            detected_regions = {"full_document": image}
        
        logger.info(f"  ✅ Detected {len(detected_regions)} region(s):")
        for region_name in detected_regions.keys():
            logger.info(f"    - {region_name}")
        
        # ========== STEP 3: LOAD PROMPTS ==========
        logger.info("\n  📝 STEP 3: Prompt Loading")
        prompts = prompt_loader.build_batch_prompts(template, detected_regions)
        logger.info(f"  ✅ Loaded {len(prompts)} prompt(s):")
        for region_name in prompts.keys():
            logger.info(f"    - {region_name}")
        
        # ========== STEP 4: BUILD OCR BATCHES ==========
        logger.info("\n  📦 STEP 4: Building OCR Batches")
        
        from utils import convert_image_to_base64
        from main import batch_ocr
        import models
        
        ocr_batches = []
        
        for region_name, region_image in detected_regions.items():
            if region_name not in prompts:
                logger.warning(f"    ⚠️  No prompt for region: {region_name}, skipping")
                continue
            
            # Extract image from dict if needed
            if isinstance(region_image, dict):
                actual_image = region_image["image"]
            else:
                actual_image = region_image
            
            # Convert region image to base64
            if isinstance(actual_image, np.ndarray):
                from PIL import Image
                region_pil = Image.fromarray(actual_image)
                region_base64 = convert_image_to_base64(region_pil)
            else:
                region_base64 = convert_image_to_base64(actual_image)
            
            # Get prompt for this region
            prompt = prompts[region_name]
            
            # Create OCRBatch object matching main.py signature
            ocr_batch = models.OCRBatch(
                page_type=category,  # e.g., "BOL"
                region_name=region_name,  # e.g., "stamp"
                prompt=prompt,
                image=region_base64,
                stamp_exist="unknown"  # Default value
            )
            
            ocr_batches.append(ocr_batch)
            logger.info(f"    ✓ Added batch for region: {region_name}")
        
        logger.info(f"  ✅ Built {len(ocr_batches)} OCR batch(es)")
        
        # ========== STEP 5: RUN OCR ==========
        logger.info("\n  🤖 STEP 5: Running OCR Extraction")
        
        if not ocr_batches:
            logger.error("  ❌ No OCR batches to process!")
            raise Exception("No valid regions/prompts for OCR processing")
        
        # Call batch_ocr with correct signature
        ocr_responses = await batch_ocr(ocr_batches)
        
        logger.info(f"  ✅ OCR complete: {len(ocr_responses)} response(s)")
        
        # ========== STEP 6: EXTRACT DATA FROM RESPONSES ==========
        logger.info("\n  📊 STEP 6: Extracting Data from OCR Responses")
        
        extracted_data = {}
        
        for ocr_response in ocr_responses:
            region_name = ocr_response.region_name
            ocr_result = ocr_response.ocr_response
            
            logger.info(f"    Processing response for: {region_name}")
            
            # Store with region prefix (e.g., "stamp.total_received")
            for field, value in ocr_result.items():
                key = f"{region_name}.{field}"
                extracted_data[key] = value
                logger.debug(f"      {key} = {value}")
            
            logger.info(f"      ✓ Extracted {len(ocr_result)} fields from {region_name}")
        
        logger.info(f"  ✅ Total extracted fields: {len(extracted_data)}")
        
        # Log extracted data for debugging
        logger.info("\n  🔍 EXTRACTED DATA SUMMARY:")
        for key, value in extracted_data.items():
            logger.info(f"    {key}: {value}")
        
        # ========== STEP 7: APPLY FIELD MAPPING ==========
        logger.info("\n  🗺️  STEP 7: Field Mapping")
        mapped_data = post_processor.apply_field_mapping(extracted_data, template)
        logger.info(f"  ✅ Mapped {len(mapped_data)} fields")
        
        # Log mapped data
        logger.info("\n  🔍 MAPPED DATA SUMMARY:")
        for key, value in mapped_data.items():
            logger.info(f"    {key}: {value}")
        
        # ========== STEP 8: APPLY POST-PROCESSING ==========
        logger.info("\n  🔧 STEP 8: Post-Processing Rules")
        final_data = post_processor.apply_rules(mapped_data, template)
        logger.info(f"  ✅ Post-processing complete")

        # ========== NOTATION: SEAL INTACT FOCUSED VALIDATION ==========
        # If all 4 qty fields have no meaningful value (empty, 0, null), the first
        # VLM pass may have hallucinated seal_intact. Run a second focused VLM call
        # on the same stamp image asking only about Y/N circle — if still uncertain, default to "empty".
        if template.get("template_id") == "TEMPLATE_NOTATION":
            notation_qty_fields = ["Received_Qty", "Damage_Qty", "Short_Qty", "Over_Qty"]
            empty_count = sum(
                1 for f in notation_qty_fields
                if final_data.get(f) in [0, None] or str(final_data.get(f, "")).lower() in ["empty", "", "0", "null"]
            )
            if empty_count == 4:
                logger.info("✅ NOTATION: all 4 qty fields are empty — running focused seal_intact validation call")
                stamp_image = next((b.image for b in ocr_batches if b.region_name == "stamp"), None)
                if stamp_image:
                    # ---- SEAL CROP: trim 30% from top and 20% from bottom to isolate the Seal Intact row ----
                    # The notation stamp layout places Seal Intact roughly in the middle third.
                    # Removing surrounding rows reduces visual noise that causes hallucination.
                    # To disable this crop and send the full stamp image, comment out the 5 lines below.
                    SEAL_CROP_ENABLED = True  # ← set False (or comment block) to deactivate
                    if SEAL_CROP_ENABLED:
                        from utils import base64topil
                        _pil = base64topil(stamp_image)
                        _w, _h = _pil.size
                        _pil_cropped = _pil.crop((0, int(_h * 0.40), _w, int(_h * 0.73)))
                        stamp_image = convert_image_to_base64(_pil_cropped)
                        logger.info(f"✅ NOTATION: seal crop applied — kept rows 30%–80% of stamp height ({int(_h*0.30)}px–{int(_h*0.80)}px of {_h}px)")
                    # ---- END SEAL CROP ----
                    seal_validation_prompt = (
                        "You are inspecting a cropped section of a delivery confirmation stamp.\n"
                        "This image shows only the middle portion of the stamp, focused around the \"Seal Intact (Y)/(N)\" row.\n\n"
                        "YOUR ONLY TASK: determine whether a person has hand-drawn a circle on EITHER the letter Y or the letter N.\n\n"
                        "CRITICAL — UNDERSTAND THE TEMPLATE LAYOUT:\n"
                        "The printed label on this form already reads: Seal Intact  (Y)  /  (N)\n"
                        "The parentheses ( ) around Y and N are PRE-PRINTED on every single document — marked or not.\n"
                        "They are part of the form's fixed text. They are NOT a hand-drawn mark.\n"
                        "If you see ( Y ) or ( N ) with nothing extra — that is an UNMARKED document. Return \"empty\".\n\n"
                        "WHAT COUNTS AS A REAL MARK:\n"
                        "A person adds a freehand pen or pencil circle that ENCLOSES the bare letter Y or N.\n"
                        "This extra circle is VISUALLY DISTINCT from the pre-printed parentheses — it is larger, rounder,\n"
                        "and clearly drawn on top of the printed text as additional ink.\n"
                        "A check mark, tick, or slash drawn through the letter Y or N also counts.\n\n"
                        "WHAT DOES NOT COUNT:\n"
                        "- The pre-printed ( ) parentheses — these are always present and never indicate a mark.\n"
                        "- Any ink, writing, or signatures below the Y/N letters (those are Receiver Name/Signature rows).\n"
                        "- Any ambiguous smudge, stray mark, or ink bleed that is not clearly on the letter itself.\n"
                        "- A partial touch, smear, or stroke that approaches the letter from below or from the side.\n"
                        "  The Receiver Signature row is directly below the Seal Intact row. Signature ink often bleeds\n"
                        "  or extends upward and may appear to graze the Y or N letter from underneath.\n"
                        "  This is NOT a circle. A genuine circle FULLY ENCLOSES the letter — it wraps all the way around.\n"
                        "  If the mark only touches the letter from one side or from below, it is signature bleed. Return \"empty\".\n\n"
                        "DEFAULT RULE — STRONG BIAS TOWARD \"empty\":\n"
                        "Only return \"yes\" or \"no\" if there is UNMISTAKABLY clear extra ink forming a COMPLETE circle on one letter.\n"
                        "A complete circle wraps around all sides of the letter — top, bottom, left, right.\n"
                        "If you are even slightly unsure, return \"empty\". False positives are worse than false negatives here.\n\n"
                        "Return JSON only — no other text:\n"
                        "{\"seal_intact\": \"yes\"}    ← extra hand-drawn circle clearly on the letter Y\n"
                        "{\"seal_intact\": \"no\"}     ← extra hand-drawn circle clearly on the letter N\n"
                        "{\"seal_intact\": \"empty\"} ← no extra circle visible, or any doubt at all"
                    )
                    seal_batch = models.OCRBatch(
                        page_type="BOL",
                        region_name="seal_validation",
                        prompt=seal_validation_prompt,
                        image=stamp_image
                    )
                    try:
                        seal_responses = await batch_ocr([seal_batch])
                        seal_result = seal_responses[0].ocr_response.get("seal_intact", "empty") if seal_responses else "empty"
                        seal_result = str(seal_result).lower().strip()
                        if seal_result in ["yes", "no"]:
                            logger.info(f"✅ NOTATION: focused seal call returned '{seal_result}' — using it")
                            final_data["Seal_Intact"] = seal_result
                        else:
                            logger.info("✅ NOTATION: focused seal call returned uncertain — defaulting Seal_Intact to 'empty'")
                            final_data["Seal_Intact"] = "empty"
                    except Exception as e:
                        logger.warning(f"⚠️ NOTATION: focused seal call failed ({e}) — defaulting Seal_Intact to 'empty'")
                        final_data["Seal_Intact"] = "empty"
                else:
                    logger.warning("⚠️ NOTATION: no stamp image in batches — defaulting Seal_Intact to 'empty'")
                    final_data["Seal_Intact"] = "empty"

        # ========== STEP 9: CREATE RESPONSE MODEL ==========
        logger.info("\n  📦 STEP 9: Creating Response Model")

        # Ensure all required fields exist with defaults
        required_fields = {
            "B_L_Number": "",
            "Stamp_Exists": "",
            "Seal_Intact": "null",
            "POD_Date": "",
            "Signature_Exists": "",
            "Issued_Qty": 0,           # ✅ CHANGED: 0 → None
            "Received_Qty": 0,         # ✅ CHANGED: 0 → None
            "Damage_Qty": "0",
            "Short_Qty": "0",
            "Over_Qty": "0",
            "Refused_Qty": "0",           # ✅ CHANGED: "null" → "0"
            "Customer_Order_Num": "null",
            "Notation_Exists": "null",
            "Status": "valid"
        }

        # Merge final_data with defaults
        for field, default_value in required_fields.items():
            if field not in final_data or final_data[field] is None:
                final_data[field] = default_value

        list_fields = ["Customer_Order_Num"]
        for field in list_fields:
            if field in final_data:
                value = final_data[field]
                
                # Convert list to comma-separated string
                if isinstance(value, list):
                    final_data[field] = ", ".join(str(v) for v in value)
                    logger.info(f"    Converted {field}: list → '{final_data[field]}'")

        # List of ALL quantity fields that must be strings in EnhancedOCRResponse
        # Convert integer-type fields (null/string → int)
        qty_int_fields = ["Received_Qty", "Issued_Qty"]
        for field in qty_int_fields:
            if field in final_data:
                value = final_data[field]
                
                # Handle "null" string → 0
                if isinstance(value, str) and value.lower() == "null":
                    final_data[field] = 0  # ✅ Use 0, not None
                    logger.debug(f"    Converted {field}: 'null' → 0")
                
                # Handle None → 0
                elif value is None:
                    final_data[field] = 0  # ✅ Use 0, not None
                    logger.debug(f"    Converted {field}: None → 0")
                
                # Already an integer - keep it
                elif isinstance(value, int):
                    logger.debug(f"    Kept {field}: {value} (int)")
                
                # Empty or blank value should pass through as "empty".
                elif isinstance(value, str) and value.lower() in ["empty", ""]:
                    final_data[field] = "empty"
                    logger.debug(f"    Kept {field}: blank → 'empty'")

                # Try to parse string to int
                elif isinstance(value, str):
                    try:
                        final_data[field] = int(value)
                        logger.debug(f"    Parsed {field}: '{value}' → {final_data[field]}")
                    except ValueError:
                        final_data[field] = 0  # ✅ Default to 0 on parse failure
                        logger.warning(f"    Failed to parse {field}: '{value}' → 0")

        # Convert string-type quantity fields (int → str)
        qty_string_fields = ["Damage_Qty", "Short_Qty", "Over_Qty", "Refused_Qty"]
        for field in qty_string_fields:
            if field in final_data:
                value = final_data[field]

                # Convert integer to string
                if isinstance(value, int):
                    final_data[field] = str(value)

                # Handle None/null values
                elif value is None or (isinstance(value, str) and value.lower() == "null"):
                    final_data[field] = "0"

        # For notation template, Refused_Qty must always be "null" — the field
        # does not exist on this template. Override regardless of what the loop set.
        # Also, Received_Qty cannot logically be 0 — force to "empty" if 0.
        if template.get("template_id") == "TEMPLATE_NOTATION":
            final_data["Refused_Qty"] = "null"
            logger.info("✅ NOTATION: Refused_Qty forced to 'null' (field not present on template)")
            if final_data.get("Received_Qty") == 0 or str(final_data.get("Received_Qty", "")).strip() == "0":
                final_data["Received_Qty"] = "empty"
                logger.info("✅ NOTATION: Received_Qty was 0 — forced to 'empty' (cannot logically be zero)")
            # Seal_Intact must be exactly "yes", "no", or "empty" — normalize any other value
            seal_val = str(final_data.get("Seal_Intact", "")).lower().strip()
            if seal_val not in ["yes", "no"]:
                final_data["Seal_Intact"] = "empty"
                logger.info("✅ NOTATION: Seal_Intact normalized to 'empty' (was not yes/no)")

        # Create Pydantic response model
        try:
            response = EnhancedOCRResponse(**final_data)
            logger.info("  ✅ Response model created successfully")
        except ValidationError as e:
            logger.error("  ❌ Pydantic validation failed:")
            for error in e.errors():
                logger.error(f"    Field: {error['loc'][0]}, Error: {error['msg']}, Input: {error['input']}")
            raise

        # Log final extracted values
        logger.info("\n  📋 FINAL EXTRACTED VALUES:")
        logger.info(f"    B/L Number: {final_data.get('B_L_Number')}")
        logger.info(f"    Received Qty: {final_data.get('Received_Qty')}")
        logger.info(f"    POD Date: {final_data.get('POD_Date')}")
        logger.info(f"    Stamp Exists: {final_data.get('Stamp_Exists')}")
        logger.info(f"    Signature Exists: {final_data.get('Signature_Exists')}")
        logger.info(f"    Damage Qty: {final_data.get('Damage_Qty')} (type: {type(final_data.get('Damage_Qty')).__name__})")
        logger.info(f"    Short Qty: {final_data.get('Short_Qty')} (type: {type(final_data.get('Short_Qty')).__name__})")
        logger.info(f"    Over Qty: {final_data.get('Over_Qty')} (type: {type(final_data.get('Over_Qty')).__name__})")
        
        logger.info("=" * 70)
        logger.info("✅ PROCESS_WITH_TEMPLATE: Processing complete")
        logger.info("=" * 70 + "\n")
        
        return response
    
    except Exception as e:
        logger.error("=" * 70)
        logger.error(f"❌ Error processing with template: {e}")
        logger.exception("Full traceback:")
        logger.error("=" * 70 + "\n")
        raise


def classify_document_category(image) -> Tuple[str, float]:
    """
    Classify document using existing YOLO model
    Returns (category, confidence)
    
    FIXED: Handles Dict[str, List[str]] response correctly
    
    Args:
        image: Either PIL Image, numpy array, or base64 string
    
    Returns:
        Tuple of (category: str, confidence: float)
    """
    logger.info("\n" + "=" * 70)
    logger.info("📄 DOCUMENT CLASSIFICATION")
    logger.info("=" * 70)
    
    try:
        from main import classify_documents
        from PIL import Image
        import numpy as np
        
        # Ensure image is base64 string
        if not isinstance(image, str):
            # Convert PIL/numpy to base64
            logger.info("  Converting image to base64...")
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
                image = convert_image_to_base64(pil_image)
            elif isinstance(image, Image.Image):
                image = convert_image_to_base64(image)
        
        logger.info("  Calling classify_documents()...")
        
        # Call classify_documents (returns Dict[str, List[str]])
        # Format: {"BOL": ["img1"], "receipt": [], "others": []}
        results = classify_documents([image])
        
        logger.info(f"  Classification results: {list(results.keys())}")
        logger.info(f"  Results detail:")
        for category, images in results.items():
            logger.info(f"    - {category}: {len(images)} image(s)")
        
        # Find which category has images
        for category, images in results.items():
            if images and len(images) > 0:
                # Found a match!
                logger.info(f"  ✅ Document classified as: {category}")
                logger.info("=" * 70 + "\n")
                return category, 0.95  # High confidence since YOLO matched
        
        # No category matched
        logger.warning("  ⚠️  No classification match, defaulting to 'others'")
        logger.info("=" * 70 + "\n")
        return "others", 0.0
    
    except Exception as e:
        logger.error("=" * 70)
        logger.error(f"❌ Error in classification: {e}")
        logger.exception("Full traceback:")
        logger.error("=" * 70 + "\n")
        return "others", 0.0


def create_unregistered_response(
    message: str,
    start_time: float,
    category: str = "Others",
    primary_confidence: float = 0.0,
    suggested_templates: List[Dict] = None  # Can be None or list
) -> EnhancedOCRResponse:
    """
    Create response for unregistered documents
    ✅ Only includes suggested_templates if provided
    """
    processing_time = int((time.time() - start_time) * 1000)
    
    return EnhancedOCRResponse(
        B_L_Number="",
        Stamp_Exists="",
        Seal_Intact="null",
        POD_Date="",
        Signature_Exists="",
        Issued_Qty=0,
        Received_Qty=0,
        Damage_Qty="0",
        Short_Qty="0",
        Over_Qty="0",
        Refused_Qty="null",
        Customer_Order_Num="null",
        id=None,  # Will be set by caller
        template_id=None,
        template_identifier=None,
        confidence=0.0,
        processing_time=processing_time,
        classification_details=ClassificationDetails(
            primary_model_prediction=category,
            primary_confidence=primary_confidence
        ),
        suggested_templates=suggested_templates if suggested_templates else []  # Only if provided
    )


def create_error_response(error_message: str) -> EnhancedOCRResponse:
    """Create error response"""
    return EnhancedOCRResponse(
        B_L_Number="",
        Stamp_Exists="",
        Seal_Intact="null",
        POD_Date="",
        Signature_Exists="",
        Issued_Qty=0,
        Received_Qty=0,
        Damage_Qty="0",
        Short_Qty="0",
        Over_Qty="0",
        Refused_Qty="null",
        Customer_Order_Num="null",
        id=None,  # Will be set by caller
        template_id=None,
        template_identifier=None,
        confidence=0.0,
        processing_time=0,
        classification_details=None,
        suggested_templates=[]  # Empty on error
    )


def create_receipt_response(
    receipt_result: ReceiptProcessingResult,
    start_time: float,
    category: str = "BOL",
    primary_confidence: float = 0.0
) -> EnhancedOCRResponse:
    """
    Create response with receipt data when BOL template matching is unavailable.
    Used as fallback when receipt is detected but no BOL templates match.
    """
    receipt_data = receipt_result.extracted_data or {}

    # POD_Date from receipt pipeline
    pod_date = receipt_data.get("POD_Date", "") or ""

    # Customer_Order_Num: from receipt (except R4 which uses appointment_id)
    customer_order_nums = receipt_data.get("Customer_Order_Num", [])
    if receipt_result.receipt_case == "R4":
        customer_order_num_str = "null"
    elif isinstance(customer_order_nums, list) and customer_order_nums:
        customer_order_num_str = ", ".join(str(v) for v in customer_order_nums)
    elif customer_order_nums:
        customer_order_num_str = str(customer_order_nums)
    else:
        customer_order_num_str = "null"

    # appointment_id only for R4
    appointment_id = "null"
    if receipt_result.receipt_case == "R4":
        appointment_id = str(receipt_data.get("appointment_id", "null") or "null")

    # delivery_confirmed_by only for R5
    delivery_confirmed_by = "null"
    if receipt_result.receipt_case == "R5":
        delivery_confirmed_by = str(receipt_data.get("delivery_confirmed_by", "null") or "null")

    processing_time = int((time.time() - start_time) * 1000)

    return EnhancedOCRResponse(
        B_L_Number="",
        Stamp_Exists="",
        Seal_Intact="null",
        POD_Date=pod_date,
        Signature_Exists="null",
        Issued_Qty=0,
        Received_Qty=0,
        Damage_Qty="0",
        Short_Qty="0",
        Over_Qty="0",
        Refused_Qty="null",
        Customer_Order_Num=customer_order_num_str,
        sticker_exist="null",
        receipt_exist="yes",
        appointment_id=appointment_id,
        delivery_confirmed_by=delivery_confirmed_by,
        id=None,  # Will be set by caller
        template_id=None,
        template_identifier=None,
        confidence=0.0,
        processing_time=processing_time,
        classification_details=ClassificationDetails(
            primary_model_prediction=category,
            primary_confidence=primary_confidence
        ),
        suggested_templates=[],
    )


# ========== Run Server ==========
if __name__ == "__main__":
    import uvicorn
    # To leverage multiple CPU cores for non-async blocking tasks (like some parts of image processing),
    # you can use multiple workers. However, for a purely async application, a single worker is often sufficient.
    # The `uvicorn.run` with `workers=1` is generally recommended for async FastAPI apps.
    uvicorn.run("app:app", host="0.0.0.0", port=8080, workers=1)
