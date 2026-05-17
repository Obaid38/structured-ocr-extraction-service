"""
Template Matching Algorithm - OPTIMIZED with Parallel Processing
✅ Async/await for concurrent template evaluation
✅ Reference image caching to avoid redundant downloads
✅ Persistent LMDeploy client
✅ ENHANCED: Detailed text pattern matching logs with ✅/🟡/❌ indicators
✅ Stamp text extracted ONCE per request and shared across all templates (eliminates N-1 redundant VLM calls)
"""

from typing import Dict, List, Tuple, Optional
import logging
import numpy as np
from PIL import Image
import imagehash
import httpx
import io
import cv2
import asyncio
from functools import lru_cache
from app_settings import get_env_or_config
logger = logging.getLogger("template_engine.matcher")


class TemplateMatcher:
    """Intelligent template matching with parallel processing"""
    
    def __init__(self, lmdeploy_client=None):
        logger.info("🎯 TemplateMatcher initialized with parallel processing")
        self.weights = {
            "text": 1.0,
            # "text": 0.35,
            # "visual": 0.30,
            # "layout": 0.20,
            # "image_hash": 0.15
        }
        logger.info(f"  Scoring weights: {self.weights}")
        
        # Persistent HTTP client for image downloads
        self.http_client = httpx.AsyncClient(timeout=10.0)
        
        # Cache for reference images (avoid re-downloading)
        self.image_cache: Dict[str, Image.Image] = {}
        
        # Cache for image hashes
        self.hash_cache: Dict[str, imagehash.ImageHash] = {}
        
        # LMDeploy client
        self.lmdeploy_client = lmdeploy_client
        self.lmdeploy_url = get_env_or_config(
            "LMDEPLOY_URL",
            ("services", "vlm_base_url"),
            "http://127.0.0.1:23333/v1",
        )
        
        if lmdeploy_client:
            logger.info("  ✓ LMDeploy client provided")
        else:
            logger.info(f"  ⚠️  No client provided, will create on-demand using {self.lmdeploy_url}")
    
    async def match_templates(
        self, 
        image: Image.Image, 
        templates: List[Dict],
        cached_regions: Optional[Dict] = None
    ) -> Tuple[Optional[Dict], float, List[Dict]]:
        """
        Match document image against templates IN PARALLEL
        
        Returns:
            (best_template, confidence, suggestions)
        """
        logger.info("\n" + "=" * 80)
        logger.info("🎯 TEMPLATE_MATCHING: Starting PARALLEL template matching")
        logger.info("=" * 80)
        
        if isinstance(image, np.ndarray):
            logger.info(f"  Input image type: numpy.ndarray (shape: {image.shape})")
            from PIL import Image
            image = Image.fromarray(image)
        else:
            logger.info(f"  Input image size: {image.size}, mode: {image.mode}")
        
        logger.info(f"  Templates to evaluate: {len(templates)}")
        
        # Store cached regions for all templates to use
        if cached_regions:
            self._cached_regions = cached_regions
            logger.info(f"  ✓ Using cached regions: {list(cached_regions.keys())}")

        if not templates:
            logger.warning("  ⚠️  No templates provided!")
            return None, 0.0, []

        # ✅ HYBRID: Parallel processing with early termination
        import time
        start_time = time.time()

        early_exit_threshold = 0.90
        logger.info(f"  Early exit threshold: {early_exit_threshold:.2f}")

        # ✅ STAMP TEXT CACHE: Extract stamp text from VLM exactly ONCE.
        # All N templates use the same stamp region image and the same generic OCR prompt,
        # so calling the VLM N times is pure waste.  We call it once here, share the result.
        # MongoDB prompts (data extraction after match) are completely separate — not affected.
        cached_stamp_text = await self._extract_stamp_text_once(image)
        if cached_stamp_text is not None:
            logger.info(
                f"  ✅ Stamp text pre-extracted ({len(cached_stamp_text)} chars) — "
                f"shared across all {len(templates)} templates (saves {len(templates) - 1} VLM calls)"
            )
        else:
            logger.warning(
                "  ⚠️  Stamp text pre-extraction failed — "
                "each template will fall back to its own VLM call"
            )

        # Strategy: Launch all tasks in parallel, but check results as they complete
        task_coroutines = [
            self._evaluate_single_template(image, template, idx, len(templates), cached_stamp_text=cached_stamp_text)
            for idx, template in enumerate(templates, 1)
        ]
        
        # Convert coroutines to Tasks so we can cancel them
        tasks = [asyncio.create_task(coro) for coro in task_coroutines]

        scored_templates = []
        best_score = 0.0
        early_exit_triggered = False

        # Use asyncio.as_completed to process results as they finish
        for completed_task in asyncio.as_completed(tasks):
            scored = await completed_task
            scored_templates.append(scored)
            
            # Track best score
            if scored["match_score"] > best_score:
                best_score = scored["match_score"]
            
            # ✅ EARLY EXIT: If we found a match >= 85%, we can stop waiting for others
            if scored["match_score"] >= early_exit_threshold and not early_exit_triggered:
                early_exit_triggered = True
                evaluated_count = len(scored_templates)
                skipped_count = len(templates) - evaluated_count
                
                logger.info(f"""
╔═══════════════════════════════════════════════════════════╗
║ ⚡ EARLY EXIT TRIGGERED (Parallel Mode)                   ║
╠═══════════════════════════════════════════════════════════╣
║ Template:   {scored['template_name']:<42} ║
║ Score:      {scored['match_score']*100:.2f}% (≥ {early_exit_threshold*100:.0f}%)     ║
║ Evaluated:  {evaluated_count}/{len(templates)} (completed first)       ║
║ Skipped:    ~{skipped_count} template(s) (may still finish)    ║
╚═══════════════════════════════════════════════════════════╝
                """)
                
                # Cancel remaining tasks to save resources
                for task in tasks:
                    if not task.done():
                        task.cancel()
                
                # Stop waiting for remaining tasks
                break

        # Collect any remaining completed tasks (should be minimal/none after cancel)
        for task in tasks:
            if task.done() and not task.cancelled():
                try:
                    result = task.result()
                    if result not in scored_templates:
                        scored_templates.append(result)
                except asyncio.CancelledError:
                    pass  # Expected for cancelled tasks

        elapsed = time.time() - start_time
        logger.info(f"\n⚡ PARALLEL MATCHING COMPLETE in {elapsed:.2f}s ({elapsed*1000:.0f}ms)")
        logger.info(f"  Templates evaluated: {len(scored_templates)}/{len(templates)}")
        if early_exit_triggered:
            logger.info(f"  ⚡ Early exit saved ~{(len(templates) - len(scored_templates))} evaluations")

        # Sort by score
        scored_templates.sort(key=lambda x: x["match_score"], reverse=True)
        
        best_template = scored_templates[0]
        threshold = best_template["template"].get("identification", {}).get("confidence_threshold", 0.75)
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 MATCHING RESULTS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"  🥇 Best Match: {best_template['template_name']}")
        logger.info(f"  📈 Score: {best_template['match_score']:.4f} ({best_template['match_score'] * 100:.2f}%)")
        logger.info(f"  📏 Threshold: {threshold:.4f} ({threshold * 100:.2f}%)")
        
        if best_template['match_score'] >= threshold:
            logger.info(f"  ✅ STATUS: TEMPLATE MATCHED!")
            logger.info(f"  ✨ Margin: +{(best_template['match_score'] - threshold) * 100:.2f}% above threshold")
        else:
            logger.info(f"  ❌ STATUS: NO MATCH (Below threshold)")
            logger.info(f"  ⚠️  Gap: -{(threshold - best_template['match_score']) * 100:.2f}% below threshold")
        
        logger.info(f"\n  📋 Template Rankings:")
        for i, scored in enumerate(scored_templates, 1):
            status_icon = "✅" if scored['passes_threshold'] else "❌"
            logger.info(f"    {i}. {status_icon} {scored['template_name']}: {scored['match_score']:.4f}")
        
        # Build suggestions
        suggestions = []
        for i, scored in enumerate(scored_templates[:5], 1):
            suggestions.append({
                "template_id": scored["template"].get("_id"),
                "template_name": scored["template_name"],
                "match_score": self._sanitize_score(round(scored["match_score"], 2)),
                "priority": i,
                "matched_patterns": self._get_matched_patterns(image, scored["template"]),
                "confidence_breakdown": {
                    k: self._sanitize_score(v) for k, v in scored["confidence_breakdown"].items()
                }
            })
        
        logger.info("=" * 80 + "\n")
        
        if best_template["match_score"] >= threshold:
            return best_template["template"], best_template["match_score"], suggestions
        else:
            return None, best_template["match_score"], suggestions
    
    async def _evaluate_single_template(
        self,
        image: Image.Image,
        template: Dict,
        idx: int,
        total: int,
        cached_stamp_text: Optional[str] = None,
    ) -> Dict:
        """
        Evaluate a single template (runs in parallel).

        cached_stamp_text: pre-extracted VLM text from the stamp region, shared across
                           all templates so we only call the VLM once per request.
                           When None, _calculate_text_score falls back to its own VLM call.

        Returns:
            Scored template dict
        """
        template_id = template.get("template_id", "Unknown")
        template_name = template.get("template_name", "Unknown")
        threshold = template.get("identification", {}).get("confidence_threshold", 0.75)

        logger.info(f"\n📋 EVALUATING TEMPLATE {idx}/{total}: {template_name}")

        scores = {}

        # Run all scoring methods in parallel
        score_tasks = [
            self._calculate_text_score(image, template, cached_stamp_text=cached_stamp_text),
            # self._calculate_visual_score(image, template),
            # self._calculate_layout_score(image, template),
            # self._calculate_image_hash_score(image, template)
        ]
        
        results = await asyncio.gather(*score_tasks)
        
        scores["text"] = results[0]
        # scores["visual"] = results[1]
        # scores["layout"] = results[2]
        # scores["image_hash"] = results[3]
        
        # Calculate weighted score
        overall_score = (
            scores["text"] * self.weights["text"] 
            # +
            # scores["visual"] * self.weights["visual"] +
            # scores["layout"] * self.weights["layout"] +
            # scores["image_hash"] * self.weights["image_hash"]
        )
        
        weighted_text = scores["text"] * self.weights["text"]
        # weighted_visual = scores["visual"] * self.weights["visual"]
        # weighted_layout = scores["layout"] * self.weights["layout"]
        # weighted_hash = scores["image_hash"] * self.weights["image_hash"]
        
        logger.info(f"  🎯 {template_name} SCORE: {overall_score:.4f}")
        logger.info(f"    Text: {scores['text']:.3f} → {weighted_text:.3f}")
        # logger.info(f"    Visual: {scores['visual']:.3f} → {weighted_visual:.3f}")
        # logger.info(f"    Layout: {scores['layout']:.3f} → {weighted_layout:.3f}")
        # logger.info(f"    Hash: {scores['image_hash']:.3f} → {weighted_hash:.3f}")
        
        return {
            "template": template,
            "template_id": template_id,
            "template_name": template_name,
            "match_score": overall_score,
            "score_breakdown": scores,
            "weighted_scores": {
                "text": weighted_text,
                # "visual": weighted_visual,
                # "layout": weighted_layout,
                # "image_hash": weighted_hash
            },
            "confidence_breakdown": {
                "text_similarity": scores["text"],
                # "visual_similarity": scores["visual"],
                # "layout_similarity": scores["layout"],
                # "image_hash_similarity": scores["image_hash"]
            },
            "passes_threshold": overall_score >= threshold
        }
    
    async def _download_image(self, url: str) -> Optional[Image.Image]:
        """
        Download and cache reference image
        
        Args:
            url: Image URL
        
        Returns:
            PIL Image or None
        """
        # Check cache first
        if url in self.image_cache:
            return self.image_cache[url]
        
        try:
            response = await self.http_client.get(url)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content)).convert('RGB')
            
            # Cache it
            self.image_cache[url] = img
            return img
            
        except Exception as e:
            logger.warning(f"Failed to download image {url[:50]}: {e}")
            return None
    
    def _normalize_image_for_histogram(self, img_array: np.ndarray) -> np.ndarray:
        """Normalize image for robust histogram comparison"""
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array
        
        height, width = img_bgr.shape[:2]
        if height != 1024 or width != 1024:
            img_bgr = cv2.resize(img_bgr, (1024, 1024), interpolation=cv2.INTER_AREA)
        
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_normalized = clahe.apply(l)
        
        lab_normalized = cv2.merge([l_normalized, a, b])
        normalized = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)
        normalized = cv2.GaussianBlur(normalized, (3, 3), 0)
        
        return normalized
    
    def _sanitize_score(self, value):
        """Convert numpy types to Python float"""
        import numpy as np
        if isinstance(value, (np.floating, np.integer)):
            return float(value)
        return value
    
    async def _extract_stamp_text_once(self, image: Image.Image) -> Optional[str]:
        """
        Call the VLM exactly once to extract text from the stamp region.

        This is the SINGLE source-of-truth call used by match_templates() before
        launching parallel template evaluations.  Every template shares the result,
        so we go from N VLM calls (one per template) to exactly 1.

        NOTE: This has nothing to do with the per-template MongoDB extraction prompts
        used after a template is matched.  Those run in process_with_template() and
        are completely unaffected.

        Returns extracted text (lowercased) on success, or None on failure so that
        individual templates can fall back to their own VLM call gracefully.
        """
        from openai import AsyncOpenAI
        import base64
        from io import BytesIO

        logger.info("  🔍 Pre-extracting stamp text (single shared VLM call)...")

        try:
            # --- resolve stamp image (same logic used inside _calculate_text_score) ---
            stamp_img = None
            if hasattr(self, '_cached_regions') and self._cached_regions:
                region_data = self._cached_regions.get("stamp", {})
                bbox = region_data.get("bbox")
                if bbox:
                    stamp_img = image.crop(bbox)
                    logger.info("    ✓ Using cached stamp region bbox")
            # Fallback: use full image (same fallback as _calculate_text_score)
            ocr_image = stamp_img if stamp_img else image

            # --- build client ---
            client = self.lmdeploy_client if self.lmdeploy_client else \
                AsyncOpenAI(api_key="EMPTY", base_url=self.lmdeploy_url)

            # --- encode image ---
            buffered = BytesIO()
            ocr_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            # --- VLM call (identical prompt that was previously sent once per template) ---
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                    },
                    {
                        "type": "text",
                        "text": (
                            "Extract ALL visible text from this receiving stamp area, "
                            "including both printed and handwritten text.\n\n"
                            "Look carefully for these field labels and their values:\n"
                            "- DATE (with value)\n"
                            "- SHORT (with value)\n"
                            "- OVER (with value)\n"
                            "- CARTONS (with value)\n"
                            "- DAMAGE (with value)\n"
                            "- PRINT NAME (with value)\n\n"
                            "List every text element you can see, even if handwritten or partially visible."
                        )
                    }
                ]
            }]

            response = await client.chat.completions.create(
                model="InternVL2-8B",
                messages=messages,
                temperature=0.1,
                max_tokens=1000,
            )

            extracted = response.choices[0].message.content.lower()
            logger.info(f"    ✓ Stamp text extracted ({len(extracted)} chars): {extracted[:120]}...")
            return extracted

        except Exception as e:
            logger.error(f"  ❌ _extract_stamp_text_once failed: {e}")
            logger.exception("  Full traceback:")
            return None

    async def _calculate_text_score(
        self,
        image: Image.Image,
        template: Dict,
        cached_stamp_text: Optional[str] = None,
    ) -> float:
        """
        Calculate text pattern matching score using VLM.

        cached_stamp_text: when provided (the normal path via match_templates), the VLM
                           call is skipped entirely — pattern matching runs directly on
                           the pre-extracted text.  When None (e.g. called standalone),
                           falls back to calling the VLM independently (original behaviour).
        """
        patterns = template.get("identification", {}).get("text_patterns", [])

        if not patterns:
            logger.warning("  ⚠️  No text patterns defined in template")
            return 0.5

        logger.info(f"""
  ╔═══════════════════════════════════════════════════════════╗
  ║ 📝 TEXT PATTERN MATCHING                                  ║
  ╠═══════════════════════════════════════════════════════════╣
  ║ Total Patterns: {len(patterns):<42} ║
  ╚═══════════════════════════════════════════════════════════╝
        """)

        try:
            # ------------------------------------------------------------------
            # FAST PATH: reuse stamp text extracted once by match_templates()
            # ------------------------------------------------------------------
            if cached_stamp_text is not None:
                extracted_text = cached_stamp_text
                logger.info("  ✓ Using pre-extracted stamp text (no VLM call needed)")

            # ------------------------------------------------------------------
            # FALLBACK PATH: called without a cache (e.g. standalone usage)
            # ------------------------------------------------------------------
            else:
                logger.info("  ℹ️  No cached stamp text — calling VLM directly")

                # Resolve stamp image
                stamp_img = None
                if hasattr(self, '_cached_regions') and self._cached_regions:
                    if "stamp" in self._cached_regions:
                        region_data = self._cached_regions["stamp"]
                        if "bbox" in region_data:
                            bbox = region_data["bbox"]
                            stamp_img = image.crop(bbox)
                            logger.info("  ✓ Using cached stamp region for text matching")
                else:
                    logger.debug("  ⚠️ No cached regions, detecting stamp region...")
                    yolo_config = template.get("region_config", {}).get("yolo_config")
                    if yolo_config:
                        try:
                            from .region_detector import RegionDetector
                            detector = RegionDetector()
                            image_np = np.array(image)
                            regions = detector.detect_regions(image_np, template, "BOL")
                            if isinstance(regions, dict) and "stamp" in regions:
                                if "bbox" in regions["stamp"]:
                                    bbox = regions["stamp"]["bbox"]
                                    stamp_img = image.crop(bbox)
                        except Exception as e:
                            logger.warning(f"  Region detection failed: {e}")

                from openai import AsyncOpenAI
                import base64
                from io import BytesIO

                ocr_image = stamp_img if stamp_img else image
                client = self.lmdeploy_client if self.lmdeploy_client else \
                    AsyncOpenAI(api_key="EMPTY", base_url=self.lmdeploy_url)

                buffered = BytesIO()
                ocr_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()

                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                        },
                        {
                            "type": "text",
                            "text": (
                                "Extract ALL visible text from this receiving stamp area, "
                                "including both printed and handwritten text.\n\n"
                                "Look carefully for these field labels and their values:\n"
                                "- DATE (with value)\n"
                                "- SHORT (with value)\n"
                                "- OVER (with value)\n"
                                "- CARTONS (with value)\n"
                                "- DAMAGE (with value)\n"
                                "- PRINT NAME (with value)\n\n"
                                "List every text element you can see, even if handwritten or partially visible."
                            )
                        }
                    ]
                }]

                response = await client.chat.completions.create(
                    model="InternVL2-8B",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1000,
                )

                extracted_text = response.choices[0].message.content.lower()

            logger.info(f"  📄 Extracted Text (first 200 chars):")
            logger.info(f"     {extracted_text[:200]}...")
            logger.info("")
            logger.info("  ╔═══════════════════════════════════════════════════════════╗")
            logger.info("  ║ 🔍 PATTERN MATCHING RESULTS                               ║")
            logger.info("  ╠═══════════════════════════════════════════════════════════╣")
            
            # Match patterns with detailed logging
            exact_matches = 0
            partial_matches = 0
            failed_patterns = []
            
            for idx, pattern in enumerate(patterns, 1):
                pattern_lower = pattern.lower()
                
                # Check for exact match
                if pattern_lower in extracted_text:
                    exact_matches += 1
                    logger.info(f"  ║ ✅ [{idx}/{len(patterns)}] EXACT:   '{pattern}'")
                else:
                    # Check for partial match (word-level)
                    pattern_words = [w for w in pattern_lower.split() if len(w) > 2]
                    matched_words = [w for w in pattern_words if w in extracted_text]
                    
                    if matched_words:
                        partial_matches += 1
                        match_pct = (len(matched_words) / len(pattern_words)) * 100 if pattern_words else 0
                        logger.info(f"  ║ 🟡 [{idx}/{len(patterns)}] PARTIAL: '{pattern}' ({match_pct:.0f}% words matched)")
                    else:
                        failed_patterns.append(pattern)
                        logger.info(f"  ║ ❌ [{idx}/{len(patterns)}] FAILED:  '{pattern}'")
            
            logger.info("  ╠═══════════════════════════════════════════════════════════╣")
            
            # Calculate score
            total_score = exact_matches + (partial_matches * 0.5)
            score = min(total_score / len(patterns), 1.0)
            
            # Summary
            logger.info(f"  ║ 📊 Summary:                                               ║")
            logger.info(f"  ║    ✅ Exact Matches:   {exact_matches}/{len(patterns):<38} ║")
            logger.info(f"  ║    🟡 Partial Matches: {partial_matches}/{len(patterns):<38} ║")
            logger.info(f"  ║    ❌ Failed:          {len(failed_patterns)}/{len(patterns):<38} ║")
            logger.info(f"  ║    🎯 Score:           {score:.4f} ({score*100:.2f}%){' '*(24-len(f'{score:.4f} ({score*100:.2f}%)'))} ║")
            logger.info("  ╚═══════════════════════════════════════════════════════════╝")
            
            return score
            
        except Exception as e:
            logger.error(f"  ❌ VLM text extraction error: {e}")
            logger.exception("  Full traceback:")
            return 0.0
    
    async def _calculate_visual_score(self, image: Image.Image, template: Dict) -> float:
        """Calculate visual feature similarity using color histograms (ASYNC)"""
        ref_images_config = template.get("identification", {}).get("reference_images", [])
        
        if not ref_images_config:
            return 0.5
        
        try:
            input_img = np.array(image.convert('RGB'))
            input_normalized = self._normalize_image_for_histogram(input_img)
            
            input_hist = cv2.calcHist([input_normalized], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
            input_hist = cv2.normalize(input_hist, input_hist).flatten()
            
            # ✅ PARALLEL: Download all reference images concurrently
            download_tasks = []
            for ref_config in ref_images_config:
                file_path = ref_config.get('file_path')
                if file_path.startswith('http'):
                    download_tasks.append(self._download_image(file_path))
            
            ref_images = await asyncio.gather(*download_tasks)
            
            # Calculate histograms
            scores = []
            for ref_img in ref_images:
                if ref_img is None:
                    continue
                
                ref_img_np = np.array(ref_img)
                ref_normalized = self._normalize_image_for_histogram(ref_img_np)
                
                ref_hist = cv2.calcHist([ref_normalized], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
                ref_hist = cv2.normalize(ref_hist, ref_hist).flatten()
                
                similarity = cv2.compareHist(input_hist, ref_hist, cv2.HISTCMP_CORREL)
                similarity_score = (similarity + 1) / 2
                scores.append(similarity_score)
            
            if scores:
                max_score = np.max(scores)
                return max_score
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error in visual comparison: {e}")
            return 0.5
    
    async def _calculate_layout_score(self, image: Image.Image, template: Dict) -> float:
        """Calculate layout similarity"""
        yolo_config = template.get("region_config", {}).get("yolo_config", {})
        classes = yolo_config.get("classes", [])
        
        if not classes:
            return 0.5
        
        return 0.85
    
    async def _calculate_image_hash_score(self, image: Image.Image, template: Dict) -> float:
        """Calculate perceptual image hash similarity (ASYNC)"""
        ref_images_config = template.get("identification", {}).get("reference_images", [])
        
        if not ref_images_config:
            return 0.5
        
        try:
            # Calculate hash for input (cache it)
            input_hash = imagehash.average_hash(image)
            
            # ✅ PARALLEL: Download all reference images
            download_tasks = []
            for ref_config in ref_images_config:
                file_path = ref_config.get('file_path')
                if file_path.startswith('http'):
                    download_tasks.append(self._download_image(file_path))
            
            ref_images = await asyncio.gather(*download_tasks)
            
            # Calculate hashes
            hash_scores = []
            for ref_img in ref_images:
                if ref_img is None:
                    continue
                
                ref_hash = imagehash.average_hash(ref_img)
                hamming_distance = input_hash - ref_hash
                similarity = 1 - (hamming_distance / 64.0)
                hash_scores.append(similarity)
            
            if hash_scores:
                max_score = np.max(hash_scores)
                return max_score
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating hash: {e}")
            return 0.5
    
    def _get_matched_patterns(self, image: Image.Image, template: Dict) -> List[str]:
        """Get list of text patterns that matched"""
        patterns = template.get("identification", {}).get("text_patterns", [])
        
        try:
            import pytesseract
            text = pytesseract.image_to_string(image).lower()
            return [p for p in patterns if p.lower() in text]
        except:
            return []
    
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()
