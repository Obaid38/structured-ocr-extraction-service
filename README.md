# Structured OCR Extraction Service

Template driven OCR service that combines layout aware region detection with VLM based field extraction.

## Overview

This project processes multi page business documents where plain OCR is not enough. It first identifies the relevant page and document regions with detection models, then sends targeted crops and prompts to a vision language model for structured extraction.

The public version of this repository is intentionally generic. Client specific weights, case packs, sample documents, business rules, and deployment exports are excluded. The remaining code focuses on the reusable platform pieces:

1. Structural page and region inference
2. Template matching against configurable document definitions
3. Prompt driven extraction through an OpenAI compatible VLM endpoint
4. Field mapping into a stable API response

## Problem Statement

Traditional OCR often performs poorly on operational documents that contain dense tables, overlapping labels, stamps, handwritten notes, and repeated numeric fields. In those layouts, a full page OCR pass can mix unrelated zones together and return values from the wrong section.

This project addresses that problem by detecting the specific regions of interest before extraction. Instead of asking a model to interpret the entire page at once, it isolates the relevant blocks such as document number regions, confirmation areas, quantity sections, or signature zones, then sends those focused crops to the VLM.

The template driven layer is equally important for enterprise use. Large organizations often receive multiple document variants for the same workflow, with different vendors, layouts, or scan qualities. The system supports multiple templates and matches incoming pages against them so extraction can adapt to each document family without changing the core application code.

## Architecture

The service is built around a two stage pipeline.

1. Document routing
   The API converts the source PDF into images, selects a relevant page, and runs a classifier or template matcher to determine which extraction profile should be used.

2. Structural extraction
   The selected template defines how to locate regions. A detector can use YOLO boxes, fixed coordinates, or a hybrid fallback. Each detected crop is paired with a prompt and sent to the VLM endpoint. The raw values are then mapped into the final response schema.

The template engine is the main reusable abstraction. A template can define identification text patterns, reference images, region detection strategy, prompts, and field mappings. This keeps document specific behavior outside the core runtime.

## Tech Stack

| Area | Technology |
| --- | --- |
| Language | Python 3 |
| API framework | FastAPI |
| Model serving | LMDeploy, OpenAI compatible client |
| Vision models | Ultralytics YOLO, DocTR orientation model |
| Data layer | MongoDB via Motor |
| Imaging | Pillow, OpenCV, PyMuPDF |
| Config | Environment variables, YAML |

## Project Structure

```text
.
├── app.py
├── main.py
├── app_settings.py
├── mongo_db.py
├── config_manager.py
├── template_loader.py
├── prompts.py
├── config/
│   ├── prompts.yml
│   └── settings.example.yml
├── examples/
│   └── generic_template.example.json
├── template_engine/
│   ├── matcher.py
│   ├── memory_store.py
│   ├── post_processor.py
│   ├── prompt_loader.py
│   └── region_detector.py
└── requirements.txt
```

## Setup

```bash
git clone <your-repo-url>
cd structured-ocr-extraction-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Start the API:

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

If you are running LMDeploy locally:

```bash
python lmdeploy_app.py
```

## Configuration

Runtime configuration is split across two layers.

1. Environment variables
   `.env.example` lists the required database, service, and model path variables.

2. YAML config
   `config/settings.example.yml` defines default model paths, service URLs, frontend settings, and runtime defaults.

Prompt text is stored in `config/prompts.yml`. This lets you revise extraction instructions without editing Python files.

The most important variables are:

1. `MONGODB_URI`
2. `MONGODB_DB_NAME`
3. `LMDEPLOY_URL`
4. `DOC_CLASSIFIER_MODEL_PATH`
5. `REGION_MODEL_PATH`
6. `ORIENTATION_MODEL_PATH`
7. `APP_CONFIG_PATH`
8. `PROMPTS_CONFIG_PATH`

## Adapting To New Documents

To support a new document family:

1. Add a template document to MongoDB or start from `examples/generic_template.example.json`
2. Define identification patterns and reference images
3. Choose a region detection strategy
4. Write region level prompts in `config/prompts.yml` or in template stored prompts
5. Map extracted fields into the API response schema

This design works well for semi structured documents where the same fields appear in stable visual zones but raw OCR alone is unreliable.

## API Reference

The main endpoints are:

1. `POST /run-ocr`
   Batch OCR endpoint for structured extraction requests.

2. `POST /api/ocr/reprocess`
   Reprocess a document with a manually assigned template.

3. `POST /api/templates/sync`
   Add, update, or remove template definitions in memory.

4. `POST /api/templates/test`
   Run a template against a document for validation.

5. `GET /api/templates/stats`
   Return in memory template statistics.

## Security Notes

This repository does not include production credentials, model weights, sample business documents, or client templates. Keep `.env`, weights, exports, and uploaded files out of version control. For production use, point model paths and service endpoints to your own infrastructure.

## License

MIT
