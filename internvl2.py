import json
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from fastapi import FastAPI
import requests
from io import BytesIO
from langsmith import traceable
import base64
import re
from dotenv import load_dotenv
from utils import dynamic_preprocess, build_transform
import models
from app_settings import get_env_or_config

load_dotenv()
app = FastAPI()
  
# Initialize model and tokenizer
MODEL_PATH = get_env_or_config(
    "INTERNVL_MODEL_ID",
    ("models", "internvl_model_id"),
    "OpenGVLab/InternVL2-8B",
)
model = (
    AutoModel.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    )
    .eval()
    .cuda()
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)

def load_image(image, input_size=448, max_num=12):
    """
    Load and preprocess image from various input formats
    Args:
        image: URL, file path, base64 string, or PIL Image
        input_size: Target size for image processing
        max_num: Maximum number of image chunks to process
    Returns:
        tensor: Processed image tensor ready for model inference
    """
    if isinstance(image, str):
        if image.startswith(("http://", "https://")):
            # Handle URL input
            response = requests.get(image, timeout=10)
            try:
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
            finally:
                response.close()
        elif image.startswith("data:image"):
            # Handle inline base64 image
            header, base64_data = image.split(",", 1)
            image = Image.open(BytesIO(base64.b64decode(base64_data))).convert("RGB")
        elif base64.b64encode(base64.b64decode(image)).decode('utf-8') == image:
            # Handle raw base64 string
            image = Image.open(BytesIO(base64.b64decode(image))).convert("RGB")
        else:
            # Handle local file path
            image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        raise ValueError("Unsupported image input type.")

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, 
        image_size=input_size, 
        use_thumbnail=True, 
        max_num=max_num
    )
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)

@app.post("/internvl-inference", response_model=models.InterVL2Response)
async def run_inference(request: models.InterVL2Request):
    """
    Process image inference request using InternVL2 model
    Args:
        request: Contains image source and prompt for analysis
    Returns:
        InterVL2Response: Processed model response in JSON format
    """
    # Prepare image input
    pixel_values = (
        load_image(request.image_url_or_path_or_base64, max_num=12)
        .to(torch.bfloat16)
        .cuda()
    )

    # Configure generation parameters
    generation_config = {
        "max_new_tokens": 1024,
        "do_sample": False,
        "temperature": 0.2,
        "top_p": 0.7,
        "repetition_penalty": 1.1
    }

    # Generate model response
    question = f"<image>\n{request.prompt}"
    response = model.chat(
        tokenizer, 
        pixel_values, 
        question, 
        generation_config
    )
    print(">>>>>>>>>>", response)

    # Extract JSON content from response
    json_pattern = r'```json\n(.*?)\n```'
    match = re.search(json_pattern, response, re.DOTALL)
    if match:
        json_content = match.group(1)
        print(json_content)

    return models.InterVL2Response(
        response=json_content.strip("```json").strip("```")
    )
