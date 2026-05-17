

# pixtral.py
import json
from PIL import Image
import base64
from io import BytesIO
from fastapi import FastAPI
import requests
from openai import AsyncOpenAI  # Already correctly imported AsyncOpenAI
import re
import models
from langsmith import traceable
import asyncio  # <--- ADDED THIS IMPORT
from datetime import datetime
from contextlib import asynccontextmanager
 
from app_settings import get_env_or_config

openai_api_key = "EMPTY"
openai_api_base = get_env_or_config(
    "LMDEPLOY_URL",
    ("services", "vlm_base_url"),
    "http://127.0.0.1:23333/v1",
)

# Global OpenAI client
_openai_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown"""
    global _openai_client
    # Startup: Create shared OpenAI client
    print("Starting up: Creating shared AsyncOpenAI client...")
    _openai_client = AsyncOpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    yield
    # Shutdown: Close OpenAI client
    print("Shutting down: Closing AsyncOpenAI client...")
    if _openai_client:
        await _openai_client.close()
    print("AsyncOpenAI client closed successfully")

app = FastAPI(lifespan=lifespan)


def image_to_base64(image: str) -> str:
    """
    Convert an image to a Base64-encoded string.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64-encoded string of the image.
    """
    # Simplified and corrected the image loading logic
    image_val = None
    if image.startswith("http://") or image.startswith("https://"):
        response = requests.get(image, timeout=10)
        try:
            response.raise_for_status()
            image_val = Image.open(BytesIO(response.content)).convert("RGB")
        finally:
            response.close()
    elif image.startswith("data:image"):
        header, base64_data = image.split(",", 1)
        image_val = Image.open(BytesIO(base64.b64decode(base64_data))).convert("RGB")
    elif base64.b64encode(base64.b64decode(image)).decode('utf-8') == image: # Check if it's a raw base64 string
        image_val = Image.open(BytesIO(base64.b64decode(image))).convert("RGB")
    else: # Assume it's a file path
        image_val = Image.open(image).convert("RGB")

    if image_val is None:
        raise ValueError(f"Unsupported image input type or invalid data for: {image[:50]}...") # Provide more context

    try:
        buffered = BytesIO()
        image_val.save(buffered, "PNG")
        base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return base64_string
    except Exception as e:
        raise ValueError(f"Error converting image to Base64: {e}")

@traceable(run_type="tool", name="llm-response-reasoning", dangerously_allow_filesystem=True)
@app.post("/pixtral-inference", response_model=models.InterVL2Response)
async def run_inference(request:models.InterVL2Request):
    global _openai_client
    client = _openai_client
    llms = await client.models.list()
    model = llms.data[0].id
    # image_base64 = image_to_base64(request.image_url_or_path_or_base64)
    chat_completion_from_base64 = await client.chat.completions.create(
        messages=[{
            "role":
            "user",
            "content": [
                {
                    "type": "text",
                    "text": request.prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{request.image_url_or_path_or_base64}"
                    },
                },
            ],
        }],
        model=model,
        temperature=0.8,
        top_p=0.7
    )

    response = chat_completion_from_base64.choices[0].message.content
    
    # return models.InterVL2Response(response=json_content.strip("```json").strip("```"))
    return models.InterVL2Response(response=response)


@traceable(run_type="tool", name="llm-batch-response-reasoning", dangerously_allow_filesystem=True)
@app.post("/batch-pixtral-inference", response_model=list[models.InterVL2Response])
async def run_batch_inference(requests: list[models.InterVL2Request]):
    print(f"starting from pixtral /batch-pixtral-inference : {datetime.now()}")

    print(f"new request come /batch-pixtral-inference : {datetime.now()}")

    global _openai_client
    client = _openai_client

    llms = await client.models.list()
    model_id = llms.data[0].id

    async def process_request(req: models.InterVL2Request):
        completion = await client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": req.prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{req.image_url_or_path_or_base64}"
                    }},
                ],
            }],
            model=model_id,
            temperature=0.8,
            top_p=0.7
        )
        print(f"gone request /batch-pixtral-inference : {datetime.now()}")
        return models.InterVL2Response(response=completion.choices[0].message.content)

    # Run all requests concurrently
    tasks = [process_request(req) for req in requests] # Correct: This creates a list of coroutine objects
    results = await asyncio.gather(*tasks) # Correct: asyncio.gather expects coroutine objects
    print(f"ending from pixtral /batch-pixtral-inference : {datetime.now()}")

    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3203)
