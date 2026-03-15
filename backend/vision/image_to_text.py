''#image->text(gemini vision)
#Take an image → return a factual product description

"""Image → Vision → Text
                 ↓
User Query → Text
                 ↓
        Embed → FAISS → Rerank → Context → LLM

"""
'''
#using the hugging face api 
import os
import base64
import requests
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}

def image_to_text(image_path: str) -> str:
    #converts the image to text using blip model
    print(f"[image_to_text] Processing: {image_path}")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    print(f"[image_to_text] Image size: {len(image_bytes)} bytes")    

    response = requests.post(
        API_URL,
        headers=HEADERS,
        data=image_bytes,
        timeout=60
    )

    print(f"[image_to_text] API Status: {response.status_code}")

    if response.status_code != 200:
        raise RuntimeError(
            f"HuggingFace Vision API failed: {response.status_code} - {response.text}"
        )

    result = response.json()
    print(f"[image_to_text] API Response: {result}")

    # BLIP returns a list with one dict containing generated text 
    if isinstance(result, list) and len(result) > 0:
        description = result[0].get("generated_text", "")
    else:
        description = result.get("generated_text", "")
    
    print(f"[image_to_text] Final description: {description}")

    return description'''

import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

print("[INFO] Loading BLIP model...")

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.to(device)

print(f"[INFO] BLIP model loaded on {device}")

def image_to_text(image_path: str) -> str:
    print(f"[image_to_text] Processing locally: {image_path}")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")

    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_length=50)

    caption = processor.decode(out[0], skip_special_tokens=True)

    print(f"[image_to_text] Caption: {caption}")

    return f"Product in image: {caption}"