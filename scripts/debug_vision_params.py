"""Debug script to investigate Vision API discrepancies."""

import sys
import base64
import json
import httpx
from pathlib import Path
from PIL import Image
import io

# Configuration
OLLAMA_URL = "http://localhost:11434"
MODEL = "moondream:latest"
TIMEOUT = 120.0

def encode_image(image_path: Path, resize: tuple = None) -> str:
    """Encode image to base64, optionally resizing."""
    with Image.open(image_path) as img:
        if resize:
            print(f"  Resizing from {img.size} to {resize}...")
            img = img.resize(resize, Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed (e.g. for PNGs with alpha)
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            bg = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode != 'RGB':
            img = img.convert('RGB')
            
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

def test_payload(name: str, payload: dict):
    """Send a payload to Ollama and print the result."""
    print(f"\n--- Test Case: {name} ---")
    print(f"Payload keys: {list(payload.keys())}")
    if "options" in payload:
        print(f"Options: {payload['options']}")
    if "messages" in payload:
        print(f"System Prompt: {[m['content'] for m in payload['messages'] if m['role'] == 'system']}")
    
    try:
        response = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=TIMEOUT
        )
        response.raise_for_status()
        result = response.json()
        content = result.get("message", {}).get("content", "").strip()
        print(f"RESULT:\n{content[:200]}..." if len(content) > 200 else f"RESULT:\n{content}")
    except Exception as e:
        print(f"ERROR: {e}")

def main(image_path_str: str):
    image_path = Path(image_path_str)
    if not image_path.exists():
        print(f"Error: File not found: {image_path}")
        return

    print(f"Testing with image: {image_path}")
    
    # 1. Baseline (Current Implementation)
    # Reads raw bytes, no PIL processing
    with open(image_path, "rb") as f:
        raw_b64 = base64.b64encode(f.read()).decode("utf-8")
        
    baseline_payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are analyzing a NEW image. Ignore any previous images or context. Describe ONLY what you see in THIS specific image."
            },
            {
                "role": "user",
                "content": "Describe this image.", # Simplified prompt for debugging
                "images": [raw_b64]
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_ctx": 8096,
        },
    }
    test_payload("1. Baseline (Raw Bytes, System Prompt, Temp 0.3)", baseline_payload)

    # 2. No System Prompt
    no_sys_payload = baseline_payload.copy()
    no_sys_payload["messages"] = [
        {
            "role": "user",
            "content": "Describe this image.",
            "images": [raw_b64]
        }
    ]
    test_payload("2. No System Prompt", no_sys_payload)

    # 3. Open WebUI Simulation (Guessing)
    # Often they use default temp (0.8 or 0.7), no system prompt, and maybe resize?
    # Let's try just changing temp first.
    ow_sim_payload = no_sys_payload.copy()
    ow_sim_payload["options"] = {"temperature": 0.8} # Default is usually higher
    test_payload("3. Higher Temp (0.8), No System Prompt", ow_sim_payload)

    # 4. Resized Image (PIL processing)
    # Moondream often works better with smaller images or specific aspect ratios?
    # Actually, let's just try standardizing to JPEG and removing alpha.
    processed_b64 = encode_image(image_path)
    processed_payload = no_sys_payload.copy()
    processed_payload["messages"][0]["images"] = [processed_b64]
    test_payload("4. PIL Processed (JPEG, RGB)", processed_payload)

    # 5. Resized to max 1024
    resized_b64 = encode_image(image_path, resize=(512, 512)) # Aggressive resize
    resized_payload = no_sys_payload.copy()
    resized_payload["messages"][0]["images"] = [resized_b64]
    test_payload("5. Resized to 512x512", resized_payload)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_vision_params.py <image_path>")
    else:
        main(sys.argv[1])
