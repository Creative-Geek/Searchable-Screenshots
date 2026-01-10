"""Diagnostic script to test vision service and embedding pipeline."""

import sys
sys.path.insert(0, ".")

from pathlib import Path
from src.services.vision import VisionService
from src.services.embedding import EmbeddingService
from src.core.config import ConfigManager

def test_vision(image_path: str):
    """Test vision service with a specific image."""
    config = ConfigManager()
    api = config.config.api
    
    print(f"Testing vision service...")
    print(f"  Ollama URL: {api.ollama_url}")
    print(f"  Vision Model: {api.vision_model}")
    print(f"  Image: {image_path}")
    print("-" * 60)
    
    vision = VisionService(api.ollama_url, api.vision_model)
    
    # Test with current prompt
    print("\n[Current Prompt Result]:")
    result = vision.describe(Path(image_path))
    print(result)
    print("-" * 60)
    
    # Test with a simpler prompt
    simple_prompt = "Describe what you see in this screenshot. Be specific about the content, applications, and any text visible."
    print("\n[Simple Prompt Result]:")
    result2 = vision.describe(Path(image_path), prompt=simple_prompt)
    print(result2)
    print("-" * 60)
    
    # Test with a more detailed prompt
    detailed_prompt = """You are analyzing a screenshot for a search index. 
Describe the screenshot content in 2-3 sentences. Include:
- What application or website is shown
- Main visible content and purpose
- Any notable UI elements or text
Be descriptive but concise."""
    
    print("\n[Detailed Prompt Result]:")
    result3 = vision.describe(Path(image_path), prompt=detailed_prompt)
    print(result3)
    print("-" * 60)


def test_embedding(text: str):
    """Test embedding service with sample text."""
    config = ConfigManager()
    api = config.config.api
    
    print(f"\nTesting embedding service...")
    print(f"  Ollama URL: {api.ollama_url}")
    print(f"  Embed Model: {api.embed_model}")
    print(f"  Text length: {len(text)} chars")
    print("-" * 60)
    
    embedding = EmbeddingService(api.ollama_url, api.embed_model)
    
    result = embedding.embed(text)
    if result:
        print(f"  Embedding dimension: {len(result)}")
        print(f"  First 5 values: {result[:5]}")
    else:
        print("  ERROR: Embedding failed!")


if __name__ == "__main__":
    # Test with the provided image
    test_path = r"M:\SSTEST\Screenshot 2025-10-27 210223.png"
    
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    
    test_vision(test_path)
    
    # Test embedding with sample text
    test_embedding("Facebook post showing Sonic, Mario, Crash Bandicoot and Rayman gaming characters meme")
