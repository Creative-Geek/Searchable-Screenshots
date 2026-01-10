
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import time
from pathlib import Path
from src.services.vision import VisionService

async def test_batch_vision(image_paths: list[Path], batch_size: int = 5):
    vision = VisionService()
    
    print(f"Processing {len(image_paths)} images with batch size {batch_size}...")
    
    start_time = time.time()
    
    # Split into batches
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        print(f"Starting batch {i//batch_size + 1} with {len(batch)} images...")
        
        batch_start = time.time()
        
        # Create tasks
        tasks = [vision.describe_async(path) for path in batch]
        
        # Run concurrently
        results = await asyncio.gather(*tasks)
        
        batch_duration = time.time() - batch_start
        print(f"Batch {i//batch_size + 1} finished in {batch_duration:.2f}s")
        
        # Print results briefly
        for path, result in zip(batch, results):
            desc_len = len(result) if result else 0
            print(f"  - {path.name}: {desc_len} chars")
            
    total_duration = time.time() - start_time
    print(f"\nTotal time: {total_duration:.2f}s")
    print(f"Average time per image: {total_duration / len(image_paths):.2f}s")

def main():
    # Find some images to test with
    # We'll look in the user's configured folders or just pick some from the current workspace if possible
    # For this test, I'll search for images in the current workspace
    
    workspace_root = Path("M:\SSTEST\Smaller-Test")
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    
    images = []
    for ext in image_extensions:
        images.extend(list(workspace_root.rglob(f"*{ext}")))
    
    # Filter out hidden folders or venv
    images = [p for p in images if ".venv" not in str(p) and ".git" not in str(p)]
    
    # Take up to 10 images for the test
    test_images = images[:10]
    
    if not test_images:
        print("No images found to test.")
        return
        
    print(f"Found {len(test_images)} images for testing.")
    
    # Run the async test
    asyncio.run(test_batch_vision(test_images, batch_size=5))

if __name__ == "__main__":
    main()
