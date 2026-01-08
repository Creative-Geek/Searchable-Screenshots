"""Vision service for visual description generation using Moondream via Ollama."""

import base64
import httpx
from pathlib import Path
from typing import Optional


class VisionService:
    """Generate visual descriptions of screenshots using Moondream."""
    
    DEFAULT_PROMPT = (
        "Describe this screenshot in detail. Include: "
        "1) What application or website is shown, "
        "2) The main content and layout, "
        "3) Any notable UI elements, colors, or visual features, "
        "4) Any text that appears prominently (but I'll get detailed OCR separately)."
    )
    
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "moondream:latest",
        timeout: float = 120.0,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
    
    def describe(self, image_path: Path, prompt: Optional[str] = None) -> Optional[str]:
        """Generate a visual description of an image.
        
        Args:
            image_path: Path to the image file
            prompt: Optional custom prompt (uses DEFAULT_PROMPT if not provided)
            
        Returns:
            Visual description text, or None if generation failed
        """
        if not image_path.exists():
            return None
        
        # Read and encode image as base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        prompt_text = prompt or self.DEFAULT_PROMPT
        
        try:
            response = self._client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt_text,
                    "images": [image_data],
                    "stream": False,
                },
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except httpx.HTTPError as e:
            print(f"Vision API error for {image_path}: {e}")
            return None
        except Exception as e:
            print(f"Vision service failed for {image_path}: {e}")
            return None
    
    async def describe_async(self, image_path: Path, prompt: Optional[str] = None) -> Optional[str]:
        """Async version of describe()."""
        if not image_path.exists():
            return None
        
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        prompt_text = prompt or self.DEFAULT_PROMPT
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt_text,
                        "images": [image_data],
                        "stream": False,
                    },
                )
                response.raise_for_status()
                result = response.json()
                return result.get("response", "").strip()
            except httpx.HTTPError as e:
                print(f"Vision API error for {image_path}: {e}")
                return None
    
    def is_available(self) -> bool:
        """Check if the vision service is available."""
        try:
            response = self._client.get(f"{self.ollama_url}/api/tags")
            if response.status_code != 200:
                return False
            
            # Check if moondream model is available
            tags = response.json()
            models = [m.get("name", "") for m in tags.get("models", [])]
            return any(self.model in m or m in self.model for m in models)
        except Exception:
            return False
    
    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
