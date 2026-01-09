"""Vision service for visual description generation using Moondream via Ollama."""

import base64
import httpx
from pathlib import Path
from typing import Optional
from PIL import Image
import io


class VisionAPIError(Exception):
    """Raised when the Vision API encounters an error (network, server, etc)."""
    pass


class VisionService:
    """Generate visual descriptions of screenshots using Moondream."""
    
    DEFAULT_PROMPT = (
        """
### SYSTEM ROLE
You are a Computer Vision Indexer. Your job is to extract searchable data from images.

### ANALYSIS STRATEGY
1.  **Classify:** Determine if the image is a Screenshot, Meme, Photograph, or Graphic.
2.  **Transcribe:** If text is present, extract the key phrases exactly.
3.  **Describe:** Write a description based on the classification.

### OUTPUT FORMAT
Strictly follow this Markdown structure:

## 1. Category & Type
*   **Type:** [Select one: UI/Screenshot, Meme, Photograph, Movie Scene, Text Graphic]
*   **Quality:** [e.g., High Res, Blurry, Grainy, Cropped, Low Light]

## 2. Content Recognition
*   **Main Subject:** [e.g., Person, Terminal Window, Airplane, Cartoon Character]
*   **Visible Text:** [Transcribe important text exactly. If Arabic or foreign language, note it. If none, write "None".]
*   **Key Objects:** [List physical items, e.g., "ThinkPad Keyboard", "Yellow Glove", "Notifications Badge"]

## 3. Semantic Description
[Write 2-3 sentences. **Crucial:**]
*   *If Meme:* Explain the joke or metaphor (e.g., "Comparing a massive plane to a tiny bike").
*   *If Screenshot:* Describe the app context (e.g., "Docker terminal output showing disk usage").
*   *If Photo:* Describe the action and mood.

## 4. Search Keywords
[Generate 20 keywords. Mix visual tags, text terms, and abstract concepts.]
*   *Format:* tag1, tag2, tag3...

### INPUT IMAGE
[Image]
        """
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
    
    def _encode_image(self, image_path: Path) -> Optional[str]:
        """Read and encode image, ensuring it's in a format Ollama accepts."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB (handles RGBA, P, etc.)
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
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

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
        
        image_data = self._encode_image(image_path)
        if not image_data:
            return None
        
        prompt_text = prompt or self.DEFAULT_PROMPT
        
        try:
            response = self._client.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": prompt_text
                        },
                        {
                            "role": "user",
                            "content": "",
                            "images": [image_data]
                        }
                    ],
                    "stream": False,
                    "keep_alive": 0,
                    "options": {
                        "temperature": 0.3,
                        "num_ctx": 8096,
                    },
                },
            )
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "").strip()
        except httpx.HTTPError as e:
            # Raise to let caller know this was an API error (network, server, etc)
            raise VisionAPIError(f"Vision API error for {image_path}: {e}") from e
        except Exception as e:
            print(f"Vision service failed for {image_path}: {e}")
            return None
    
    async def describe_async(self, image_path: Path, prompt: Optional[str] = None) -> Optional[str]:
        """Async version of describe()."""
        if not image_path.exists():
            return None
        
        image_data = self._encode_image(image_path)
        if not image_data:
            return None
        
        prompt_text = prompt or self.DEFAULT_PROMPT
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": prompt_text
                            },
                            {
                                "role": "user",
                                "content": "",
                                "images": [image_data]
                            }
                        ],
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                        },
                    },
                )
                response.raise_for_status()
                result = response.json()
                return result.get("message", {}).get("content", "").strip()
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
