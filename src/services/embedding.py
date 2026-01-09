"""Embedding service for text vectorization via Ollama."""

import httpx
from typing import Optional


class EmbeddingService:
    """Generate text embeddings using Ollama."""
    
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "mxbai-embed-large",
        timeout: float = 60.0,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        self._dimension: Optional[int] = None
    
    def embed(self, text: str, max_retries: int = 3) -> Optional[list[float]]:
        """Generate embedding vector for text.
        
        Args:
            text: Text to embed
            max_retries: Number of retry attempts for transient failures
            
        Returns:
            Embedding vector as list of floats, or None if failed
        """
        # Validate input - check for empty or whitespace-only text
        if not text:
            return None
        
        cleaned_text = text.strip()
        if not cleaned_text:
            return None
        
        # Retry loop for transient failures
        import time
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = self._client.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={
                        "model": self.model,
                        "prompt": cleaned_text,
                    },
                )
                response.raise_for_status()
                result = response.json()
                embedding = result.get("embedding")
                
                if embedding:
                    self._dimension = len(embedding)
                    return embedding
                else:
                    # Empty embedding returned - don't retry
                    return None
                    
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code >= 500:
                    # Server error - retry with backoff
                    if attempt < max_retries - 1:
                        time.sleep(0.5 * (attempt + 1))
                        continue
                # Client error or final attempt - don't retry
                print(f"Embedding API error: {e}")
                return None
            except httpx.HTTPError as e:
                last_error = e
                # Network error - retry with backoff
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                print(f"Embedding API error: {e}")
                return None
            except Exception as e:
                print(f"Embedding service failed: {e}")
                return None
        
        return None
    
    async def embed_async(self, text: str) -> Optional[list[float]]:
        """Async version of embed()."""
        if not text or not text.strip():
            return None
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={
                        "model": self.model,
                        "prompt": text,
                    },
                )
                response.raise_for_status()
                result = response.json()
                embedding = result.get("embedding")
                
                if embedding:
                    self._dimension = len(embedding)
                
                return embedding
            except httpx.HTTPError as e:
                print(f"Embedding API error: {e}")
                return None
    
    def embed_batch(self, texts: list[str]) -> list[Optional[list[float]]]:
        """Generate embeddings for multiple texts.
        
        Note: Ollama doesn't have a batch API, so this calls embed() sequentially.
        For better performance, use embed_batch_async().
        """
        return [self.embed(text) for text in texts]
    
    @property
    def dimension(self) -> Optional[int]:
        """Get the embedding dimension (available after first embedding)."""
        return self._dimension
    
    def is_available(self) -> bool:
        """Check if the embedding service is available."""
        try:
            response = self._client.get(f"{self.ollama_url}/api/tags")
            if response.status_code != 200:
                return False
            
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
