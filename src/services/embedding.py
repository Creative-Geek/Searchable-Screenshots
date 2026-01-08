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
    
    def embed(self, text: str) -> Optional[list[float]]:
        """Generate embedding vector for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats, or None if failed
        """
        if not text or not text.strip():
            return None
        
        try:
            response = self._client.post(
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
        except Exception as e:
            print(f"Embedding service failed: {e}")
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
