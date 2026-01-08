"""Qdrant vector store wrapper for semantic search."""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)


@dataclass
class VectorSearchResult:
    """Result from vector similarity search."""
    id: int
    score: float
    file_path: Optional[str] = None


class VectorStore:
    """Qdrant vector store for semantic screenshot search."""
    
    COLLECTION_NAME = "screenshots"
    DEFAULT_DIMENSION = 1024  # mxbai-embed-large dimension
    
    def __init__(self, path: Path, dimension: int = DEFAULT_DIMENSION):
        """Initialize the vector store.
        
        Args:
            path: Directory path for persistent storage
            dimension: Embedding vector dimension
        """
        self.path = path
        self.dimension = dimension
        path.mkdir(parents=True, exist_ok=True)
        
        self.client = QdrantClient(path=str(path))
        self._ensure_collection()
    
    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.COLLECTION_NAME not in collection_names:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE,
                ),
            )
    
    def add(
        self,
        id: int,
        vector: list[float],
        file_path: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Add a vector to the store.
        
        Args:
            id: Unique ID (should match SQLite screenshot ID)
            vector: Embedding vector
            file_path: Path to the screenshot file
            metadata: Additional metadata to store
        """
        payload = metadata or {}
        if file_path:
            payload["file_path"] = file_path
        
        point = PointStruct(
            id=id,
            vector=vector,
            payload=payload,
        )
        
        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[point],
        )
    
    def add_batch(
        self,
        ids: list[int],
        vectors: list[list[float]],
        file_paths: Optional[list[str]] = None,
        metadata_list: Optional[list[dict]] = None,
    ) -> None:
        """Add multiple vectors to the store.
        
        Args:
            ids: List of unique IDs
            vectors: List of embedding vectors
            file_paths: Optional list of file paths
            metadata_list: Optional list of metadata dicts
        """
        points = []
        for i, (id_, vector) in enumerate(zip(ids, vectors)):
            payload = {}
            if metadata_list and i < len(metadata_list):
                payload = metadata_list[i] or {}
            if file_paths and i < len(file_paths):
                payload["file_path"] = file_paths[i]
            
            points.append(PointStruct(
                id=id_,
                vector=vector,
                payload=payload,
            ))
        
        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points,
        )
    
    def search(
        self,
        query_vector: list[float],
        limit: int = 20,
        score_threshold: Optional[float] = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score (optional)
            
        Returns:
            List of search results ordered by similarity
        """
        # Use query_points for newer qdrant-client versions
        results = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
        )
        
        return [
            VectorSearchResult(
                id=int(r.id),
                score=r.score,
                file_path=r.payload.get("file_path") if r.payload else None,
            )
            for r in results.points
        ]
    
    def delete(self, id: int) -> None:
        """Delete a vector by ID."""
        self.client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=[id],
        )
    
    def delete_batch(self, ids: list[int]) -> None:
        """Delete multiple vectors by ID."""
        if ids:
            self.client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=ids,
            )
    
    def get_count(self) -> int:
        """Get the number of vectors in the store."""
        info = self.client.get_collection(self.COLLECTION_NAME)
        return info.points_count
    
    def clear(self) -> None:
        """Delete and recreate the collection."""
        self.client.delete_collection(self.COLLECTION_NAME)
        self._ensure_collection()
    
    def close(self) -> None:
        """Close the client connection."""
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
