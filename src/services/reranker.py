"""Optional reranker service for improving search result quality."""

from typing import Optional


class RerankerService:
    """Rerank search results using cross-encoder model."""
    
    def __init__(self, model_name: str = "mixedbread-ai/mxbai-rerank-large-v1"):
        """Initialize the reranker.
        
        Args:
            model_name: HuggingFace model name for reranking
        """
        self.model_name = model_name
        self._model = None
    
    def _load_model(self):
        """Lazy load the model to avoid startup overhead."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
    
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: Optional[int] = None,
    ) -> list[tuple[int, float]]:
        """Rerank documents based on relevance to query.
        
        Args:
            query: Search query
            documents: List of document texts to rerank
            top_k: Return only top K results (all if None)
            
        Returns:
            List of (original_index, score) tuples, sorted by score descending
        """
        if not documents:
            return []
        
        self._load_model()
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Get scores
        scores = self._model.predict(pairs)
        
        # Create indexed scores and sort
        indexed_scores = [(i, float(score)) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            indexed_scores = indexed_scores[:top_k]
        
        return indexed_scores
    
    def rerank_with_ids(
        self,
        query: str,
        items: list[tuple[int, str]],
        top_k: Optional[int] = None,
    ) -> list[tuple[int, float]]:
        """Rerank items that have IDs.
        
        Args:
            query: Search query
            items: List of (id, text) tuples
            top_k: Return only top K results
            
        Returns:
            List of (id, score) tuples, sorted by score descending
        """
        if not items:
            return []
        
        ids, texts = zip(*items)
        reranked = self.rerank(query, list(texts), top_k)
        
        return [(ids[idx], score) for idx, score in reranked]
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None
