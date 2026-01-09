"""Unified search interface with query routing and hybrid search."""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from ..core.database import Database, Screenshot
from ..services.vector_store import VectorStore, VectorSearchResult
from ..services.embedding import EmbeddingService
from ..services.sparse_embedding import SparseEmbeddingService
from ..services.reranker import RerankerService


@dataclass
class SearchResult:
    """A search result with screenshot data and relevance score."""
    screenshot: Screenshot
    score: float
    search_type: str  # 'fts', 'vector', 'hybrid', 'hybrid+rerank'
    
    @property
    def file_path(self) -> Path:
        return Path(self.screenshot.file_path)
    
    @property
    def id(self) -> int:
        return self.screenshot.id


class SearchEngine:
    """Unified search engine with FTS5, vector, and hybrid search."""
    
    def __init__(
        self,
        db: Database,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        sparse_embedding: Optional[SparseEmbeddingService] = None,
        reranker: Optional[RerankerService] = None,
        use_reranker: bool = False,
        hybrid_weight: float = 0.5,
    ):
        """Initialize the search engine.
        
        Args:
            db: Database instance
            vector_store: Qdrant vector store
            embedding_service: Dense embedding service
            sparse_embedding: Optional BM25 sparse embedding service
            reranker: Optional reranker service
            use_reranker: Whether to apply reranking
            hybrid_weight: Weight for hybrid search (0.0 = sparse only, 1.0 = dense only)
        """
        self.db = db
        self.vector_store = vector_store
        self.embedding = embedding_service
        self.sparse_embedding = sparse_embedding
        self.reranker = reranker
        self.use_reranker = use_reranker and reranker is not None
        self.hybrid_weight = max(0.0, min(1.0, hybrid_weight))  # Clamp to [0, 1]
    
    def search(
        self,
        query: str,
        limit: int = 20,
    ) -> list[SearchResult]:
        """Search for screenshots matching query.
        
        Uses query routing:
        - Quoted queries ("like this") → FTS5 exact match
        - Unquoted queries → Hybrid search (sparse + dense) if available, otherwise vector search
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of search results ordered by relevance
        """
        query = query.strip()
        
        if not query:
            return []
        
        # Query routing based on quotes
        if self._is_exact_query(query):
            # Strip quotes and do FTS search
            exact_query = query[1:-1]
            return self.fts_search(exact_query, limit)
        else:
            # Use hybrid search if sparse embedding is available
            if self.sparse_embedding and self.sparse_embedding.is_fitted:
                return self.hybrid_search(query, limit)
            else:
                return self.vector_search(query, limit)
    
    def fts_search(
        self,
        query: str,
        limit: int = 20,
    ) -> list[SearchResult]:
        """Perform FTS5 full-text search.
        
        Args:
            query: Search query (will be used in MATCH)
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        # Escape special FTS5 characters and format query
        fts_query = self._format_fts_query(query)
        
        screenshots = self.db.fts_search(fts_query, limit)
        
        # FTS5 results are already ranked by BM25
        return [
            SearchResult(
                screenshot=s,
                score=1.0 - (i * 0.01),  # Approximate score based on rank
                search_type="fts",
            )
            for i, s in enumerate(screenshots)
        ]
    
    def vector_search(
        self,
        query: str,
        limit: int = 20,
    ) -> list[SearchResult]:
        """Perform vector similarity search.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of search results ordered by similarity
        """
        # Generate query embedding
        query_vector = self.embedding.embed(query)
        if query_vector is None:
            return []
        
        # Search vector store
        # Request more results if reranking
        search_limit = limit * 3 if self.use_reranker else limit
        vector_results = self.vector_store.search(query_vector, search_limit)
        
        # Fetch full screenshot data for results
        results = []
        for vr in vector_results:
            screenshot = self.db.get_by_id(vr.id)
            if screenshot:
                results.append(SearchResult(
                    screenshot=screenshot,
                    score=vr.score,
                    search_type="vector",
                ))
        
        # Apply reranking if enabled
        if self.use_reranker and results:
            results = self._rerank_results(query, results, limit)
        else:
            results = results[:limit]
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        limit: int = 20,
    ) -> list[SearchResult]:
        """Perform hybrid search combining sparse BM25 and dense vector scores.
        
        The final score is computed as:
        final = (1 - hybrid_weight) * sparse_score + hybrid_weight * dense_score
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of search results ordered by combined score
        """
        # Generate query embedding for dense search
        query_vector = self.embedding.embed(query)
        if query_vector is None:
            # Fall back to sparse-only search
            return self._sparse_only_search(query, limit)
        
        # Get dense vector scores (request more for merging)
        search_limit = limit * 3
        vector_results = self.vector_store.search(query_vector, search_limit)
        
        # Get sparse BM25 scores (normalized)
        sparse_scores = {}
        if self.sparse_embedding and self.sparse_embedding.is_fitted:
            for doc_id, score in self.sparse_embedding.get_scores_normalized(query):
                sparse_scores[doc_id] = score
        
        # Build dense scores map
        dense_scores = {vr.id: vr.score for vr in vector_results}
        
        # Get all candidate doc IDs
        all_doc_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        
        # Combine scores
        combined_results = []
        for doc_id in all_doc_ids:
            dense_score = dense_scores.get(doc_id, 0.0)
            sparse_score = sparse_scores.get(doc_id, 0.0)
            
            # Weighted combination
            final_score = (1 - self.hybrid_weight) * sparse_score + self.hybrid_weight * dense_score
            
            combined_results.append((doc_id, final_score))
        
        # Sort by combined score descending
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        # Fetch screenshot data and build results
        results = []
        for doc_id, score in combined_results[:search_limit]:
            screenshot = self.db.get_by_id(doc_id)
            if screenshot:
                results.append(SearchResult(
                    screenshot=screenshot,
                    score=score,
                    search_type="hybrid",
                ))
        
        # Apply reranking if enabled
        if self.use_reranker and results:
            results = self._rerank_results(query, results, limit, search_type="hybrid+rerank")
        else:
            results = results[:limit]
        
        return results
    
    def _sparse_only_search(
        self,
        query: str,
        limit: int,
    ) -> list[SearchResult]:
        """Perform sparse-only search when dense embedding fails."""
        if not self.sparse_embedding or not self.sparse_embedding.is_fitted:
            return []
        
        results = []
        for doc_id, score in self.sparse_embedding.get_scores_normalized(query)[:limit]:
            screenshot = self.db.get_by_id(doc_id)
            if screenshot:
                results.append(SearchResult(
                    screenshot=screenshot,
                    score=score,
                    search_type="sparse",
                ))
        
        return results
    
    def _rerank_results(
        self,
        query: str,
        results: list[SearchResult],
        limit: int,
        search_type: str = "vector+rerank",
    ) -> list[SearchResult]:
        """Rerank results using cross-encoder."""
        if not self.reranker or not results:
            return results[:limit]
        
        # Prepare documents for reranking
        items = []
        for r in results:
            # Combine visual description and OCR text
            doc_text = ""
            if r.screenshot.visual_description:
                doc_text += r.screenshot.visual_description
            if r.screenshot.ocr_text:
                doc_text += "\n" + r.screenshot.ocr_text
            items.append((r, doc_text))
        
        # Rerank
        reranked = self.reranker.rerank(
            query,
            [doc for _, doc in items],
            top_k=limit,
        )
        
        # Rebuild results with new scores
        result_map = {i: r for i, (r, _) in enumerate(items)}
        reranked_results = []
        for idx, score in reranked:
            result = result_map[idx]
            reranked_results.append(SearchResult(
                screenshot=result.screenshot,
                score=score,
                search_type=search_type,
            ))
        
        return reranked_results
    
    def _is_exact_query(self, query: str) -> bool:
        """Check if query should use exact matching (quoted)."""
        return (
            len(query) >= 2 and
            query.startswith('"') and
            query.endswith('"')
        )
    
    def _format_fts_query(self, query: str) -> str:
        """Format a query for FTS5 MATCH.
        
        FTS5 uses specific syntax, so we need to handle:
        - Multiple words → implicit AND
        - Special characters → escape or remove
        """
        # Simple approach: wrap each word in quotes for exact phrase matching
        # and join with OR for flexibility
        words = query.split()
        if len(words) == 1:
            # Single word: simple match
            return words[0]
        else:
            # Multiple words: phrase match
            return f'"{query}"'

