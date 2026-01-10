"""Test script for hybrid BM25 + dense search functionality."""

import sys
sys.path.insert(0, ".")

from src.services.sparse_embedding import SparseEmbeddingService


def test_sparse_embedding_service():
    """Test basic BM25 functionality."""
    print("Testing SparseEmbeddingService...")
    
    # Create service
    svc = SparseEmbeddingService()
    assert not svc.is_fitted, "Should not be fitted initially"
    
    # Test corpus
    documents = [
        (1, "Discord chat application with dark mode interface showing messages"),
        (2, "Visual Studio Code editor Python file syntax highlighting"),
        (3, "Chrome browser Google search page white background"),
        (4, "File explorer showing folders and documents"),
        (5, "Spotify music player dark theme album art visible"),
    ]
    
    # Fit the model
    svc.fit(documents)
    assert svc.is_fitted, "Should be fitted after fit()"
    assert svc.document_count == 5, f"Expected 5 docs, got {svc.document_count}"
    
    # Test search
    results = svc.get_scores("discord chat")
    print(f"  Query 'discord chat': Top result ID={results[0][0]}, score={results[0][1]:.3f}")
    assert results[0][0] == 1, "Discord document should be top result"
    
    # Test normalized scores
    results_norm = svc.get_scores_normalized("code python editor")
    print(f"  Query 'code python editor': Top result ID={results_norm[0][0]}, score={results_norm[0][1]:.3f}")
    assert results_norm[0][0] == 2, "VSCode document should be top result"
    assert 0 <= results_norm[0][1] <= 1, "Normalized score should be in [0, 1]"
    
    # Test add_document
    svc.add_document(6, "Terminal command line interface black background green text")
    assert svc.document_count == 6, "Should have 6 docs after add"
    
    results = svc.get_scores("terminal command")
    print(f"  Query 'terminal command': Top result ID={results[0][0]}, score={results[0][1]:.3f}")
    assert results[0][0] == 6, "Newly added terminal doc should be top"
    
    # Test remove_document
    svc.remove_document(6)
    assert svc.document_count == 5, "Should have 5 docs after remove"
    
    # Test persistence
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_bm25.pkl"
        svc.save(save_path)
        assert save_path.exists(), "Index file should exist"
        
        # Load into new service
        svc2 = SparseEmbeddingService()
        loaded = svc2.load(save_path)
        assert loaded, "Load should return True"
        assert svc2.is_fitted, "Loaded service should be fitted"
        assert svc2.document_count == 5, "Loaded service should have 5 docs"
        
        # Verify search works on loaded model
        results2 = svc2.get_scores("browser google")
        assert results2[0][0] == 3, "Chrome doc should be top result in loaded model"
    
    print("✅ All SparseEmbeddingService tests passed!")


def test_hybrid_search_config():
    """Test hybrid search configuration."""
    print("\nTesting ConfigManager hybrid settings...")
    
    from src.core.config import AppConfig
    
    # Default weight
    config = AppConfig()
    assert config.hybrid_search_weight == 0.5, f"Default weight should be 0.5, got {config.hybrid_search_weight}"
    
    # Serialization
    data = config.to_dict()
    assert "hybrid_search_weight" in data, "hybrid_search_weight should be in dict"
    
    # Deserialization
    data["hybrid_search_weight"] = 0.7
    config2 = AppConfig.from_dict(data)
    assert config2.hybrid_search_weight == 0.7, f"Weight should be 0.7, got {config2.hybrid_search_weight}"
    
    print("✅ ConfigManager hybrid settings tests passed!")


if __name__ == "__main__":
    test_sparse_embedding_service()
    test_hybrid_search_config()
    print("\n🎉 All tests passed!")
