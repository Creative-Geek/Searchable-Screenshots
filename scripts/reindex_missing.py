"""Reindex screenshots that are missing embeddings."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import ConfigManager
from src.core.database import Database
from src.services.vector_store import VectorStore
from src.services.ocr import OCRService
from src.services.vision import VisionService
from src.services.embedding import EmbeddingService
from src.services.sparse_embedding import SparseEmbeddingService
from src.core.processor import ScreenshotProcessor


def find_missing_embeddings():
    """Find screenshots without embeddings."""
    config = ConfigManager()
    db = Database(config.db_path)
    vector_store = VectorStore(config.vector_store_path)
    
    print("Finding screenshots without embeddings...\n")
    
    # Get all screenshot IDs from database
    with db._connection() as conn:
        cursor = conn.execute("SELECT id, file_path FROM screenshots ORDER BY id")
        screenshots = cursor.fetchall()
    
    missing = []
    
    for screenshot_id, file_path in screenshots:
        try:
            result = vector_store.client.retrieve(
                collection_name=vector_store.COLLECTION_NAME,
                ids=[screenshot_id]
            )
            
            if not result or len(result) == 0:
                missing.append((screenshot_id, file_path))
        except Exception:
            missing.append((screenshot_id, file_path))
    
    vector_store.close()
    return missing


def reindex_missing(dry_run: bool = False):
    """Reindex screenshots that are missing embeddings.
    
    Args:
        dry_run: If True, only show what would be reindexed
    """
    missing = find_missing_embeddings()
    
    if not missing:
        print("✓ All screenshots have embeddings!")
        return
    
    print(f"Found {len(missing)} screenshots missing embeddings\n")
    
    if dry_run:
        print("Dry run - would reindex:")
        for screenshot_id, file_path in missing:
            print(f"  ID {screenshot_id}: {Path(file_path).name}")
        return
    
    # Initialize services
    config = ConfigManager()
    db = Database(config.db_path)
    vector_store = VectorStore(config.vector_store_path)
    
    api_config = config.config.api
    ocr = OCRService()
    vision = VisionService(api_config.ollama_url, api_config.vision_model)
    embedding = EmbeddingService(api_config.ollama_url, api_config.embed_model)
    
    # Load sparse embedding if exists
    sparse_embedding = SparseEmbeddingService()
    if config.sparse_index_path.exists():
        sparse_embedding.load(config.sparse_index_path)
    
    processor = ScreenshotProcessor(
        config,
        db,
        vector_store,
        ocr,
        vision,
        embedding,
        sparse_embedding=sparse_embedding,
    )
    
    # Reindex each missing one
    success = 0
    failed = 0
    
    for i, (screenshot_id, file_path) in enumerate(missing, 1):
        print(f"[{i}/{len(missing)}] Reindexing: {Path(file_path).name}... ", end="", flush=True)
        
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            print("✗ File not found")
            failed += 1
            continue
        
        try:
            result = processor.process_single(file_path_obj, force=True)
            if result is not None:
                print("✓")
                success += 1
            else:
                print("✗ Failed")
                failed += 1
        except Exception as e:
            print(f"✗ Error: {e}")
            failed += 1
    
    # Save sparse embedding index
    if sparse_embedding and sparse_embedding.is_fitted:
        sparse_embedding.save(config.sparse_index_path)
    
    print(f"\n{'='*60}")
    print(f"Reindex complete:")
    print(f"  Success: {success}")
    print(f"  Failed: {failed}")
    print(f"{'='*60}")
    
    vector_store.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Reindex screenshots missing embeddings")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be reindexed without doing it")
    args = parser.parse_args()
    
    reindex_missing(dry_run=args.dry_run)
