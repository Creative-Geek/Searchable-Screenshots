"""Clean invalid entries from all databases.

Removes screenshots that have:
1. Missing files on disk
2. Empty visual descriptions
3. Missing embeddings in vector store
4. Other data integrity issues
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import ConfigManager
from src.core.database import Database
from src.services.vector_store import VectorStore
from src.services.sparse_embedding import SparseEmbeddingService


def clean_invalid_entries(dry_run: bool = False):
    """Find and remove invalid entries.
    
    Args:
        dry_run: If True, only show what would be removed
    """
    config = ConfigManager()
    db = Database(config.db_path)
    vector_store = VectorStore(config.vector_store_path)
    
    # Load sparse embedding if exists
    sparse_embedding = SparseEmbeddingService()
    if config.sparse_index_path.exists():
        sparse_embedding.load(config.sparse_index_path)
    
    print("Scanning for invalid entries...\n")
    
    # Get all screenshots from database
    with db._connection() as conn:
        cursor = conn.execute("SELECT id, file_path, visual_description, ocr_text FROM screenshots ORDER BY id")
        screenshots = cursor.fetchall()
    
    invalid_ids = []
    reasons = {}  # id -> list of reasons
    
    # Batch check vector store existence for efficiency
    # We'll check in chunks of 100
    all_ids = [s[0] for s in screenshots]
    existing_vector_ids = set()
    
    chunk_size = 100
    for i in range(0, len(all_ids), chunk_size):
        chunk_ids = all_ids[i:i+chunk_size]
        try:
            results = vector_store.client.retrieve(
                collection_name=vector_store.COLLECTION_NAME,
                ids=chunk_ids
            )
            for point in results:
                existing_vector_ids.add(point.id)
        except Exception as e:
            print(f"Warning: Failed to check vector store chunk: {e}")
    
    for screenshot_id, file_path, visual_desc, ocr_text in screenshots:
        current_reasons = []
        
        # Check 1: File exists
        path_obj = Path(file_path)
        if not path_obj.exists():
            current_reasons.append("File not found on disk")
        
        # Check 2: Visual description
        if not visual_desc or not visual_desc.strip():
            current_reasons.append("Empty visual description")
        
        # Check 3: Vector store existence
        if screenshot_id not in existing_vector_ids:
            current_reasons.append("Missing embedding in vector store")
            
        # Check 4: Combined content check (similar to processor logic)
        # At least one of visual_desc or ocr_text must be present
        has_content = False
        if visual_desc and visual_desc.strip():
            has_content = True
        if ocr_text and ocr_text.strip():
            has_content = True
            
        if not has_content:
            current_reasons.append("No content (empty visual desc AND empty OCR)")
            
        if current_reasons:
            invalid_ids.append(screenshot_id)
            reasons[screenshot_id] = current_reasons
            print(f"Invalid ID {screenshot_id} ({path_obj.name}): {', '.join(current_reasons)}")
    
    if not invalid_ids:
        print("\n✓ No invalid entries found!")
        vector_store.close()
        return
    
    print(f"\nFound {len(invalid_ids)} invalid entries out of {len(screenshots)} total.")
    
    if dry_run:
        print("\nDry run - no changes made.")
        vector_store.close()
        return
    
    print("\nCleaning up...")
    
    # 1. Remove from SQLite
    with db._connection() as conn:
        # SQLite doesn't support list parameters directly in a clean way for IN clause with many items
        # So we'll do it in chunks or loop
        placeholders = ','.join('?' * len(invalid_ids))
        conn.execute(f"DELETE FROM screenshots WHERE id IN ({placeholders})", invalid_ids)
        conn.commit()
    print("✓ Removed from SQLite database")
    
    # 2. Remove from Vector Store
    try:
        vector_store.delete_batch(invalid_ids)
        print("✓ Removed from Vector Store")
    except Exception as e:
        print(f"✗ Failed to remove from Vector Store: {e}")
    
    # 3. Remove from Sparse Index
    if sparse_embedding and sparse_embedding.is_fitted:
        count = 0
        for doc_id in invalid_ids:
            sparse_embedding.remove_document(doc_id)
            count += 1
        sparse_embedding.save(config.sparse_index_path)
        print(f"✓ Removed {count} documents from Sparse Index")
    
    print(f"\nSuccessfully cleaned {len(invalid_ids)} entries.")
    vector_store.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean invalid entries from databases")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without doing it")
    args = parser.parse_args()
    
    clean_invalid_entries(dry_run=args.dry_run)
