"""Verify that all indexed screenshots have embeddings in the vector store."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import ConfigManager
from src.core.database import Database
from src.services.vector_store import VectorStore


def verify_embeddings(fix: bool = False):
    """Check which screenshots are missing embeddings.
    
    Args:
        fix: If True, attempt to retrieve the vectors (for debugging)
    """
    config = ConfigManager()
    db = Database(config.db_path)
    vector_store = VectorStore(config.vector_store_path)
    
    print("Checking for screenshots without embeddings...\n")
    
    # Get all screenshots from database
    total_screenshots = db.get_count()
    print(f"Total screenshots in database: {total_screenshots}")
    
    # Get count from vector store
    vector_count = vector_store.get_count()
    print(f"Total vectors in vector store: {vector_count}")
    
    if total_screenshots == vector_count:
        print("\n✓ All screenshots have embeddings!")
        return
    
    print(f"\n⚠ Mismatch detected: {total_screenshots - vector_count} screenshots may be missing embeddings")
    print("\nChecking individual screenshots...\n")
    
    # Get all screenshot IDs from database
    with db._connection() as conn:
        cursor = conn.execute("SELECT id, file_path FROM screenshots ORDER BY id")
        screenshots = cursor.fetchall()
    
    missing = []
    
    for screenshot_id, file_path in screenshots:
        # Try to search for this specific ID in vector store
        # We'll do a dummy search and check if the ID appears
        try:
            # The vector store uses ID as the point ID
            # We can try to retrieve it directly (though Qdrant client doesn't have a direct "has" method)
            # Instead, we'll check by trying to search and seeing if results include this ID
            result = vector_store.client.retrieve(
                collection_name=vector_store.COLLECTION_NAME,
                ids=[screenshot_id]
            )
            
            if not result or len(result) == 0:
                missing.append((screenshot_id, file_path))
                print(f"✗ Missing: ID {screenshot_id} - {Path(file_path).name}")
        except Exception as e:
            # If retrieve fails, the point doesn't exist
            missing.append((screenshot_id, file_path))
            print(f"✗ Missing: ID {screenshot_id} - {Path(file_path).name}")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total screenshots: {total_screenshots}")
    print(f"  With embeddings: {total_screenshots - len(missing)}")
    print(f"  Missing embeddings: {len(missing)}")
    print(f"{'='*60}")
    
    if missing:
        print("\nTo fix missing embeddings, you can:")
        print("1. Use the 'Index Now' button in the GUI (will skip unchanged files)")
        print("2. Use the 'Reindex' button in the image info dialog for specific images")
        print("3. Run: uv run scripts/reindex_missing.py (if you create this script)")
        
        # Save list to file
        output_file = Path(__file__).parent / "missing_embeddings.txt"
        with open(output_file, "w") as f:
            f.write("Screenshots missing embeddings:\n\n")
            for screenshot_id, file_path in missing:
                f.write(f"ID {screenshot_id}: {file_path}\n")
        
        print(f"\nList saved to: {output_file}")
    
    vector_store.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify screenshot embeddings")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix missing embeddings")
    args = parser.parse_args()
    
    verify_embeddings(fix=args.fix)
