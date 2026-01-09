"""Screenshot processor that orchestrates the ingestion pipeline."""

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Callable
import os
import asyncio

from ..core.database import Database, Screenshot, compute_file_hash
from ..core.config import ConfigManager, ScanFolder
from ..services.ocr import OCRService
from ..services.vision import VisionService, VisionAPIError
from ..services.embedding import EmbeddingService
from ..services.sparse_embedding import SparseEmbeddingService
from ..services.vector_store import VectorStore


@dataclass
class ProcessingStats:
    """Statistics from a processing run."""
    total_files: int = 0
    new_indexed: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    
    @property
    def processed(self) -> int:
        return self.new_indexed + self.updated


@dataclass
class ProcessingProgress:
    """Progress update during processing."""
    current_file: str
    current_index: int
    total_files: int
    status: str  # 'processing', 'skipped', 'failed', 'cancelled'
    error_message: Optional[str] = None


class ScreenshotProcessor:
    """Orchestrates the ingestion pipeline for screenshots."""
    
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
    
    def __init__(
        self,
        config: ConfigManager,
        db: Database,
        vector_store: VectorStore,
        ocr: OCRService,
        vision: VisionService,
        embedding: EmbeddingService,
        sparse_embedding: Optional[SparseEmbeddingService] = None,
    ):
        self.config = config
        self.db = db
        self.vector_store = vector_store
        self.ocr = ocr
        self.vision = vision
        self.embedding = embedding
        self.sparse_embedding = sparse_embedding
    
    def discover_images(self, folders: Optional[list[ScanFolder]] = None) -> list[Path]:
        """Discover all image files in configured folders.
        
        Args:
            folders: Specific folders to scan (uses config if None)
            
        Returns:
            List of image file paths
        """
        folders = folders or self.config.config.scan_folders
        images = []
        
        for folder in folders:
            folder_path = Path(folder.path)
            if not folder_path.exists():
                continue
            
            if folder.include_subfolders:
                for ext in self.IMAGE_EXTENSIONS:
                    images.extend(folder_path.rglob(f"*{ext}"))
            else:
                for ext in self.IMAGE_EXTENSIONS:
                    images.extend(folder_path.glob(f"*{ext}"))
        
        return sorted(set(images))
    
    def check_changes(self, images: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
        """Check which images are new, changed, or unchanged.
        
        Returns:
            Tuple of (new_images, changed_images, unchanged_images)
        """
        existing = self.db.get_all_paths_and_hashes()
        
        new_images = []
        changed_images = []
        unchanged_images = []
        
        for image_path in images:
            path_str = str(image_path)
            
            if path_str not in existing:
                new_images.append(image_path)
            else:
                current_hash = compute_file_hash(image_path)
                if current_hash != existing[path_str]:
                    changed_images.append(image_path)
                else:
                    unchanged_images.append(image_path)
        
        return new_images, changed_images, unchanged_images
    
    def process_single(
        self,
        image_path: Path,
        force: bool = False,
    ) -> Optional[int]:
        """Process a single image through the pipeline.
        
        All-or-nothing: only saves to database if ALL processing steps succeed.
        If any step fails, nothing is stored so the file can be retried later.
        
        Args:
            image_path: Path to the image file
            force: Force reprocessing even if unchanged
            
        Returns:
            Screenshot ID if successful, None if skipped
            
        Raises:
            Exception if processing fails (so caller knows to skip this file)
        """
        path_str = str(image_path)
        current_hash = compute_file_hash(image_path)
        
        # Check if already indexed
        existing = self.db.get_by_path(path_str)
        if existing and not force:
            if existing.file_hash == current_hash:
                return None  # Unchanged, skip
        
        # Step 1: Extract OCR text (can be empty, that's OK)
        ocr_text = self.ocr.extract_text(image_path)
        
        # Step 2: Generate visual description (raises VisionAPIError on failure)
        visual_desc = self.vision.describe(image_path)
        
        # Step 3: Combine text for embedding
        combined_text = self._combine_for_embedding(ocr_text, visual_desc)
        
        # Step 4: Generate embedding - REQUIRED for storage
        if not combined_text:
            raise ValueError(f"No text content to embed for {image_path}")
        
        embedding_vector = self.embedding.embed(combined_text)
        if embedding_vector is None:
            raise RuntimeError(f"Failed to generate embedding for {image_path}")
        
        # All processing succeeded - now save everything
        
        # Extract metadata
        app_name, window_title = self._extract_metadata(image_path)
        captured_at = self._get_capture_time(image_path)
        
        # Create database record
        screenshot = Screenshot(
            id=existing.id if existing else None,
            file_path=path_str,
            file_hash=current_hash,
            app_name=app_name,
            window_title=window_title,
            captured_at=captured_at,
            indexed_at=datetime.now(),
            ocr_text=ocr_text,
            visual_description=visual_desc,
        )
        
        if existing:
            self.db.update(screenshot)
            screenshot_id = existing.id
        else:
            screenshot_id = self.db.insert(screenshot)
        
        # Add to vector store
        self.vector_store.add(
            id=screenshot_id,
            vector=embedding_vector,
            file_path=path_str,
            metadata={"app_name": app_name, "window_title": window_title},
        )
        
        # Add to sparse embedding index
        if self.sparse_embedding:
            self.sparse_embedding.add_document(screenshot_id, combined_text)
        
        return screenshot_id
    
    def process_all(
        self,
        folders: Optional[list[ScanFolder]] = None,
        force: bool = False,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> ProcessingStats:
        """Process all images in configured folders.
        
        Args:
            folders: Specific folders to process (uses config if None)
            force: Force reprocessing of all images
            progress_callback: Called with progress updates
            cancel_check: Callable that returns True if processing should be cancelled
            
        Returns:
            Processing statistics
        """
        stats = ProcessingStats()
        images = self.discover_images(folders)
        stats.total_files = len(images)
        
        if not force:
            new_images, changed_images, unchanged_images = self.check_changes(images)
            stats.skipped = len(unchanged_images)
            to_process = new_images + changed_images
        else:
            to_process = images
        
        for i, image_path in enumerate(to_process):
            # Check for cancellation
            if cancel_check and cancel_check():
                if progress_callback:
                    progress_callback(ProcessingProgress(
                        current_file="",
                        current_index=i,
                        total_files=len(to_process),
                        status="cancelled",
                    ))
                break
            
            if progress_callback:
                progress_callback(ProcessingProgress(
                    current_file=str(image_path),
                    current_index=i + 1,
                    total_files=len(to_process),
                    status="processing",
                ))
            
            try:
                existing = self.db.get_by_path(str(image_path))
                result = self.process_single(image_path, force=force)
                
                if result is not None:
                    if existing:
                        stats.updated += 1
                    else:
                        stats.new_indexed += 1
                else:
                    stats.skipped += 1
            except VisionAPIError as e:
                # Vision API error - skip this image so it can be reindexed later
                print(f"Vision API error, skipping {image_path}: {e}")
                stats.failed += 1
                
                if progress_callback:
                    progress_callback(ProcessingProgress(
                        current_file=str(image_path),
                        current_index=i + 1,
                        total_files=len(to_process),
                        status="api_error",
                        error_message=str(e),
                    ))
            except Exception as e:
                print(f"Failed to process {image_path}: {e}")
                stats.failed += 1
                
                if progress_callback:
                    progress_callback(ProcessingProgress(
                        current_file=str(image_path),
                        current_index=i + 1,
                        total_files=len(to_process),
                        status="failed",
                        error_message=str(e),
                    ))
        
        return stats
    
    def _combine_for_embedding(
        self,
        ocr_text: Optional[str],
        visual_desc: Optional[str],
    ) -> Optional[str]:
        """Combine OCR and visual description for embedding."""
        parts = []
        
        if visual_desc and visual_desc.strip():
            parts.append(f"Visual: {visual_desc.strip()}")
        
        if ocr_text and ocr_text.strip():
            parts.append(f"Text content: {ocr_text.strip()}")
        
        combined = "\n\n".join(parts) if parts else None
        
        # Final validation - ensure we have meaningful content
        if combined and len(combined.strip()) < 5:
            return None
        
        return combined
    
    def _extract_metadata(self, image_path: Path) -> tuple[Optional[str], Optional[str]]:
        """Extract app name and window title from image path/metadata.
        
        For now, this is a simple implementation based on folder structure.
        Could be enhanced to read EXIF data or use OS-specific APIs.
        """
        # Simple heuristic: use parent folder as "app name"
        app_name = image_path.parent.name if image_path.parent.name else None
        window_title = None
        return app_name, window_title
    
    def _get_capture_time(self, image_path: Path) -> Optional[datetime]:
        """Get the capture time of an image."""
        try:
            # Use file modification time as a fallback
            mtime = os.path.getmtime(image_path)
            return datetime.fromtimestamp(mtime)
        except Exception:
            return None
    
    # =========================================================================
    # Async Processing Methods (for parallel image processing)
    # =========================================================================
    
    async def process_single_async(
        self,
        image_path: Path,
        force: bool = False,
    ) -> Optional[int]:
        """Async version of process_single with parallel OCR+Vision.
        
        Runs OCR (CPU) and Vision (GPU) concurrently for each image,
        then generates embedding and saves to database.
        
        Args:
            image_path: Path to the image file
            force: Force reprocessing even if unchanged
            
        Returns:
            Screenshot ID if successful, None if skipped
            
        Raises:
            Exception if processing fails
        """
        path_str = str(image_path)
        current_hash = compute_file_hash(image_path)
        
        # Check if already indexed
        existing = self.db.get_by_path(path_str)
        if existing and not force:
            if existing.file_hash == current_hash:
                return None  # Unchanged, skip
        
        # Run OCR (CPU) and Vision (GPU) in parallel
        ocr_task = asyncio.to_thread(self.ocr.extract_text, image_path)
        vision_task = self.vision.describe_async(image_path)
        
        ocr_text, visual_desc = await asyncio.gather(ocr_task, vision_task)
        
        # Handle vision failure
        if visual_desc is None:
            raise VisionAPIError(f"Vision API returned None for {image_path}")
        
        # Combine text for embedding
        combined_text = self._combine_for_embedding(ocr_text, visual_desc)
        
        if not combined_text:
            raise ValueError(f"No text content to embed for {image_path}")
        
        # Generate embedding (async)
        embedding_vector = await self.embedding.embed_async(combined_text)
        if embedding_vector is None:
            raise RuntimeError(f"Failed to generate embedding for {image_path}")
        
        # All processing succeeded - now save everything (sync, fast)
        app_name, window_title = self._extract_metadata(image_path)
        captured_at = self._get_capture_time(image_path)
        
        screenshot = Screenshot(
            id=existing.id if existing else None,
            file_path=path_str,
            file_hash=current_hash,
            app_name=app_name,
            window_title=window_title,
            captured_at=captured_at,
            indexed_at=datetime.now(),
            ocr_text=ocr_text,
            visual_description=visual_desc,
        )
        
        if existing:
            self.db.update(screenshot)
            screenshot_id = existing.id
        else:
            screenshot_id = self.db.insert(screenshot)
        
        # Add to vector store
        self.vector_store.add(
            id=screenshot_id,
            vector=embedding_vector,
            file_path=path_str,
            metadata={"app_name": app_name, "window_title": window_title},
        )
        
        # Add to sparse embedding index
        if self.sparse_embedding:
            self.sparse_embedding.add_document(screenshot_id, combined_text)
        
        return screenshot_id
    
    async def process_all_async(
        self,
        folders: Optional[list[ScanFolder]] = None,
        force: bool = False,
        concurrency: int = 1,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> ProcessingStats:
        """Process all images with bounded parallelism.
        
        Uses asyncio.Semaphore to limit concurrent processing based on
        the concurrency parameter (typically from config.parallel_processing).
        
        Args:
            folders: Specific folders to process (uses config if None)
            force: Force reprocessing of all images
            concurrency: Maximum number of images to process concurrently
            progress_callback: Called with progress updates (may be called from multiple tasks)
            cancel_check: Callable that returns True if processing should be cancelled
            
        Returns:
            Processing statistics
        """
        stats = ProcessingStats()
        images = self.discover_images(folders)
        stats.total_files = len(images)
        
        if not force:
            new_images, changed_images, unchanged_images = self.check_changes(images)
            stats.skipped = len(unchanged_images)
            to_process = new_images + changed_images
        else:
            to_process = images
        
        if not to_process:
            return stats
        
        semaphore = asyncio.Semaphore(concurrency)
        completed_count = 0
        lock = asyncio.Lock()  # For thread-safe stats updates
        
        async def process_one(image_path: Path, index: int):
            nonlocal completed_count
            
            async with semaphore:
                # Check cancellation before starting
                if cancel_check and cancel_check():
                    return None
                
                if progress_callback:
                    progress_callback(ProcessingProgress(
                        current_file=str(image_path),
                        current_index=index + 1,
                        total_files=len(to_process),
                        status="processing",
                    ))
                
                try:
                    existing = self.db.get_by_path(str(image_path))
                    result = await self.process_single_async(image_path, force=force)
                    
                    async with lock:
                        if result is not None:
                            if existing:
                                stats.updated += 1
                            else:
                                stats.new_indexed += 1
                        else:
                            stats.skipped += 1
                        completed_count += 1
                    
                    if progress_callback:
                        progress_callback(ProcessingProgress(
                            current_file=str(image_path),
                            current_index=completed_count,
                            total_files=len(to_process),
                            status="done",
                        ))
                    
                    return result
                    
                except VisionAPIError as e:
                    async with lock:
                        stats.failed += 1
                        completed_count += 1
                    
                    print(f"Vision API error, skipping {image_path}: {e}")
                    if progress_callback:
                        progress_callback(ProcessingProgress(
                            current_file=str(image_path),
                            current_index=completed_count,
                            total_files=len(to_process),
                            status="api_error",
                            error_message=str(e),
                        ))
                    return None
                    
                except Exception as e:
                    async with lock:
                        stats.failed += 1
                        completed_count += 1
                    
                    print(f"Failed to process {image_path}: {e}")
                    if progress_callback:
                        progress_callback(ProcessingProgress(
                            current_file=str(image_path),
                            current_index=completed_count,
                            total_files=len(to_process),
                            status="failed",
                            error_message=str(e),
                        ))
                    return None
        
        # Create all tasks
        tasks = [process_one(img, i) for i, img in enumerate(to_process)]
        
        # Run with concurrency limit (semaphore handles it)
        await asyncio.gather(*tasks)
        
        return stats
