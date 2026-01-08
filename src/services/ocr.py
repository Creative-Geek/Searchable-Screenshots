"""OCR service for text extraction from screenshots."""

import sys
from pathlib import Path
from typing import Optional


class OCRService:
    """OS-specific OCR text extraction."""
    
    def __init__(self):
        self._ocr = None
        self._backend = "none"
        self._init_ocr()
    
    def _init_ocr(self) -> None:
        """Initialize the appropriate OCR backend for the current OS."""
        if sys.platform == "win32":
            try:
                import oneocr
                self._ocr = oneocr.OcrEngine()
                self._backend = "oneocr"
            except ImportError:
                raise RuntimeError("oneocr is required on Windows. Install with: uv add oneocr")
        elif sys.platform == "darwin":
            # macOS: VNRecognizeText - not implemented yet
            raise NotImplementedError("macOS OCR (VNRecognizeText) is not implemented yet")
        else:
            # Linux: TBD - need Arabic support
            raise NotImplementedError("Linux OCR is not implemented yet (need Arabic support)")
    
    def extract_text(self, image_path: Path) -> Optional[str]:
        """Extract text from an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text, or None if extraction failed
        """
        if not image_path.exists():
            return None
        
        if self._backend == "oneocr":
            return self._extract_with_oneocr(image_path)
        
        return None
    
    def _extract_with_oneocr(self, image_path: Path) -> Optional[str]:
        """Extract text using OneOCR (Windows)."""
        try:
            from PIL import Image
            
            img = Image.open(str(image_path))
            result = self._ocr.recognize_pil(img)
            
            # oneocr returns a dict with 'text' key
            if result and isinstance(result, dict):
                return result.get('text', None)
            
            return None
        except Exception as e:
            print(f"OCR extraction failed for {image_path}: {e}")
            return None
    
    @property
    def backend_name(self) -> str:
        """Get the name of the active OCR backend."""
        return self._backend

