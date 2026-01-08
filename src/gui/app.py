"""Main application window for Searchable Screenshots using PySide6."""

import sys
import os
from pathlib import Path
from typing import Optional
import threading

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QScrollArea, QFrame,
    QFileDialog, QMessageBox, QProgressBar, QSplitter,
)
from PySide6.QtCore import Qt, Signal, QObject, QSize
from PySide6.QtGui import QPixmap, QFont, QIcon

from ..core.config import ConfigManager
from ..core.database import Database
from ..core.search import SearchEngine, SearchResult
from ..core.processor import ScreenshotProcessor, ProcessingStats
from ..services.ocr import OCRService
from ..services.vision import VisionService
from ..services.embedding import EmbeddingService
from ..services.vector_store import VectorStore
from ..services.reranker import RerankerService


class WorkerSignals(QObject):
    """Signals for background worker threads."""
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(str)


class ResultCard(QFrame):
    """A card displaying a single search result."""
    
    clicked = Signal(object)
    
    def __init__(self, result: SearchResult, parent=None):
        super().__init__(parent)
        self.result = result
        self.setup_ui()
    
    def setup_ui(self):
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setStyleSheet("""
            ResultCard {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                margin: 4px;
            }
            ResultCard:hover {
                background-color: #f5f5f5;
                border-color: #2196F3;
            }
        """)
        self.setCursor(Qt.PointingHandCursor)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # Thumbnail
        thumbnail = QLabel()
        thumbnail.setFixedSize(80, 80)
        thumbnail.setStyleSheet("background-color: #e0e0e0; border-radius: 4px;")
        
        # Try to load the actual image as thumbnail
        file_path = Path(self.result.screenshot.file_path)
        if file_path.exists():
            pixmap = QPixmap(str(file_path))
            if not pixmap.isNull():
                scaled = pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                thumbnail.setPixmap(scaled)
                thumbnail.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(thumbnail)
        
        # Info section
        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)
        
        # Filename
        name_label = QLabel(file_path.name)
        name_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        name_label.setStyleSheet("color: #212121;")
        info_layout.addWidget(name_label)
        
        # Score and type
        score_label = QLabel(f"Score: {self.result.score:.3f} | {self.result.search_type}")
        score_label.setStyleSheet("color: #666666; font-size: 10px;")
        info_layout.addWidget(score_label)
        
        # App name
        app_label = QLabel(self.result.screenshot.app_name or "Unknown app")
        app_label.setStyleSheet("color: #888888; font-size: 10px;")
        info_layout.addWidget(app_label)
        
        # Preview of OCR text
        if self.result.screenshot.ocr_text:
            preview = self.result.screenshot.ocr_text[:100].replace('\n', ' ')
            if len(self.result.screenshot.ocr_text) > 100:
                preview += "..."
            ocr_label = QLabel(preview)
            ocr_label.setStyleSheet("color: #9e9e9e; font-size: 9px;")
            ocr_label.setWordWrap(True)
            info_layout.addWidget(ocr_label)
        
        info_layout.addStretch()
        layout.addLayout(info_layout, 1)
        
        # Open button
        open_btn = QPushButton("Open")
        open_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        open_btn.clicked.connect(self.on_open_clicked)
        layout.addWidget(open_btn)
    
    def on_open_clicked(self):
        self.clicked.emit(self.result)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.result)
        super().mousePressEvent(event)


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Searchable Screenshots")
        self.setMinimumSize(900, 700)
        
        # Initialize services
        self.config_manager = ConfigManager()
        self.db = Database(self.config_manager.db_path)
        self.vector_store = VectorStore(self.config_manager.vector_store_path)
        
        api_config = self.config_manager.config.api
        self.ocr = OCRService()
        self.vision = VisionService(api_config.ollama_url, api_config.vision_model)
        self.embedding = EmbeddingService(api_config.ollama_url, api_config.embed_model)
        
        self.reranker = None
        if self.config_manager.config.use_reranker:
            self.reranker = RerankerService()
        
        self.search_engine = SearchEngine(
            self.db,
            self.vector_store,
            self.embedding,
            self.reranker,
            self.config_manager.config.use_reranker,
        )
        
        self.processor = ScreenshotProcessor(
            self.config_manager,
            self.db,
            self.vector_store,
            self.ocr,
            self.vision,
            self.embedding,
        )
        
        self.results = []
        self.is_indexing = False
        
        self.setup_ui()
        self.apply_styles()
        self.update_status()
    
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        
        self.add_folder_btn = QPushButton("📁 Add Folder")
        self.add_folder_btn.clicked.connect(self.on_add_folder)
        toolbar.addWidget(self.add_folder_btn)
        
        self.index_btn = QPushButton("🔄 Index Now")
        self.index_btn.clicked.connect(self.on_index)
        toolbar.addWidget(self.index_btn)
        
        self.settings_btn = QPushButton("⚙️ Settings")
        self.settings_btn.clicked.connect(self.on_settings)
        toolbar.addWidget(self.settings_btn)
        
        toolbar.addStretch()
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #666666;")
        toolbar.addWidget(self.status_label)
        
        layout.addLayout(toolbar)
        
        # Search bar
        search_layout = QHBoxLayout()
        search_layout.setSpacing(8)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search screenshots... (use quotes for exact match)")
        self.search_input.returnPressed.connect(self.on_search)
        search_layout.addWidget(self.search_input, 1)
        
        self.search_btn = QPushButton("🔍 Search")
        self.search_btn.clicked.connect(self.on_search)
        search_layout.addWidget(self.search_btn)
        
        layout.addLayout(search_layout)
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Results area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #fafafa;
            }
        """)
        
        self.results_widget = QWidget()
        self.results_widget.setStyleSheet("background-color: #fafafa;")
        self.results_layout = QVBoxLayout(self.results_widget)
        self.results_layout.setContentsMargins(8, 8, 8, 8)
        self.results_layout.setSpacing(8)
        self.results_layout.addStretch()
        
        self.scroll_area.setWidget(self.results_widget)
        layout.addWidget(self.scroll_area, 1)
        
        # Footer
        footer = QHBoxLayout()
        self.indexed_label = QLabel("")
        self.indexed_label.setStyleSheet("color: #888888; font-size: 11px;")
        footer.addWidget(self.indexed_label)
        footer.addStretch()
        layout.addLayout(footer)
    
    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #fafafa;
            }
            QPushButton {
                background-color: #f5f5f5;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #eeeeee;
                border-color: #bdbdbd;
            }
            QLineEdit {
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                padding: 12px 16px;
                font-size: 14px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #2196F3;
            }
            QProgressBar {
                border: none;
                border-radius: 4px;
                background-color: #e0e0e0;
                height: 6px;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 4px;
            }
        """)
    
    def update_status(self):
        count = self.db.get_count()
        folders = len(self.config_manager.config.scan_folders)
        self.indexed_label.setText(f"📸 {count} screenshots indexed | 📁 {folders} folders configured")
    
    def on_add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select a folder to scan")
        if folder:
            self.config_manager.add_scan_folder(folder, include_subfolders=True)
            QMessageBox.information(self, "Folder Added", f"Added folder: {folder}")
            self.update_status()
    
    def on_index(self):
        if self.is_indexing:
            return
        
        self.is_indexing = True
        self.index_btn.setEnabled(False)
        self.index_btn.setText("⏳ Indexing...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.status_label.setText("Indexing...")
        
        # Create worker signals
        signals = WorkerSignals()
        signals.finished.connect(self.on_index_complete)
        signals.error.connect(self.on_index_error)
        
        def do_index():
            try:
                stats = self.processor.process_all()
                signals.finished.emit(stats)
            except Exception as e:
                signals.error.emit(str(e))
        
        thread = threading.Thread(target=do_index, daemon=True)
        thread.start()
    
    def on_index_complete(self, stats: ProcessingStats):
        self.is_indexing = False
        self.index_btn.setEnabled(True)
        self.index_btn.setText("🔄 Index Now")
        self.progress_bar.setVisible(False)
        self.status_label.setText(
            f"✅ Indexed {stats.new_indexed} new, {stats.updated} updated, {stats.skipped} skipped"
        )
        self.update_status()
    
    def on_index_error(self, error: str):
        self.is_indexing = False
        self.index_btn.setEnabled(True)
        self.index_btn.setText("🔄 Index Now")
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"❌ Error: {error}")
        QMessageBox.critical(self, "Indexing Error", f"Failed to index: {error}")
    
    def on_settings(self):
        config = self.config_manager.config
        msg = (
            f"Config location:\n{self.config_manager.config_path}\n\n"
            f"Ollama URL: {config.api.ollama_url}\n"
            f"Vision Model: {config.api.vision_model}\n"
            f"Embed Model: {config.api.embed_model}\n"
            f"Reranker: {'Enabled' if config.use_reranker else 'Disabled'}\n\n"
            f"Scan Folders:\n"
        )
        for folder in config.scan_folders:
            msg += f"  • {folder.path} (subfolders: {folder.include_subfolders})\n"
        
        QMessageBox.information(self, "Settings", msg)
    
    def on_search(self):
        query = self.search_input.text().strip()
        if not query:
            return
        
        self.status_label.setText(f"Searching: {query}...")
        QApplication.processEvents()
        
        try:
            self.results = self.search_engine.search(query, limit=20)
            self.display_results()
            self.status_label.setText(f"Found {len(self.results)} results")
        except Exception as e:
            self.status_label.setText(f"❌ Search error: {e}")
            QMessageBox.warning(self, "Search Error", str(e))
    
    def display_results(self):
        # Clear existing results
        while self.results_layout.count() > 1:  # Keep the stretch
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if not self.results:
            no_results = QLabel("No results found. Try a different search term.")
            no_results.setStyleSheet("color: #888888; padding: 40px; font-size: 14px;")
            no_results.setAlignment(Qt.AlignCenter)
            self.results_layout.insertWidget(0, no_results)
            return
        
        for result in self.results:
            card = ResultCard(result)
            card.clicked.connect(self.on_result_clicked)
            self.results_layout.insertWidget(self.results_layout.count() - 1, card)
    
    def on_result_clicked(self, result: SearchResult):
        file_path = result.screenshot.file_path
        if sys.platform == "win32":
            os.startfile(file_path)
        elif sys.platform == "darwin":
            os.system(f'open "{file_path}"')
        else:
            os.system(f'xdg-open "{file_path}"')


def run_app():
    """Entry point for the application."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Force light theme colors using QPalette
    from PySide6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(250, 250, 250))
    palette.setColor(QPalette.WindowText, QColor(33, 33, 33))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(33, 33, 33))
    palette.setColor(QPalette.Text, QColor(33, 33, 33))
    palette.setColor(QPalette.Button, QColor(245, 245, 245))
    palette.setColor(QPalette.ButtonText, QColor(33, 33, 33))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(33, 150, 243))
    palette.setColor(QPalette.Highlight, QColor(33, 150, 243))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()

