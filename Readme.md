# Searchable Screenshots

A Python application that indexes and enables fast searching of your screenshots using OCR, vision models, and vector embeddings.

⚠️ Under active development. Use at your own risk.

## Features

- **Folder scanning** – Recursively scan selected folders for image files.
- **OCR extraction** – Pull text from screenshots via Tesseract.
- **Vision description** – Generate concise image captions using a local LLM (e.g., Gemma3).
- **Embeddings** – Create vector embeddings for hybrid text‑image search.
- **Hybrid search** – Combine keyword and visual similarity for powerful queries.
- **Configurable parallelism** – Adjust concurrency for indexing performance.

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd Searchable-Screenshots

# Create a virtual environment (Windows example)
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run the GUI

```bash
python main.py
```

The application will open a window where you can add folders, configure settings, and start indexing.

### Index from the command line

```bash
python -m src.core.processor --folder "C:\path\to\screenshots"
```

This will process the images and populate the local database.

## Configuration

Open **Settings** in the GUI to adjust:

- Ollama URL
- Vision model name
- Embedding model name
- Parallel processing count (default = 1)

## Contributing

Feel free to open issues or submit pull requests. Please follow the existing code style and run the test suite before submitting.

## License

This project is licensed under the MIT License.
