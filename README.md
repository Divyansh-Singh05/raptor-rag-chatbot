# Raptor RAG Chatbot (TDS Assistant)

This project is a Python + Streamlit Retrieval-Augmented Generation (RAG) application for TDS-related query support, including document upload and invoice analysis.

## Features

- Streamlit-based chatbot UI for TDS guidance
- RAG pipeline for knowledge ingestion and retrieval
- Invoice/document parsing with OCR support
- Azure OpenAI integration via environment variables
- Optional launcher/build flow for packaged executable usage

## Project Structure

- `tds_app6.py` - Main Streamlit application
- `unified_rag_pipeline.py` - RAG preprocessing/ingestion flow
- `launcher.py` - Launcher that initializes data and starts the app
- `invoice_processor.py`, `ocr.py`, `advanced_parsing.py` - Document extraction helpers
- `data/` - Knowledge/input files
- `embeddings/` - Generated vector index artifacts
- `chunks.json` - Generated chunk metadata
- `requirements.txt` - Python dependencies

## Prerequisites

- Python 3.10+ recommended
- `pip` package manager
- (Optional) Tesseract OCR installed locally for OCR-heavy workflows

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:

```env
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
AZURE_OPENAI_VERSION=2024-10-21
```

## Run the Application

### Option 1: Run Streamlit app directly

```bash
streamlit run tds_app6.py
```

### Option 2: Run launcher

```bash
python launcher.py
```

The launcher can reuse existing embeddings/chunks and trigger RAG processing only when needed.

## Regenerate RAG Artifacts (if needed)

If data changes and you want to rebuild embeddings/chunks:

```bash
python unified_rag_pipeline.py
```

## Notes for GitHub Upload

- Do **not** commit `.env` (contains secrets)
- Do **not** commit generated folders/files (`__pycache__/`, `embeddings/`, temporary outputs)
- Add a `.gitignore` before pushing (I can create this next)

## Troubleshooting

- Missing environment variables: verify `.env` keys are present
- Streamlit not opening: run `streamlit run tds_app6.py` manually and check terminal logs
- OCR quality issues: ensure Tesseract is installed and available in PATH

## License

Add your preferred license here (for example: MIT).
