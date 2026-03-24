# Flask RAG Pages

Flask app to browse PDF pages and chat with RAG grounded on those PDFs.

## What It Does

- Converts PDFs to page images (`.webp`)
- Ingests PDF text into FAISS vector indexes
- Answers chat questions from retrieved document context
- Supports chat response streaming (SSE)

## Requirements

- Python 3.10+
- Poppler (needed by `pdf2image`)
- One provider:
  - Gemini API key, or
  - OpenAI API key, or
  - Local Ollama (`qwen2.5:latest` + `nomic-embed-text`)

## Quick Start

```bash
git clone https://github.com/sufianstek/flask-rag-pages.git
cd flask-rag-pages

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# choose one key (or use local Ollama without keys)
export GEMINI_API_KEY="your_key"
# export OPENAI_API_KEY="your_key"

flask --app app ingest-pdf
flask --app app run --debug
```

Open `http://127.0.0.1:5000`.

## Basic Usage

1. Place PDFs in `static/pdf/`.
2. Run `flask --app app ingest-pdf`.
3. Browse documents at `/`.
4. Chat at `/chat`.

Generated outputs:
- Page images: `static/pages/<pdf_name>_pages/*.webp`
- Vectors: `vector_store/*.index.faiss` + metadata


## Notes

- Provider order: Gemini -> OpenAI -> Ollama
- Keep API keys out of source control
