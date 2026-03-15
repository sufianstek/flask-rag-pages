# Flask RAG Pages

Flask web app for browsing PDF pages and chatting with a Retrieval-Augmented Generation (RAG) assistant grounded on those PDFs.

The app supports:
- PDF page rendering (`.pdf` -> `.webp` pages)
- Vector ingestion with FAISS
- Context-aware chat (regular JSON and SSE streaming)
- Multiple LLM/embedding providers with fallback: Gemini -> OpenAI -> Ollama

## Features

- Document gallery and per-document page viewer
- Optional document-scoped chat (`doc_id`) for targeted answers
- Strict RAG prompt grounding (answers only from indexed PDF context)
- Flask CLI commands for ingestion and quick search
- REST endpoints for ingest, search, chat, and chat streaming

## Tech Stack

- Python + Flask
- FAISS (`faiss-cpu`) for vector similarity search
- `pypdf` for PDF text extraction
- `pdf2image` + Pillow for page image conversion
- LLM providers:
  - Gemini (`google-genai`)
  - OpenAI (HTTP API)
  - Ollama (local API)

## Project Structure

```text
.
|- app.py                # Flask app, routes, and CLI commands
|- rag_pipeline.py       # PDF chunking, embeddings, FAISS ingest/search
|- functions.py          # Chat provider routing + generation/stream helpers
|- pdf_to_webp.py        # PDF page conversion to static WebP folders
|- config.py             # Paths, model names, runtime provider settings
|- static/
|  |- pdf/               # Source PDFs
|  |- pages/             # Generated page images: <doc>_pages/1.webp...
|  |- css/ js/
|- templates/
|  |- index.html
|  |- document_pages.html
|  |- chat.html
```

## Requirements

- Linux/macOS/Windows
- Python 3.10+
- `pip`
- Poppler (required by `pdf2image`)

Poppler install examples:
- Ubuntu/Debian: `sudo apt-get install poppler-utils`
- macOS (Homebrew): `brew install poppler`
- Windows: install Poppler binaries and add to `PATH`

If using Ollama provider:
- Install Ollama
- Pull required models (based on `config.py` defaults):
  - `ollama pull qwen2.5:latest`
  - `ollama pull nomic-embed-text`

## Quick Start

1. Clone and enter the repo.

```bash
git clone https://github.com/sufianstek/flask-rag-pages.git
cd flask-rag-pages
```

2. Create and activate a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

4. Configure provider credentials.

Preferred via environment variables:

```bash
export GEMINI_API_KEY="your_key"
# or
export OPENAI_API_KEY="your_key"
```

Optional custom OpenAI-compatible endpoint:

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

Notes:
- Provider selection is automatic:
  - Gemini key present -> use Gemini first
  - else OpenAI key present -> use OpenAI first
  - else fallback to Ollama
- `config.py` also contains constants for keys/models, but environment variables are recommended.

5. Add your PDFs into `static/pdf/`.

6. Ingest PDFs and generate vectors/pages.

```bash
flask --app app ingest-pdf
```

7. Run the app.

```bash
flask --app app run --debug
```

Open `http://127.0.0.1:5000`.

## Usage Flow

1. Put PDFs in `static/pdf/`.
2. Run `flask --app app ingest-pdf`.
3. Open home page and select a document.
4. Open chat (`/chat`) optionally scoped to a document.

During ingestion:
- PDF filenames are sanitized
- Pages are converted to `static/pages/<pdf_stem>_pages/*.webp`
- Text chunks are embedded and indexed into `vector_store/`

## Flask CLI Commands

- `flask --app app ingest-pdf`
  - Converts PDF pages to WebP and ingests vectors
- `RAG_QUERY="..." RAG_TOP_K=5 flask --app app rag-search`
  - Runs a local search against the vector store

## API Endpoints

Base URL: `http://127.0.0.1:5000`

- `POST /api/rag/ingest`
  - Ingest all PDFs under `static/pdf/`
- `POST /api/rag/search`
  - Body: `{"query":"...","top_k":5,"doc_id":"optional","use_context":true}`
- `POST /api/chat`
  - Body: `{"message":"...","history":[],"top_k":3,"doc_id":"optional","use_context":true}`
- `POST /api/chat/stream`
  - Same body as `/api/chat`
  - Returns `text/event-stream` with `chunk`, `done`, and `error` events

Example search call:

```bash
curl -X POST http://127.0.0.1:5000/api/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query":"What is STEMI protocol?","top_k":5,"use_context":true}'
```

Example chat call:

```bash
curl -X POST http://127.0.0.1:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Summarize first-line treatment","history":[],"top_k":3,"use_context":true}'
```

## Configuration

Main settings are in `config.py`:
- Paths: `PDF_DIR`, `VECTOR_DIR`
- Runtime flag: `IS_GAE`
- Chat retrieval depth: `CHAT_RAG_TOP_K`
- Model names for Gemini/OpenAI/Ollama

Environment variables:
- `GEMINI_API_KEY`
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OLLAMA_BASE_URL`
- `RAG_QUERY` and `RAG_TOP_K` (for CLI `rag-search`)

## Deployment Notes

When `IS_GAE` is true (Google App Engine standard), ingestion/search mutation routes are disabled:
- `/api/rag/ingest` returns 403
- `/api/rag/search` returns 403

Prepare vector data before deployment if your production environment is read-only.

## Troubleshooting

- `Vector store not found. Run /api/rag/ingest first.`
  - Run `flask --app app ingest-pdf` to create `vector_store/*.faiss` and metadata.

- `No PDF files found in ...`
  - Ensure PDFs exist in `static/pdf/` and have `.pdf` extension.

- `pdf2image` conversion errors
  - Verify Poppler is installed and available on `PATH`.

- Provider/API errors
  - Verify API keys and connectivity.
  - If cloud providers are unavailable, ensure Ollama is running locally.

## Security Notes

- Do not commit API keys to source control.
- Prefer environment variables or secret managers for credentials.
