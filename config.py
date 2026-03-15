import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / 'static' / 'pdf'
VECTOR_DIR = BASE_DIR / 'vector_store'
IS_GAE = os.getenv('GAE_ENV', '').strip().lower().startswith('standard')

GEMINI_API_KEY = '' #leave empty to use ollama

GEMINI_MODEL = 'gemini-2.0-flash'
GEMINI_FALLBACK_MODELS = 'gemini-2.0-flash-lite,gemini-2.5-flash'
GEMINI_EMBED_MODEL = 'gemini-embedding-001'

OLLAMA_MODEL = 'qwen2.5:latest'
OLLAMA_EMBED_MODEL = 'nomic-embed-text'
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434').strip()

CHAT_RAG_TOP_K = 3