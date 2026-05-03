import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / 'static' / 'pdf'
VECTOR_DIR = BASE_DIR / 'vector_store'
IS_GAE = os.getenv('GAE_ENV', '').strip().lower().startswith('standard')

# Leave empty to skip Gemini; runtime fallback order is Gemini -> OpenAI -> Ollama.
# run flask ingest-pdf once to populate vector store before using the app and every time change provider.
GEMINI_API_KEY = '' 
GEMINI_MODEL = 'gemini-2.0-flash'
GEMINI_FALLBACK_MODELS = 'gemini-2.0-flash-lite,gemini-2.5-flash'
GEMINI_EMBED_MODEL = 'gemini-embedding-001'

OPENAI_API_KEY = ''
OPENAI_MODEL = 'gpt-4.1-mini'
OPENAI_EMBED_MODEL = 'text-embedding-3-small'

#make sure install ollama and pull qwen3.5:4b and nomic-embed-text models before using Ollama provider.
OLLAMA_MODEL = 'phi4:latest' #'qwen3:4b'
OLLAMA_EMBED_MODEL =  'nomic-embed-text' 
#OLLAMA_EMBED_MODEL =  'qwen3-embedding'
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434').strip()
OLLAMA_KEEP_ALIVE = os.getenv('OLLAMA_KEEP_ALIVE', '24h').strip() or '24h'

CHAT_RAG_TOP_K = 3