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

# Make sure to install Ollama and run: ollama pull qwen2.5:7b && ollama pull nomic-embed-text
# qwen2.5:7b  : ~4.7 GB VRAM (Q4_K_M), excellent reasoning-to-latency ratio within 12 GB budget
# qwen2.5:14b : ~8.5 GB VRAM (Q4_K_M), better reasoning, still fits in 12 GB with num_ctx ≤ 4096
# nomic-embed-text: 274 MB, 768-dim, very fast; alternative: mxbai-embed-large (670 MB, 1024-dim)
OLLAMA_MODEL = 'qwen2.5:7b'
OLLAMA_EMBED_MODEL = 'nomic-embed-text'
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434').strip()
# GPU VRAM budget: 12 GB
# num_ctx controls KV-cache size (the main variable VRAM cost beyond model weights).
# 8192 tokens ≈ 1-2 GB KV-cache for 7B; total ≈ 6-7 GB, well within 12 GB.
OLLAMA_NUM_CTX = 8192
# -1 = offload all layers to GPU (Ollama will auto-cap to available VRAM); set 0 to force CPU.
OLLAMA_NUM_GPU = -1

CHAT_RAG_TOP_K = 3