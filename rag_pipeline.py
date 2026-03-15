import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import requests
from google import genai
from pypdf import PdfReader

from config import GEMINI_EMBED_MODEL, OLLAMA_EMBED_MODEL, OLLAMA_BASE_URL


@dataclass
class IngestionResult:
    pdf_path: str
    chunks_count: int
    vector_dimension: int
    index_path: str
    metadata_path: str

    def to_dict(self) -> Dict[str, str | int]:
        return {
            "pdf_path": self.pdf_path,
            "chunks_count": self.chunks_count,
            "vector_dimension": self.vector_dimension,
            "index_path": self.index_path,
            "metadata_path": self.metadata_path,
        }


def _extract_pdf_pages(pdf_path: Path) -> List[Dict[str, str | int]]:
    reader = PdfReader(str(pdf_path))
    pages: List[Dict[str, str | int]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        normalized = " ".join(text.split())
        if normalized:
            pages.append(
                {
                    "page": page_number,
                    "text": normalized,
                }
            )

    return pages


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    words = text.split()
    chunks: List[str] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        segment = " ".join(words[start:end]).strip()
        if segment:
            chunks.append(segment)
        if end == len(words):
            break
        start = end - overlap

    return chunks


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def _load_gemini_key() -> str:
    key = os.getenv('GEMINI_API_KEY', '').strip()
    if key:
        return key

    try:
        from config import GEMINI_API_KEY as configured_key
        configured_value = str(configured_key or '').strip()
        if configured_value:
            return configured_value
    except Exception:
        pass

    cfg = Path(__file__).resolve().parent / 'config.json'
    if not cfg.exists():
        return ''
    try:
        data = json.loads(cfg.read_text(encoding='utf-8').strip())
        if isinstance(data, str):
            return data.strip()
        if isinstance(data, dict):
            for k in ('api_key', 'gemini_api_key', 'GEMINI_API_KEY', 'key'):
                v = str(data.get(k, '')).strip()
                if v:
                    return v
    except (json.JSONDecodeError, Exception):
        pass
    return ''



def _resolve_embed_provider() -> str:
    if _load_gemini_key():
        return 'gemini'
    return 'ollama'


def _embed_texts_gemini(texts: List[str], model_name: str, gemini_api_key: str = '') -> np.ndarray:
    api_key = gemini_api_key or _load_gemini_key()
    if not api_key:
        raise ValueError('Gemini API key is required for embedding')

    client = genai.Client(api_key=api_key)
    batch_size = 100
    all_emb: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = client.models.embed_content(
            model=model_name,
            contents=batch,
        )
        for emb in result.embeddings:
            all_emb.append(emb.values)

    return np.asarray(all_emb, dtype=np.float32)



def _embed_texts_ollama(texts: List[str], model_name: str) -> np.ndarray:
    all_emb: List[List[float]] = []
    base_url = OLLAMA_BASE_URL.rstrip('/')
    for text in texts:
        response = requests.post(
            f'{base_url}/api/embeddings',
            json={
                'model': model_name,
                'prompt': text,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        embedding = data.get('embedding')
        if not embedding:
            raise ValueError('Ollama returned empty embedding')
        all_emb.append(embedding)

    return np.asarray(all_emb, dtype=np.float32)


def _embed_texts(
    texts: List[str],
    model_name: str = '',
    gemini_api_key: str = '',
) -> np.ndarray:
    errors: List[str] = []

    if _load_gemini_key() or gemini_api_key:
        try:
            target_model = (model_name or GEMINI_EMBED_MODEL).strip()
            return _embed_texts_gemini(texts=texts, model_name=target_model, gemini_api_key=gemini_api_key)
        except Exception as exc:
            errors.append(f'gemini={exc}')

    try:
        target_model = (model_name or OLLAMA_EMBED_MODEL).strip()
        return _embed_texts_ollama(texts=texts, model_name=target_model)
    except Exception as exc:
        errors.append(f'ollama={exc}')

    raise ValueError(f'No embedding provider available. Errors: {"; ".join(errors)}')


def ingest_pdf_to_vectors(
    pdf_path: str,
    output_dir: str,
    chunk_size: int = 220,
    overlap: int = 40,
    model_name: str = "",
    gemini_api_key: str = "",
) -> IngestionResult:
    pdf = Path(pdf_path)
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    index_path = out / f"{pdf.stem}.index.faiss"
    metadata_path = out / f"{pdf.stem}.metadata.json"

    # If vectors for this PDF already exist, reuse them instead of re-ingesting.
    if index_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        chunks = metadata.get("chunks", [])
        if not isinstance(chunks, list):
            chunks = []

        index = faiss.read_index(str(index_path))
        return IngestionResult(
            pdf_path=str(pdf),
            chunks_count=len(chunks),
            vector_dimension=int(index.d),
            index_path=str(index_path),
            metadata_path=str(metadata_path),
        )

    pages = _extract_pdf_pages(pdf)
    records: List[Dict[str, str | int]] = []

    for page_data in pages:
        page_number = int(page_data["page"])
        page_text = str(page_data["text"])
        page_chunks = _chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
        for chunk in page_chunks:
            records.append(
                {
                    "page": page_number,
                    "chunk": chunk,
                }
            )

    if not records:
        raise ValueError("No text chunks generated from PDF")

    chunk_texts = [str(item["chunk"]) for item in records]
    vectors = _embed_texts(
        texts=chunk_texts,
        model_name=model_name,
        gemini_api_key=gemini_api_key,
    )
    vectors = _normalize_rows(vectors.astype(np.float32))

    if vectors.ndim != 2:
        raise ValueError("Embedding output has unexpected shape")

    dimension = int(vectors.shape[1])
    index = faiss.IndexFlatIP(dimension)
    index.add(np.asarray(vectors, dtype=np.float32))

    faiss.write_index(index, str(index_path))

    selected_provider = _resolve_embed_provider()
    selected_model = model_name or (
        GEMINI_EMBED_MODEL if selected_provider == 'gemini'
        else OLLAMA_EMBED_MODEL
    )

    metadata = {
        "pdf": str(pdf),
        "model": selected_model,
        "provider": selected_provider,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunks": records,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return IngestionResult(
        pdf_path=str(pdf),
        chunks_count=len(records),
        vector_dimension=dimension,
        index_path=str(index_path),
        metadata_path=str(metadata_path),
    )


def ingest_multiple_pdfs(
    pdf_paths: List[str],
    output_dir: str,
    chunk_size: int = 220,
    overlap: int = 40,
    model_name: str = "",
    gemini_api_key: str = "",
) -> List[IngestionResult]:
    """Ingest a list of PDFs, writing one index per PDF into output_dir."""
    results: List[IngestionResult] = []
    for pdf_path in pdf_paths:
        result = ingest_pdf_to_vectors(
            pdf_path=pdf_path,
            output_dir=output_dir,
            chunk_size=chunk_size,
            overlap=overlap,
            model_name=model_name,
            gemini_api_key=gemini_api_key,
        )
        results.append(result)
    return results


def _list_index_names(output_dir: Path) -> List[str]:
    """Return stems of all *.index.faiss files in output_dir."""
    return [p.stem.replace(".index", "") for p in sorted(output_dir.glob("*.index.faiss"))]


def search_vectors(
    query: str,
    output_dir: str,
    top_k: int = 5,
    model_name: str = "",
    gemini_api_key: str = "",
    index_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Search across one or more vector indexes.

    Args:
        query: The search query.
        output_dir: Directory containing *.index.faiss and *.metadata.json files.
        top_k: Number of top results to return (across all indexes combined).
        model_name: Embedding model name.
        gemini_api_key: Gemini API key.
        index_names: List of index stems to search (e.g. ['paedsprotocolv5']).
                     When None (default), all indexes in output_dir are searched.
    """
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    out = Path(output_dir)

    # Resolve which indexes to search
    if index_names is None:
        names = _list_index_names(out)
    else:
        names = list(index_names)

    if not names:
        raise FileNotFoundError("No vector indexes found. Run ingest first.")

    # Embed the query once
    query_vector = _embed_texts(
        texts=[query],
        model_name=model_name,
        gemini_api_key=gemini_api_key,
    )
    query_vector = _normalize_rows(query_vector.astype(np.float32))

    all_hits: List[Dict[str, Any]] = []

    for name in names:
        index_path = out / f"{name}.index.faiss"
        metadata_path = out / f"{name}.metadata.json"

        if not index_path.exists() or not metadata_path.exists():
            continue

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        chunks = metadata.get("chunks", [])
        if not isinstance(chunks, list) or not chunks:
            continue

        index = faiss.read_index(str(index_path))
        k = min(top_k, len(chunks))
        scores, ids = index.search(query_vector, k)

        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(chunks):
                continue
            chunk = chunks[idx]
            chunk_text = str(chunk.get("chunk", ""))
            all_hits.append(
                {
                    "score": float(score),
                    "source": name,
                    "page": chunk.get("page"),
                    "chunk": chunk_text,
                    "preview": chunk_text[:200],
                }
            )

    # Sort by score descending, keep top_k, assign ranks
    all_hits.sort(key=lambda h: h["score"], reverse=True)
    results: List[Dict[str, Any]] = []
    for rank, hit in enumerate(all_hits[:top_k], start=1):
        hit["rank"] = rank
        results.append(hit)

    return results