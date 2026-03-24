import json
import os
from pathlib import Path

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    stream_with_context,
    url_for,
)

from config import (
    IS_GAE, PDF_DIR, VECTOR_DIR,
    CHAT_RAG_TOP_K,
)
from functions import (
    _build_rag_prompt,
    _generate_reply,
    _stream_reply,
    _sse_message,
)
from rag_pipeline import ingest_multiple_pdfs, search_vectors
from pdf_to_webp import convert_all_pdfs

app = Flask(__name__)


PAGES_ROOT = Path(app.static_folder) / 'pages'

DOC_TITLE_OVERRIDES = {
    'cpg_stemi': 'CPG STEMI',
    'etdhtaa_medication_protocol': 'ETDHTAA Medication Protocol',
    'paedsprotocolv5': 'Paeds Protocol MY',
    'oscc': 'OSCC',
    'Tintinallis_Emergency_Medicine': "Tintinalli's Emergency Medicine",
}


def _image_sort_key(filename: str) -> tuple[int, str]:
    stem = Path(filename).stem
    if stem.isdigit():
        return (int(stem), filename)
    return (10**9, filename)


def _build_document_title(doc_id: str) -> str:
    if doc_id in DOC_TITLE_OVERRIDES:
        return DOC_TITLE_OVERRIDES[doc_id]
    return doc_id.replace('_', ' ').strip().title()


def _load_documents() -> list[dict]:
    documents: list[dict] = []
    if not PAGES_ROOT.exists():
        return documents

    for doc_dir in sorted(PAGES_ROOT.glob('*_pages')):
        if not doc_dir.is_dir():
            continue

        page_files = sorted(
            [f.name for f in doc_dir.iterdir() if f.is_file() and f.suffix.lower() == '.webp'],
            key=_image_sort_key,
        )
        if not page_files:
            continue

        doc_id = doc_dir.name[:-6]
        documents.append({
            'id': doc_id,
            'title': _build_document_title(doc_id),
            'folder': doc_dir.name,
            'thumbnail': page_files[0],
            'page_files': page_files,
            'page_count': len(page_files),
        })

    return documents


def _get_document_or_404(doc_id: str) -> dict:
    for doc in _load_documents():
        if doc['id'] == doc_id:
            return doc
    abort(404)


def _resolve_doc_filter(doc_id: str | None) -> list[str] | None:
    """Return index names to search for a selected document, or None for all."""
    if not doc_id:
        return None

    normalized = str(doc_id).strip()
    if not normalized:
        return None

    _get_document_or_404(normalized)
    return [normalized]


def _parse_bool(value: object, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0

    normalized = str(value).strip().lower()
    if normalized in {'1', 'true', 'yes', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'off'}:
        return False
    return default


def _list_current_pdf_paths() -> list[str]:
    return [str(p) for p in sorted(PDF_DIR.glob('*.pdf'))]


@app.route('/')
def index():
    return render_template('index.html', documents=_load_documents())


@app.route('/documents/<doc_id>')
def document_pages(doc_id: str):
    document = _get_document_or_404(doc_id)
    return render_template('document_pages.html', document=document)


@app.route('/chat')
def chat_page():
    selected_doc_id = (request.args.get('doc') or '').strip() or None
    selected_document = _get_document_or_404(selected_doc_id) if selected_doc_id else None
    return render_template(
        'chat.html',
        selected_document=selected_document,
        chat_rag_top_k=CHAT_RAG_TOP_K,
    )

@app.route('/cpg-stemi')
def cpg_stemi():
    return redirect(url_for('document_pages', doc_id='cpg_stemi'))

@app.route('/etdhtaa')
def etdhtaa():
    return redirect(url_for('document_pages', doc_id='etdhtaa_medication_protocol'))


@app.route('/paedsprotocolv5')
def paedsprotocolv5():
    return redirect(url_for('document_pages', doc_id='paedsprotocolv5'))

@app.route('/paeds_pdf')
def paeds_pdf():
    return send_from_directory(app.static_folder, 'paedsprotocolv5.pdf')


@app.route('/api/rag/ingest', methods=['POST'])
def ingest_pdf():
    if IS_GAE:
        return jsonify({'status': 'error', 'message': 'Ingestion disabled in production.'}), 403
    pdf_paths = _list_current_pdf_paths()
    if not pdf_paths:
        return jsonify({'status': 'error', 'message': f'No PDF files found in {PDF_DIR}'}), 400

    results = ingest_multiple_pdfs(
        pdf_paths=pdf_paths,
        output_dir=str(VECTOR_DIR),
    )
    return jsonify({
        'status': 'ok',
        'results': [r.to_dict() for r in results],
    })


@app.route('/api/rag/search', methods=['POST'])
def rag_search():
    if IS_GAE:
        return jsonify({'status': 'error', 'message': 'Search endpoint disabled in production.'}), 403
    payload = request.get_json(silent=True) or {}
    query = (payload.get('query') or '').strip()
    top_k = int(payload.get('top_k', 5))
    doc_id = (payload.get('doc_id') or '').strip() or None
    use_context = _parse_bool(payload.get('use_context'), default=True)

    if not query:
        return jsonify({'status': 'error', 'message': 'query is required'}), 400

    results = []
    if use_context:
        try:
            index_names = _resolve_doc_filter(doc_id)
        except Exception:
            return jsonify({'status': 'error', 'message': 'Invalid document selection'}), 400

        results = search_vectors(
            query=query,
            output_dir=str(VECTOR_DIR),
            top_k=top_k,
            index_names=index_names,
        )
    return jsonify({
        'status': 'ok',
        'query': query,
        'top_k': top_k,
        'doc_id': doc_id,
        'use_context': use_context,
        'results': results,
    })


@app.route('/api/chat', methods=['POST'])
def chat_api():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get('message', '')).strip()
    history = payload.get('history', [])
    top_k = int(payload.get('top_k', CHAT_RAG_TOP_K))
    doc_id = (payload.get('doc_id') or '').strip() or None
    use_context = _parse_bool(payload.get('use_context'), default=True)

    if not message:
        return jsonify({'status': 'error', 'message': 'message is required'}), 400
    if not isinstance(history, list):
        return jsonify({'status': 'error', 'message': 'history must be a list'}), 400
    if top_k <= 0:
        return jsonify({'status': 'error', 'message': 'top_k must be greater than 0'}), 400

    index_names = None
    if use_context:
        try:
            index_names = _resolve_doc_filter(doc_id)
        except Exception:
            return jsonify({'status': 'error', 'message': 'Invalid document selection'}), 400

    try:
        contexts = []
        grounded_message = message
        if use_context:
            contexts = search_vectors(
                query=message,
                output_dir=str(VECTOR_DIR),
                top_k=top_k,
                index_names=index_names,
            )
            grounded_message = _build_rag_prompt(message=message, contexts=contexts)
        reply, model_info = _generate_reply(message=grounded_message, history=history)
    except FileNotFoundError:
        return jsonify({
            'status': 'error',
            'message': 'Vector store not found. Run /api/rag/ingest first.',
        }), 400
    except ValueError as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 400
    except Exception as exc:
        return jsonify({'status': 'error', 'message': f'Upstream request failed: {exc}'}), 502

    return jsonify({
        'status': 'ok',
        'reply': reply,
        'provider': model_info.get('provider'),
        'model': model_info.get('model'),
        'doc_id': doc_id,
        'use_context': use_context,
        'contexts': contexts,
    })


@app.route('/api/chat/stream', methods=['POST'])
def chat_api_stream():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get('message', '')).strip()
    history = payload.get('history', [])
    top_k = int(payload.get('top_k', CHAT_RAG_TOP_K))
    doc_id = (payload.get('doc_id') or '').strip() or None
    use_context = _parse_bool(payload.get('use_context'), default=True)

    if not message:
        return jsonify({'status': 'error', 'message': 'message is required'}), 400
    if not isinstance(history, list):
        return jsonify({'status': 'error', 'message': 'history must be a list'}), 400
    if top_k <= 0:
        return jsonify({'status': 'error', 'message': 'top_k must be greater than 0'}), 400

    index_names = None
    if use_context:
        try:
            index_names = _resolve_doc_filter(doc_id)
        except Exception:
            return jsonify({'status': 'error', 'message': 'Invalid document selection'}), 400

    @stream_with_context
    def generate():
        try:
            contexts = []
            grounded_message = message
            if use_context:
                contexts = search_vectors(
                    query=message,
                    output_dir=str(VECTOR_DIR),
                    top_k=top_k,
                    index_names=index_names,
                )
                grounded_message = _build_rag_prompt(message=message, contexts=contexts)
            model_info = {'provider': None, 'model': None}
            stream = _stream_reply(message=grounded_message, history=history)
            while True:
                try:
                    chunk = next(stream)
                    yield _sse_message({'type': 'chunk', 'content': chunk})
                except StopIteration as done:
                    if isinstance(done.value, dict):
                        model_info = done.value
                    break

            yield _sse_message({
                'type': 'done',
                'doc_id': doc_id,
                'use_context': use_context,
                'contexts': contexts,
                'model_info': model_info,
            })
        except FileNotFoundError:
            yield _sse_message({
                'type': 'error',
                'message': 'Vector store not found. Run /api/rag/ingest first.',
            })
        except ValueError as exc:
            yield _sse_message({'type': 'error', 'message': str(exc)})
        except Exception as exc:
            yield _sse_message({'type': 'error', 'message': f'Unexpected server error: {exc}'})

    return Response(generate(), mimetype='text/event-stream')


@app.cli.command('ingest-pdf')
def ingest_pdf_command():
    if IS_GAE:
        print('Ingestion disabled in production.')
        return

    conversion_summary = convert_all_pdfs(pdf_root=str(PDF_DIR))
    if conversion_summary['total'] == 0:
        print(f"No PDF files found in {conversion_summary['pdf_root']}")
        return

    print(
        f"PDF conversion summary: converted {conversion_summary['converted']}, "
        f"skipped {conversion_summary['skipped']}, failed {conversion_summary['failed']}, "
        f"total {conversion_summary['total']}"
    )

    results = ingest_multiple_pdfs(
        pdf_paths=conversion_summary['pdf_files'],
        output_dir=str(VECTOR_DIR),
    )
    print(json.dumps([r.to_dict() for r in results], indent=2))


@app.cli.command('rag-search')
def rag_search_command():
    if IS_GAE:
        print('RAG search disabled in production.')
        return
    query = os.getenv('RAG_QUERY', '').strip()
    top_k = int(os.getenv('RAG_TOP_K', '5'))
    if not query:
        raise ValueError('Set RAG_QUERY environment variable to run rag-search')

    results = search_vectors(
        query=query,
        output_dir=str(VECTOR_DIR),
        top_k=top_k,
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)