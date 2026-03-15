import json
import os
from typing import Generator

import requests
from google import genai

from config import (
    BASE_DIR,
    GEMINI_MODEL,
    GEMINI_FALLBACK_MODELS,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
)


def _get_genai_client() -> genai.Client:
    api_key = _load_gemini_api_key()
    if not api_key:
        raise ValueError('Gemini API key is missing. Set GEMINI_API_KEY or config.json')
    return genai.Client(api_key=api_key)


def _load_gemini_api_key() -> str:
    key = os.getenv('GEMINI_API_KEY', '').strip()
    if key:
        return key

    # Backward-compatible fallback from config.py constant.
    try:
        from config import GEMINI_API_KEY as configured_key  # local import to avoid stale values
        configured_value = str(configured_key or '').strip()
        if configured_value:
            return configured_value
    except Exception:
        pass

    config_path = BASE_DIR / 'config.json'
    if not config_path.exists():
        return ''

    raw = config_path.read_text(encoding='utf-8').strip()
    if not raw:
        return ''

    try:
        data = json.loads(raw)
        if isinstance(data, str):
            return data.strip()
        if isinstance(data, dict):
            for candidate in ('api_key', 'gemini_api_key', 'GEMINI_API_KEY', 'key'):
                value = str(data.get(candidate, '')).strip()
                if value:
                    return value
            return ''
    except json.JSONDecodeError:
        return raw

    return ''



def _get_active_provider() -> str:
    if _load_gemini_api_key():
        return 'gemini'
    return 'ollama'


def _resolve_provider_order() -> list[str]:
    return ['gemini', 'ollama']


def _build_rag_prompt(message: str, contexts: list[dict]) -> str:
    context_lines = []
    for idx, item in enumerate(contexts, start=1):
        page = item.get('page', '?')
        chunk = str(item.get('chunk', '')).strip()
        if not chunk:
            continue
        context_lines.append(f'[{idx}] (page {page}) {chunk}')

    context_block = '\n\n'.join(context_lines).strip()

    return (
        'You must answer strictly and only from the provided PDF context below. '
        'Do not use outside knowledge, assumptions, training data, or prior conversation. '
        'If the answer is not explicitly supported by the context, reply exactly: '
        '"The document does not provide this information."\n\n'
        f'PDF Context:\n{context_block}\n\n'
        f'User Question: {message}'
    )


def _build_contents(message: str, history: list[dict] | None = None) -> list[dict]:
    contents = []
    for item in history or []:
        role = str(item.get('role', '')).strip().lower()
        text = str(item.get('content', '')).strip()
        if not text:
            continue
        mapped_role = 'user' if role == 'user' else 'model'
        contents.append({
            'role': mapped_role,
            'parts': [{'text': text}],
        })
    contents.append({
        'role': 'user',
        'parts': [{'text': message}],
    })
    return contents


def _get_models_to_try() -> list[str]:
    primary_model = GEMINI_MODEL.replace('models/', '').strip()
    fallback_models = [
        model.strip().replace('models/', '')
        for model in GEMINI_FALLBACK_MODELS.split(',')
        if model.strip()
    ]
    return [primary_model] + [m for m in fallback_models if m != primary_model]


def _gemini_generate_reply(message: str, history: list[dict] | None = None) -> str:
    client = _get_genai_client()
    contents = _build_contents(message, history)
    models_to_try = _get_models_to_try()

    last_error = None
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
            )
            reply = (response.text or '').strip()
            if reply:
                return reply
        except Exception as exc:
            last_error = exc
            continue

    raise ValueError(
        f'Gemini request failed for models {models_to_try}: {last_error}'
    )


def _gemini_stream_reply(message: str, history: list[dict] | None = None):
    client = _get_genai_client()
    contents = _build_contents(message, history)
    models_to_try = _get_models_to_try()

    last_error = None
    for model_name in models_to_try:
        try:
            emitted_any = False
            for chunk in client.models.generate_content_stream(
                model=model_name,
                contents=contents,
            ):
                text = chunk.text or ''
                if text:
                    emitted_any = True
                    yield text
            if emitted_any:
                return
        except Exception as exc:
            last_error = exc
            continue

    raise ValueError(
        f'Gemini stream request failed for models {models_to_try}: {last_error}'
    )




def _ollama_generate_reply(message: str, history: list[dict] | None = None) -> str:
    messages = []
    for item in history or []:
        role = str(item.get('role', '')).strip().lower()
        text = str(item.get('content', '')).strip()
        if not text:
            continue
        if role not in {'user', 'assistant', 'system'}:
            role = 'assistant'
        messages.append({'role': role, 'content': text})
    messages.append({'role': 'user', 'content': message})

    response = requests.post(
        f'{OLLAMA_BASE_URL.rstrip("/")}/api/chat',
        json={
            'model': OLLAMA_MODEL,
            'messages': messages,
            'stream': False,
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    content = str(data.get('message', {}).get('content', '')).strip()
    if not content:
        raise ValueError('Ollama returned empty content')
    return content


def _ollama_stream_reply(message: str, history: list[dict] | None = None) -> Generator[str, None, None]:
    messages = []
    for item in history or []:
        role = str(item.get('role', '')).strip().lower()
        text = str(item.get('content', '')).strip()
        if not text:
            continue
        if role not in {'user', 'assistant', 'system'}:
            role = 'assistant'
        messages.append({'role': role, 'content': text})
    messages.append({'role': 'user', 'content': message})

    with requests.post(
        f'{OLLAMA_BASE_URL.rstrip("/")}/api/chat',
        json={
            'model': OLLAMA_MODEL,
            'messages': messages,
            'stream': True,
        },
        timeout=300,
        stream=True,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            content = str(data.get('message', {}).get('content', '') or '')
            if content:
                yield content

            if data.get('done'):
                return


def _generate_reply(message: str, history: list[dict] | None = None) -> str:
    errors: list[str] = []

    for provider in _resolve_provider_order():
        if provider == 'gemini':
            if not _load_gemini_api_key():
                continue
            try:
                return _gemini_generate_reply(message=message, history=history)
            except Exception as exc:
                errors.append(f'gemini={exc}')
                continue

        if provider == 'ollama':
            try:
                return _ollama_generate_reply(message=message, history=history)
            except Exception as exc:
                errors.append(f'ollama={exc}')
                continue

    raise ValueError(f'No provider available. Errors: {"; ".join(errors)}')


def _stream_reply(message: str, history: list[dict] | None = None) -> Generator[str, None, None]:
    errors: list[str] = []

    for provider in _resolve_provider_order():
        if provider == 'gemini':
            if not _load_gemini_api_key():
                continue
            try:
                yield from _gemini_stream_reply(message=message, history=history)
                return
            except Exception as exc:
                errors.append(f'gemini={exc}')
                continue

        if provider == 'ollama':
            try:
                yield from _ollama_stream_reply(message=message, history=history)
                return
            except Exception as exc:
                errors.append(f'ollama={exc}')
                continue

    raise ValueError(f'No streaming provider available. Errors: {"; ".join(errors)}')


def _sse_message(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
