import json
import os
from typing import Generator

import requests
from google import genai

from config import (
    BASE_DIR,
    GEMINI_MODEL,
    GEMINI_FALLBACK_MODELS,
    OPENAI_MODEL,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
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


def _load_openai_api_key() -> str:
    key = os.getenv('OPENAI_API_KEY', '').strip()
    if key:
        return key

    # Backward-compatible fallback from config.py constant.
    try:
        from config import OPENAI_API_KEY as configured_key  # local import to avoid stale values
        configured_value = str(configured_key or '').strip()
        if configured_value:
            return configured_value
    except Exception:
        pass

    return ''


def _get_openai_base_url() -> str:
    return os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1').strip().rstrip('/')


def _build_ollama_chat_payload(messages: list[dict], stream: bool) -> dict:
    payload = {
        'model': OLLAMA_MODEL,
        'messages': messages,
        'stream': stream,
    }

    # Ollama /api/chat may reject keep_alive='-1' on some versions.
    keep_alive = str(OLLAMA_KEEP_ALIVE or '').strip()
    if keep_alive and keep_alive != '-1':
        payload['keep_alive'] = keep_alive

    return payload



def _get_active_provider() -> str:
    if _load_gemini_api_key():
        return 'gemini'
    if _load_openai_api_key():
        return 'openai'
    return 'ollama'


def _resolve_provider_order() -> list[str]:
    if _load_gemini_api_key():
        return ['gemini', 'openai', 'ollama']
    if _load_openai_api_key():
        return ['openai', 'ollama']
    return ['ollama']


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
        'If the answer is not explicitly supported by the context, do NOT say the answer is unavailable. '
        'Instead, respond with:\n'
        '"Suggested questions:\n" '
        'followed by 2 suggested questions (as a numbered list) that CAN be answered from the provided PDF context. '
        'Base the suggested questions only on topics and facts present in the context chunks.\n'
        'Always format your answer using bullet points or numbered lists. '
        'Use short, concise point-form sentences. Avoid long paragraphs.\n\n'
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


def _gemini_generate_reply(message: str, history: list[dict] | None = None) -> tuple[str, str]:
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
                return reply, model_name
        except Exception as exc:
            last_error = exc
            continue

    raise ValueError(
        f'Gemini request failed for models {models_to_try}: {last_error}'
    )


def _gemini_stream_reply(message: str, history: list[dict] | None = None) -> Generator[str, None, str]:
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
                return model_name
        except Exception as exc:
            last_error = exc
            continue

    raise ValueError(
        f'Gemini stream request failed for models {models_to_try}: {last_error}'
    )


def _build_openai_messages(message: str, history: list[dict] | None = None) -> list[dict]:
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
    return messages


def _openai_generate_reply(message: str, history: list[dict] | None = None) -> tuple[str, str]:
    api_key = _load_openai_api_key()
    if not api_key:
        raise ValueError('OpenAI API key is missing. Set OPENAI_API_KEY or config.py')

    response = requests.post(
        f'{_get_openai_base_url()}/chat/completions',
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        },
        json={
            'model': OPENAI_MODEL,
            'messages': _build_openai_messages(message=message, history=history),
            'stream': False,
        },
        timeout=120,
    )
    response.raise_for_status()

    data = response.json()
    choices = data.get('choices') or []
    content = str((choices[0] if choices else {}).get('message', {}).get('content', '')).strip()
    if not content:
        raise ValueError('OpenAI returned empty content')

    model_name = str(data.get('model') or OPENAI_MODEL).strip() or OPENAI_MODEL
    return content, model_name


def _openai_stream_reply(message: str, history: list[dict] | None = None) -> Generator[str, None, str]:
    api_key = _load_openai_api_key()
    if not api_key:
        raise ValueError('OpenAI API key is missing. Set OPENAI_API_KEY or config.py')

    model_name = OPENAI_MODEL

    with requests.post(
        f'{_get_openai_base_url()}/chat/completions',
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        },
        json={
            'model': OPENAI_MODEL,
            'messages': _build_openai_messages(message=message, history=history),
            'stream': True,
        },
        timeout=300,
        stream=True,
    ) as response:
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            raw_line = str(line).strip()
            if not raw_line.startswith('data:'):
                continue

            payload = raw_line[5:].strip()
            if payload == '[DONE]':
                return model_name

            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue

            reported_model = str(data.get('model') or '').strip()
            if reported_model:
                model_name = reported_model

            choices = data.get('choices') or []
            delta = (choices[0] if choices else {}).get('delta', {})
            content = str(delta.get('content', '') or '')
            if content:
                yield content

    return model_name




def _ollama_generate_reply(message: str, history: list[dict] | None = None) -> tuple[str, str]:
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
        json=_build_ollama_chat_payload(messages=messages, stream=False),
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    content = str(data.get('message', {}).get('content', '')).strip()
    if not content:
        raise ValueError('Ollama returned empty content')
    return content, OLLAMA_MODEL


def _ollama_stream_reply(message: str, history: list[dict] | None = None) -> Generator[str, None, str]:
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
        json=_build_ollama_chat_payload(messages=messages, stream=True),
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
                return OLLAMA_MODEL


def _generate_reply(message: str, history: list[dict] | None = None) -> tuple[str, dict[str, str]]:
    errors: list[str] = []

    for provider in _resolve_provider_order():
        if provider == 'gemini':
            if not _load_gemini_api_key():
                continue
            try:
                reply, model_name = _gemini_generate_reply(message=message, history=history)
                return reply, {'provider': 'gemini', 'model': model_name}
            except Exception as exc:
                errors.append(f'gemini={exc}')
                continue

        if provider == 'openai':
            if not _load_openai_api_key():
                continue
            try:
                reply, model_name = _openai_generate_reply(message=message, history=history)
                return reply, {'provider': 'openai', 'model': model_name}
            except Exception as exc:
                errors.append(f'openai={exc}')
                continue

        if provider == 'ollama':
            try:
                reply, model_name = _ollama_generate_reply(message=message, history=history)
                return reply, {'provider': 'ollama', 'model': model_name}
            except Exception as exc:
                errors.append(f'ollama={exc}')
                continue

    raise ValueError(f'No provider available. Errors: {"; ".join(errors)}')


def _stream_reply(message: str, history: list[dict] | None = None) -> Generator[str, None, dict[str, str]]:
    errors: list[str] = []

    for provider in _resolve_provider_order():
        if provider == 'gemini':
            if not _load_gemini_api_key():
                continue
            try:
                model_name = yield from _gemini_stream_reply(message=message, history=history)
                return {'provider': 'gemini', 'model': model_name}
            except Exception as exc:
                errors.append(f'gemini={exc}')
                continue

        if provider == 'openai':
            if not _load_openai_api_key():
                continue
            try:
                model_name = yield from _openai_stream_reply(message=message, history=history)
                return {'provider': 'openai', 'model': model_name}
            except Exception as exc:
                errors.append(f'openai={exc}')
                continue

        if provider == 'ollama':
            try:
                model_name = yield from _ollama_stream_reply(message=message, history=history)
                return {'provider': 'ollama', 'model': model_name}
            except Exception as exc:
                errors.append(f'ollama={exc}')
                continue

    raise ValueError(f'No streaming provider available. Errors: {"; ".join(errors)}')


def _sse_message(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
