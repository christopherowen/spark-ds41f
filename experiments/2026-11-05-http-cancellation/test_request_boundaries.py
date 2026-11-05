"""CPU HTTP/protocol checks; a stub replaces inference, not the API router."""
import asyncio
import hashlib
import json
import socket
import threading
import time
from types import SimpleNamespace

from fastapi import FastAPI
import httpx
import pytest
import uvicorn

from vllm.entrypoints.openai.completion.api_router import attach_router
from vllm.entrypoints.openai.completion.protocol import CompletionResponse
from vllm.entrypoints.serve.exception_handling.register import init_exception_handler
from vllm.renderers import TokenizeParams

LIMIT = 1_048_576


class Handler:
    async def create_completion(self, request, raw_request):
        prompt = request.prompt
        if isinstance(prompt, list):
            TokenizeParams(max_total_tokens=LIMIT, max_output_tokens=request.max_tokens
                           ).apply_post_tokenization(None, {'prompt_token_ids': prompt})
            summary = {'tokens': len(prompt), 'last': prompt[-1]}
        else:
            summary = {'bytes': len(prompt.encode()),
                       'sha256': hashlib.sha256(prompt.encode()).hexdigest()}
        if request.stream:
            async def generate():
                try:
                    raw_request.app.state.active += 1
                    await asyncio.sleep(0.25)  # queued/prefilling, no model tokens yet
                    yield 'data: ' + json.dumps(summary) + '\n\n'
                    yield 'data: [DONE]\n\n'
                finally:
                    raw_request.app.state.active -= 1
            return generate()
        return CompletionResponse(id='boundary', created=0, model='test', choices=[],
                                  usage={'prompt_tokens': summary.get('tokens', 1),
                                         'completion_tokens': 0, 'total_tokens': summary.get('tokens', 1)},
                                  kv_transfer_params=summary)


def wait_for_idle(app):
    # The network client can receive the body before the background callback runs.
    for _ in range(100):
        if app.state.server_load_metrics == 0:
            return
        time.sleep(0.01)
    assert app.state.server_load_metrics == 0


@pytest.fixture(scope='module')
def endpoint():
    app = FastAPI()
    app.state.openai_serving_completion = Handler()
    app.state.enable_server_load_tracking = True
    app.state.server_load_metrics = 0
    app.state.active = 0
    app.state.args = SimpleNamespace(sse_keep_alive_interval=0.02, log_error_stack=False)
    init_exception_handler(app)
    attach_router(app)
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    config = uvicorn.Config(app, log_level='error', lifespan='off', loop='asyncio')
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.01)
    assert server.started
    try:
        yield f'http://127.0.0.1:{sock.getsockname()[1]}/v1/completions', app
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()
        assert not thread.is_alive()


@pytest.mark.parametrize('wire_bytes', [2*1024**2-1, 2*1024**2+1,
                                       64*1024**2-1, 64*1024**2+1])
def test_json_body_byte_boundaries(endpoint, wire_bytes):
    url, app = endpoint
    # Exact HTTP JSON byte boundary, with non-ASCII four-byte code points.
    shell = json.dumps({'model':'test', 'prompt':'', 'max_tokens':1}, separators=(',', ':'))
    available = wire_bytes - len(shell.encode())
    prompt = '\U0001f680' * (available // 4) + 'x' * (available % 4)
    body = json.dumps({'model':'test', 'prompt':prompt, 'max_tokens':1},
                      separators=(',', ':'), ensure_ascii=False).encode()
    assert len(body) == wire_bytes
    with httpx.Client(timeout=30) as client:
        response = client.post(url, content=body, headers={'Content-Type':'application/json'})
    assert response.status_code == 200, response.text[:500]
    assert response.json()['kv_transfer_params'] == {
        'bytes': len(prompt.encode()), 'sha256': hashlib.sha256(prompt.encode()).hexdigest()}
    wait_for_idle(app)


def test_chunked_unicode_and_escaped_json(endpoint):
    url, _ = endpoint
    prompt = '\U0001f680\n"\\' * 150000
    body = json.dumps({'model':'test', 'prompt':prompt, 'max_tokens':1}).encode()
    # Stream without Content-Length; deliberately split JSON escapes.
    with httpx.Client(timeout=30) as client:
        response = client.post(url, content=(body[i:i+8191] for i in range(0,len(body),8191)),
                               headers={'Content-Type':'application/json'})
    assert response.status_code == 200
    assert response.json()['kv_transfer_params']['sha256'] == hashlib.sha256(prompt.encode()).hexdigest()


@pytest.mark.parametrize('length, output, status', [(LIMIT-1024,1024,200),
                          (LIMIT-1024+1,1024,400), (LIMIT-1,1,200), (LIMIT,1,400)])
def test_input_plus_output_limit(endpoint, length, output, status):
    url, app = endpoint
    with httpx.Client(timeout=30) as client:
        response = client.post(url, json={'model':'test','prompt':[7]*length,'max_tokens':output})
    assert response.status_code == status, response.text[:500]
    if status == 400:
        assert 'maximum context length' in response.json()['error']['message']
    else:
        assert response.json()['usage']['prompt_tokens'] == length
    wait_for_idle(app)


@pytest.mark.parametrize('body', [b'{broken', b'{"model":"test","prompt":[-1]}'])
def test_invalid_body_is_an_explicit_error(endpoint, body):
    url, app = endpoint
    with httpx.Client(timeout=5) as client:
        response = client.post(url, content=body, headers={'Content-Type':'application/json'})
    assert response.status_code == 400
    assert response.json()['error']['message']
    wait_for_idle(app)


def test_idle_stream_survives_read_timeout_and_disconnect_cleans_up(endpoint):
    url, app = endpoint
    with httpx.Client(timeout=httpx.Timeout(5, read=0.1)) as client:
        with client.stream('POST', url, json={'model':'test','prompt':'short','stream':True}) as response:
            lines = list(response.iter_lines())
        assert ': keep-alive' in lines
        assert sum(line == 'data: [DONE]' for line in lines) == 1
        assert sum(line.startswith('data: {') for line in lines) == 1
        with client.stream('POST', url, json={'model':'test','prompt':'cancel','stream':True}) as response:
            assert next(response.iter_lines()) == ': keep-alive'
    for _ in range(100):
        if app.state.active == app.state.server_load_metrics == 0:
            break
        time.sleep(0.01)
    assert app.state.active == app.state.server_load_metrics == 0
