import json
import threading
from types import SimpleNamespace

import pytest
import requests

import lazyllm
from lazyllm.module.llms.onlinemodule.supplier.claude import ClaudeChat


pytestmark = pytest.mark.usefixtures('model_call_session')


class Response:
    status_code = 200
    headers = {}

    def __init__(self, frames):
        self.frames = frames
        self.raw = SimpleNamespace(enforce_content_length=False)

    def __enter__(self): return self
    def __exit__(self, *args): pass

    def iter_lines(self, **kwargs):
        for frame in self.frames:
            if isinstance(frame, Exception):
                raise frame
            yield frame

    def iter_content(self, **kwargs):
        for frame in self.iter_lines():
            yield frame + b'\n'


def test_model_call_preserves_bound_sink_identity(monkeypatch):
    class Sink:
        def __init__(self):
            self.lock = threading.Lock()
            self.received = []

        def emit(self, event):
            self.received.append(event)

    sink = Sink()
    model = lazyllm.OnlineChatModule(source='openai', model='gpt-test', api_key='fixture')
    monkeypatch.setattr(requests, 'post', lambda *a, **kw: Response([
        b'data: {"choices":[{"delta":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}']))
    assert model('hello', stream_output={'_stream_sink': sink.emit}) == 'ok'
    assert any(event.get('delta') == 'ok' for event in sink.received)


def test_nonempty_tools_respect_prompter_filter(monkeypatch):
    model = lazyllm.OnlineChatModule(source='openai', model='gpt-test', api_key='fixture', stream=True)
    tools = [{'type': 'function', 'function': {'name': 'keep'}},
             {'type': 'function', 'function': {'name': 'drop'}}]
    generate = model._prompt.generate_prompt

    def filtered(**kwargs):
        result = generate(**kwargs)
        result['tools'] = tools[:1]
        return result

    monkeypatch.setattr(model._prompt, 'generate_prompt', filtered)
    seen = []

    def post(url, **kwargs):
        seen.append(kwargs['json'])
        return Response([b'data: {"choices":[{"delta":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}'])

    monkeypatch.setattr(requests, 'post', post)
    assert model('hello', tools=tools) == 'ok'
    assert seen[0]['tools'] == tools[:1]


def test_claude_accepts_sse_comments_and_metadata(monkeypatch):
    model = lazyllm.OnlineChatModule(source='claude', model='claude-test', api_key='fixture', stream=True)
    frames = [b': keep-alive', b'id: 1', b'retry: 1000', b'event: message_delta',
              b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"ok"}}',
              b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}']
    monkeypatch.setattr(requests, 'post', lambda *a, **kw: Response(frames))
    assert model('hello') == 'ok'


def test_claude_usage_processing_is_linear(monkeypatch):
    model = lazyllm.OnlineChatModule(source='claude', model='claude-test', api_key='fixture', stream=True)
    messages = [{'type': 'message_start', 'message': {'usage': {'input_tokens': 11}}}]
    messages.extend({'type': 'content_block_delta', 'delta': {'type': 'text_delta', 'text': 'x'}} for _ in range(30))
    messages.append({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 7}})
    frames = [('data: ' + json.dumps(message)).encode() for message in messages]
    extract = ClaudeChat._extract_usage
    visited = []

    def counted(items):
        visited.extend(items)
        return extract(items)

    monkeypatch.setattr(ClaudeChat, '_extract_usage', staticmethod(counted))
    monkeypatch.setattr(requests, 'post', lambda *a, **kw: Response(frames))
    assert model('hello') == 'x' * 30
    assert len(visited) <= 2 * len(messages)


def test_claude_failure_keeps_incremental_usage(monkeypatch):
    from lazyllm.module.llms.onlinemodule.base import ModelCallControl, ModelCallError, ModelFailureCode

    model = lazyllm.OnlineChatModule(source='claude', model='claude-test', api_key='fixture', stream=True)
    frames = [b'data: {"type":"message_start","message":{"usage":{"input_tokens":11}}}',
              b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"partial"}}',
              requests.ConnectionError('fixture disconnect')]
    monkeypatch.setattr(requests.Session, 'post', lambda *a, **kw: Response(frames))
    with pytest.raises(ModelCallError) as failure:
        model('hello', call_control=ModelCallControl())
    assert failure.value.terminal.failure.code == ModelFailureCode.TRANSPORT_ERROR
    assert failure.value.usage['prompt_tokens'] == 11
    assert failure.value.usage['completion_tokens'] == -1
    assert failure.value.terminal.usage == failure.value.usage


@pytest.mark.parametrize('frame,origin', [
    (b'data: {broken', 'protocol'),
    (b'data: {"type":"error","error":{"type":"overloaded_error"}}', 'provider'),
])
def test_claude_still_rejects_invalid_payloads_and_provider_errors(monkeypatch, frame, origin):
    from lazyllm.module.llms.onlinemodule.base import ModelCallControl, ModelCallError

    model = lazyllm.OnlineChatModule(source='claude', model='claude-test', api_key='fixture', stream=True)
    monkeypatch.setattr(requests.Session, 'post', lambda *a, **kw: Response([frame]))
    with pytest.raises(ModelCallError) as failure:
        model('hello', call_control=ModelCallControl())
    assert failure.value.terminal.failure.origin.value == origin


def test_controlled_reader_handles_decompressed_chunks_and_all_line_endings():
    from lazyllm.module.llms.onlinemodule.base.model_request import ModelCallControl, response_frames

    chunks = [b': comment\rdata: ', '中文'.encode()[:2], '中文'.encode()[2:], b'\r\nid: 1\nlast']
    response = SimpleNamespace(iter_content=lambda chunk_size: iter(chunks))
    assert list(response_frames(response, True, ModelCallControl())) == [
        b': comment', 'data: 中文'.encode(), b'', b'id: 1', b'last',
    ]


def test_claude_usage_updates_replace_unknowns_without_summing_requests():
    from lazyllm.module.llms.onlinemodule.base.model_call_runner import ModelAttemptState

    model = lazyllm.OnlineChatModule(source='claude', model='claude-test', api_key='fixture')
    frames = [{'usage': {'prompt_tokens': 11, 'completion_tokens': -1}},
              {'usage': {'prompt_tokens': -1, 'completion_tokens': 3}},
              {'usage': {'prompt_tokens': None, 'completion_tokens': 7}},
              {'usage': {'prompt_tokens': 'unknown', 'completion_tokens': -1, 'extra': 999}}]
    state = ModelAttemptState()
    for frame in frames:
        model._update_usage(frame, state)
    expected = {'prompt_tokens': 11, 'completion_tokens': 7,
                'provider_usage': {'prompt_tokens': 11, 'completion_tokens': 7}}
    assert state.usage == model._extract_usage(frames) == expected


def test_controlled_cancel_does_not_read_another_block():
    import asyncio
    from lazyllm.module.llms.onlinemodule.base.model_request import ModelCallControl, response_chunks

    cancel = threading.Event()

    def chunks(**kwargs):
        yield b'first'
        pytest.fail('read again after cancellation')

    iterator = response_chunks(SimpleNamespace(iter_content=chunks), ModelCallControl(cancel_check=cancel.is_set))
    assert next(iterator) == b'first'
    cancel.set()
    with pytest.raises(asyncio.CancelledError):
        next(iterator)


@pytest.mark.parametrize('status,code', [(401, 'authentication_failed'), (429, 'rate_limited')])
@pytest.mark.parametrize('body_failure', ['truncated', 'timeout'])
@pytest.mark.parametrize('cancelled', [False, True])
def test_controlled_http_error_keeps_status_when_body_fails(status, code, body_failure, cancelled):
    import asyncio
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from lazyllm.module.llms.onlinemodule.base import ModelCallControl, ModelCallError

    sent, release = threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args): pass

        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(status)
            self.send_header('Content-Length', '100')
            self.end_headers()
            self.wfile.write(b'{')
            self.wfile.flush()
            sent.set()
            if body_failure == 'timeout': release.wait(2)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    model = lazyllm.OnlineChatModule(
        source='openai', model='test', api_key='fixture', stream=False,
        url=f'http://127.0.0.1:{server.server_port}/v1/')
    try:
        with pytest.raises(asyncio.CancelledError if cancelled else ModelCallError) as failure:
            model('hello', timeout=(0.1, 0.1), max_retries=1,
                  call_control=ModelCallControl(cancel_check=sent.is_set if cancelled else None))
        if not cancelled:
            error = failure.value.terminal.failure
            assert (error.origin.value, error.code.value, error.provider_http_status) == ('http', code, status)
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(2)


@pytest.mark.parametrize('last_frame', [
    b'data: {broken', b'data: {"type":"error","error":{"type":"overloaded_error"}}',
    requests.ConnectionError('fixture disconnect'),
])
def test_ordinary_claude_failure_records_usage_without_control_or_callback(monkeypatch, last_frame):
    from lazyllm.module.module import ModuleExecutionError

    model = lazyllm.OnlineChatModule(source='claude', model='claude-test', api_key='fixture', stream=True)
    model.used_by('ordinary-usage-owner')
    frames = [b'data: {"type":"message_start","message":{"usage":{"input_tokens":11}}}', last_frame]
    monkeypatch.setattr(requests, 'post', lambda *a, **kw: Response(frames))
    with pytest.raises(ModuleExecutionError):
        model('hello')
    assert lazyllm.globals['usage']['ordinary-usage-owner']['prompt_tokens'] == 11
    assert lazyllm.globals['usage']['ordinary-usage-owner']['completion_tokens'] == -1
