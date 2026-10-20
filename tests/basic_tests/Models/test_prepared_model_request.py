import asyncio
import base64
import copy
import json
import socket
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

import lazyllm
from lazyllm.module.llms.onlinemodule.base import ModelCallControl, ModelCallError, ModelFailureCode, ModelFinish
from lazyllm.module.llms.onlinemodule.base.model_call_runner import _ModelCallRunner
from lazyllm.module.llms.onlinemodule.chat import dynamic_chat_config
from lazyllm.hook import LazyLLMHook


pytestmark = pytest.mark.usefixtures('model_call_session')


@pytest.fixture
def endpoint():
    seen, release, received = [], threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append(body)
            received.set()
            if body['model'] == 'slow-head':
                release.wait(3)
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream' if body['stream'] else 'application/json')
            self.end_headers()
            try:
                if body['model'] == 'slow-stream':
                    while not release.wait(0.01):
                        frame = {'choices': [{'index': 0, 'delta': {'content': 'partial '}, 'finish_reason': None}]}
                        self.wfile.write(('data: ' + json.dumps(frame) + '\n\n').encode())
                        self.wfile.flush()
                elif self.path.endswith('/messages'):
                    finish = 'max_tokens' if body['model'] == 'length' else 'end_turn'
                    if body['stream']:
                        frames = [
                            {'type': 'message_start', 'message': {'usage': {'input_tokens': 11}}},
                            {'type': 'content_block_delta', 'delta': {'type': 'text_delta', 'text': 'prepared'}},
                            {'type': 'message_delta', 'delta': {'stop_reason': finish}, 'usage': {'output_tokens': 7}},
                            {'type': 'message_stop'},
                        ]
                        for frame in frames:
                            encoded = 'event: ' + frame['type'] + '\ndata: ' + json.dumps(frame) + '\n\n'
                            self.wfile.write(encoded.encode())
                    else:
                        self.wfile.write(json.dumps({
                            'type': 'message', 'content': [{'type': 'text', 'text': 'prepared'}],
                                                    'stop_reason': finish,
                                                    'usage': {'input_tokens': 11, 'output_tokens': 7}}).encode())
                else:
                    frame = {'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'prepared'},
                                          'finish_reason': 'stop'}],
                             'usage': {'prompt_tokens': 11, 'completion_tokens': 7}}
                    self.wfile.write((('data: ' + json.dumps(frame) + '\n\ndata: [DONE]\n\n') if body['stream']
                                      else json.dumps(frame)).encode())
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1/', seen, received, release
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join()


def model(endpoint, *, source='openai', name='gpt-test', stream=False):
    return lazyllm.OnlineChatModule(source=source, model=name, url=endpoint[0], api_key='fixture-key', stream=stream)


def test_prepare_is_local_and_execution_uses_the_exact_snapshot(endpoint, monkeypatch):
    llm = model(endpoint, name='gpt-5')
    llm._static_params.update(max_tokens=111, tools=[{'invalid': 'must be cleared'}])
    history = [{'role': 'user', 'content': 'old input'}]
    with monkeypatch.context() as patcher:
        patcher.setattr(requests, 'post', lambda *a, **kw: pytest.fail('preparation sent a network request'))
        prepared = llm.prepare_request('hello', llm_chat_history=history, tools=[], output_limit=300)
    snapshot = prepared.payload
    changed = prepared.payload
    changed['messages'].clear()
    history[0]['content'] = 'mutated input'
    llm._static_params['max_tokens'] = 999
    assert prepared.is_estimate is True
    assert prepared.estimate_tokens(len) > 0
    assert 'hello' not in repr(prepared)
    assert snapshot['max_completion_tokens'] == 111
    assert 'max_tokens' not in snapshot and not snapshot.get('tools')
    result = prepared.execute()
    assert result.value == 'prepared'
    assert endpoint[1] == [snapshot]
    assert result.terminal.finish == ModelFinish.STOP
    assert result.usage['prompt_tokens'] == 11
    assert result.usage['completion_tokens'] == 7


def test_dynamic_model_is_bound_when_prepared(endpoint):
    llm = lazyllm.OnlineChatModule(source='dynamic', api_key='auto', type='llm', stream=False)
    with lazyllm.config.temp('openai_api_key', 'fixture-key'):
        with dynamic_chat_config(modules=llm, source='openai', model='first-model', url=endpoint[0]):
            prepared = llm.prepare_request('hello')
        with dynamic_chat_config(modules=llm, source='openai', model='second-model', url=endpoint[0]):
            assert prepared.execute().value == 'prepared'
    assert endpoint[1][0]['model'] == 'first-model'


@pytest.mark.parametrize('dynamic', [False, True])
@pytest.mark.parametrize('cancelled', [False, True])
def test_prepared_preserves_query_hooks_and_value_output(endpoint, dynamic, cancelled):
    calls = []

    class QueryHook(LazyLLMHook):
        __hook_error_mode__ = 'raise'

        def __init__(self, obj):
            pass

        def pre_hook(self, query):
            calls.append(('pre', query))

        def post_hook(self, output):
            assert isinstance(output, str)
            calls.append(('post', output))

        def on_error(self, exc):
            calls.append(('error', type(exc).__name__))

        def finalize(self):
            calls.append(('finalize',))

    llm = lazyllm.OnlineChatModule(
        source='dynamic' if dynamic else 'openai', model='gpt-test', type='llm',
        api_key='auto' if dynamic else 'fixture-key', url=endpoint[0], stream=False)
    llm.register_hook(QueryHook(llm))
    with lazyllm.config.temp('openai_api_key', 'fixture-key'), dynamic_chat_config(
        modules=llm, source='openai', model='gpt-test', url=endpoint[0],
    ):
        assert llm('query') == 'prepared'
        ordinary = list(calls)
        calls.clear()
        prepared = llm.prepare_request('query')
        assert calls == []
        if cancelled:
            with pytest.raises(asyncio.CancelledError):
                prepared.execute(control=ModelCallControl(cancel_check=lambda: True))
        else:
            result = prepared.execute()
    if cancelled:
        assert calls == [('pre', 'query'), ('error', 'CancelledError'), ('finalize',)]
        assert len(endpoint[1]) == 1
        return
    assert calls == ordinary == [('pre', 'query'), ('post', 'prepared'), ('finalize',)]
    assert result.value == 'prepared'
    assert result.terminal.finish == ModelFinish.STOP
    assert result.usage['prompt_tokens'] == 11 and result.usage['completion_tokens'] == 7


@pytest.mark.parametrize('dynamic', [False, True])
def test_prepared_freezes_session_history_files_and_parameters(endpoint, tmp_path, dynamic):
    image = tmp_path / 'input.png'
    image.write_bytes(base64.b64decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jFZkAAAAASUVORK5CYII='))
    llm = lazyllm.OnlineChatModule(
        source='dynamic' if dynamic else 'openai', model='vision-test', type='vlm',
        api_key='auto' if dynamic else 'fixture-key', url=endpoint[0], stream=False).prompt('System constraints.')
    lazyllm.locals['chat_history'][llm._module_id] = [{'role': 'user', 'content': 'SESSION_HISTORY_MARKER'}]
    lazyllm.locals['lazyllm_files'][llm._module_id] = [str(image)]
    lazyllm.locals['global_parameters'][llm._module_id] = {'temperature': 0.123}
    with lazyllm.config.temp('openai_api_key', 'fixture-key'), dynamic_chat_config(
        modules=llm, source='openai', model='vision-test', url=endpoint[0],
    ):
        llm('query', temperature=0.9)
        ordinary = copy.deepcopy(endpoint[1][-1])
        prepared = llm.prepare_request('query', temperature=0.9)
        assert prepared.payload == ordinary
        lazyllm.locals['chat_history'][llm._module_id][0]['content'] = 'LATER_HISTORY'
        lazyllm.locals['lazyllm_files'][llm._module_id] = []
        lazyllm.locals['global_parameters'][llm._module_id]['temperature'] = 0.8
        image.write_bytes(b'changed after preparation')
        assert prepared.execute().value == 'prepared'
    assert endpoint[1] == [ordinary, ordinary]
    assert 'SESSION_HISTORY_MARKER' in str(ordinary['messages'])
    assert 'data:image/' in str(ordinary['messages'])
    assert ordinary['temperature'] == 0.123


@pytest.mark.parametrize('dynamic', [False, True])
@pytest.mark.parametrize('controlled', [False, True])
def test_repeated_prepared_requests_cache_the_complete_result(endpoint, dynamic, controlled):
    llm = lazyllm.OnlineChatModule(
        source='dynamic' if dynamic else 'openai', model='gpt-test', type='llm',
        api_key='auto' if dynamic else 'fixture-key', url=endpoint[0], stream=False,
    ).use_cache('prepared-' + uuid.uuid4().hex)
    with lazyllm.config.temp('openai_api_key', 'fixture-key'), dynamic_chat_config(
        modules=llm, source='openai', model='gpt-test', url=endpoint[0],
    ):
        results = [
            llm.prepare_request('same query').execute(
                control=ModelCallControl(time.monotonic() + 2) if controlled else None)
            for _ in range(3)
        ]
        assert len(endpoint[1]) == 1
        assert all(result.value == 'prepared' and result.terminal.finish == ModelFinish.STOP for result in results)
        assert all(result.usage['prompt_tokens'] == 11 and result.usage['completion_tokens'] == 7 for result in results)
        llm.prepare_request('different query').execute()
        assert len(endpoint[1]) == 2
        with pytest.raises(asyncio.CancelledError):
            llm.prepare_request('same query').execute(control=ModelCallControl(cancel_check=lambda: True))
        assert len(endpoint[1]) == 2


def test_output_limit_retains_the_smaller_static_bound(endpoint):
    llm = model(endpoint)
    llm._static_params['max_tokens'] = 64
    prepared = llm.prepare_request('hello', max_tokens=512, output_limit=256)
    assert prepared.payload['max_tokens'] == 64
    assert llm._static_params['max_tokens'] == 64


def test_control_callback_failure_is_local_and_does_not_send_a_request(endpoint):
    from lazyllm.module.module import ModuleExecutionError

    def broken_check():
        raise RuntimeError('fixture control failure')

    with pytest.raises(ModuleExecutionError, match='cancellation check failed'):
        model(endpoint).prepare_request('hello').execute(control=ModelCallControl(cancel_check=broken_check))
    assert endpoint[1] == []


@pytest.mark.parametrize('source', ['openai', 'claude'])
@pytest.mark.parametrize('stream', [False, True])
def test_public_outcome_and_usage_across_protocols(endpoint, source, stream):
    events = []
    llm = model(endpoint, source=source, stream=stream)
    prepared = llm.prepare_request('hello', stream={'_stream_sink': events.append} if stream else False)
    result = prepared.execute(control=ModelCallControl(time.monotonic() + 2))
    assert result.value == 'prepared'
    assert result.terminal.kind == 'finish' and result.terminal.finish == ModelFinish.STOP
    assert result.terminal.attempt_count == 1
    assert result.usage['prompt_tokens'] == 11 and result.usage['completion_tokens'] == 7
    if stream:
        terminals = [event for event in events
                     if (event.get('runtime_event') or {}).get('type') == 'model_call_finished']
        assert len(terminals) == 1


@pytest.mark.parametrize('stream', [False, True])
def test_native_length_is_a_public_incomplete_terminal(endpoint, stream):
    prepared = model(endpoint, source='claude', name='length').prepare_request('hello', stream=stream)
    with pytest.raises(ModelCallError) as failure:
        prepared.execute()
    assert failure.value.terminal.kind == 'finish'
    assert failure.value.terminal.finish == ModelFinish.LENGTH
    assert failure.value.usage['completion_tokens'] == 7


@pytest.mark.parametrize('name', ['slow-head', 'slow-stream'])
def test_deadline_interrupts_actual_headers_and_stream_reads(endpoint, name):
    prepared = model(endpoint, name=name).prepare_request('hello', stream=True, timeout=3)
    started = time.monotonic()
    with pytest.raises(ModelCallError) as failure:
        prepared.execute(control=ModelCallControl(started + 0.15))
    assert time.monotonic() - started < 1.5
    assert failure.value.terminal.failure.code == ModelFailureCode.REQUEST_TIMEOUT
    assert len(endpoint[1]) == 1


@pytest.mark.parametrize('name', ['slow-head', 'slow-stream'])
def test_cancellation_interrupts_actual_http_without_another_attempt(endpoint, name):
    cancel = threading.Event()
    prepared = model(endpoint, name=name).prepare_request('hello', stream=True, timeout=3)

    def stop():
        if endpoint[2].wait(1):
            cancel.set()

    thread = threading.Thread(target=stop)
    thread.start()
    started = time.monotonic()
    try:
        with pytest.raises(asyncio.CancelledError):
            prepared.execute(control=ModelCallControl(started + 2, cancel.is_set))
    finally:
        thread.join()
    assert time.monotonic() - started < 1.5
    assert len(endpoint[1]) == 1


def test_retry_wait_uses_the_same_deadline():
    attempts, events = [], []

    def fail(state):
        attempts.append(state)
        raise requests.ConnectTimeout('fixture timeout')

    runner = _ModelCallRunner(emit_event=lambda kind, data: events.append((kind, data)),
                              control=ModelCallControl(time.monotonic() + 0.05))
    with pytest.raises(ModelCallError) as failure:
        runner.run(fail, max_attempts=3)
    assert len(attempts) == 1
    assert failure.value.terminal.failure.code == ModelFailureCode.REQUEST_TIMEOUT
    assert len([kind for kind, _ in events if kind == 'model_call_finished']) == 1


@pytest.mark.parametrize('cancelled', [False, True])
def test_deadline_and_cancel_stop_waiting_for_dns_before_any_http(endpoint, monkeypatch, cancelled):
    import socket

    resolving, finish, finished, cancel = (threading.Event() for _ in range(4))
    resolve = socket.getaddrinfo

    def slow_dns(*args, **kwargs):
        resolving.set()
        try:
            assert finish.wait(2)
            return resolve(*args, **kwargs)
        finally:
            finished.set()

    def stop():
        if resolving.wait(1):
            cancel.set()

    prepared = model(endpoint).prepare_request('hello', timeout=3)
    monkeypatch.setattr(socket, 'getaddrinfo', slow_dns)
    thread = threading.Thread(target=stop)
    if cancelled:
        thread.start()
    started = time.monotonic()
    try:
        control = ModelCallControl(started + (2 if cancelled else 0.15), cancel.is_set)
        with pytest.raises(asyncio.CancelledError if cancelled else ModelCallError) as error:
            prepared.execute(control=control)
        elapsed = time.monotonic() - started
        if not cancelled:
            assert error.value.terminal.failure.code == ModelFailureCode.REQUEST_TIMEOUT
    finally:
        finish.set()
        assert finished.wait(1)
        if cancelled:
            thread.join()
    assert elapsed < 1
    assert endpoint[1] == []


@pytest.mark.parametrize('cancelled', [False, True])
def test_deadline_and_cancel_interrupt_tls_handshake(cancelled):
    received, release, cancel = (threading.Event() for _ in range(3))
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        listener.settimeout(2)

        def hold_handshake():
            with listener.accept()[0] as peer:
                assert peer.recv(4096)
                received.set()
                if cancelled:
                    cancel.set()
                release.wait(4)

        thread = threading.Thread(target=hold_handshake, daemon=True)
        thread.start()
        llm = lazyllm.OnlineChatModule(
            source='openai', model='handshake-test', api_key='fixture-key', stream=False,
            url=f'https://127.0.0.1:{listener.getsockname()[1]}/v1/', timeout=3)
        prepared = llm.prepare_request('hello', max_retries=1)
        started = time.monotonic()
        try:
            with pytest.raises(asyncio.CancelledError if cancelled else ModelCallError) as error:
                prepared.execute(control=ModelCallControl(started + (3 if cancelled else 0.3), cancel.is_set))
            elapsed = time.monotonic() - started
            if not cancelled:
                assert error.value.terminal.failure.code == ModelFailureCode.REQUEST_TIMEOUT
        finally:
            release.set()
            thread.join(2)
    assert received.is_set()
    assert elapsed < 1.5
