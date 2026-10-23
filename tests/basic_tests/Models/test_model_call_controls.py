import asyncio
import base64
import copy
import threading
import time
import uuid

import pytest
import requests

import lazyllm
from lazyllm.hook import LazyLLMHook
from lazyllm.module.module import ModuleExecutionError
from lazyllm.module.llms.onlinemodule.base import ModelCallControl, ModelCallError, ModelFailureCode, ModelFinish
from lazyllm.module.llms.onlinemodule.base.model_call_runner import _ModelCallRunner
from lazyllm.module.llms.onlinemodule.chat import dynamic_chat_config
from model_call_test_support import model


pytestmark = pytest.mark.usefixtures('model_call_session')


def test_request_check_observes_exact_sent_body_without_executing_it(endpoint):
    llm = model(endpoint, name='gpt-5')
    llm._static_params.update(max_tokens=111, tools=[{'invalid': 'must be cleared'}])
    observed = []

    def check(request):
        assert endpoint[1] == []
        assert request.is_estimate and request.estimate_tokens(len) > 0
        assert 'hello' not in repr(request)
        observed.append(request.payload)
        request.payload['messages'].clear()

    assert llm('hello', tools=[], output_limit=300, call_control=ModelCallControl(request_check=check)) == 'prepared'
    assert endpoint[1] == observed
    assert observed[0]['max_completion_tokens'] == 111
    assert 'max_tokens' not in observed[0] and not observed[0].get('tools')


@pytest.mark.parametrize('dynamic', [False, True])
def test_normal_call_preserves_hooks_history_files_and_parameters(endpoint, tmp_path, dynamic):
    calls = []

    class Hook(LazyLLMHook):
        __hook_error_mode__ = 'raise'

        def __init__(self, obj): pass

        def pre_hook(self, query, **kwargs):
            calls.append(('pre', query))

        def post_hook(self, value):
            assert isinstance(value, str)
            calls.append(('post', value))

        def finalize(self): calls.append(('finalize',))

    llm = lazyllm.OnlineChatModule(source='dynamic' if dynamic else 'openai', model='gpt-test', type='vlm',
                                   api_key='auto' if dynamic else 'fixture', url=endpoint[0], stream=False)
    llm.prompt('system instruction')
    llm.register_hook(Hook(llm))
    image = tmp_path / 'image.png'
    image.write_bytes(base64.b64decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jQmQAAAAASUVORK5CYII='))
    history = [{'role': 'user', 'content': 'prior input'}]
    lazyllm.globals['chat_history'][llm._module_id] = copy.deepcopy(history)
    lazyllm.globals['lazyllm_files'][llm._module_id] = [str(image)]
    lazyllm.globals['global_parameters'][llm._module_id] = {'temperature': 0.17}
    with lazyllm.config.temp('openai_api_key', 'fixture'), dynamic_chat_config(
        modules=llm, source='openai', model='resolved-model', url=endpoint[0],
    ):
        assert llm('query', call_control=ModelCallControl()) == 'prepared'
    body = endpoint[1][0]
    assert body['model'] == ('resolved-model' if dynamic else 'gpt-test')
    assert body['temperature'] == 0.17
    assert 'prior input' in str(body['messages']) and 'data:image/png;base64,' in str(body['messages'])
    assert lazyllm.globals['chat_history'][llm._module_id] == history
    assert calls == [('pre', 'query'), ('post', 'prepared'), ('finalize',)]


@pytest.mark.parametrize('dynamic', [False, True])
def test_controlled_normal_calls_cache_and_check_cancel_on_hits(endpoint, dynamic):
    llm = lazyllm.OnlineChatModule(source='dynamic' if dynamic else 'openai', model='gpt-test', type='llm',
                                   api_key='auto' if dynamic else 'fixture', url=endpoint[0], stream=False)
    llm.use_cache('call-test-' + uuid.uuid4().hex)
    with lazyllm.config.temp('openai_api_key', 'fixture'), dynamic_chat_config(
        modules=llm, source='openai', model='gpt-test', url=endpoint[0],
    ):
        for _ in range(3):
            assert llm('same query', call_control=ModelCallControl(cancel_check=lambda: False)) == 'prepared'
        assert len(endpoint[1]) == 1
        with pytest.raises(asyncio.CancelledError):
            llm('same query', call_control=ModelCallControl(cancel_check=lambda: True))
        assert len(endpoint[1]) == 1
        assert llm('different query', call_control=ModelCallControl()) == 'prepared'
        assert len(endpoint[1]) == 2


@pytest.mark.parametrize('source,name,key', [
    ('openai', 'gpt-test', 'max_tokens'), ('openai', 'gpt-5', 'max_completion_tokens'),
    ('claude', 'claude-test', 'max_tokens'),
])
def test_output_limit_retains_smaller_static_bound(endpoint, source, name, key):
    llm = model(endpoint, source=source, name=name)
    llm._static_params['max_tokens'] = 64
    assert llm('hello', max_tokens=512, output_limit=256) == 'prepared'
    assert endpoint[1][0][key] == 64
    assert llm._static_params['max_tokens'] == 64


def test_control_callback_failure_is_local(endpoint):
    def broken(): raise RuntimeError('fixture control failure')

    with pytest.raises(ModuleExecutionError, match='cancellation check failed'):
        model(endpoint)('hello', call_control=ModelCallControl(cancel_check=broken))
    assert endpoint[1] == []


@pytest.mark.parametrize('source', ['openai', 'claude'])
@pytest.mark.parametrize('stream', [False, True])
def test_public_outcome_and_usage_across_protocols(endpoint, source, stream):
    terminals = []
    llm = model(endpoint, source=source, stream=stream)
    assert llm('hello', on_terminal=terminals.append, call_control=ModelCallControl()) == 'prepared'
    assert len(terminals) == 1
    terminal = terminals[0]
    assert terminal.kind == 'finish' and terminal.finish == ModelFinish.STOP and terminal.attempt_count == 1
    assert terminal.usage['prompt_tokens'] == 11 and terminal.usage['completion_tokens'] == 7


@pytest.mark.parametrize('stream', [False, True])
def test_native_length_has_public_incomplete_terminal(endpoint, stream):
    with pytest.raises(ModelCallError) as failure:
        model(endpoint, source='claude', name='length', stream=stream)('hello', call_control=ModelCallControl())
    assert failure.value.terminal.kind == 'finish' and failure.value.terminal.finish == ModelFinish.LENGTH
    assert failure.value.usage['completion_tokens'] == 7


@pytest.mark.parametrize('name,stream', [('slow-head', True), ('slow-read', True), ('slow-body', False)])
def test_idle_read_timeout_bounds_waits(endpoint, name, stream):
    started = time.monotonic()
    with pytest.raises(ModelCallError) as failure:
        model(endpoint, name=name, stream=stream)(
            'hello', timeout=(0.15, 0.15), max_retries=1, call_control=ModelCallControl())
    assert time.monotonic() - started < 1.5
    assert failure.value.terminal.failure.code == ModelFailureCode.REQUEST_TIMEOUT
    assert len(endpoint[1]) == 1


def test_continuous_stream_outlives_read_timeout(endpoint):
    started = time.monotonic()
    assert model(endpoint, name='long-stream', stream=True)(
        'hello', timeout=(0.1, 0.1), call_control=ModelCallControl()) == 'partial ' * 12
    assert time.monotonic() - started > 0.2
    assert len(endpoint[1]) == 1


@pytest.mark.parametrize('name,stream', [('slow-head', True), ('slow-stream', True), ('slow-body', False)])
def test_cancel_ends_at_read_boundary_or_timeout_without_retry(endpoint, name, stream):
    cancel = threading.Event()

    def stop():
        if endpoint[2].wait(1): cancel.set()

    thread = threading.Thread(target=stop)
    thread.start()
    started = time.monotonic()
    try:
        with pytest.raises(asyncio.CancelledError):
            model(endpoint, name=name, stream=stream)(
                'hello', timeout=(0.2, 0.2), call_control=ModelCallControl(cancel_check=cancel.is_set))
    finally:
        thread.join(2)
    assert time.monotonic() - started < 1.5
    assert len(endpoint[1]) == 1


def test_retry_backoff_is_cancellable():
    cancel = threading.Event()
    attempts = []

    def fail(state):
        attempts.append(state)
        raise requests.ConnectTimeout('fixture timeout')

    def event(kind, data):
        if kind == 'model_retry_scheduled': cancel.set()

    runner = _ModelCallRunner(emit_event=event, control=ModelCallControl(cancel_check=cancel.is_set))
    with pytest.raises(asyncio.CancelledError):
        runner.run(fail, max_attempts=3)
    assert len(attempts) == 1


def test_cancel_wins_over_a_returning_network_error():
    cancel = threading.Event()
    events = []

    def fail(state):
        cancel.set()
        raise requests.ReadTimeout('fixture error after cancel')

    runner = _ModelCallRunner(emit_event=lambda kind, data: events.append(kind),
                              control=ModelCallControl(cancel_check=cancel.is_set))
    with pytest.raises(asyncio.CancelledError):
        runner.run(fail, max_attempts=3)
    assert 'model_retry_scheduled' not in events


def test_request_rejection_is_local_and_does_not_send_http(endpoint):
    from lazyllm.common import HandledException

    class Rejected(HandledException): pass

    def check(request): raise Rejected('fixture input rejected')

    with pytest.raises(Rejected):
        model(endpoint)('hello', call_control=ModelCallControl(request_check=check))
    assert endpoint[1] == []


def test_terminal_callback_failure_is_local(endpoint):
    def broken(terminal): raise RuntimeError('fixture callback failed')

    with pytest.raises(ModuleExecutionError, match='terminal callback failed'):
        model(endpoint)('hello', on_terminal=broken)
    assert len(endpoint[1]) == 1
