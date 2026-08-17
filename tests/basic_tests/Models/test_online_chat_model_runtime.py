import json

import pytest
import requests

from lazyllm.module.llms.onlinemodule.base.model_call_runner import ModelAttemptState, ModelCallRunner
from lazyllm.module.llms.onlinemodule.base.model_outcome import (
    ModelCallFailed,
    ModelCallInterrupted,
    ModelFinish,
)
from lazyllm.module.llms.onlinemodule.base.onlineChatModuleBase import LazyLLMOnlineChatModuleBase


class _Response:
    def __init__(self, *, frames=None, status_code=200, body=''):
        self.frames = frames or []
        self.status_code = status_code
        self.text = body

    def __enter__(self): return self
    def __exit__(self, *args): return None

    def iter_lines(self):
        for frame in self.frames:
            if isinstance(frame, Exception): raise frame
            yield frame

    def json(self):
        return json.loads(self.text or '{}')


def _module():
    module = LazyLLMOnlineChatModuleBase.__new__(LazyLLMOnlineChatModuleBase)
    module._stream_sink = None
    module._dynamic_auth = False
    module._LazyLLMOnlineBase__headers = [{}]
    module._LazyLLMOnlineBase__api_keys = ''
    return module


def _frame(delta, finish_reason=None):
    return ('data: ' + json.dumps({
        'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish_reason}],
    })).encode()


@pytest.mark.parametrize(('raw', 'expected'), [
    ('stop', ModelFinish.STOP),
    ('tool_calls', ModelFinish.TOOL_CALLS),
    ('function_call', ModelFinish.TOOL_CALLS),
    ('length', ModelFinish.LENGTH),
    ('content_filter', ModelFinish.CONTENT_FILTER),
    ('provider_extension', ModelFinish.UNKNOWN),
])
def test_finish_reason_mapping_is_limited_to_openai_values(raw, expected):
    assert _module()._map_finish_reason(raw) is expected


def test_strict_stream_requires_finish_reason(monkeypatch):
    module = _module()
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: _Response(frames=[
        _frame({'role': 'assistant', 'content': 'partial'}),
        b'data: [DONE]',
    ]))
    state = ModelAttemptState()

    with pytest.raises(Exception, match='without a finish_reason'):
        module._forward_impl(
            {'messages': []}, runtime_url='http://provider.test', stream_output=True,
            proxies=None, request_timeout=1, state=state,
        )

    assert state.semantic_output is True


def test_malformed_json_is_protocol_failure(monkeypatch):
    module = _module()
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: _Response(frames=[b'data: not-json']))

    with pytest.raises(Exception, match='invalid JSON frame'):
        module._forward_impl(
            {'messages': []}, runtime_url='http://provider.test', stream_output=True,
            proxies=None, request_timeout=1, state=ModelAttemptState(),
        )


def test_non_stream_response_requires_finish_reason(monkeypatch):
    module = _module()
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: _Response(body=json.dumps({
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'partial'}}],
    })))

    with pytest.raises(Exception, match='without a finish_reason'):
        module._forward_impl(
            {'messages': []}, runtime_url='http://provider.test', stream_output=False,
            proxies=None, request_timeout=1, state=ModelAttemptState(),
        )


def test_deprecated_function_call_fragment_is_semantic_output():
    state = ModelAttemptState()
    _module()._update_attempt_state({
        'choices': [{
            'message': {'function_call': {'name': 'lookup', 'arguments': '{'}},
            'finish_reason': 'function_call',
        }],
    }, state)

    assert state.semantic_output is True
    assert state.finish is ModelFinish.TOOL_CALLS


def test_transport_error_after_finish_reason_keeps_terminal(monkeypatch):
    module = _module()
    calls = 0

    def post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _Response(frames=[
            _frame({'role': 'assistant', 'content': 'ok'}),
            _frame({}, 'stop'),
            requests.exceptions.ChunkedEncodingError('late disconnect'),
        ])

    monkeypatch.setattr(requests, 'post', post)
    events = []
    runner = ModelCallRunner(
        emit_event=lambda event_type, data: events.append((event_type, data)),
        is_retryable_transport_error=module._is_retryable_transport_error,
        sleep=lambda delay: None,
    )
    result = runner.run(
        lambda state: module._forward_impl(
            {'messages': []}, runtime_url='http://provider.test', stream_output=True,
            proxies=None, request_timeout=1, state=state,
        ),
        max_attempts=3,
    )

    assert result
    assert calls == 1
    assert events[-1][0] == 'model_call_finished'
    assert events[-1][1]['finish'] == 'stop'


def test_runner_retries_only_pre_output_transport_failures():
    events = []
    attempts = 0

    def execute(state):
        nonlocal attempts
        attempts += 1
        if attempts < 3: raise requests.ConnectTimeout('timeout')
        state.semantic_output = True
        state.finish = ModelFinish.STOP
        state.raw_finish_reason = 'stop'
        return [{'choices': [{'delta': {'content': 'ok'}, 'finish_reason': 'stop'}]}]

    runner = ModelCallRunner(
        emit_event=lambda event_type, data: events.append((event_type, data)),
        is_retryable_transport_error=lambda exc: isinstance(exc, requests.ConnectTimeout),
        sleep=lambda delay: None,
    )
    result = runner.run(execute, max_attempts=3)

    assert result
    assert attempts == 3
    assert [event[0] for event in events] == [
        'model_retry_scheduled', 'model_retry_scheduled', 'model_call_finished',
    ]
    assert len({event[1]['model_call_id'] for event in events}) == 1


def test_runner_does_not_retry_after_semantic_output():
    events = []
    attempts = 0

    def execute(state):
        nonlocal attempts
        attempts += 1
        state.semantic_output = True
        raise requests.exceptions.ChunkedEncodingError('cut')

    runner = ModelCallRunner(
        emit_event=lambda event_type, data: events.append((event_type, data)),
        is_retryable_transport_error=lambda exc: True,
        sleep=lambda delay: None,
    )
    with pytest.raises(ModelCallInterrupted):
        runner.run(execute, max_attempts=3)

    assert attempts == 1
    assert [event[0] for event in events] == ['model_call_finished']
    assert events[0][1]['failure']['origin'] == 'transport'


def test_unknown_finish_interrupts_after_one_terminal():
    events = []

    def execute(state):
        state.finish = ModelFinish.UNKNOWN
        state.raw_finish_reason = 'provider_extension'
        return []

    runner = ModelCallRunner(
        emit_event=lambda event_type, data: events.append((event_type, data)),
        is_retryable_transport_error=lambda exc: False,
        sleep=lambda delay: None,
    )
    with pytest.raises(ModelCallInterrupted):
        runner.run(execute, max_attempts=3)

    assert [event[0] for event in events] == ['model_call_finished']
    assert events[0][1]['finish'] == 'unknown'


def test_http_failure_is_not_retried(monkeypatch):
    module = _module()
    events = []
    calls = 0

    def post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _Response(status_code=429, body='{"error":{"code":"rate_limit_exceeded"}}')

    monkeypatch.setattr(requests, 'post', post)
    runner = ModelCallRunner(
        emit_event=lambda event_type, data: events.append((event_type, data)),
        is_retryable_transport_error=module._is_retryable_transport_error,
        sleep=lambda delay: None,
    )
    with pytest.raises(ModelCallFailed):
        runner.run(
            lambda state: module._forward_impl(
                {'messages': []}, runtime_url='http://provider.test', stream_output=True,
                proxies=None, request_timeout=1, state=state,
            ),
            max_attempts=3,
        )

    assert calls == 1
    assert [event[0] for event in events] == ['model_call_finished']
    assert events[0][1]['failure']['origin'] == 'http'
    assert 'provider_error_code' not in events[0][1]['failure']
