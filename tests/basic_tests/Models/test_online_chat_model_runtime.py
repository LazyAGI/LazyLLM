import json
import socket

import pytest
import requests

from lazyllm.module.llms.onlinemodule.base.model_call_runner import ModelAttemptState, ModelCallRunner
from lazyllm.module.llms.onlinemodule.base.model_outcome import (
    ModelCallFailed,
    ModelCallInterrupted,
    ModelFailure,
    ModelFailureCode,
    ModelFailureOrigin,
    ModelFinish,
    ModelResponseError,
)
from lazyllm.module.llms.onlinemodule.base.onlineChatModuleBase import LazyLLMOnlineChatModuleBase
from lazyllm.module.llms.onlinemodule.supplier.deepseek import DeepSeekChat
from lazyllm.module.llms.onlinemodule.supplier.minimax import MinimaxChat
from lazyllm.module.llms.onlinemodule.supplier.openai import OpenAIChat
from lazyllm.module.llms.onlinemodule.supplier.qwen import QwenChat


class _Response:
    def __init__(self, *, frames=None, status_code=200, body='', headers=None):
        self.frames = frames or []
        self.status_code = status_code
        self.text = body
        self.headers = headers or {}

    def __enter__(self): return self
    def __exit__(self, *args): return None

    def iter_lines(self):
        for frame in self.frames:
            if isinstance(frame, Exception): raise frame
            yield frame

    def json(self):
        return json.loads(self.text or '{}')


def _module(module_cls=LazyLLMOnlineChatModuleBase):
    module = module_cls.__new__(module_cls)
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


def test_deepseek_maps_insufficient_system_resource_finish_reason():
    assert _module(DeepSeekChat)._map_finish_reason(
        'insufficient_system_resource',
    ) is ModelFinish.INSUFFICIENT_SYSTEM_RESOURCE
    assert _module()._map_finish_reason('insufficient_system_resource') is ModelFinish.UNKNOWN


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

    with pytest.raises(ModelResponseError, match='invalid JSON frame') as exc_info:
        module._forward_impl(
            {'messages': []}, runtime_url='http://provider.test', stream_output=True,
            proxies=None, request_timeout=1, state=ModelAttemptState(),
        )
    assert exc_info.value.failure.code is ModelFailureCode.PROTOCOL_ERROR


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


@pytest.mark.parametrize(('errno', 'retryable'), [
    (socket.EAI_AGAIN, True),
    (socket.EAI_NONAME, False),
])
def test_transport_retry_only_accepts_temporary_dns_errors(errno, retryable):
    error = requests.ConnectionError('dns lookup failed')
    error.__cause__ = socket.gaierror(errno, 'dns lookup failed')

    assert _module()._is_retryable_transport_error(error) is retryable


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
    assert events[0][1]['failure']['code'] == 'transport_error'


def test_runner_does_not_retry_after_response_headers(monkeypatch):
    module = _module()
    events = []
    failures = []
    calls = 0

    def post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _Response(frames=[requests.exceptions.ChunkedEncodingError('cut after headers')])

    monkeypatch.setattr(requests, 'post', post)
    runner = ModelCallRunner(
        emit_event=lambda event_type, data: events.append((event_type, data)),
        is_retryable_transport_error=module._is_retryable_transport_error,
        report_failure=failures.append,
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
    assert events[0][1]['failure']['code'] == 'transport_error'
    assert len(failures) == 1
    assert failures[0].response_started is True


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
        return _Response(
            status_code=429,
            body='{"error":{"type":"rate_limit_error","code":"rate_limit_exceeded"}}',
            headers={'Retry-After': '2'},
        )

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
    assert events[0][1]['failure']['code'] == 'too_many_requests'
    assert 'provider_http_status' not in events[0][1]['failure']
    assert 'retry_after_ms' not in events[0][1]['failure']
    assert 'provider_error_code' not in events[0][1]['failure']
    assert 'provider_error_type' not in events[0][1]['failure']


def test_retry_after_accepts_http_date_and_rejects_invalid_value(monkeypatch):
    monkeypatch.setattr(
        'lazyllm.module.llms.onlinemodule.base.onlineChatModuleBase.time.time',
        lambda: 1445412478,
    )

    assert _module()._retry_after_ms({'Retry-After': 'Wed, 21 Oct 2015 07:28:00 GMT'}) == 2000
    assert _module()._retry_after_ms({'Retry-After': 'not-a-date'}) is None


@pytest.mark.parametrize(('module_cls', 'status', 'body', 'expected'), [
    (LazyLLMOnlineChatModuleBase, 400, '{"error":{}}', ModelFailureCode.INVALID_REQUEST),
    (LazyLLMOnlineChatModuleBase, 401, '{"error":{}}', ModelFailureCode.AUTHENTICATION_FAILED),
    (LazyLLMOnlineChatModuleBase, 403, '{"error":{}}', ModelFailureCode.PERMISSION_DENIED),
    (LazyLLMOnlineChatModuleBase, 404, '{"error":{}}', ModelFailureCode.NOT_FOUND),
    (LazyLLMOnlineChatModuleBase, 409, '{"error":{}}', ModelFailureCode.CONFLICT),
    (LazyLLMOnlineChatModuleBase, 422, '{"error":{}}', ModelFailureCode.UNPROCESSABLE_ENTITY),
    (LazyLLMOnlineChatModuleBase, 429, '{"error":{}}', ModelFailureCode.TOO_MANY_REQUESTS),
    (LazyLLMOnlineChatModuleBase, 500, '{"error":{}}', ModelFailureCode.PROVIDER_INTERNAL_ERROR),
    (LazyLLMOnlineChatModuleBase, 503, '{"error":{}}', ModelFailureCode.SERVICE_UNAVAILABLE),
    (LazyLLMOnlineChatModuleBase, 402, '{"error":{}}', ModelFailureCode.PROVIDER_REJECTED),
])
def test_provider_http_mapping_uses_supplier_source(monkeypatch, module_cls, status, body, expected):
    module = _module(module_cls)
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: _Response(
        status_code=status,
        body=body,
    ))

    with pytest.raises(ModelResponseError) as exc_info:
        module._forward_impl(
            {'messages': []}, runtime_url='http://provider.test', stream_output=True,
            proxies=None, request_timeout=1, state=ModelAttemptState(),
        )

    assert exc_info.value.failure.code is expected
    assert exc_info.value.failure.provider_http_status == status


@pytest.mark.parametrize(('provider_code', 'expected'), [
    ('credit_balance_exhausted', ModelFailureCode.BALANCE_EXHAUSTED),
    ('organization_spend_limit_exceeded', ModelFailureCode.ORGANIZATION_SPEND_LIMIT_EXCEEDED),
    ('project_spend_limit_exceeded', ModelFailureCode.PROJECT_SPEND_LIMIT_EXCEEDED),
    ('organization_usage_limit_exceeded', ModelFailureCode.ORGANIZATION_USAGE_LIMIT_EXCEEDED),
])
def test_openai_billing_codes_preserve_specific_reason(monkeypatch, provider_code, expected):
    module = _module(OpenAIChat)
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: _Response(
        status_code=429,
        body=json.dumps({'error': {'code': provider_code, 'type': 'insufficient_quota'}}),
    ))

    with pytest.raises(ModelResponseError) as exc_info:
        module._forward_impl(
            {'messages': []}, runtime_url='http://provider.test', stream_output=True,
            proxies=None, request_timeout=1, state=ModelAttemptState(),
        )

    assert exc_info.value.failure.code is expected


def test_openai_insufficient_quota_type_is_generic_fallback(monkeypatch):
    module = _module(OpenAIChat)
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: _Response(
        status_code=429,
        body='{"error":{"type":"insufficient_quota"}}',
    ))

    with pytest.raises(ModelResponseError) as exc_info:
        module._forward_impl(
            {'messages': []}, runtime_url='http://provider.test', stream_output=True,
            proxies=None, request_timeout=1, state=ModelAttemptState(),
        )

    assert exc_info.value.failure.code is ModelFailureCode.QUOTA_EXHAUSTED


@pytest.mark.parametrize(('status', 'body', 'expected'), [
    (429, '{"error":{"code":"rate_limit_exceeded","type":"rate_limit_error"}}',
     ModelFailureCode.TOO_MANY_REQUESTS),
    (401, '{"error":{"code":"invalid_api_key"}}', ModelFailureCode.AUTHENTICATION_FAILED),
    (400, '{"error":{"code":"content_policy_violation"}}', ModelFailureCode.INVALID_REQUEST),
])
def test_openai_undocumented_error_aliases_do_not_override_http(monkeypatch, status, body, expected):
    module = _module(OpenAIChat)
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: _Response(status_code=status, body=body))

    with pytest.raises(ModelResponseError) as exc_info:
        module._forward_impl(
            {'messages': []}, runtime_url='http://provider.test', stream_output=True,
            proxies=None, request_timeout=1, state=ModelAttemptState(),
        )

    assert exc_info.value.failure.code is expected


def test_failure_public_dict_excludes_provider_diagnostics():
    failure = ModelFailure(
        origin=ModelFailureOrigin.HTTP,
        code=ModelFailureCode.TOO_MANY_REQUESTS,
        provider_error_code='private-code',
        provider_error_type='private-type',
        provider_http_status=429,
        retry_after_ms=2000,
        diagnostic_id='diag-safe',
        response_started=True,
    )

    assert failure.public_dict() == {
        'origin': 'http',
        'code': 'too_many_requests',
        'has_semantic_output': False,
        'diagnostic_id': 'diag-safe',
    }


def test_minimax_http_200_base_resp_is_provider_failure():
    module = _module(MinimaxChat)

    with pytest.raises(ModelResponseError) as exc_info:
        module._str_to_json(json.dumps({
            'base_resp': {'status_code': 2056, 'status_msg': 'must stay private'},
        }), stream_output=False)

    failure = exc_info.value.failure
    assert failure.origin is ModelFailureOrigin.PROVIDER
    assert failure.code is ModelFailureCode.QUOTA_EXHAUSTED
    assert failure.provider_http_status is None
    assert 'status_msg' not in failure.public_dict()


def test_qwen_top_level_data_inspection_error_uses_provider_mapping():
    module = _module(QwenChat)

    with pytest.raises(ModelResponseError) as exc_info:
        module._str_to_json(json.dumps({
            'code': 'DataInspectionFailed',
            'type': 'data_inspection_failed',
            'message': 'must stay private',
        }), stream_output=False)

    failure = exc_info.value.failure
    assert failure.origin is ModelFailureOrigin.PROVIDER
    assert failure.code is ModelFailureCode.INPUT_FILTERED
    assert failure.provider_error_code == 'DataInspectionFailed'
    assert 'message' not in failure.public_dict()


@pytest.mark.parametrize(('module_cls', 'expected'), [
    (LazyLLMOnlineChatModuleBase, ModelFailureCode.TOO_MANY_REQUESTS),
    (QwenChat, ModelFailureCode.RATE_LIMITED),
])
def test_provider_profiles_isolate_bare_http_429(monkeypatch, module_cls, expected):
    module = _module(module_cls)
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: _Response(
        status_code=429,
        body='{"error":{}}',
    ))

    with pytest.raises(ModelResponseError) as exc_info:
        module._forward_impl(
            {'messages': []}, runtime_url='http://provider.test', stream_output=True,
            proxies=None, request_timeout=1, state=ModelAttemptState(),
        )

    assert exc_info.value.failure.code is expected


def test_minimax_stream_business_error_preserves_partial_output(monkeypatch):
    module = _module(MinimaxChat)
    events = []
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: _Response(frames=[
        _frame({'role': 'assistant', 'content': 'partial'}),
        ('data: ' + json.dumps({
            'base_resp': {'status_code': 1027, 'status_msg': 'private output detail'},
        })).encode(),
    ]))
    runner = ModelCallRunner(
        emit_event=lambda event_type, data: events.append((event_type, data)),
        is_retryable_transport_error=module._is_retryable_transport_error,
        sleep=lambda delay: None,
    )

    with pytest.raises(ModelCallInterrupted):
        runner.run(
            lambda state: module._forward_impl(
                {'messages': []}, runtime_url='http://provider.test', stream_output=True,
                proxies=None, request_timeout=1, state=state,
            ),
            max_attempts=3,
        )

    terminal = events[0][1]
    assert terminal['has_semantic_output'] is True
    assert terminal['failure']['origin'] == 'provider'
    assert terminal['failure']['code'] == 'output_filtered'
