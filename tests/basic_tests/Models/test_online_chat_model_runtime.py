import json
import socket

import pytest
import requests

import lazyllm
from lazyllm.module.llms.onlinemodule.base.model_call_runner import (
    ModelAttemptState,
    _ModelCallRunner,
    is_retryable_transport_error,
)
from lazyllm.module.llms.onlinemodule.base.model_outcome import (
    ModelCallError,
    ModelFailure,
    ModelFailureCode,
    ModelFailureOrigin,
    ModelFinish,
    _ModelResponseError,
)
from lazyllm.module.llms.onlinemodule.base.onlineChatModuleBase import LazyLLMOnlineChatModuleBase
from lazyllm.module.llms.onlinemodule.base.provider_response import (
    OPENAI_COMPATIBLE_PROFILE,
    raise_for_http_error,
    usage_from_frames,
)
from lazyllm.module.llms.onlinemodule.supplier.claude import ClaudeChat
from lazyllm.module.llms.onlinemodule.supplier.deepseek import DeepSeekChat
from lazyllm.module.llms.onlinemodule.supplier.glm import GLMChat
from lazyllm.module.llms.onlinemodule.supplier.minimax import MinimaxChat
from lazyllm.module.llms.onlinemodule.supplier.openai import OpenAIChat
from lazyllm.module.llms.onlinemodule.supplier.qwen import QwenChat
from lazyllm.module.llms.onlinemodule.supplier.siliconflow import SiliconFlowChat


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


def _parser(module_cls=LazyLLMOnlineChatModuleBase):
    return _module(module_cls)._response_parser(False)


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
    assert LazyLLMOnlineChatModuleBase.RESPONSE_PROFILE.map_finish(raw) is expected


def test_deepseek_maps_insufficient_system_resource_finish_reason():
    assert DeepSeekChat.RESPONSE_PROFILE.map_finish(
        'insufficient_system_resource',
    ) is ModelFinish.INSUFFICIENT_SYSTEM_RESOURCE
    assert LazyLLMOnlineChatModuleBase.RESPONSE_PROFILE.map_finish(
        'insufficient_system_resource',
    ) is ModelFinish.UNKNOWN


def test_glm_maps_sensitive_finish_reason():
    assert GLMChat.RESPONSE_PROFILE.map_finish('sensitive') is ModelFinish.CONTENT_FILTER
    assert LazyLLMOnlineChatModuleBase.RESPONSE_PROFILE.map_finish('sensitive') is ModelFinish.UNKNOWN


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

    with pytest.raises(_ModelResponseError, match='invalid JSON frame') as exc_info:
        module._forward_impl(
            {'messages': []}, runtime_url='http://provider.test', stream_output=True,
            proxies=None, request_timeout=1, state=ModelAttemptState(),
        )
    assert exc_info.value.failure.code is ModelFailureCode.PROTOCOL_ERROR


def test_response_frame_and_json_payload_have_distinct_boundaries():
    parser = _parser()
    payload = json.dumps({'choices': []})

    assert parser.parse_response_frame(f'data: {payload}') == {'choices': []}
    with pytest.raises(_ModelResponseError, match='invalid JSON frame'):
        parser.parse_json_payload(f'data: {payload}')


@pytest.mark.parametrize('late_frame', [
    b'data: not-json',
    b'data: {"error":{"code":"late_provider_error"}}',
])
def test_protocol_or_provider_error_after_finish_is_not_swallowed(monkeypatch, late_frame):
    module = _module()
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: _Response(frames=[
        _frame({'role': 'assistant', 'content': 'ok'}),
        _frame({}, 'stop'),
        late_frame,
    ]))

    with pytest.raises(_ModelResponseError):
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
    _parser().update_attempt_state({
        'choices': [{
            'message': {'function_call': {'name': 'lookup', 'arguments': '{'}},
            'finish_reason': 'function_call',
        }],
    }, state)

    assert state.semantic_output is True
    assert state.finish is ModelFinish.TOOL_CALLS


@pytest.mark.parametrize(('choices', 'expected'), [
    ([
        {'index': 0, 'message': {'content': 'complete'}, 'finish_reason': 'stop'},
        {'index': 1, 'message': {'content': 'partial'}, 'finish_reason': 'length'},
    ], ModelFinish.STOP),
    ([
        {'index': 1, 'message': {'content': 'complete'}, 'finish_reason': 'stop'},
        {'index': 0, 'message': {'content': 'partial'}, 'finish_reason': 'length'},
    ], ModelFinish.LENGTH),
])
def test_attempt_terminal_tracks_primary_choice(choices, expected):
    module = _module()
    state = ModelAttemptState()

    _parser().update_attempt_state({'choices': choices}, state)

    assert state.finish is expected
    assert module._extract_specified_key_fields({'choices': choices})['content'] \
        == next(choice for choice in choices if choice['index'] == 0)['message']['content']


def test_attempt_ignores_frame_for_non_primary_choice():
    state = ModelAttemptState(finish=ModelFinish.STOP)

    _parser().update_attempt_state({'choices': [{
        'index': 1,
        'message': {'content': 'secondary partial'},
        'finish_reason': 'length',
    }]}, state)

    assert state.finish is ModelFinish.STOP


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
    runner = _ModelCallRunner(
        emit_event=lambda event_type, data: events.append((event_type, data)),
        is_retryable_transport_error=is_retryable_transport_error,
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
        return [{'choices': [{'delta': {'content': 'ok'}, 'finish_reason': 'stop'}]}]

    runner = _ModelCallRunner(
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

    assert is_retryable_transport_error(error) is retryable


def test_runner_does_not_retry_after_semantic_output():
    events = []
    attempts = 0

    def execute(state):
        nonlocal attempts
        attempts += 1
        state.semantic_output = True
        raise requests.exceptions.ChunkedEncodingError('cut')

    runner = _ModelCallRunner(
        emit_event=lambda event_type, data: events.append((event_type, data)),
        is_retryable_transport_error=lambda exc: True,
        sleep=lambda delay: None,
    )
    with pytest.raises(ModelCallError):
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
    runner = _ModelCallRunner(
        emit_event=lambda event_type, data: events.append((event_type, data)),
        is_retryable_transport_error=is_retryable_transport_error,
        report_failure=failures.append,
        sleep=lambda delay: None,
    )
    with pytest.raises(ModelCallError):
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
        return []

    runner = _ModelCallRunner(
        emit_event=lambda event_type, data: events.append((event_type, data)),
        is_retryable_transport_error=lambda exc: False,
        sleep=lambda delay: None,
    )
    with pytest.raises(ModelCallError):
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
    runner = _ModelCallRunner(
        emit_event=lambda event_type, data: events.append((event_type, data)),
        is_retryable_transport_error=is_retryable_transport_error,
        sleep=lambda delay: None,
    )
    with pytest.raises(ModelCallError):
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
    assert events[0][1]['failure']['code'] == 'rate_limited'
    assert 'provider_http_status' not in events[0][1]['failure']
    assert 'retry_after_ms' not in events[0][1]['failure']
    assert 'provider_error_code' not in events[0][1]['failure']
    assert 'provider_error_type' not in events[0][1]['failure']


def test_http_status_survives_error_body_transport_failure():
    module = _module()
    response = _Response(status_code=429)

    def broken_json():
        raise requests.exceptions.ChunkedEncodingError('truncated error body')

    response.json = broken_json
    with pytest.raises(_ModelResponseError) as exc_info:
        raise_for_http_error(response, module.RESPONSE_PROFILE)

    assert exc_info.value.failure.provider_http_status == 429
    assert exc_info.value.failure.code is ModelFailureCode.RATE_LIMITED


@pytest.mark.parametrize(('module_cls', 'status', 'body', 'expected'), [
    (LazyLLMOnlineChatModuleBase, 400, '{"error":{}}', ModelFailureCode.INVALID_REQUEST),
    (LazyLLMOnlineChatModuleBase, 401, '{"error":{}}', ModelFailureCode.AUTHENTICATION_FAILED),
    (LazyLLMOnlineChatModuleBase, 403, '{"error":{}}', ModelFailureCode.PERMISSION_DENIED),
    (LazyLLMOnlineChatModuleBase, 404, '{"error":{}}', ModelFailureCode.NOT_FOUND),
    (LazyLLMOnlineChatModuleBase, 409, '{"error":{}}', ModelFailureCode.CONFLICT),
    (LazyLLMOnlineChatModuleBase, 422, '{"error":{}}', ModelFailureCode.UNPROCESSABLE_ENTITY),
    (LazyLLMOnlineChatModuleBase, 429, '{"error":{}}', ModelFailureCode.RATE_LIMITED),
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

    with pytest.raises(_ModelResponseError) as exc_info:
        module._forward_impl(
            {'messages': []}, runtime_url='http://provider.test', stream_output=True,
            proxies=None, request_timeout=1, state=ModelAttemptState(),
        )

    assert exc_info.value.failure.code is expected
    assert exc_info.value.failure.provider_http_status == status


def test_deepseek_http_402_is_balance_exhausted(monkeypatch):
    module = _module(DeepSeekChat)
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: _Response(
        status_code=402,
        body='{"error":{}}',
    ))

    with pytest.raises(_ModelResponseError) as exc_info:
        module._forward_impl(
            {'messages': []}, runtime_url='http://provider.test', stream_output=True,
            proxies=None, request_timeout=1, state=ModelAttemptState(),
        )

    assert exc_info.value.failure.code is ModelFailureCode.BALANCE_EXHAUSTED


@pytest.mark.parametrize(('provider_code', 'expected'), [
    ('credit_balance_exhausted', ModelFailureCode.BALANCE_EXHAUSTED),
    ('organization_spend_limit_exceeded', ModelFailureCode.ORGANIZATION_SPEND_LIMIT_EXCEEDED),
    ('project_spend_limit_exceeded', ModelFailureCode.PROJECT_SPEND_LIMIT_EXCEEDED),
    ('organization_usage_limit_exceeded', ModelFailureCode.USAGE_LIMIT_EXCEEDED),
])
def test_openai_billing_codes_preserve_specific_reason(monkeypatch, provider_code, expected):
    module = _module(OpenAIChat)
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: _Response(
        status_code=429,
        body=json.dumps({'error': {'code': provider_code, 'type': 'insufficient_quota'}}),
    ))

    with pytest.raises(_ModelResponseError) as exc_info:
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

    with pytest.raises(_ModelResponseError) as exc_info:
        module._forward_impl(
            {'messages': []}, runtime_url='http://provider.test', stream_output=True,
            proxies=None, request_timeout=1, state=ModelAttemptState(),
        )

    assert exc_info.value.failure.code is ModelFailureCode.QUOTA_EXHAUSTED


@pytest.mark.parametrize(('status', 'body', 'expected'), [
    (429, '{"error":{"code":"rate_limit_exceeded","type":"rate_limit_error"}}',
     ModelFailureCode.RATE_LIMITED),
    (401, '{"error":{"code":"invalid_api_key"}}', ModelFailureCode.AUTHENTICATION_FAILED),
    (400, '{"error":{"code":"content_policy_violation"}}', ModelFailureCode.INVALID_REQUEST),
])
def test_openai_undocumented_error_aliases_do_not_override_http(monkeypatch, status, body, expected):
    module = _module(OpenAIChat)
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: _Response(status_code=status, body=body))

    with pytest.raises(_ModelResponseError) as exc_info:
        module._forward_impl(
            {'messages': []}, runtime_url='http://provider.test', stream_output=True,
            proxies=None, request_timeout=1, state=ModelAttemptState(),
        )

    assert exc_info.value.failure.code is expected


def test_failure_public_dict_excludes_provider_diagnostics():
    failure = ModelFailure(
        origin=ModelFailureOrigin.HTTP,
        code=ModelFailureCode.RATE_LIMITED,
        provider_error_code='private-code',
        provider_error_type='private-type',
        provider_http_status=429,
        diagnostic_id='diag-safe',
        response_started=True,
    )

    assert failure.public_dict() == {
        'origin': 'http',
        'code': 'rate_limited',
        'diagnostic_id': 'diag-safe',
    }


def test_minimax_http_200_base_resp_is_provider_failure():
    parser = _parser(MinimaxChat)

    with pytest.raises(_ModelResponseError) as exc_info:
        parser.parse_json_payload(json.dumps({
            'base_resp': {'status_code': 2056, 'status_msg': 'must stay private'},
        }))

    failure = exc_info.value.failure
    assert failure.origin is ModelFailureOrigin.PROVIDER
    assert failure.code is ModelFailureCode.QUOTA_EXHAUSTED
    assert failure.provider_http_status is None
    assert 'status_msg' not in failure.public_dict()


def test_qwen_data_inspection_without_phase_uses_generic_failure(monkeypatch):
    module = _module(QwenChat)
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: _Response(
        status_code=400,
        body=json.dumps({'error': {
            'code': 'DataInspectionFailed',
            'type': 'data_inspection_failed',
            'message': 'must stay private',
        }}),
    ))

    with pytest.raises(_ModelResponseError) as exc_info:
        module._forward_impl(
            {'messages': []}, runtime_url='http://provider.test', stream_output=False,
            proxies=None, request_timeout=1, state=ModelAttemptState(),
        )

    failure = exc_info.value.failure
    assert failure.origin is ModelFailureOrigin.HTTP
    assert failure.code is ModelFailureCode.PROVIDER_REJECTED
    assert failure.provider_error_code == 'DataInspectionFailed'
    assert 'message' not in failure.public_dict()


def test_siliconflow_top_level_model_not_found_uses_provider_mapping():
    parser = _parser(SiliconFlowChat)

    with pytest.raises(_ModelResponseError) as exc_info:
        parser.parse_json_payload(json.dumps({
            'code': 20012,
            'message': 'must stay private',
            'data': None,
        }))

    failure = exc_info.value.failure
    assert failure.origin is ModelFailureOrigin.PROVIDER
    assert failure.code is ModelFailureCode.NOT_FOUND
    assert failure.provider_error_code == '20012'
    assert 'message' not in failure.public_dict()


def test_bare_http_429_is_generic_rate_limit(monkeypatch):
    module = _module()
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: _Response(
        status_code=429,
        body='{"error":{}}',
    ))

    with pytest.raises(_ModelResponseError) as exc_info:
        module._forward_impl(
            {'messages': []}, runtime_url='http://provider.test', stream_output=True,
            proxies=None, request_timeout=1, state=ModelAttemptState(),
        )

    assert exc_info.value.failure.code is ModelFailureCode.RATE_LIMITED


@pytest.mark.parametrize(('provider_code', 'expected'), [
    ('Throttling.RateQuota', ModelFailureCode.RATE_LIMITED),
    ('Throttling.BurstRate', ModelFailureCode.RATE_LIMITED),
    ('Throttling.Concurrency', ModelFailureCode.CONCURRENCY_LIMITED),
])
def test_qwen_throttling_codes_preserve_normalized_semantics(provider_code, expected):
    failure = QwenChat.RESPONSE_PROFILE.error(
        'Provider rejected request.',
        ModelFailureOrigin.PROVIDER,
        provider_error_code=provider_code,
    ).failure

    assert failure.code is expected


def test_provider_profiles_are_immutable_and_isolated():
    child = OPENAI_COMPATIBLE_PROFILE.extend(
        code_map={'MixedCase': ModelFailureCode.CONCURRENCY_LIMITED},
        http_map={429: ModelFailureCode.CONCURRENCY_LIMITED},
        finish_map={'custom_stop': ModelFinish.STOP},
        error_at_top_level=True,
    )

    assert child.code_map['mixedcase'] is ModelFailureCode.CONCURRENCY_LIMITED
    assert child.http_map[429] is ModelFailureCode.CONCURRENCY_LIMITED
    assert child.map_finish('custom_stop') is ModelFinish.STOP
    assert child.error_at_top_level is True
    assert 'mixedcase' not in OPENAI_COMPATIBLE_PROFILE.code_map
    assert OPENAI_COMPATIBLE_PROFILE.http_map[429] is ModelFailureCode.RATE_LIMITED
    assert OPENAI_COMPATIBLE_PROFILE.map_finish('custom_stop') is ModelFinish.UNKNOWN
    assert OPENAI_COMPATIBLE_PROFILE.error_at_top_level is False
    with pytest.raises(TypeError):
        child.code_map['another'] = ModelFailureCode.RATE_LIMITED


def test_provider_profile_declaration_is_reentrant():
    first = LazyLLMOnlineChatModuleBase.RESPONSE_PROFILE.extend(
        http_map={402: ModelFailureCode.BALANCE_EXHAUSTED},
    )
    second = LazyLLMOnlineChatModuleBase.RESPONSE_PROFILE.extend(
        http_map={402: ModelFailureCode.BALANCE_EXHAUSTED},
    )

    assert first == second
    assert first is not second


def test_minimax_stream_business_error_preserves_partial_output(monkeypatch):
    module = _module(MinimaxChat)
    events = []
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: _Response(frames=[
        _frame({'role': 'assistant', 'content': 'partial'}),
        ('data: ' + json.dumps({
            'base_resp': {'status_code': 1027, 'status_msg': 'private output detail'},
        })).encode(),
    ]))
    runner = _ModelCallRunner(
        emit_event=lambda event_type, data: events.append((event_type, data)),
        is_retryable_transport_error=is_retryable_transport_error,
        sleep=lambda delay: None,
    )

    with pytest.raises(ModelCallError) as exc_info:
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
    assert exc_info.value.partial_response[0]['choices'][0]['delta']['content'] == 'partial'


def test_non_stream_interruption_survives_module_boundary_with_partial_and_usage(monkeypatch):
    module = OpenAIChat(
        base_url='http://provider.test/v1/', model='test-model', api_key='',
        stream=False, skip_auth=True,
    )

    def post(*args, **kwargs):
        assert kwargs['stream'] is True
        return _Response(body=json.dumps({
            'choices': [{
                'index': 0,
                'message': {'role': 'assistant', 'content': 'partial answer'},
                'finish_reason': 'length',
            }],
            'usage': {'prompt_tokens': 11, 'completion_tokens': 7},
        }))

    monkeypatch.setattr(requests, 'post', post)
    with pytest.raises(ModelCallError) as exc_info:
        module('hello', max_retries=1)

    error = exc_info.value
    assert error.terminal.finish is ModelFinish.LENGTH
    assert error.partial_response[0]['choices'][0]['message']['content'] == 'partial answer'
    assert error.usage['prompt_tokens'] == 11
    assert error.usage['completion_tokens'] == 7
    assert error.usage['provider_usage'] == {'prompt_tokens': 11, 'completion_tokens': 7}
    assert lazyllm.globals['usage'][module._module_id]['prompt_tokens'] == 11
    assert lazyllm.globals['usage'][module._module_id]['provider_usages'] == [
        {'prompt_tokens': 11, 'completion_tokens': 7},
    ]


def test_claude_keeps_pre_response_transport_retry(monkeypatch):
    module = _module(ClaudeChat)
    calls = 0

    def post(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise requests.ConnectTimeout('connect timeout')
        return _Response(body=json.dumps({
            'type': 'message',
            'content': [{'type': 'text', 'text': 'ok'}],
            'usage': {'input_tokens': 1, 'output_tokens': 1},
        }))

    monkeypatch.setattr(requests, 'post', post)
    monkeypatch.setattr(
        'lazyllm.module.llms.onlinemodule.base.onlineChatModuleBase.time.sleep',
        lambda delay: None,
    )

    result = module._forward_with_retry(
        {'messages': [], 'stream': False}, runtime_url='http://provider.test',
        stream_output=False, proxies=None, max_retries=2, request_timeout=1,
    )

    assert calls == 2
    assert result[0]['choices'][0]['message']['content'] == 'ok'


def test_usage_from_frames_scans_backwards():
    frames = [
        {'choices': [{'delta': {'content': 'hi'}}]},
        {'usage': {'prompt_tokens': 3, 'completion_tokens': 1}},
        {'choices': []},
    ]
    assert usage_from_frames(frames) == {'prompt_tokens': 3, 'completion_tokens': 1}


def test_record_usage_accumulates_repeated_calls_on_same_module():
    module = OpenAIChat(
        base_url='http://provider.test/v1/', model='test-model', api_key='',
        stream=False, skip_auth=True,
    )
    module._record_usage({
        'prompt_tokens': 100,
        'completion_tokens': 10,
        'provider_usage': {
            'prompt_tokens': 100,
            'completion_tokens': 10,
            'prompt_tokens_details': {'cached_tokens': 80},
        },
    })
    module._record_usage({
        'prompt_tokens': 50,
        'completion_tokens': 20,
        'provider_usage': {
            'prompt_tokens': 50,
            'completion_tokens': 20,
            'prompt_tokens_details': {'cached_tokens': 0},
        },
    })
    recorded = lazyllm.globals['usage'][module._module_id]
    assert recorded['prompt_tokens'] == 150
    assert recorded['completion_tokens'] == 30
    assert recorded['provider_usages'] == [
        {
            'prompt_tokens': 100,
            'completion_tokens': 10,
            'prompt_tokens_details': {'cached_tokens': 80},
        },
        {
            'prompt_tokens': 50,
            'completion_tokens': 20,
            'prompt_tokens_details': {'cached_tokens': 0},
        },
    ]
    assert 'provider_usage' not in recorded
