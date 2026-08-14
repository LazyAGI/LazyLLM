import json

import pytest
import requests

from lazyllm.module.llms.onlinemodule.base.onlineChatModuleBase import (
    LazyLLMOnlineChatModuleBase,
    _ProviderHTTPError,
    _ProviderProtocolError,
)
from lazyllm.module.llms.onlinemodule.supplier.claude import ClaudeChat


class _TestChat(LazyLLMOnlineChatModuleBase):
    def _get_system_prompt(self):
        return 'test'


class _Response:
    def __init__(self, status_code=200, frames=(), text=None):
        self.status_code = status_code
        self._frames = frames
        self.text = text if text is not None else ''

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def iter_lines(self):
        for frame in self._frames:
            if isinstance(frame, Exception): raise frame
            yield frame


def _module():
    return _TestChat(api_key='', base_url='http://provider.test/v1/', model_name='test',
                     stream=True, skip_auth=True)


def _stream_frame(delta=None, finish_reason=None, usage=None):
    choice = {'index': 0, 'delta': delta or {}, 'finish_reason': finish_reason}
    payload = {'choices': [choice]}
    if usage is not None: payload['usage'] = usage
    return f'data: {json.dumps(payload)}'.encode()


def _call(module, monkeypatch, responses, *, max_retries=1):
    events = []
    iterator = iter(responses)

    def post(*args, **kwargs):
        response = next(iterator)
        if isinstance(response, Exception): raise response
        return response

    monkeypatch.setattr(requests, 'post', post)
    result = module._forward_with_retry(
        {'messages': []}, runtime_url='http://provider.test/v1/chat/completions',
        stream_output={'_stream_sink': events.append}, proxies=None, max_retries=max_retries,
        request_timeout=1,
    )
    return result, events


@pytest.mark.parametrize('finish_reason', ['stop', 'tool_calls', 'length', 'content_filter', 'VendorLimit'])
def test_finish_reason_is_forwarded_without_normalization(monkeypatch, finish_reason):
    module = _module()
    _, events = _call(module, monkeypatch, [
        _Response(frames=[_stream_frame({'content': 'answer'}, finish_reason), b'data: [DONE]']),
    ])

    statuses = [event for event in events if event['tag'] == 'provider_status']
    assert statuses == [{
        'tag': 'provider_status',
        'model_call_id': statuses[0]['model_call_id'],
        'http_status': 200,
        'finish_reason': finish_reason,
    }]


@pytest.mark.parametrize('status_code,error_body', [
    (429, '{"error":{"message":"rate limited"}}'),
    (503, 'temporarily unavailable'),
])
def test_non_200_body_is_forwarded_once_and_never_retried(monkeypatch, status_code, error_body):
    module = _module()
    events = []
    calls = 0

    def post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _Response(status_code=status_code, text=error_body)

    monkeypatch.setattr(requests, 'post', post)
    with pytest.raises(_ProviderHTTPError) as exc_info:
        module._forward_with_retry(
            {'messages': []}, runtime_url='http://provider.test',
            stream_output={'_stream_sink': events.append}, proxies=None, max_retries=6, request_timeout=1,
        )

    assert calls == 1
    assert error_body not in str(exc_info.value)
    assert [event['tag'] for event in events] == ['provider_status']
    assert events[0]['http_status'] == status_code
    assert events[0]['error_body'] == error_body


def test_http_200_error_frame_preserves_payload(monkeypatch):
    module = _module()
    payload = '{"error":{"code":"bad_request","message":"invalid"}}'
    events = []
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: _Response(frames=[f'data: {payload}'.encode()]))

    with pytest.raises(_ProviderProtocolError) as exc_info:
        module._forward_with_retry(
            {'messages': []}, runtime_url='http://provider.test',
            stream_output={'_stream_sink': events.append}, proxies=None, max_retries=6, request_timeout=1,
        )

    assert payload not in str(exc_info.value)
    assert events[-1] == {
        'tag': 'provider_status',
        'model_call_id': events[-1]['model_call_id'],
        'http_status': 200,
        'finish_reason': None,
        'error_body': payload,
    }


@pytest.mark.parametrize('frame,error_body', [
    (b'data: not-json', 'not-json'),
    (b'data: [DONE]', '[DONE]'),
])
def test_protocol_errors_are_forwarded_and_not_retried(monkeypatch, frame, error_body):
    module = _module()
    events = []
    calls = 0

    def post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _Response(frames=[frame])

    monkeypatch.setattr(requests, 'post', post)
    with pytest.raises(_ProviderProtocolError):
        module._forward_with_retry(
            {'messages': []}, runtime_url='http://provider.test',
            stream_output={'_stream_sink': events.append}, proxies=None, max_retries=6, request_timeout=1,
        )

    assert calls == 1
    assert events[-1]['error_body'] == error_body


def test_clean_eof_without_finish_reason_is_protocol_error(monkeypatch):
    module = _module()
    events = []
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: _Response(frames=[
        _stream_frame({'role': 'assistant'}),
    ]))

    with pytest.raises(_ProviderProtocolError):
        module._forward_with_retry(
            {'messages': []}, runtime_url='http://provider.test',
            stream_output={'_stream_sink': events.append}, proxies=None, max_retries=6, request_timeout=1,
        )

    assert events[-1]['http_status'] == 200
    assert events[-1]['finish_reason'] is None
    assert events[-1]['error_body'] == ''


def test_network_failures_retry_with_same_call_id_then_succeed(monkeypatch):
    module = _module()
    sleeps = []
    monkeypatch.setattr('lazyllm.module.llms.onlinemodule.base.onlineChatModuleBase.random.uniform',
                        lambda low, high: (low + high) / 2)
    monkeypatch.setattr('lazyllm.module.llms.onlinemodule.base.onlineChatModuleBase.time.sleep', sleeps.append)
    _, events = _call(module, monkeypatch, [
        requests.exceptions.ConnectTimeout('connect timeout'),
        requests.exceptions.ReadTimeout('read timeout'),
        _Response(frames=[_stream_frame({'content': 'ok'}, 'stop'), b'data: [DONE]']),
    ], max_retries=3)

    retries = [event for event in events if event['tag'] == 'model_retry']
    status = next(event for event in events if event['tag'] == 'provider_status')
    assert [event['retry_index'] for event in retries] == [1, 2]
    assert [event['max_retries'] for event in retries] == [2, 2]
    assert sleeps == [1.0, 2.0]
    assert {event['model_call_id'] for event in retries} == {status['model_call_id']}


def test_default_retry_budget_is_five_extra_attempts(monkeypatch):
    module = _module()
    events = []
    calls = 0
    monkeypatch.setattr('lazyllm.module.llms.onlinemodule.base.onlineChatModuleBase.random.uniform',
                        lambda low, high: (low + high) / 2)
    monkeypatch.setattr('lazyllm.module.llms.onlinemodule.base.onlineChatModuleBase.time.sleep', lambda delay: None)

    def post(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise requests.exceptions.ReadTimeout('read timeout')

    monkeypatch.setattr(requests, 'post', post)
    with pytest.raises(requests.exceptions.ReadTimeout):
        module._forward_with_retry(
            {'messages': []}, runtime_url='http://provider.test',
            stream_output={'_stream_sink': events.append}, proxies=None, max_retries=None, request_timeout=1,
        )

    assert calls == 6
    assert len([event for event in events if event['tag'] == 'model_retry']) == 5
    assert [event['tag'] for event in events][-1] == 'model_transport_error'


@pytest.mark.parametrize('delta', [
    {'content': 'partial'},
    {'reasoning_content': 'thinking'},
    {'tool_calls': [{'index': 0, 'function': {'arguments': '{'}}]},
])
def test_disconnect_after_semantic_output_is_not_retried(monkeypatch, delta):
    module = _module()
    events = []
    calls = 0

    def post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _Response(frames=[_stream_frame(delta), requests.exceptions.ChunkedEncodingError('cut')])

    monkeypatch.setattr(requests, 'post', post)
    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        module._forward_with_retry(
            {'messages': []}, runtime_url='http://provider.test',
            stream_output={'_stream_sink': events.append}, proxies=None, max_retries=6, request_timeout=1,
        )

    assert calls == 1
    assert not [event for event in events if event['tag'] == 'model_retry']
    assert events[-1]['tag'] == 'model_transport_error'


@pytest.mark.parametrize('error', [
    requests.exceptions.SSLError('certificate verify failed'),
    requests.exceptions.ProxyError('proxy unavailable'),
])
def test_certificate_and_proxy_errors_are_not_retried(monkeypatch, error):
    module = _module()
    events = []
    calls = 0

    def post(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise error

    monkeypatch.setattr(requests, 'post', post)
    with pytest.raises(type(error)):
        module._forward_with_retry(
            {'messages': []}, runtime_url='http://provider.test',
            stream_output={'_stream_sink': events.append}, proxies=None, max_retries=6, request_timeout=1,
        )

    assert calls == 1
    assert [event['tag'] for event in events] == ['model_transport_error']


def test_finish_reason_remains_authoritative_if_connection_closes_before_done(monkeypatch):
    module = _module()
    _, events = _call(module, monkeypatch, [
        _Response(frames=[
            _stream_frame({'content': 'complete'}, 'stop'),
            requests.exceptions.ChunkedEncodingError('missing done'),
        ]),
    ])

    assert events[-1]['tag'] == 'provider_status'
    assert events[-1]['finish_reason'] == 'stop'
    assert not [event for event in events if event['tag'] == 'model_transport_error']


def test_claude_native_protocol_does_not_emit_openai_provider_status(monkeypatch):
    module = ClaudeChat(api_key='test', base_url='http://provider.test/', stream=True)
    _, events = _call(module, monkeypatch, [
        _Response(frames=[
            b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hello"}}',
            b'data: {"type":"message_stop"}',
        ]),
    ])

    assert not [event for event in events if event['tag'] == 'provider_status']
    assert any(event.get('tag') == 'text' and event.get('delta') == 'hello' for event in events)
