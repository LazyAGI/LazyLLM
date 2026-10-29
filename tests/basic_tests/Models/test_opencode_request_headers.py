import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import requests
from urllib3.response import HTTPResponse

import lazyllm
from lazyllm.module.llms.onlinemodule.base.model_call_runner import ModelAttemptState
from lazyllm.module.llms.onlinemodule.base.model_request import ModelCallControl
from lazyllm.module.llms.onlinemodule.supplier.claude import ClaudeChat
from lazyllm.module.llms.onlinemodule.supplier.deepseek import DeepSeekChat
from lazyllm.module.llms.onlinemodule.supplier.openai import OpenAIChat


GO_URL = 'https://opencode.ai/zen/go/v1/chat/completions'


@pytest.fixture(autouse=True)
def no_external_requests(monkeypatch):
    def unexpected_send(*args, **kwargs):
        pytest.fail('Request-header tests must mock every HTTP request')

    monkeypatch.setattr(requests.Session, 'send', unexpected_send)


class RequestHeaderChat(OpenAIChat):
    def _prepare_request_headers(self, runtime_url):
        headers = super()._prepare_request_headers(runtime_url)
        headers['X-Request-Session'] = lazyllm.globals._sid
        return headers


def _respond():
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps({
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}],
    }).encode()
    response._content_consumed = True
    response.raw = HTTPResponse()
    return response


def _call(module, url=GO_URL, call_control=None):
    return module._forward_impl(
        {'model': 'deepseek-v4.1-flash', 'messages': [{'role': 'user', 'content': 'hello'}]},
        runtime_url=url, stream_output=False, proxies=None, request_timeout=10, state=ModelAttemptState(),
        call_control=call_control,
    )


@pytest.mark.parametrize('conversation_metadata', [False, True])
@pytest.mark.parametrize('controlled', [False, True])
def test_opencode_uses_stable_conversation_session_without_mutating_auth(monkeypatch, conversation_metadata, controlled):
    module = OpenAIChat(api_key='test-key', stream=False)
    original_headers = dict(module._header)
    sent = []

    def post(url, **kwargs):
        sent.append(kwargs['headers'])
        return _respond()

    monkeypatch.setattr(requests, 'post', post)
    monkeypatch.setattr(requests.Session, 'post', lambda self, url, **kwargs: post(url, **kwargs))
    for request_id in ['request-1', 'request-2']:
        with lazyllm.new_session(request_id):
            lazyllm.set_trace_context({
                'session_id': request_id if conversation_metadata else 'conversation-1',
                'trace_metadata': {'conversation_id': 'conversation-1'} if conversation_metadata else {},
            })
            _call(module, call_control=ModelCallControl() if controlled else None)
    assert [headers['x-opencode-session'] for headers in sent] == ['conversation-1', 'conversation-1']
    assert all(headers['User-Agent'].startswith('LazyLLM/') for headers in sent)
    assert all(headers['Authorization'] == 'Bearer test-key' for headers in sent)
    assert module._header == original_headers


def test_shared_module_keeps_concurrent_sessions_separate(monkeypatch):
    module = OpenAIChat(api_key='test-key', stream=False)
    barrier = Barrier(2)

    def post(url, **kwargs):
        barrier.wait(timeout=10)
        assert kwargs['headers']['x-opencode-session'] == lazyllm.globals._sid
        return _respond()

    def run(session):
        with lazyllm.new_session(session):
            return _call(module)

    monkeypatch.setattr(requests, 'post', post)
    with ThreadPoolExecutor(max_workers=2) as executor:
        assert all(executor.map(run, ['conversation-a', 'conversation-b']))
    assert 'x-opencode-session' not in module._header


@pytest.mark.parametrize('url', [
    'https://api.openai.com/v1/chat/completions',
    'https://opencode.ai.example.com/v1/chat/completions',
    'https://example.com/opencode.ai/chat/completions',
    'http://opencode.ai/zen/go/v1/chat/completions',
])
def test_other_destinations_keep_original_headers(monkeypatch, url):
    module = OpenAIChat(base_url='https://opencode.ai/zen/go/v1/', api_key='test-key', stream=False)
    original_headers = dict(module._header)

    def post(destination, **kwargs):
        assert destination == url
        assert kwargs['headers'] == original_headers
        return _respond()

    monkeypatch.setattr(requests, 'post', post)
    with lazyllm.new_session('conversation-other'):
        _call(module, url)


def test_supplier_request_header_extension_is_applied_without_mutating_auth(monkeypatch):
    module = RequestHeaderChat(api_key='test-key', stream=False)
    sent = []

    def post(url, **kwargs):
        sent.append(kwargs['headers'])
        return _respond()

    monkeypatch.setattr(requests, 'post', post)
    for session in ['conversation-a', 'conversation-b']:
        with lazyllm.new_session(session):
            _call(module, 'https://example.com/v1/chat/completions')
    assert sent == [
        {'Content-Type': 'application/json', 'Authorization': 'Bearer test-key',
         'X-Request-Session': session}
        for session in ['conversation-a', 'conversation-b']
    ]
    assert module._header == {'Content-Type': 'application/json', 'Authorization': 'Bearer test-key'}


def test_non_openai_supplier_does_not_inherit_opencode_headers(monkeypatch):
    module = DeepSeekChat(api_key='test-key', model='deepseek-v4-flash', stream=False)
    sent = []

    def post(url, **kwargs):
        sent.append(kwargs['headers'])
        return _respond()

    monkeypatch.setattr(requests, 'post', post)
    with lazyllm.new_session('conversation-deepseek'):
        _call(module)
    assert sent == [{'Content-Type': 'application/json', 'Authorization': 'Bearer test-key'}]


def test_native_supplier_keeps_its_authentication_headers(monkeypatch):
    module = ClaudeChat(api_key='test-key', stream=False)
    sent = []

    def post(url, **kwargs):
        sent.append(kwargs['headers'])
        response = _respond()
        response._content = json.dumps({
            'type': 'message', 'role': 'assistant', 'content': [{'type': 'text', 'text': 'ok'}],
            'stop_reason': 'end_turn', 'usage': {'input_tokens': 1, 'output_tokens': 1},
        }).encode()
        return response

    monkeypatch.setattr(requests, 'post', post)
    with lazyllm.new_session('conversation-claude'):
        result = _call(module, 'https://api.anthropic.com/v1/messages')
    assert result[0]['choices'][0]['message']['content'] == 'ok'
    assert sent == [{'Content-Type': 'application/json', 'anthropic-version': '2023-06-01',
                     'x-api-key': 'test-key'}]
