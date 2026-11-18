import io
import json

import pytest
import requests

import lazyllm
from lazyllm.module.llms.onlinemodule.supplier.minimax import MinimaxChat, MinimaxText2Image, MinimaxTTS


@pytest.fixture
def transport(monkeypatch):
    requests_sent = []

    def send(session, request, **kwargs):
        payload = json.loads(request.body)
        requests_sent.append((request, payload))
        message = {'role': 'assistant', 'content': 'Hello'}
        choice = {'index': 0, 'finish_reason': 'stop'}
        choice['delta' if payload['stream'] else 'message'] = message
        body = json.dumps({'choices': [choice]})
        if payload['stream']:
            body = f'data: {body}\n\ndata: [DONE]\n\n'
        response = requests.Response()
        response.status_code = 200
        response.encoding = 'utf-8'
        response.raw = io.BytesIO(body.encode())
        response.request = request
        response.url = request.url
        return response

    # Intercept after requests has serialized the real provider payload.
    monkeypatch.setattr(requests.Session, 'send', send)
    return requests_sent


@pytest.mark.parametrize('factory', [lazyllm.OnlineChatModule, lazyllm.OnlineModule])
@pytest.mark.parametrize('region,host', [('global', 'api.minimax.io'), ('cn', 'api.minimaxi.com')])
@pytest.mark.parametrize('stream', [False, True])
def test_default_flash_request(transport, factory, region, host, stream):
    module = factory(source='minimax', api_key='test-key', region=region, stream=stream)
    assert isinstance(module, MinimaxChat)
    assert module('Hi') == 'Hello'
    request, payload = transport.pop()
    assert request.url == f'https://{host}/v1/chat/completions'
    assert request.headers['Authorization'] == 'Bearer test-key'
    assert payload['model'] == 'MiniMax-M3.1-Flash'
    assert payload['messages'][-1] == {'role': 'user', 'content': 'Hi'}
    assert payload['stream'] is stream
    assert 'thinking' not in payload
    assert 'reasoning_effort' not in payload
    assert 'service_tier' not in payload
    assert 'reasoning_split' not in payload


@pytest.mark.parametrize('model', ['MiniMax-M3', 'MiniMax-M2.7', 'saved-custom-model'])
@pytest.mark.parametrize('runtime_override', [False, True])
def test_explicit_models_and_custom_urls_are_preserved(transport, model, runtime_override):
    module = lazyllm.OnlineChatModule(
        source='minimax', api_key='test-key', stream=False,
        model='MiniMax-M3' if runtime_override else model, url='https://custom.example/v1/')
    options = {'model': model, 'url': 'https://runtime.example/v1/'} if runtime_override else {}
    assert module('Hi', **options) == 'Hello'
    request, payload = transport.pop()
    host = 'runtime.example' if runtime_override else 'custom.example'
    assert request.url == f'https://{host}/v1/chat/completions'
    assert payload['model'] == model
    assert module._model_name == ('MiniMax-M3' if runtime_override else model)
    assert module._base_url == 'https://custom.example/v1/'


@pytest.mark.parametrize('effort', ['low', 'medium', 'high', 'xhigh', 'max'])
def test_flash_adaptive_thinking_and_effort_are_forwarded(transport, effort):
    module = lazyllm.OnlineChatModule(source='minimax', api_key='test-key', stream=False)
    module('Hi', thinking={'type': 'adaptive'}, reasoning_effort=effort)
    _, payload = transport.pop()
    assert payload['model'] == 'MiniMax-M3.1-Flash'
    assert payload['thinking'] == {'type': 'adaptive'}
    assert payload['reasoning_effort'] == effort


@pytest.mark.parametrize('thinking', ['adaptive', 'disabled'])
def test_m3_thinking_is_preserved(transport, thinking):
    module = lazyllm.OnlineChatModule(source='minimax', api_key='test-key', model='MiniMax-M3', stream=False)
    module('Hi', thinking={'type': thinking})
    _, payload = transport.pop()
    assert payload['model'] == 'MiniMax-M3'
    assert payload['thinking'] == {'type': thinking}
    assert 'reasoning_effort' not in payload


def test_structured_messages_and_tools_are_preserved(transport):
    module = lazyllm.OnlineChatModule(source='minimax', api_key='test-key', stream=False)
    messages = [
        {'role': 'user', 'content': [
            {'type': 'text', 'text': 'Describe this image'},
            {'type': 'image_url', 'image_url': {'url': 'https://media.example/image.png'}},
        ]},
        {'role': 'assistant', 'content': 'A landscape', 'reasoning_content': 'Visible hills'},
        {'role': 'user', 'content': 'What is visible?'},
    ]
    tools = [{'type': 'function', 'function': {'name': 'lookup', 'parameters': {'type': 'object'}}}]
    module('Hi', messages=messages, tools=tools)
    _, payload = transport.pop()
    assert payload['messages'] == messages
    assert payload['tools'] == tools


def test_nontext_supplier_defaults_are_unchanged():
    assert MinimaxText2Image.MODEL_NAME == 'image-01'
    assert MinimaxTTS.MODEL_NAME == 'speech-2.6-hd'
