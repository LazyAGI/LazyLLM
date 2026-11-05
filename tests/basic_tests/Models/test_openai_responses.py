import json

import requests

from lazyllm.module.llms.onlinemodule.base.model_call_runner import ModelAttemptState
from lazyllm.module.llms.onlinemodule.supplier.openai_responses import (
    OpenAIResponsesChat, _response_stream_as_chat, chat_to_responses_request, responses_to_chat_chunk,
)


def test_responses_request_preserves_tool_conversation():
    chat = {
        'model': 'gpt-5.6-sol', 'stream': False, 'max_completion_tokens': 128,
        'messages': [
            {'role': 'user', 'content': 'weather?'},
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'call_1', 'type': 'function', 'function': {'name': 'weather', 'arguments': '{"city":"Taipei"}'}}
            ]},
            {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'sunny'},
        ],
        'tools': [{'type': 'function', 'function': {'name': 'weather', 'parameters': {'type': 'object'}}}],
    }
    body = chat_to_responses_request(chat)
    assert body['input'][1] == {'type': 'function_call', 'call_id': 'call_1', 'name': 'weather',
                                'arguments': '{"city":"Taipei"}'}
    assert body['input'][2] == {'type': 'function_call_output', 'call_id': 'call_1', 'output': 'sunny'}
    assert body['tools'][0] == {'type': 'function', 'name': 'weather', 'parameters': {'type': 'object'}}
    assert body['max_output_tokens'] == 128
    assert 'messages' not in body and 'max_completion_tokens' not in body


def test_responses_request_flattens_forced_function_choice():
    body = chat_to_responses_request({
        'messages': [{'role': 'user', 'content': 'weather?'}],
        'tool_choice': {'type': 'function', 'function': {'name': 'weather'}},
        'tools': [{'type': 'function', 'function': {'name': 'weather', 'parameters': {'type': 'object'}}}],
    })
    assert body['tool_choice'] == {'type': 'function', 'name': 'weather'}


def test_responses_output_preserves_function_calls_and_usage():
    response = {'output': [
        {'type': 'message', 'content': [{'type': 'output_text', 'text': 'Checking.'}]},
        {'type': 'function_call', 'call_id': 'call_2', 'name': 'weather', 'arguments': '{}'},
    ], 'usage': {'input_tokens': 12, 'output_tokens': 8}}
    chunk = responses_to_chat_chunk(response)
    assert chunk['choices'][0]['message']['content'] == 'Checking.'
    assert chunk['choices'][0]['message']['tool_calls'][0]['id'] == 'call_2'
    assert chunk['choices'][0]['finish_reason'] == 'tool_calls'
    assert chunk['usage']['prompt_tokens'] == 12
    assert chunk['usage']['completion_tokens'] == 8


def test_responses_json_schema_format_is_flattened():
    body = chat_to_responses_request({'messages': [], 'response_format': {
        'type': 'json_schema', 'json_schema': {'name': 'answer', 'strict': True,
                                              'schema': {'type': 'object'}}}})
    assert body['text']['format'] == {'type': 'json_schema', 'name': 'answer', 'strict': True,
                                      'schema': {'type': 'object'}}


def test_responses_completed_event_supplies_text_without_deltas():
    event = {'type': 'response.completed', 'response': {'output': [
        {'type': 'message', 'content': [{'type': 'output_text', 'text': 'Hello'}]}]}}
    chunks = [json.loads(frame[6:]) for frame in _response_stream_as_chat(
        [('data: ' + json.dumps(event)).encode()]) if frame != 'data: [DONE]']
    assert chunks[0]['choices'][0]['delta']['content'] == 'Hello'
    assert chunks[-1]['choices'][0]['finish_reason'] == 'stop'


def test_responses_nonstream_http_result(monkeypatch):
    class Response:
        status_code = 200
        headers = {}
        raw = type('Raw', (), {})()
        text = json.dumps({'output': [{'type': 'message', 'content': [
            {'type': 'output_text', 'text': 'Hello'}]}],
                           'usage': {'input_tokens': 2, 'output_tokens': 3, 'total_tokens': 5}})

        def __enter__(self): return self
        def __exit__(self, *args): return None

    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: Response())
    module = OpenAIResponsesChat(base_url='https://provider.test/v1/', model='gpt-5.6-sol',
                                 api_key='test-key', stream=False)
    state = ModelAttemptState()
    result = module._forward_impl({'model': 'gpt-5.6-sol', 'input': [], 'stream': False},
                                  runtime_url=module._chat_url, stream_output=False,
                                  proxies=None, request_timeout=1, state=state)
    assert module._chat_url == 'https://provider.test/v1/responses'
    assert result[0]['choices'][0]['message']['content'] == 'Hello'
    assert state.usage['prompt_tokens'] == 2
    assert state.usage['completion_tokens'] == 3


def test_responses_module_call_uses_responses_wire_format(monkeypatch):
    class Response:
        status_code = 200
        headers = {}
        raw = type('Raw', (), {})()
        text = json.dumps({'output': [{'type': 'message', 'content': [
            {'type': 'output_text', 'text': 'Hello'}]}]})

        def __enter__(self): return self
        def __exit__(self, *args): return None

    sent = []
    monkeypatch.setattr(requests, 'post', lambda url, **kwargs: (sent.append((url, kwargs['json'])), Response())[1])
    model = OpenAIResponsesChat(base_url='https://provider.test/v1/', model='gpt-5.6-sol',
                                api_key='test-key', stream=False)
    assert model('Hi', stream_output=False) == 'Hello'
    assert sent[0][0] == 'https://provider.test/v1/responses'
    assert sent[0][1]['input'][-1] == {'role': 'user', 'content': 'Hi'}
    assert 'messages' not in sent[0][1]


def test_responses_stream_yields_text_and_tool_call(monkeypatch):
    class Response:
        status_code = 200
        headers = {}
        raw = type('Raw', (), {})()

        def __enter__(self): return self
        def __exit__(self, *args): return None

        def iter_lines(self):
            for event in [
                {'type': 'response.output_text.delta', 'delta': 'Checking.'},
                {'type': 'response.completed', 'response': {'output': [
                    {'type': 'message', 'content': [{'type': 'output_text', 'text': 'Checking.'}]},
                    {'type': 'function_call', 'call_id': 'call_2', 'name': 'weather', 'arguments': '{}'},
                ], 'usage': {'input_tokens': 12, 'output_tokens': 8}}},
            ]:
                yield ('data: ' + json.dumps(event)).encode()

    sent = []
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: (sent.append(kwargs['json']), Response())[1])
    module = OpenAIResponsesChat(base_url='https://provider.test/v1/', model='gpt-5.6-sol',
                                 api_key='test-key', stream=True)
    state = ModelAttemptState()
    result = module._forward_impl({'model': 'gpt-5.6-sol', 'input': [], 'stream': True},
                                  runtime_url='https://provider.test/v1/responses', stream_output=False,
                                  proxies=None, request_timeout=1, state=state)
    assert sent[0]['stream'] is True
    assert result[0]['choices'][0]['delta']['content'] == 'Checking.'
    assert result[-1]['choices'][0]['delta']['tool_calls'][0]['id'] == 'call_2'
    assert state.usage['prompt_tokens'] == 12
