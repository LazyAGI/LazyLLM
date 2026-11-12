import asyncio
import json

import pytest
import requests

from lazyllm.module.llms.onlinemodule.base.model_call_runner import ModelAttemptState, _ModelCallRunner
from lazyllm.module.llms.onlinemodule.base.model_outcome import ModelCallError
from lazyllm.module.llms.onlinemodule.base.model_request import _ModelCallDeadlineExceeded
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


def test_responses_request_drops_sampling_for_default_gpt6_reasoning():
    body = chat_to_responses_request({
        'model': 'gpt-6-sol', 'messages': [{'role': 'user', 'content': 'title?'}],
        'temperature': 0, 'top_p': 0.7, 'top_logprobs': 3, 'logprobs': True,
        'include': ['reasoning.encrypted_content', 'message.output_text.logprobs'],
    })
    assert all(key not in body for key in ('temperature', 'top_p', 'top_logprobs', 'logprobs'))
    assert body['include'] == ['reasoning.encrypted_content']


def test_responses_request_preserves_sampling_with_no_reasoning():
    body = chat_to_responses_request({
        'model': 'gpt-6-sol', 'messages': [{'role': 'user', 'content': 'title?'}],
        'reasoning_effort': 'none', 'temperature': 0, 'top_p': 0.7,
        'top_logprobs': 3, 'logprobs': True,
    })
    assert body['reasoning'] == {'effort': 'none'}
    assert body['temperature'] == 0
    assert body['top_p'] == 0.7
    assert body['top_logprobs'] == 3
    assert 'logprobs' not in body
    assert body['include'] == ['message.output_text.logprobs']


def test_responses_request_preserves_gpt4o_sampling():
    body = chat_to_responses_request({
        'model': 'gpt-4o', 'messages': [{'role': 'user', 'content': 'title?'}],
        'temperature': 0, 'top_p': 0.7,
    })
    assert body['temperature'] == 0
    assert body['top_p'] == 0.7


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
        text = json.dumps({
            'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'Hello'}]}],
            'usage': {'input_tokens': 2, 'output_tokens': 3, 'total_tokens': 5},
        })

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

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

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

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

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

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
    merged = module._merge_stream_result(result)
    assert merged['choices'][0]['delta']['tool_calls'][0]['id'] == 'call_2'
    assert state.usage['prompt_tokens'] == 12


@pytest.mark.parametrize('option', ['stop', 'frequency_penalty', 'presence_penalty', 'seed', 'logit_bias',
                                    'audio', 'modalities', 'prediction'])
def test_responses_rejects_unmappable_options(option):
    with pytest.raises(ValueError, match=option):
        chat_to_responses_request({'messages': [], option: 1})


def test_responses_maps_legacy_functions_and_rejects_unsupported_tools():
    body = chat_to_responses_request({'messages': [], 'functions': [{'name': 'lookup'}],
                                      'function_call': {'name': 'lookup'}})
    assert body['tools'] == [{'type': 'function', 'name': 'lookup'}]
    assert body['tool_choice'] == {'type': 'function', 'name': 'lookup'}
    with pytest.raises(ValueError, match='n'):
        chat_to_responses_request({'messages': [], 'n': 2})
    with pytest.raises(ValueError, match='tool'):
        chat_to_responses_request({'messages': [], 'tools': [{'type': 'image_generation'}]})


@pytest.mark.parametrize('payload', ['{bad', '[]', '{"type":"response.completed"}'])
def test_responses_bad_stream_payload_is_protocol_failure(payload):
    model = OpenAIResponsesChat(model='test', api_key='test')
    runner = _ModelCallRunner(emit_event=lambda *args: None)
    with pytest.raises(ModelCallError) as caught:
        runner.run(lambda state: model._response_parser(False).collect(
            _response_stream_as_chat(['data: ' + payload]), state), max_attempts=1)
    assert caught.value.terminal.failure.origin.value == 'protocol'


def test_responses_failed_frame_preserves_partial_and_usage():
    frames = list(_response_stream_as_chat(['data: ' + json.dumps({
        'type': 'response.failed', 'response': {
            'status': 'failed',
            'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'partial'}]}],
            'usage': {'input_tokens': 11, 'output_tokens': 3},
            'error': {'code': 'server_error', 'message': 'failed'}}})]))
    chunks = [json.loads(frame[6:]) for frame in frames]
    assert chunks[0]['choices'][0]['delta']['content'] == 'partial'
    assert any(chunk.get('usage', {}).get('prompt_tokens') == 11 for chunk in chunks)
    assert chunks[-1]['error']['code'] == 'server_error'


@pytest.mark.parametrize('exception,kind', [
    (asyncio.CancelledError, 'cancelled'), (_ModelCallDeadlineExceeded, 'failure'),
])
def test_runner_interrupt_has_one_terminal(exception, kind):
    events = []
    runner = _ModelCallRunner(emit_event=lambda event, data: events.append((event, data)))

    def interrupt(state):
        state.usage = {'prompt_tokens': 11, 'completion_tokens': 3}
        state.semantic_output = True
        raise exception('interrupted')
    with pytest.raises(exception):
        runner.run(interrupt, max_attempts=2)
    terminals = [data for event, data in events if event == 'model_call_finished']
    assert len(terminals) == 1 and terminals[0]['kind'] == kind
    assert terminals[0]['usage']['prompt_tokens'] == 11
    assert terminals[0]['has_semantic_output']


def test_responses_incremental_tool_reasoning_refusal_and_final_snapshot():
    call = {'type': 'function_call', 'id': 'fc1', 'call_id': 'c1', 'name': 'lookup', 'arguments': ''}
    events = [
        {'type': 'response.output_item.added', 'output_index': 1, 'item': call},
        {'type': 'response.function_call_arguments.delta', 'output_index': 1, 'item_id': 'fc1', 'delta': '{"x":'},
        {'type': 'response.function_call_arguments.delta', 'output_index': 1, 'item_id': 'fc1', 'delta': '1}'},
        {'type': 'response.reasoning_summary_text.delta', 'output_index': 0, 'summary_index': 0, 'delta': 'Thinking'},
        {'type': 'response.refusal.delta', 'output_index': 2, 'content_index': 0, 'delta': 'Cannot'},
        {'type': 'response.completed', 'response': {'status': 'completed', 'output': [
            {**call, 'arguments': '{"x":1}'},
            {'type': 'message', 'content': [{'type': 'refusal', 'refusal': 'Cannot comply'}]},
        ]}},
    ]
    chunks = [json.loads(frame[6:]) for frame in _response_stream_as_chat(
        ['data: ' + json.dumps(event) for event in events]) if frame != 'data: [DONE]']
    deltas = [chunk['choices'][0]['delta'] for chunk in chunks if chunk.get('choices')]
    assert any(delta.get('reasoning_content') == 'Thinking' for delta in deltas)
    assert ''.join(delta.get('refusal', '') for delta in deltas) == 'Cannot comply'
    calls = [call for delta in deltas for call in delta.get('tool_calls', [])]
    assert ''.join(call.get('function', {}).get('arguments', '') for call in calls) == '{"x":1}'
    assert calls[0]['id'] == 'c1'


@pytest.mark.parametrize('stream', [False, True])
def test_responses_failed_http_keeps_accounting_and_partial(monkeypatch, stream):
    payload = {'status': 'failed', 'output': [{'type': 'message', 'content': [
        {'type': 'output_text', 'text': 'partial answer'}]}],
        'usage': {'input_tokens': 11, 'output_tokens': 3},
        'error': {'code': 'server_error', 'message': 'failed'}}

    class Response:
        status_code = 200
        headers = {}
        raw = type('Raw', (), {})()
        text = json.dumps(payload)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def iter_lines(self):
            yield ('data: ' + json.dumps({'type': 'response.failed', 'response': payload})).encode()
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: Response())
    model = OpenAIResponsesChat(model='test', api_key='test', stream=stream)
    with pytest.raises(ModelCallError) as caught:
        model('hello')
    error = caught.value
    assert error.terminal.failure.origin.value == 'provider'
    assert error.terminal.usage['prompt_tokens'] == 11
    assert error.usage['completion_tokens'] == 3
    assert model._extract_partial_content(error.partial_response) == 'partial answer'


@pytest.mark.parametrize('payload', ['[]', '{bad', '{"output":[null]}'])
def test_responses_bad_nonstream_http_is_protocol_failure(monkeypatch, payload):

    class Response:
        status_code = 200
        headers = {}
        raw = type('Raw', (), {})()
        text = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: Response())
    model = OpenAIResponsesChat(model='test', api_key='test', stream=False)
    with pytest.raises(ModelCallError) as caught:
        model('hello')
    assert caught.value.terminal.failure.origin.value == 'protocol'


def test_responses_truncated_tool_stream_keeps_non_executable_partial():
    events = [
        {'type': 'response.output_item.added', 'output_index': 0, 'item': {
            'type': 'function_call', 'id': 'fc1', 'call_id': 'c1', 'name': 'lookup', 'arguments': ''}},
        {'type': 'response.function_call_arguments.delta', 'item_id': 'fc1', 'delta': '{"x":'},
    ]
    model = OpenAIResponsesChat(model='test', api_key='test')
    runner = _ModelCallRunner(emit_event=lambda *args: None)
    with pytest.raises(ModelCallError) as caught:
        runner.run(lambda state: model._response_parser(False).collect(_response_stream_as_chat(
            ['data: ' + json.dumps(event) for event in events]), state), max_attempts=1)
    assert caught.value.terminal.has_semantic_output
    assert caught.value.terminal.failure.origin.value == 'protocol'
    merged = model._merge_stream_result(caught.value.partial_response)
    assert merged['choices'][0]['delta']['tool_calls'][0]['function']['arguments'] == '{"x":'


def test_runner_cancellation_in_backoff_closes_call():
    from lazyllm.module.llms.onlinemodule.base.model_request import ModelCallControl
    events = []
    cancelled = False

    def emit(event, data):
        nonlocal cancelled
        events.append((event, data))
        if event == 'model_retry_scheduled': cancelled = True

    def fail(state):
        raise requests.ConnectTimeout('connect timeout')
    runner = _ModelCallRunner(emit_event=emit, control=ModelCallControl(cancel_check=lambda: cancelled))
    with pytest.raises(asyncio.CancelledError):
        runner.run(fail, max_attempts=3)
    terminals = [data for event, data in events if event == 'model_call_finished']
    assert len(terminals) == 1 and terminals[0]['kind'] == 'cancelled'
    assert terminals[0]['attempt_count'] == 1


def test_responses_terminal_cannot_silently_drop_streamed_tool():
    events = [
        {'type': 'response.output_item.added', 'output_index': 0, 'item': {
            'type': 'function_call', 'id': 'fc1', 'call_id': 'c1', 'name': 'lookup', 'arguments': ''}},
        {'type': 'response.completed', 'response': {'status': 'completed', 'output': [
            {'type': 'message', 'content': [{'type': 'output_text', 'text': 'Done'}]}]}},
    ]
    model = OpenAIResponsesChat(model='test', api_key='test')
    runner = _ModelCallRunner(emit_event=lambda *args: None)
    with pytest.raises(ModelCallError) as caught:
        runner.run(lambda state: model._response_parser(False).collect(_response_stream_as_chat(
            ['data: ' + json.dumps(event) for event in events]), state), max_attempts=1)
    assert caught.value.terminal.failure.origin.value == 'protocol'


def test_responses_equal_reasoning_deltas_are_not_deduplicated():
    model = OpenAIResponsesChat(model='test', api_key='test')
    events = [
        {'type': 'response.reasoning_summary_text.delta', 'delta': 'Again'},
        {'type': 'response.reasoning_summary_text.delta', 'delta': 'Again'},
        {'type': 'response.completed', 'response': {'output': [
            {'type': 'message', 'content': [{'type': 'output_text', 'text': 'Done'}]}]}},
    ]
    state = ModelAttemptState()
    frames = model._response_parser(False).collect(_response_stream_as_chat(
        ['data: ' + json.dumps(event) for event in events]), state)
    message = model._extract_specified_key_fields(model._merge_stream_result(frames))
    assert message['reasoning_content'] == 'AgainAgain'


def test_responses_mapped_options_preserve_native_siblings():
    body = chat_to_responses_request({'messages': [], 'reasoning': {'summary': 'auto'},
                                      'reasoning_effort': 'low', 'text': {'verbosity': 'low'},
                                      'response_format': {'type': 'json_object'}})
    assert body['reasoning'] == {'summary': 'auto', 'effort': 'low'}
    assert body['text'] == {'verbosity': 'low', 'format': {'type': 'json_object'}}


def test_responses_rejects_background_jobs_without_a_polling_lifecycle():
    with pytest.raises(ValueError, match='background'):
        chat_to_responses_request({'messages': [], 'background': True})
    assert chat_to_responses_request({'messages': [], 'background': False})['background'] is False
