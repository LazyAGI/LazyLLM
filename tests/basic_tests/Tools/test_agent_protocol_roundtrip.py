'''Transport-level regressions for Agent protocol state across tool rounds.'''
from copy import deepcopy
from io import BytesIO
import json

import pytest
import requests

import lazyllm
from lazyllm.module.llms.onlinemodule.supplier.openai_responses import OpenAIResponsesChat
from lazyllm.tools import ReactAgent


def protocol_increment(value: int) -> int:
    '''Increment an integer.

    Args:
        value (int): Integer to increment.

    Returns:
        int: Incremented integer.
    '''
    return value + 1


def _tool_response(round_number):
    return {
        'status': 'completed',
        'output': [
            {'type': 'reasoning', 'id': f'rs_{round_number}', 'summary': [],
             'encrypted_content': f'opaque-continuation-{round_number}'},
            {'type': 'message', 'id': f'msg_{round_number}', 'role': 'assistant',
             'status': 'completed', 'content': [
                 {'type': 'output_text', 'text': f'Step {round_number}.', 'annotations': []}]},
            {'type': 'function_call', 'id': f'fc_{round_number}', 'call_id': f'call_{round_number}',
             'name': 'protocol_increment', 'arguments': json.dumps({'value': round_number}),
             'status': 'completed'},
        ],
    }


def _http_response(payload, stream):
    response = requests.Response()
    response.status_code = 200
    response.encoding = 'utf-8'
    if stream:
        events = []
        for index, item in enumerate(payload['output']):
            if item['type'] == 'message':
                text = item['content'][0]['text']
                events.extend({'type': 'response.output_text.delta', 'output_index': index,
                               'content_index': 0, 'delta': chunk}
                              for chunk in (text[:2], text[2:]))
            elif item['type'] == 'function_call':
                events.append({'type': 'response.output_item.added', 'output_index': index,
                               'item': {**item, 'arguments': '', 'status': 'in_progress'}})
                arguments = item['arguments']
                events.extend({'type': 'response.function_call_arguments.delta', 'output_index': index,
                               'item_id': item['id'], 'delta': chunk}
                              for chunk in (arguments[:5], arguments[5:]))
        events.append({'type': 'response.completed', 'response': payload})
        content = ''.join('data: ' + json.dumps(event) + '\n\n' for event in events).encode()
        response.headers['Content-Type'] = 'text/event-stream'
    else:
        content = json.dumps(payload).encode()
        response.headers['Content-Type'] = 'application/json'
    response._content = content
    response._content_consumed = True
    response.raw = BytesIO(content)
    return response


@pytest.mark.parametrize('stream', [False, True], ids=['nonstream', 'stream'])
def test_responses_agent_preserves_protocol_state_across_tool_rounds(monkeypatch, model_call_session, stream):
    '''Dropping message metadata must break the next wire request, not just a unit assertion.'''
    outputs = [_tool_response(1), _tool_response(2), {
        'status': 'completed', 'output': [
            {'type': 'message', 'id': 'msg_final', 'role': 'assistant', 'status': 'completed',
             'content': [{'type': 'output_text', 'text': 'Final result: 3.', 'annotations': []}]},
        ],
    }]
    requests_sent = []

    def send(session, request, **kwargs):
        assert request.url == 'https://provider.test/v1/responses'
        body = json.loads(request.body)
        requests_sent.append(deepcopy(body))
        assert body['stream'] is stream
        assert len(requests_sent) <= len(outputs), 'Agent unexpectedly made an extra model call'
        return _http_response(outputs[len(requests_sent) - 1], stream)

    monkeypatch.setattr(requests.Session, 'send', send)
    model = OpenAIResponsesChat(base_url='https://provider.test/v1/', model='gpt-5.6-sol',
                                api_key='test-key', stream=stream)
    agent = ReactAgent(llm=model, tools=[protocol_increment], max_retries=5,
                       stream=stream, enable_builtin_tools=False)
    queue = lazyllm.FileSystemQueue()
    queue.clear()
    try:
        assert agent('Increment 1 twice using the tool.') == 'Final result: 3.'
        assert len(requests_sent) == 3
        for round_index in (1, 2):
            items = requests_sent[round_index]['input']
            # Every completed round must retain the complete native output, in order,
            # followed by the matching result. No wire request contains the internal wrapper.
            for previous in range(round_index):
                native = outputs[previous]['output']
                start = next(i for i, item in enumerate(items) if item.get('id') == native[0]['id'])
                assert items[start:start + len(native)] == native
                tool_output = items[start + len(native)]
                assert tool_output['type'] == 'function_call_output'
                assert tool_output['call_id'] == f'call_{previous + 1}'
                assert tool_output['output'].split('\n')[0] == str(previous + 2)
            assert '_lazyllm_provider_items' not in json.dumps(items)

        history = lazyllm.locals['_lazyllm_agent']['history']
        tool_messages = [message for message in history if message.get('role') == 'assistant'
                         and message.get('tool_calls')]
        assert len(tool_messages) == 2
        for index, message in enumerate(tool_messages):
            assert message['_lazyllm_provider_items']['output'] == outputs[index]['output']

        events = [json.loads(raw) for raw in queue.dequeue() if raw.startswith('{')]
        if stream:
            calls = [event for event in events if event.get('tag') == 'tool_calls']
            assert len(calls) == 2
            for index, event in enumerate(calls):
                assert event['assistant_message']['_lazyllm_provider_items']['output'] == outputs[index]['output']
            final = [event for event in events if event.get('tag') == 'assistant_message']
            assert final[-1]['assistant_message']['_lazyllm_provider_items']['output'] == outputs[-1]['output']
    finally:
        queue.clear()


def _chat_response(message, stream):
    finish_reason = 'tool_calls' if message.get('tool_calls') else 'stop'
    response = requests.Response()
    response.status_code = 200
    response.encoding = 'utf-8'
    if stream:
        deltas = [{'role': 'assistant'}, {'reasoning_content': message['reasoning_content']},
                  {'content': message['content']}]
        for index, call in enumerate(message.get('tool_calls', [])):
            function = call['function']
            deltas.append({'tool_calls': [{'index': index, 'id': call['id'], 'type': 'function',
                                           'function': {'name': function['name'], 'arguments': ''}}]})
            arguments = function['arguments']
            deltas.extend({'tool_calls': [{'index': index, 'function': {'arguments': chunk}}]}
                          for chunk in (arguments[:5], arguments[5:]))
        chunks = [{'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]} for delta in deltas]
        chunks.append({'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish_reason}]})
        content = (''.join('data: ' + json.dumps(chunk) + '\n\n' for chunk in chunks)
                   + 'data: [DONE]\n\n').encode()
    else:
        content = json.dumps({'choices': [{'index': 0, 'message': message,
                                           'finish_reason': finish_reason}]}).encode()
    response._content = content
    response._content_consumed = True
    response.raw = BytesIO(content)
    return response


@pytest.mark.parametrize('stream', [False, True], ids=['nonstream', 'stream'])
def test_chat_agent_preserves_reasoning_and_tool_history(monkeypatch, model_call_session, stream):
    from lazyllm.module.llms.onlinemodule.supplier.openai import OpenAIChat

    messages = [{
        'role': 'assistant', 'content': f'Step {number}.', 'reasoning_content': f'Reasoning {number}.',
        'tool_calls': [{'id': f'call_{number}', 'type': 'function',
                        'function': {'name': 'protocol_increment', 'arguments': json.dumps({'value': number})}}],
    } for number in (1, 2)]
    messages.append({'role': 'assistant', 'content': 'Final result: 3.', 'reasoning_content': 'Finished.'})
    requests_sent = []

    def send(session, request, **kwargs):
        assert request.url == 'https://provider.test/v1/chat/completions'
        body = json.loads(request.body)
        requests_sent.append(deepcopy(body))
        assert body['stream'] is stream
        assert len(requests_sent) <= len(messages)
        return _chat_response(messages[len(requests_sent) - 1], stream)

    monkeypatch.setattr(requests.Session, 'send', send)
    model = OpenAIChat(base_url='https://provider.test/v1/', model='test', api_key='test-key', stream=stream)
    agent = ReactAgent(llm=model, tools=[protocol_increment], max_retries=5,
                       stream=stream, enable_builtin_tools=False)
    queue = lazyllm.FileSystemQueue()
    queue.clear()
    try:
        assert agent('Increment 1 twice using the tool.') == 'Final result: 3.'
        assert len(requests_sent) == 3
        for round_index in (1, 2):
            history = requests_sent[round_index]['messages']
            for previous in range(round_index):
                start = next(i for i, message in enumerate(history)
                             if message.get('tool_calls', [{}])[0].get('id') == f'call_{previous + 1}')
                assert history[start] == messages[previous]
                assert history[start + 1]['role'] == 'tool'
                assert history[start + 1]['tool_call_id'] == f'call_{previous + 1}'
                assert history[start + 1]['content'].split('\n')[0] == str(previous + 2)
            assert '_lazyllm_provider_items' not in json.dumps(history)
    finally:
        queue.clear()
