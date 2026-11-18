import json
import types

import requests

import lazyllm
from lazyllm.module.llms.onlinemodule.supplier.claude import ClaudeChat
from lazyllm.tools import FunctionCallAgent, fc_register


def test_convert_messages_turns_openai_tool_history_into_anthropic_blocks():
    messages = [
        {'role': 'user', 'content': 'What is the weather in Paris and Rome?'},
        {'role': 'assistant', 'content': 'Let me check.', 'reasoning_content': '', 'tool_calls': [
            {'id': 'toolu_01', 'type': 'function',
             'function': {'name': 'get_weather', 'arguments': '{"city": "Paris"}'}},
            {'id': 'toolu_02', 'type': 'function',
             'function': {'name': 'get_weather', 'arguments': {'city': 'Rome'}}},
        ]},
        {'role': 'tool', 'content': '18C in Paris', 'tool_call_id': 'toolu_01', 'name': 'get_weather'},
        {'role': 'tool', 'content': {'temp': 21}, 'tool_call_id': 'toolu_02', 'name': 'get_weather'},
        {'role': 'assistant', 'content': 'Paris 18C, Rome 21C.', 'reasoning_content': ''},
    ]

    assert ClaudeChat._convert_messages(messages) == [
        {'role': 'user', 'content': 'What is the weather in Paris and Rome?'},
        {'role': 'assistant', 'content': [
            {'type': 'text', 'text': 'Let me check.'},
            {'type': 'tool_use', 'id': 'toolu_01', 'name': 'get_weather', 'input': {'city': 'Paris'}},
            {'type': 'tool_use', 'id': 'toolu_02', 'name': 'get_weather', 'input': {'city': 'Rome'}},
        ]},
        {'role': 'user', 'content': [
            {'type': 'tool_result', 'tool_use_id': 'toolu_01', 'content': '18C in Paris'},
            {'type': 'tool_result', 'tool_use_id': 'toolu_02', 'content': '{"temp": 21}'},
        ]},
        {'role': 'assistant', 'content': 'Paris 18C, Rome 21C.'},
    ]


def test_convert_messages_handles_empty_and_invalid_tool_arguments():
    messages = [
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'a', 'type': 'function', 'function': {'name': 'f', 'arguments': ''}},
            {'id': 'b', 'type': 'function', 'function': {'name': 'g', 'arguments': 'not json'}},
        ]},
    ]
    assert ClaudeChat._convert_messages(messages) == [
        {'role': 'assistant', 'content': [
            {'type': 'tool_use', 'id': 'a', 'name': 'f', 'input': {}},
            {'type': 'tool_use', 'id': 'b', 'name': 'g', 'input': {}},
        ]},
    ]


@fc_register('tool')
def claude_history_get_weather(city: str) -> str:
    '''Get the current weather for a city.

    Args:
        city (str): the city name
    '''
    return f'18C in {city}'


class _FakeResponse:
    status_code = 200
    headers = {}

    def __init__(self, body):
        self._body = json.dumps(body).encode()
        self.text = self._body.decode()
        self.raw = types.SimpleNamespace(enforce_content_length=False)

    def __enter__(self): return self
    def __exit__(self, *args): return False
    def iter_content(self, *args, **kwargs): yield self._body
    def iter_lines(self, *args, **kwargs): yield self._body

    def json(self):
        return json.loads(self.text)


def test_function_call_agent_sends_anthropic_tool_history(monkeypatch):
    replies = [
        {'id': 'msg_1', 'type': 'message', 'role': 'assistant', 'model': 'claude-sonnet-4-5',
         'content': [{'type': 'tool_use', 'id': 'toolu_01', 'name': 'claude_history_get_weather',
                      'input': {'city': 'Paris'}}],
         'stop_reason': 'tool_use', 'usage': {'input_tokens': 20, 'output_tokens': 5}},
        {'id': 'msg_2', 'type': 'message', 'role': 'assistant', 'model': 'claude-sonnet-4-5',
         'content': [{'type': 'text', 'text': 'It is 18C in Paris.'}],
         'stop_reason': 'end_turn', 'usage': {'input_tokens': 30, 'output_tokens': 8}},
    ]
    sent = []

    def fake_post(url, **kwargs):
        sent.append(kwargs['json'])
        return _FakeResponse(replies[min(len(sent), len(replies)) - 1])

    monkeypatch.setattr(requests, 'post', fake_post)
    llm = lazyllm.OnlineChatModule(source='claude', model='claude-sonnet-4-5', api_key='fake-key', stream=False)
    agent = FunctionCallAgent(llm, tools=['claude_history_get_weather'], enable_builtin_tools=False)
    agent('What is the weather in Paris?')

    assert len(sent) >= 2
    second = sent[1]['messages']
    for msg in second:
        assert msg['role'] in ('user', 'assistant')
        assert set(msg) == {'role', 'content'}
    uses = [b for m in second if isinstance(m['content'], list) for b in m['content'] if b['type'] == 'tool_use']
    results = [b for m in second if isinstance(m['content'], list)
               for b in m['content'] if b['type'] == 'tool_result']
    assert uses == [{'type': 'tool_use', 'id': 'toolu_01', 'name': 'claude_history_get_weather',
                     'input': {'city': 'Paris'}}]
    assert [r['tool_use_id'] for r in results] == ['toolu_01']
    assert '18C in Paris' in json.dumps(results[0]['content'])
