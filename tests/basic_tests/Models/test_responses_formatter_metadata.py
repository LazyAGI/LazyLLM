import json
from io import BytesIO

import pytest
import requests
from pydantic import BaseModel, ConfigDict

from lazyllm import OnlineChatModule
from lazyllm.components import FunctionCallFormatter, JsonFormatter
from lazyllm.tools.writer.tools.base import WriterToolBase


def _mock_responses(monkeypatch, outputs):
    outputs = iter(outputs)
    sent = []

    def send(session, request, **kwargs):
        assert request.url == 'https://provider.test/v1/responses'
        body = json.loads(request.body)
        sent.append(body)
        payload = {'status': 'completed', 'output': next(outputs)}
        response = requests.Response()
        response.status_code = 200
        if body['stream']:
            event = {'type': 'response.completed', 'response': payload}
            response._content = ('data: ' + json.dumps(event) + '\n\n').encode()
        else:
            response._content = json.dumps(payload).encode()
        response._content_consumed = True
        response.raw = BytesIO()
        return response

    monkeypatch.setattr(requests.Session, 'send', send)
    return sent


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('reasoning', [False, True])
@pytest.mark.parametrize('formatter', ['json', 'writer'])
def test_responses_json_and_writer_outputs_exclude_protocol_metadata(monkeypatch, stream, reasoning, formatter):
    class ReviewSelection(BaseModel):
        model_config = ConfigDict(extra='forbid')
        material_roles: dict[str, str]
        role: str
        content: str

    # Business JSON may itself contain message-like keys; it is still not a protocol envelope.
    expected = {'material_roles': {'paper_1': 'paper'}, 'role': 'assistant', 'content': 'selected materials'}
    output = [{'type': 'message', 'id': 'msg_1', 'role': 'assistant', 'status': 'completed',
               'content': [{'type': 'output_text', 'text': json.dumps(expected)}]}]
    if reasoning:
        output.insert(0, {'type': 'reasoning', 'id': 'rs_1', 'summary': [], 'encrypted_content': 'opaque'})
    sent = _mock_responses(monkeypatch, [output, output])
    model = OnlineChatModule(source='openai_responses', model='gpt-6.1-sol', api_key='test',
                             base_url='https://provider.test/v1/', stream=stream)
    if formatter == 'writer':
        result = WriterToolBase(llm=model)._call_llm_structured(
            'Select materials', ReviewSelection, stream_output=stream).model_dump()
    else:
        result = model.formatter(JsonFormatter())('Select materials')
    assert result == expected
    assert ReviewSelection.model_validate(result).model_dump() == expected
    assert len(sent) == 1
    assert all(body['stream'] is stream for body in sent)


@pytest.mark.parametrize('stream', [False, True])
def test_responses_function_call_metadata_survives_tool_and_followup_turns(monkeypatch, stream):
    call_output = [
        {'type': 'reasoning', 'id': 'rs_1', 'summary': [], 'encrypted_content': 'tool-state'},
        {'type': 'function_call', 'id': 'fc_1', 'call_id': 'call_1', 'name': 'lookup', 'arguments': '{"id":1}'},
    ]
    final_output = [
        {'type': 'reasoning', 'id': 'rs_2', 'summary': [], 'encrypted_content': 'answer-state'},
        {'type': 'message', 'id': 'msg_2', 'role': 'assistant', 'status': 'completed',
         'content': [{'type': 'output_text', 'text': 'Found it', 'annotations': []}]},
    ]
    sent = _mock_responses(monkeypatch, [call_output, final_output, final_output])
    model = OnlineChatModule(source='openai_responses', model='gpt-6.1-sol', api_key='test',
                             base_url='https://provider.test/v1/', stream=stream).formatter(FunctionCallFormatter())
    tools = [{'type': 'function', 'function': {'name': 'lookup', 'parameters': {'type': 'object'}}}]
    call = model('Look up paper 1', tools=tools)
    assert call['tool_calls'] == [{'id': 'call_1', 'type': 'function',
                                  'function': {'name': 'lookup', 'arguments': '{"id":1}'}}]
    assert call['_lazyllm_provider_items']['output'] == call_output
    history = [{'role': 'user', 'content': 'Look up paper 1'}, call]
    tool_result = {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'paper 1'}
    answer = model({'input': tool_result}, llm_chat_history=history, tools=tools)
    assert answer['content'] == 'Found it'
    assert answer['_lazyllm_provider_items']['output'] == final_output
    history.extend([tool_result, answer])
    assert model('Explain it', llm_chat_history=history, tools=tools)['content'] == 'Found it'
    assert sent[1]['input'] == [*sent[0]['input'], *call_output,
                                {'type': 'function_call_output', 'call_id': 'call_1', 'output': 'paper 1'}]
    assert sent[2]['input'][:-1] == [*sent[1]['input'], *final_output]
    assert all('_lazyllm_provider_items' not in json.dumps(body) for body in sent)
