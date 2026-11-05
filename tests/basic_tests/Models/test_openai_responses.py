import json

import pytest

from lazyllm.module.llms.onlinemodule.base.openai_responses import responses_request
from lazyllm.module.llms.onlinemodule.base.model_call_runner import ModelAttemptState
from lazyllm.module.llms.onlinemodule.base.model_outcome import ModelFinish, _ModelResponseError
from lazyllm.module.llms.onlinemodule.supplier.openai import OpenAIChat


def module(model='gpt-6.1-sol', url='https://api.openai.com/v1/'):
    return OpenAIChat(model=model, base_url=url, api_key='test-key', stream=False)


def test_request_routes_and_converts_tools_history_and_options():
    llm = module()
    data, url, *_ = llm._prepare_request(
        'continue', temperature=0.2, top_p=0.8, max_tokens=500,
        tools=[{'type': 'function', 'function': {'name': 'read', 'parameters': {'type': 'object'}}}],
        llm_chat_history=[{'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'call_1', 'type': 'function', 'function': {'name': 'read', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'saved'}])
    assert url == 'https://api.openai.com/v1/responses'
    assert data['max_output_tokens'] == 500
    assert not {'messages', 'temperature', 'top_p', 'max_tokens', 'stream_options'} & data.keys()
    assert data['tools'][0]['name'] == 'read'
    assert data['tools'][0]['strict'] is False
    assert {'type': 'function_call_output', 'call_id': 'call_1', 'output': 'saved'} in data['input']
    assert any(x.get('type') == 'function_call' for x in data['input'])


@pytest.mark.parametrize(('model_name', 'url'), [
    ('gpt-4o', 'https://api.openai.com/v1/'), ('gpt-6.1-sol', 'http://provider.test/v1/'),
])
def test_existing_chat_routes_unchanged(model_name, url):
    data, endpoint, *_ = module(model_name, url)._prepare_request('hi', temperature=0.2)
    assert endpoint.endswith('/chat/completions')
    assert 'messages' in data and data['temperature'] == 0.2


def frames(events):
    return [('data: ' + json.dumps(e)).encode() for e in events]


def test_stream_tool_arguments_merge_and_usage():
    llm = module()
    state = ModelAttemptState()
    output = llm._response_parser(False).collect(frames([
        {'type': 'response.output_item.added', 'output_index': 1,
         'item': {'type': 'function_call', 'call_id': 'call_1', 'name': 'read'}},
        {'type': 'response.function_call_arguments.delta', 'output_index': 1, 'delta': '{"key":'},
        {'type': 'response.function_call_arguments.delta', 'output_index': 1, 'delta': '"prompt"}'},
        {'type': 'response.completed', 'response': {'status': 'completed',
         'output': [{'type': 'function_call'}], 'usage': {'input_tokens': 10, 'output_tokens': 5}}},
    ]), state)
    merged = llm._extract_specified_key_fields(llm._merge_stream_result(output))
    call = merged['tool_calls'][0] if isinstance(merged['tool_calls'], list) else merged['tool_calls']
    assert call['function']['arguments'] == '{"key":"prompt"}'
    assert call['id'] == 'call_1'
    assert state.finish == ModelFinish.TOOL_CALLS
    assert llm._extract_usage(output)['prompt_tokens'] == 10


def test_nonstream_text_and_incomplete():
    llm = module()
    state = ModelAttemptState()
    result = llm._response_parser(False).collect([json.dumps({
        'object': 'response', 'status': 'incomplete',
        'incomplete_details': {'reason': 'max_output_tokens'},
        'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'partial'}]}]})], state)
    assert result[0]['choices'][0]['delta']['content'] == 'partial'
    assert state.finish == ModelFinish.LENGTH


def test_failed_response_and_missing_terminal_are_errors():
    llm = module()
    with pytest.raises(_ModelResponseError):
        llm._response_parser(False).collect(frames([{'type': 'response.failed', 'response': {
            'error': {'code': 'server_error', 'type': 'server_error'}}}]), ModelAttemptState())
    with pytest.raises(_ModelResponseError):
        llm._response_parser(False).collect(frames([{'type': 'response.output_text.delta', 'delta': 'partial'}]),
                                            ModelAttemptState())


def test_image_and_named_tool_conversion():
    data = responses_request({'messages': [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'look'}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,abc'}}]}],
        'tool_choice': {'type': 'function', 'function': {'name': 'read'}}})
    assert data['input'][0]['content'][1]['type'] == 'input_image'
    assert data['tool_choice'] == {'type': 'function', 'name': 'read'}


def test_stream_text_is_not_duplicated_by_completed_snapshot():
    llm = module()
    state = ModelAttemptState()
    result = llm._response_parser(False).collect(frames([
        {'type': 'response.output_text.delta', 'delta': 'hello '},
        {'type': 'response.output_text.delta', 'delta': 'world'},
        {'type': 'response.completed', 'response': {'status': 'completed', 'output': [
            {'type': 'message', 'content': [{'type': 'output_text', 'text': 'hello world'}]}]}},
    ]), state)
    assert llm._extract_specified_key_fields(llm._merge_stream_result(result))['content'] == 'hello world'
    assert state.finish == ModelFinish.STOP


def test_nonstream_failure_does_not_look_successful():
    with pytest.raises(_ModelResponseError):
        module()._response_parser(False).collect([json.dumps({
            'object': 'response', 'status': 'failed',
            'error': {'code': 'server_error'}})], ModelAttemptState())


def test_html_survives_default_formatter_without_dict_stringification():
    llm = module()
    html = "<!DOCTYPE html>\n<html><style>body {font-family: 'Arial';}</style><body>正文</body></html>"
    for events in [
        [{'type': 'response.output_text.delta', 'delta': html},
         {'type': 'response.completed', 'response': {'status': 'completed', 'output': []}}],
        [{'object': 'response', 'status': 'completed', 'output': [
            {'type': 'message', 'content': [{'type': 'output_text', 'text': html}]}]}],
    ]:
        result = llm._response_parser(False).collect(frames(events), ModelAttemptState())
        message = llm._extract_specified_key_fields(llm._merge_stream_result(result))
        output = llm._formatter(message)
        assert isinstance(output, str)
        assert output == html
