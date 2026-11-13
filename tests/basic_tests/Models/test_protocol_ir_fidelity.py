from copy import deepcopy

from lazyllm.module.llms.onlinemodule.base.protocol_ir import ResponseIR


def test_output_metadata_and_refusal_survive_projection_and_replay():
    output = [{'type': 'message', 'id': 'msg_1', 'role': 'assistant', 'status': 'completed', 'content': [
        {'type': 'output_text', 'text': 'Source', 'annotations': [
            {'type': 'url_citation', 'start_index': 0, 'end_index': 6,
             'url': 'https://example.test', 'title': 'Source'}]},
        {'type': 'refusal', 'refusal': 'Cannot comply'},
    ]}]
    ir = ResponseIR.from_responses({'status': 'completed', 'output': output})
    message = ir.to_chat_chunk()['choices'][0]['message']
    assert message['refusal'] == 'Cannot comply'
    assert ir.message.to_responses() == output
    from lazyllm.module.llms.onlinemodule.base.protocol_ir import MessageIR
    assert MessageIR.from_chat(message).to_responses() == output
    modified = deepcopy(message)
    modified['content'] = 'Compacted'
    replay = MessageIR.from_chat(modified).to_responses()
    assert 'Source' not in str(replay)
    assert 'Compacted' in str(replay)


def test_changed_tool_arguments_do_not_replay_original_snapshot():
    from lazyllm.module.llms.onlinemodule.base.protocol_ir import MessageIR
    ir = ResponseIR.from_responses({'status': 'completed', 'output': [
        {'type': 'function_call', 'id': 'fc_1', 'call_id': 'call_1', 'name': 'render', 'arguments': '{"page":1}'},
    ]})
    message = ir.to_chat_chunk()['choices'][0]['message']
    message['tool_calls'][0]['function']['arguments'] = '{"page":2}'
    assert MessageIR.from_chat(message).to_responses()[0]['arguments'] == '{"page":2}'


def test_completions_and_responses_have_same_semantic_response():
    chat = ResponseIR.from_chat({'choices': [{'message': {'role': 'assistant', 'content': [
        {'type': 'text', 'text': 'Done'}]}, 'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 2, 'completion_tokens': 1}})
    native = ResponseIR.from_responses({'status': 'completed', 'output': [
        {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'Done'}]}],
        'usage': {'input_tokens': 2, 'output_tokens': 1}})
    assert chat.message.text == native.message.text == 'Done'
    assert chat.finish == native.finish
    assert chat.usage['prompt_tokens'] == native.usage['prompt_tokens']


def test_role_and_media_edit_invalidate_native_snapshot():
    from lazyllm.module.llms.onlinemodule.base.protocol_ir import MessageIR
    output = [{'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'Hello'}]}]
    message = ResponseIR.from_responses({'output': output}).to_chat_chunk()['choices'][0]['message']
    message['role'] = 'user'
    assert MessageIR.from_chat(message).to_responses()[0]['role'] == 'user'
    message['role'] = 'assistant'
    message['content'] = [{'type': 'text', 'text': 'Hello'},
                          {'type': 'image_url', 'image_url': {'url': 'https://example.test/a.png'}}]
    assert 'input_image' in str(MessageIR.from_chat(message).to_responses())


def test_default_text_formatter_projects_typed_content(monkeypatch):
    import json
    from io import BytesIO
    import requests
    from lazyllm import OnlineChatModule

    def send(session, request, **kwargs):
        response = requests.Response()
        response.status_code = 200
        message = {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Done'}],
                   'reasoning_content': 'Checked'}
        response._content = json.dumps({'choices': [{'message': message, 'finish_reason': 'stop'}]}).encode()
        response._content_consumed = True
        response.raw = BytesIO()
        return response

    monkeypatch.setattr(requests.Session, 'send', send)
    model = OnlineChatModule(source='openai', model='test', api_key='test', stream=False)
    assert model('Finish') == '<think>Checked</think>Done'
