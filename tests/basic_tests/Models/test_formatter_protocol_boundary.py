'''Provider continuation belongs to message projections, never business values.'''
import json
from io import BytesIO

import pytest
import requests

import lazyllm
from pydantic import BaseModel, ConfigDict

from lazyllm.components import FunctionCallFormatter, JsonFormatter
from lazyllm.components.formatter.formatterbase import PipelineFormatter
from lazyllm.flow import Pipeline
from lazyllm.module.llms.onlinemodule.supplier.openai_responses import OpenAIResponsesChat
from lazyllm.tools.writer.tools.base import WriterToolBase


@pytest.fixture
def model_response(monkeypatch, model_call_session):
    def install(value):
        output = [{'type': 'message', 'id': 'msg_1', 'role': 'assistant', 'status': 'completed',
                   'content': [{'type': 'output_text', 'text': json.dumps(value), 'annotations': []}]}]

        def send(session, request, **kwargs):
            body = json.loads(request.body)
            response = requests.Response()
            response.status_code = 200
            payload = {'status': 'completed', 'output': output}
            if body['stream']:
                data = 'data: ' + json.dumps({'type': 'response.completed', 'response': payload}) + '\n\n'
                response.raw = BytesIO(data.encode())
            else:
                response._content = json.dumps(payload).encode()
                response._content_consumed = True
                response.raw = BytesIO()
            return response

        monkeypatch.setattr(requests.Session, 'send', send)
        return OpenAIResponsesChat(model='test', api_key='test', base_url='https://provider.test/v1/'), output
    try:
        yield install
    finally:
        lazyllm.FileSystemQueue().clear()


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('projection', ['json', 'pipeline', 'callable'])
@pytest.mark.parametrize('payload', [
    {'answer': 42}, [{'answer': 42}],
    {'role': 'assistant', 'content': 'business', '_lazyllm_provider_items': 'user-owned'},
])
def test_business_projection_is_not_modified(model_response, stream, projection, payload):
    model, _ = model_response(payload)
    formatter = {
        'json': JsonFormatter(),
        'pipeline': PipelineFormatter(Pipeline(json.loads, lambda value: value)),
        'callable': lambda message: json.loads(message['content']),
    }[projection]
    assert model.formatter(formatter)('question', stream_output=stream) == payload


@pytest.mark.parametrize('stream', [False, True])
def test_writer_strict_schema_accepts_clean_business_result(model_response, stream):
    class Selection(BaseModel):
        model_config = ConfigDict(extra='forbid')
        answer: int

    model, _ = model_response({'answer': 42})
    result = WriterToolBase(llm=model)._call_llm_structured('question', Selection, stream_output=stream)
    assert result.model_dump() == {'answer': 42}


@pytest.mark.parametrize('stream', [False, True])
def test_explicit_message_projection_preserves_replay_without_mutating_formatter_value(model_response, stream):
    model, output = model_response({'answer': 42})
    owned = {'role': 'assistant', 'content': '{"answer": 42}'}

    def message_formatter(message):
        return owned

    message_formatter.output_kind = 'message'
    result = model.formatter(message_formatter)('question', stream_output=stream)
    assert result['_lazyllm_provider_items']['output'] == output
    assert owned == {'role': 'assistant', 'content': '{"answer": 42}'}


def test_invalid_message_projection_fails_explicitly(model_response):
    model, _ = model_response({'answer': 42})

    def message_formatter(message):
        return 'lost message'

    message_formatter.output_kind = 'message'
    with pytest.raises(Exception, match='message.*dictionary'):
        model.formatter(message_formatter)('question', stream_output=False)


def test_function_call_formatter_share_preserves_protocol_state(model_response):
    model, output = model_response({'answer': 42})
    shared = model.share(format=FunctionCallFormatter())
    result = shared('question', stream_output=False)
    assert result['_lazyllm_provider_items']['output'] == output
    assert model.formatter(JsonFormatter())('question', stream_output=False) == {'answer': 42}


def test_dynamic_router_share_keeps_value_and_message_contracts(model_response):
    from lazyllm.module.llms.onlinemodule.chat import OnlineChatModule, dynamic_chat_config

    _, output = model_response({'answer': 42})
    model = OnlineChatModule(source='dynamic')
    with dynamic_chat_config(source='openai_responses', model='test',
                             url='https://provider.test/v1/', skip_auth=True):
        message = model.share(format=FunctionCallFormatter())('question', stream_output=False)
        assert message['_lazyllm_provider_items']['output'] == output
        value = model.share(format=JsonFormatter())('question', stream_output=False)
        assert value == {'answer': 42}


def test_permissive_writer_model_does_not_store_protocol_metadata(model_response):
    from lazyllm.tools.writer.data_models.writer_ir import WriterBlock

    model, _ = model_response({'node_id': 'section-1', 'type': 'paragraph', 'content': 'Evidence'})
    block = WriterToolBase(llm=model)._call_llm_structured('question', WriterBlock)
    assert block.content == 'Evidence'
    assert '_lazyllm_provider_items' not in block.model_dump_json()
