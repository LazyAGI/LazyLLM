import json
from unittest.mock import Mock

import pytest

import lazyllm
import lazyllm.tracing.backends.langfuse.backend as langfuse_backend_module
from lazyllm.tracing.consume.api import get_single_trace
from lazyllm.tracing.consume.reconstruction.extractors import extract_semantic
from lazyllm.tracing.consume.reconstruction.extractors.utils import doc_node_summaries
from lazyllm.tracing.consume.reconstruction.tree import rebuild
from lazyllm.tracing.datamodel.raw import RawSpanRecord, RawTraceRecord
from lazyllm.tracing.errors import ConsumeBackendError
from lazyllm.tracing.semantics import SemanticType

from .conftest import (
    CHILD_SPAN_ID,
    ROOT_SPAN_ID,
    TRACE_ID,
    make_langfuse_observation,
    make_langfuse_trace_payload,
    make_response,
    set_langfuse_env,
)


def span_id(value):
    return value * 16


def make_trace(trace_id=TRACE_ID, **overrides):
    data = {
        'trace_id': trace_id,
        'name': 'trace-name',
        'session_id': 'session-1',
        'user_id': 'user-1',
        'tags': ['tag-a'],
        'metadata': {'source': 'test'},
        'input': {'query': 'trace input'},
        'output': {'answer': 'trace output'},
        'start_time': 100.0,
        'end_time': None,
        'status': 'ok',
        'raw': {},
    }
    data.update(overrides)
    return RawTraceRecord(**data)


def make_span(
    span_id=ROOT_SPAN_ID, *,
    parent_span_id=None, name='span-name', start_time=100.0, end_time=101.0,
    status='ok', attributes=None, input=None, output=None, error_message=None, raw=None,
    **overrides,
):
    data = {
        'trace_id': TRACE_ID,
        'span_id': span_id,
        'parent_span_id': parent_span_id,
        'name': name,
        'start_time': start_time,
        'end_time': end_time,
        'status': status,
        'attributes': {'lazyllm.span.kind': 'callable'} if attributes is None else attributes,
        'input': {'args': ['input'], 'kwargs': {}} if input is None else input,
        'output': 'output' if output is None else output,
        'metadata': {},
        'error_message': error_message,
        'raw': {} if raw is None else raw,
    }
    data.update(overrides)
    return RawSpanRecord(**data)


def test_get_single_trace_rebuilds_structured_trace_from_langfuse_payload(monkeypatch):
    set_langfuse_env(monkeypatch)

    monkeypatch.setattr(
        langfuse_backend_module.requests, 'request',
        Mock(return_value=make_response(make_langfuse_trace_payload(
            trace_id=TRACE_ID, name='trace-name',
            sessionId='session-1', userId='user-1',
            tags=['tag-a'], metadata={'source': 'api-test'},
            observations=[
                make_langfuse_observation(
                    ROOT_SPAN_ID, name='root-step',
                    obs_input={'args': ['hello'], 'kwargs': {}},
                    obs_output={'root': 'ok'},
                    attrs={'lazyllm.semantic_type': 'workflow_control'},
                ),
                make_langfuse_observation(
                    CHILD_SPAN_ID, parent_id=ROOT_SPAN_ID,
                    obs_type='SPAN', name='child-step',
                    obs_output='child-output',
                    attrs={'lazyllm.span.kind': 'callable'},
                ),
            ],
        ))),
    )
    with lazyllm.config.temp('trace_consume_backend', 'langfuse'):
        trace = get_single_trace(TRACE_ID)

    assert trace.trace_id == TRACE_ID
    assert trace.metadata.name == 'trace-name'
    assert trace.metadata.session_id == 'session-1'
    assert trace.metadata.user_id == 'user-1'
    assert trace.metadata.tags == ['tag-a']
    assert trace.metadata.metadata == {'source': 'api-test'}
    assert trace.execution_tree.name == 'root-step'
    assert trace.execution_tree.node_type == 'flow'
    assert trace.execution_tree.semantic_type == 'workflow_control'
    assert trace.execution_tree.raw_data.input == {'args': ['hello'], 'kwargs': {}}
    assert trace.execution_tree.raw_data.output == {'root': 'ok'}
    assert [child.name for child in trace.execution_tree.children] == ['child-step']
    assert trace.execution_tree.children[0].raw_data.output == 'child-output'



def test_rebuild_nested_workflow_trace_with_semantic_steps():
    trace = make_trace(start_time=None)
    root = make_span(
        name='workflow-root',
        start_time=10.0,
        end_time=14.0,
        attributes={
            'lazyllm.span.kind': 'flow',
            'lazyllm.semantic_type': 'workflow_control',
        },
        input={'args': ['question'], 'kwargs': {}},
        output={'answer': 'done'},
    )
    first_child = make_span(
        CHILD_SPAN_ID,
        parent_span_id=ROOT_SPAN_ID,
        name='call-a',
        start_time=11.0,
    )
    second_child = make_span(
        span_id('3'),
        parent_span_id=ROOT_SPAN_ID,
        name='call-b',
        start_time=12.0,
    )

    structured = rebuild(trace, [second_child, root, first_child])

    assert structured.trace_id == TRACE_ID
    assert structured.metadata.start_time == 10.0
    assert structured.execution_tree.step_id == ROOT_SPAN_ID
    assert structured.execution_tree.name == 'workflow-root'
    assert structured.execution_tree.node_type == 'flow'
    assert structured.execution_tree.semantic_type == 'workflow_control'
    assert structured.execution_tree.raw_data.input == {'args': ['question'], 'kwargs': {}}
    assert structured.execution_tree.raw_data.output == {'answer': 'done'}
    assert [child.step_id for child in structured.execution_tree.children] == [CHILD_SPAN_ID, span_id('3')]


@pytest.mark.parametrize(
    ('spans', 'expected_child_ids'),
    [
        ([], []),
        ([
            make_span(ROOT_SPAN_ID, name='root-a', start_time=10.0),
            make_span(CHILD_SPAN_ID, name='root-b', start_time=11.0),
        ], [ROOT_SPAN_ID, CHILD_SPAN_ID]),
        ([
            make_span(span_id('4'), parent_span_id=span_id('7'), name='orphan'),
        ], [span_id('4')]),
        ([
            make_span(span_id('5'), parent_span_id=span_id('6'), name='cycle-a'),
            make_span(span_id('6'), parent_span_id=span_id('5'), name='cycle-b'),
        ], [span_id('5'), span_id('6')]),
    ],
)
def test_rebuild_uses_virtual_root_for_empty_or_ambiguous_graphs(spans, expected_child_ids):
    trace = make_trace(name='virtual-trace', input = {'query': 'I am query.'}, start_time=9.0)

    structured = rebuild(trace, spans)

    assert structured.execution_tree.step_id == '__root__'
    assert structured.execution_tree.name == 'virtual-trace'
    assert structured.execution_tree.raw_data.input == {'query': 'I am query.'}
    assert [child.step_id for child in structured.execution_tree.children] == expected_child_ids


def test_rebuild_aggregates_error_status_latency_and_trace_metadata():
    trace = make_trace(
        name='metadata-trace',
        start_time=None,
        end_time=None,
        status=None,
        tags=['prod', 'debug'],
        metadata={'tenant': 'acme'},
    )
    root = make_span(
        ROOT_SPAN_ID,
        name='root',
        start_time=5.0,
        end_time=8.0,
        attributes={'lazyllm.span.kind': 'flow'},
    )
    child = make_span(
        CHILD_SPAN_ID,
        parent_span_id=ROOT_SPAN_ID,
        name='failed-child',
        start_time=6.5,
        end_time=7.0,
        status='error',
        error_message='boom',
    )

    structured = rebuild(trace, [root, child])

    assert structured.metadata.name == 'metadata-trace'
    assert structured.metadata.status == 'error'
    assert structured.metadata.error_message == 'boom'
    assert structured.metadata.start_time == 5.0
    assert structured.metadata.end_time == 8.0
    assert structured.metadata.latency_ms == 3000.0
    assert structured.metadata.tags == ['prod', 'debug']
    assert structured.metadata.metadata == {'tenant': 'acme'}
    assert structured.execution_tree.children[0].status == 'error'
    assert structured.execution_tree.children[0].error_message == 'boom'


def test_llm_extractor_maps_model_usage_and_config():
    data = extract_semantic(make_span(
        attributes={
            'lazyllm.semantic_type': SemanticType.LLM,
            'lazyllm.span.kind': 'llm',
            'gen_ai.request.model': 'gpt-test',
            'gen_ai.usage.input_tokens': '3',
            'gen_ai.usage.output_tokens': 5,
            'lazyllm.entity.config.static_params': {'temperature': 0.2, 'top_p': '0.9'},
            'lazyllm.entity.config.max_tokens': '128',
        },
        input={'args': [{'query': 'hello', 'context_str': 'ctx'}], 'kwargs': {}},
        output='answer',
    ))

    assert data['model_name'] == 'gpt-test'
    assert data['query'] == 'hello'
    assert data['context_str'] == 'ctx'
    assert data['usage'] == {'input_tokens': 3, 'output_tokens': 5}
    assert data['temperature'] == 0.2
    assert data['top_p'] == 0.9
    assert data['max_tokens'] == 128
    assert data['answer_length'] == 6


def test_retriever_extractor_maps_query_filters_and_doc_nodes():
    data = extract_semantic(make_span(
        attributes={
            'lazyllm.semantic_type': SemanticType.RETRIEVER,
            'lazyllm.output.doc_count': '2',
            'lazyllm.output.similarity_scores': json.dumps([0.7, '0.6', 'bad']),
            'lazyllm.entity.config.topk': '5',
        },
        input={'args': ['find docs'], 'kwargs': {'filters': {'source': 'kb'}}},
        output=[
            {'id': 'doc-1', 'group': 'g1', 'content': 'alpha'},
            {'node_id': 'doc-2', 'text': 'beta'},
        ],
    ))

    assert data['query'] == 'find docs'
    assert data['filters'] == {'source': 'kb'}
    assert data['topk'] == 5
    assert data['node_count'] == 2
    assert data['returned_node_ids'] == ['doc-1', 'doc-2']
    assert data['scores'] == [0.7, 0.6]
    assert data['returned_nodes'][0] == {'id': 'doc-1', 'group': 'g1', 'content_preview': 'alpha'}


def test_embedding_extractor_maps_model_dimension_and_batch():
    data = extract_semantic(make_span(
        attributes={
            'lazyllm.semantic_type': SemanticType.EMBEDDING,
            'gen_ai.request.model': 'embed-model',
            'lazyllm.entity.config.batch_size': '16',
        },
        input={'args': [['a', 'b', 'c']], 'kwargs': {}},
        output=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    ))

    assert data['model_name'] == 'embed-model'
    assert data['input_count'] == 3
    assert data['dim'] == 3
    assert data['batch_size'] == 16
    assert data['output_truncated'] is False


def test_rerank_extractor_maps_scores_ranking_and_model():
    data = extract_semantic(make_span(
        attributes={
            'lazyllm.semantic_type': SemanticType.RERANK,
            'lazyllm.output.relevance_scores': json.dumps([0.9, 0.4]),
            'lazyllm.entity.config.topk': '2',
            'lazyllm.entity.config.model': 'rerank-model',
        },
        input={
            'args': [[{'id': 'doc-1', 'content': 'alpha'}, {'id': 'doc-2', 'content': 'beta'}]],
            'kwargs': {'query': 'rank docs'},
        },
        output=[{'id': 'doc-2', 'content': 'beta'}, {'id': 'doc-1', 'content': 'alpha'}],
    ))

    assert data['query'] == 'rank docs'
    assert data['candidate_node_count'] == 2
    assert data['topk'] == 2
    assert data['scores'] == [0.9, 0.4]
    assert data['candidate_doc_ids'] == ['doc-1', 'doc-2']
    assert data['ranked_doc_ids'] == ['doc-2', 'doc-1']
    assert data['rerank_model'] == 'rerank-model'


def test_doc_node_summaries_normalizes_supported_inputs_and_limits():
    docs = [
        {'id': 'dict-id', 'group_name': 'dict-group', 'text': 'abcdef'},
        'DocNode(id:repr-id, group:repr-group, content:hello world) parent:None',
        {'content': 'missing id'},
    ]

    assert doc_node_summaries(docs, limit=2, content_limit=4) == [
        {'id': 'dict-id', 'group': 'dict-group', 'content_preview': 'abcd...<truncated>'},
        {'id': 'repr-id', 'group': 'repr-group', 'content_preview': 'hell...<truncated>'},
    ]
    assert doc_node_summaries({'text': 'missing id'}) is None
