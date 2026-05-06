import json
from types import SimpleNamespace

from lazyllm.tracing.consume.reconstruction.extractors import extract_semantic
from lazyllm.tracing.consume.reconstruction.extractors.utils import doc_node_summaries
from lazyllm.tracing.consume.reconstruction.tree import rebuild
from lazyllm.tracing.datamodel.raw import RawSpanRecord, RawTraceRecord
from lazyllm.tracing.semantics import SemanticType


def span(
    trace_id, span_id, parent_span_id, name, start_time, end_time, *,
    kind='callable', semantic_type=None, attributes=None,
    status='ok', input=None, output=None, error_message=None,
):
    attrs = {'lazyllm.span.kind': kind}
    if semantic_type is not None:
        attrs['lazyllm.semantic_type'] = semantic_type
    attrs.update(attributes or {})
    return RawSpanRecord(
        trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id, name=name,
        start_time=start_time, end_time=end_time, status=status, attributes=attrs,
        input=input, output=output, metadata={}, error_message=error_message, raw={},
    )


def test_rebuild_handles_complex_tree_metadata_error_and_orphan():
    trace_id = '0' * 32
    root_id, parallel_id, retriever_a_id, retriever_b_id, merge_id, rerank_id, llm_id, orphan_id = [
        value * 16 for value in '12345678'
    ]

    trace = RawTraceRecord(
        trace_id=trace_id, name='complex-trace', session_id='session-1', user_id='user-1',
        tags=['prod', 'debug'], metadata={'tenant': 'acme'},
        input={'query': 'trace input'}, output={'answer': 'trace output'},
        start_time=None, end_time=None, status=None, raw={},
    )

    merge = span(trace_id, merge_id, root_id, 'merge-docs', 13.0, 13.4, output=['doc-a', 'doc-b'])
    root = span(
        trace_id, root_id, None, 'workflow-root', 10.0, 18.0,
        kind='flow', semantic_type=SemanticType.WORKFLOW_CONTROL,
        input={'args': ['question'], 'kwargs': {}}, output={'answer': 'done'},
    )
    retriever_b = span(
        trace_id, retriever_b_id, parallel_id, 'retriever-b', 12.0, 12.5,
        kind='module', semantic_type=SemanticType.RETRIEVER, output=[{'id': 'doc-b'}],
    )
    parallel = span(trace_id, parallel_id, root_id, 'parallel-retrieval', 11.0, 12.8, kind='flow')
    retriever_a = span(
        trace_id, retriever_a_id, parallel_id, 'retriever-a', 11.2, 11.8,
        kind='module', semantic_type=SemanticType.RETRIEVER, output=[{'id': 'doc-a'}],
    )
    rerank = span(
        trace_id, rerank_id, root_id, 'rerank-docs', 14.0, 14.5,
        kind='module', semantic_type=SemanticType.RERANK,
        output=[{'id': 'doc-b'}, {'id': 'doc-a'}],
    )
    llm = span(
        trace_id, llm_id, root_id, 'call-llm', 15.0, 16.0,
        kind='module', semantic_type=SemanticType.LLM, status='error',
        attributes={'gen_ai.request.model': 'gpt-test'},
        input={'args': [{'query': 'question', 'context_str': 'ctx'}], 'kwargs': {}},
        error_message='provider boom',
    )
    orphan = span(trace_id, orphan_id, 'a' * 16, 'orphan-cleanup', 19.0, 20.0, output='orphan-output')
    spans = [merge, root, retriever_b, parallel, retriever_a, rerank, llm, orphan]

    structured = rebuild(trace, spans)

    assert structured.trace_id == trace_id
    assert structured.metadata.status == 'error'
    assert structured.metadata.error_message == 'provider boom'
    assert structured.metadata.latency_ms == 10000.0
    assert structured.metadata.tags == ['prod', 'debug']
    assert structured.metadata.metadata == {'tenant': 'acme'}

    tree = structured.execution_tree
    assert (tree.step_id, tree.name, tree.node_type) == ('__root__', 'complex-trace', 'flow')
    assert tree.raw_data.input == {'query': 'trace input'}
    assert [child.step_id for child in tree.children] == [root_id, orphan_id]

    root = tree.children[0]
    parallel, merge, rerank, llm = root.children
    assert (root.name, root.semantic_type, root.latency_ms) == (
        'workflow-root', SemanticType.WORKFLOW_CONTROL, 8000.0,
    )
    assert [child.name for child in parallel.children] == ['retriever-a', 'retriever-b']
    assert (merge.raw_data.output, rerank.semantic_type) == (['doc-a', 'doc-b'], SemanticType.RERANK)
    assert (llm.status, llm.error_message, llm.semantic_data['model_name']) == (
        'error', 'provider boom', 'gpt-test',
    )
    assert tree.children[1].raw_data.output == 'orphan-output'


def test_semantic_extractors_map_representative_fields():
    llm = extract_semantic(SimpleNamespace(
        span_id='llm',
        name='llm',
        attributes={
            'lazyllm.semantic_type': SemanticType.LLM,
            'gen_ai.request.model': 'gpt-test',
            'gen_ai.usage.input_tokens': '3',
            'gen_ai.usage.output_tokens': 5,
            'lazyllm.entity.config.static_params': {'temperature': 0.2},
            'lazyllm.entity.config.max_tokens': '128',
        },
        input={'args': [{'query': 'hello', 'context_str': 'ctx'}], 'kwargs': {}},
        output='answer',
        raw={},
    ))
    retriever = extract_semantic(SimpleNamespace(
        span_id='retriever',
        name='retriever',
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
        raw={},
    ))
    embedding = extract_semantic(SimpleNamespace(
        span_id='embedding',
        name='embedding',
        attributes={
            'lazyllm.semantic_type': SemanticType.EMBEDDING,
            'gen_ai.request.model': 'embed-model',
            'lazyllm.entity.config.batch_size': '16',
        },
        input={'args': [['a', 'b', 'c']], 'kwargs': {}},
        output=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        raw={},
    ))
    rerank = extract_semantic(SimpleNamespace(
        span_id='rerank',
        name='rerank',
        attributes={
            'lazyllm.semantic_type': SemanticType.RERANK,
            'lazyllm.output.relevance_scores': json.dumps([0.9, 0.4]),
            'lazyllm.entity.config.model': 'rerank-model',
        },
        input={
            'args': [[{'id': 'doc-1', 'content': 'alpha'}, {'id': 'doc-2', 'content': 'beta'}]],
            'kwargs': {'query': 'rank docs'},
        },
        output=[{'id': 'doc-2', 'content': 'beta'}, {'id': 'doc-1', 'content': 'alpha'}],
        raw={},
    ))

    assert llm['model_name'] == 'gpt-test'
    assert llm['usage'] == {'input_tokens': 3, 'output_tokens': 5}
    assert (llm['temperature'], llm['max_tokens'], llm['answer_length']) == (0.2, 128, 6)

    assert retriever['query'] == 'find docs'
    assert retriever['scores'] == [0.7, 0.6]
    assert retriever['returned_nodes'][0] == {'id': 'doc-1', 'group': 'g1', 'content_preview': 'alpha'}

    assert (embedding['model_name'], embedding['input_count'], embedding['dim']) == ('embed-model', 3, 3)
    assert (rerank['rerank_model'], rerank['ranked_doc_ids']) == ('rerank-model', ['doc-2', 'doc-1'])
    assert doc_node_summaries(
        [
            {'id': 'dict-id', 'group_name': 'dict-group', 'text': 'abcdef'},
            'DocNode(id:repr-id, group:repr-group, content:hello world) parent:None',
        ],
        limit=1,
        content_limit=4,
    ) == [{'id': 'dict-id', 'group': 'dict-group', 'content_preview': 'abcd...<truncated>'}]
