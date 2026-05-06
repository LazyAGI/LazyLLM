import json
from unittest.mock import Mock, patch

import pytest

import lazyllm
from lazyllm import (
    ChatPrompter,
    Document,
    LazyTraceContext,
    OnlineChatModule,
    OnlineEmbeddingModule,
    Reranker,
    Retriever,
    bind,
    parallel,
    pipeline,
    set_trace_context,
)
from lazyllm.tools.rag.doc_node import DocNode


def test_trace_controls_disable_and_overrides(exporter):
    with pipeline() as disabled_context_flow:
        disabled_context_flow.add_one = lambda value: value + 1
    set_trace_context(LazyTraceContext(enabled=False, sampled=True))

    assert disabled_context_flow(1) == 2
    assert not exporter.get_finished_spans()
    exporter.clear()

    set_trace_context(LazyTraceContext(enabled=True, sampled=True))
    with lazyllm.config.temp('trace_enabled', False):
        with pipeline() as disabled_config_flow:
            disabled_config_flow.add_one = lambda value: value + 1
        assert disabled_config_flow(1) == 2
    assert not exporter.get_finished_spans()
    exporter.clear()

    set_trace_context(LazyTraceContext(enabled=True, module_trace={
        'by_class': {'OnlineEmbeddingModule': False, 'OnlineChatModule': False},
        'by_name': {'test_embed': True},
    }))
    with patch.object(OnlineChatModule, 'forward', return_value=1), \
            patch.object(OnlineEmbeddingModule, 'forward', return_value=1):
        llm = OnlineChatModule(source='dynamic', type='llm', model='mock-chat')
        embedding = OnlineEmbeddingModule(
            source='dynamic', type='embed', model='mock-embedding', name='test_embed')
        with pipeline() as flow:
            flow.llm = llm
            flow.embedding = embedding
            flow.add_one = lambda value: value + 1

        assert flow('hello') == 2

    assert [span.name for span in exporter.get_finished_spans()] == ['test_embed', '<lambda>', 'Pipeline']


def test_nested_parallel_error_propagates_status_to_parent_flows(exporter):
    def first(value):
        return value + 1

    def raises_error(value):
        raise ValueError(f'boom:{value}')

    def unreachable(value):
        return value * 2

    with pipeline() as flow:
        with parallel(_concurrent=2) as branches:
            branches.first = first
            branches.raises_error = raises_error
        flow.unreachable = unreachable

    with pytest.raises(Exception, match='boom:1'):
        flow(1)

    spans = exporter.get_finished_spans()

    by_name = {span.name: span for span in spans}
    assert all(
        by_name[name].attributes.get('lazyllm.status') == 'error'
        for name in ('raises_error', 'Parallel', 'Pipeline')
    )
    assert by_name['raises_error'].attributes.get('lazyllm.error.message') == 'boom:1'


def merge_doc_groups(*groups):
    docs = []
    for group in groups:
        docs.extend(group)
    return docs


def format_rag_input(nodes, query):
    return {
        'context_str': ' & '.join(node.get_content() for node in nodes),
        'query': query,
    }


def rerank_model(query, documents, top_n):
    return [(1, 0.95), (0, 0.85)]


def test_rag_tracing_records_core_spans(exporter):
    prompt = (
        'Answer based only on context. Context:\n{context_str}\n\n'
        'Question: {query}\nAnswer:'
    )
    primary_nodes, secondary_nodes = [], []
    for prefix, scores, target in (
        ('primary', [0.9, 0.7], primary_nodes),
        ('secondary', [0.8, 0.6], secondary_nodes),
    ):
        for index, score in enumerate(scores):
            node = DocNode(uid=f'{prefix}-{index}', text=f'{prefix} doc {index}')
            node.similarity_score = score
            target.append(node)

    with patch.object(Retriever, '_init_submodules_and_embed_keys', return_value=None):
        primary = Retriever(Mock(spec=Document), group_name='sentences')
        secondary = Retriever(Mock(spec=Document), group_name='sentences')

    reranker = Reranker(name='ModuleReranker', model=rerank_model, topk=2)

    with (
        patch.object(primary, 'forward', return_value=primary_nodes),
        patch.object(secondary, 'forward', return_value=secondary_nodes),
        patch.object(OnlineChatModule, 'forward', return_value='mock answer'),
    ):
        llm = OnlineChatModule(source='dynamic', type='llm', model='mock-chat')
        with pipeline() as rag:
            with parallel() as retrieval:
                retrieval.primary = primary
                retrieval.secondary = secondary
            rag.merge_doc_groups = merge_doc_groups
            rag.reranker = reranker
            rag.format_rag_input = format_rag_input | bind(query=rag.input)
            rag.llm = llm.prompt(ChatPrompter(prompt, extra_keys=['context_str']))
        result = rag('What is LazyLLM?')

    spans = exporter.get_finished_spans()
    assert result == 'mock answer'
    assert {span.name for span in spans[:2]} == {'primary', 'secondary'}
    assert [span.name for span in spans[2:]] == [
        'Parallel',
        'merge_doc_groups',
        'ModuleReranker',
        'format_rag_input',
        'llm',
        'Pipeline',
    ]

    by_name = {span.name: span for span in spans}
    assert json.loads(by_name['primary'].attributes['lazyllm.output.similarity_scores']) == [0.9, 0.7]
    assert json.loads(by_name['ModuleReranker'].attributes['lazyllm.output.relevance_scores']) == [0.95, 0.85]
    assert json.loads(by_name['llm'].attributes['lazyllm.io.input']) == {
        'args': [{'context_str': 'primary doc 1 & primary doc 0', 'query': 'What is LazyLLM?'}],
        'kwargs': {},
    }
