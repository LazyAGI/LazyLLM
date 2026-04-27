import json
from unittest.mock import Mock, patch

from lazyllm import ChatPrompter, Document, OnlineChatModule, Reranker, Retriever, bind, parallel, pipeline
from lazyllm.tools.rag.doc_node import DocNode


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


def test_rag_tracing(exporter):
    prompt = (
        'Answer based only on context. Context:\n{context_str}\n\n'
        'Question: {query}\nAnswer:'
    )
    primary_nodes = []
    for index, score in enumerate([0.9, 0.7]):
        node = DocNode(uid=f'primary-{index}', text=f'primary doc {index}')
        node.similarity_score = score
        primary_nodes.append(node)

    secondary_nodes = []
    for index, score in enumerate([0.8, 0.6]):
        node = DocNode(uid=f'secondary-{index}', text=f'secondary doc {index}')
        node.similarity_score = score
        secondary_nodes.append(node)

    with patch.object(Retriever, '_init_submodules_and_embed_keys', return_value=None):
        primary = Retriever(Mock(spec=Document), group_name='sentences')
        secondary = Retriever(Mock(spec=Document), group_name='sentences')

    reranker = Reranker(name='ModuleReranker', model=rerank_model, topk=2)

    with patch.object(primary, 'forward', return_value=primary_nodes):
        with patch.object(secondary, 'forward', return_value=secondary_nodes):
            with patch.object(OnlineChatModule, 'forward', return_value='mock answer'):
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
    assert len(spans) == 8
    assert {s.name for s in spans[:2]} == {'primary', 'secondary'}
    assert [s.name for s in spans[2:]] == [
        'Parallel',
        'merge_doc_groups',
        'ModuleReranker',
        'format_rag_input',
        'llm',
        'Pipeline',
    ]
    assert all(span.context.trace_id == spans[0].context.trace_id for span in spans)

    parallel_span, merge_span, rerank_span, prompt_span, llm_span, pipeline_span = spans[2:]
    primary_span = [s for s in spans if s.name == 'primary'][0]
    secondary_span = [s for s in spans if s.name == 'secondary'][0]

    assert pipeline_span.parent is None
    assert all(s.parent.span_id == parallel_span.context.span_id for s in [primary_span, secondary_span])
    assert all(s.parent.span_id == pipeline_span.context.span_id for s in [
        parallel_span, merge_span, rerank_span, prompt_span, llm_span
    ])
    assert json.loads(llm_span.attributes.get('lazyllm.io.input')) == {
        'args': [{'context_str': 'primary doc 1 & primary doc 0', 'query': 'What is LazyLLM?'}],
        'kwargs': {},
    }
    assert json.loads(primary_span.attributes.get('lazyllm.output.similarity_scores')) == [0.9, 0.7]
    assert json.loads(secondary_span.attributes.get('lazyllm.output.similarity_scores')) == [0.8, 0.6]
    assert json.loads(rerank_span.attributes.get('lazyllm.output.relevance_scores')) == [0.95, 0.85]
