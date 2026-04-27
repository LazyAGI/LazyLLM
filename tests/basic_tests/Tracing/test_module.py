import json
from unittest.mock import Mock, patch

from lazyllm import (
    Document,
    OnlineChatModule,
    OnlineEmbeddingModule,
    Reranker,
    Retriever,
    ToolManager,
)
from lazyllm.tools.agent.reactAgent import ReactAgent
from lazyllm.tools.rag.doc_node import DocNode


def test_online_chat_module_tracing(exporter):
    with patch.object(OnlineChatModule, 'forward', return_value='mock response'):
        module = OnlineChatModule(
            source='dynamic', type='llm', model='mock-chat',
            url='http://mock-api.example.com',
        )
        result = module('hello')

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == 'OnlineChatModule'
    assert spans[0].attributes.get('lazyllm.span.kind') == 'module'
    assert spans[0].attributes.get('lazyllm.semantic_type') == 'llm'
    assert spans[0].attributes.get('lazyllm.entity.config.model') == 'mock-chat'
    assert spans[0].attributes.get('gen_ai.request.model') == 'mock-chat'
    assert spans[0].attributes.get('lazyllm.entity.class') == 'OnlineChatModule'
    assert spans[0].attributes.get('lazyllm.entity.config.base_url') == 'http://mock-api.example.com'
    assert json.loads(spans[0].attributes.get('lazyllm.io.input')) == {'args': ['hello'], 'kwargs': {}}
    assert spans[0].attributes.get('lazyllm.io.output') == 'mock response'
    assert result == 'mock response'


def test_online_embedding_module_tracing(exporter):
    embeddings = [[0.1, 0.2], [0.3, 0.4]]

    with patch.object(OnlineEmbeddingModule, 'forward', return_value=embeddings):
        module = OnlineEmbeddingModule(source='dynamic', type='embed', model='mock-embedding')
        result = module(['first', 'second'])

    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ['OnlineEmbeddingModule']
    assert spans[0].attributes.get('lazyllm.semantic_type') == 'embedding'
    assert spans[0].attributes.get('lazyllm.entity.config.model') == 'mock-embedding'
    assert json.loads(spans[0].attributes.get('lazyllm.io.output')) == embeddings
    assert result == embeddings


def test_retriever_tracing(exporter):
    document = Mock(spec=Document)
    nodes = []
    for index, score in enumerate([0.9, 0.8, 0.7]):
        node = DocNode(uid=str(index), text=f'doc-{index}')
        node.similarity_score = score
        nodes.append(node)

    with patch.object(Retriever, '_init_submodules_and_embed_keys', return_value=None):
        retriever = Retriever(document, group_name='sentences', similarity='cosine')
    with patch.object(retriever, 'forward', return_value=nodes):
        result = retriever('query')

    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ['Retriever']
    assert spans[0].attributes.get('lazyllm.semantic_type') == 'retriever'
    assert spans[0].attributes.get('lazyllm.output.doc_count') == 3
    assert json.loads(spans[0].attributes.get('lazyllm.output.similarity_scores')) == [0.9, 0.8, 0.7]
    assert spans[0].attributes.get('lazyllm.entity.config.similarity') == 'cosine'
    assert result == nodes


def test_reranker_tracing(exporter):
    nodes = [DocNode(uid='0', text='doc-0'), DocNode(uid='1', text='doc-1')]

    def mock_reranker_fn(query, documents, top_n):
        return [(0, 0.95), (1, 0.85)]

    with patch('lazyllm.tools.rag.rerank.lazyllm.TrainableModule', return_value=mock_reranker_fn):
        reranker = Reranker(name='ModuleReranker', model='mock-reranker', topk=2)
    result = reranker(nodes, query='query')

    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ['ModuleReranker']
    assert spans[0].attributes.get('lazyllm.semantic_type') == 'rerank'
    assert spans[0].attributes.get('lazyllm.entity.config.model') == 'mock-reranker'
    assert spans[0].attributes.get('lazyllm.output.doc_count') == 2
    assert json.loads(spans[0].attributes.get('lazyllm.output.relevance_scores')) == [0.95, 0.85]
    assert [node.uid for node in result] == ['0', '1']
    assert [node.relevance_score for node in result] == [0.95, 0.85]


def test_agent_module_tracing(exporter, tmp_path):
    mock_llm = Mock()
    with patch('lazyllm.tools.agent.base.create_sandbox', return_value=None), \
         patch('lazyllm.tools.agent.base.ToolManager'):
        agent = ReactAgent(llm=mock_llm, tools=['mock_tool'], workspace=str(tmp_path))

    def build_agent():
        agent._agent = lambda query, history: f'agent:{query}'

    with patch.object(agent, 'build_agent', side_effect=build_agent):
        result = agent('input')

    spans = exporter.get_finished_spans()
    assert len(spans) == 1 and spans[0].name == 'ReactAgent'
    assert spans[0].attributes.get('lazyllm.semantic_type') == 'agent'
    assert spans[0].attributes.get('lazyllm.span.kind') == 'module'
    assert spans[0].attributes.get('lazyllm.entity.class') == 'ReactAgent'
    assert result == 'agent:input'


def test_tool_manager_module_tracing(exporter):
    manager = ToolManager([])

    with patch.object(manager, 'forward', return_value=['tool result']):
        result = manager({'function': {'name': 'missing', 'arguments': '{}'}})

    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ['ToolManager']
    assert spans[0].attributes.get('lazyllm.semantic_type') == 'tool'
    assert result == ['tool result']
