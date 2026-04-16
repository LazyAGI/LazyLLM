from unittest.mock import MagicMock

import pytest

from lazyllm.tools.rag.doc_node import DocNode
from lazyllm.tools.rag.global_metadata import RAG_DOC_ID, RAG_KB_ID
from lazyllm.tools.rag.store.document_store import _DocumentStore


def test_update_nodes_raises_when_store_upsert_returns_false():
    document_store = _DocumentStore(algo_name='test_algo', store={'type': 'map'})
    document_store.activate_group('group1')
    document_store.impl.upsert = MagicMock(return_value=False)

    node = DocNode(
        uid='node1',
        text='text1',
        group='group1',
        global_metadata={RAG_KB_ID: 'kb1', RAG_DOC_ID: 'doc1'},
    )

    with pytest.raises(RuntimeError, match='Failed to upsert segments for group group1'):
        document_store.update_nodes([node], copy=True)


def test_update_doc_meta_raises_when_store_upsert_returns_false():
    document_store = _DocumentStore(algo_name='test_algo', store={'type': 'map'})
    document_store.activate_group('group1')

    node = DocNode(
        uid='node1',
        text='text1',
        group='group1',
        global_metadata={RAG_KB_ID: 'kb1', RAG_DOC_ID: 'doc1'},
    )
    document_store.update_nodes([node], copy=True)
    document_store.impl.upsert = MagicMock(return_value=False)

    with pytest.raises(RuntimeError, match='Failed to upsert segments for group group1'):
        document_store.update_doc_meta(doc_id='doc1', metadata={'foo': 'bar'}, kb_id='kb1')
