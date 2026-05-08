import tempfile

from lazyllm.tools.rag.doc_node import DocNode
from lazyllm.tools.rag.global_metadata import RAG_DOC_ID, RAG_KB_ID
from lazyllm.tools.rag.parsing_service import server as parsing_server
from lazyllm.tools.rag.parsing_service.server import DocumentProcessor
from lazyllm.tools.rag.store import MapStore
from lazyllm.tools.rag.store.document_store import _DocumentStore


def test_parsing_service_lists_doc_chunks_sorted_with_offset(monkeypatch):
    store = _DocumentStore(algo_name='__default__', store=MapStore())
    store.activate_group('line')
    store.update_nodes([
        DocNode(
            uid='chunk-3',
            text='third',
            group='line',
            metadata={'lazyllm_store_num': 3},
            global_metadata={RAG_KB_ID: 'kb_test', RAG_DOC_ID: 'doc_test'},
        ),
        DocNode(
            uid='chunk-1',
            text='first',
            group='line',
            metadata={'lazyllm_store_num': 1},
            global_metadata={RAG_KB_ID: 'kb_test', RAG_DOC_ID: 'doc_test'},
        ),
        DocNode(
            uid='chunk-2',
            text='second',
            group='line',
            metadata={'lazyllm_store_num': 2},
            global_metadata={RAG_KB_ID: 'kb_test', RAG_DOC_ID: 'doc_test'},
        ),
    ], copy=True)

    impl = DocumentProcessor._Impl(
        db_config={
            'db_type': 'sqlite',
            'user': None,
            'password': None,
            'host': None,
            'port': None,
            'db_name': tempfile.mktemp(suffix='.db'),
        },
        num_workers=0,
        post_func=lambda *args, **kwargs: True,
    )
    impl._lazy_init = lambda: None
    monkeypatch.setattr(parsing_server, 'load_obj', lambda _: {
        'store': store,
        'node_groups': {'line': {'display_name': 'Line', 'group_type': 'chunk'}},
    })
    impl._get_algo = lambda algo_id: {
        'id': algo_id,
        'info_pickle': 'mock-info',
    }

    data = impl._list_doc_chunks_data(
        algo_id='__default__',
        kb_id='kb_test',
        doc_id='doc_test',
        group='line',
        offset=1,
        limit=2,
    )

    assert data['total'] == 3
    assert data['offset'] == 1
    assert data['page_size'] == 2
    assert [item['number'] for item in data['items']] == [2, 3]
    assert [item['content'] for item in data['items']] == ['second', 'third']
