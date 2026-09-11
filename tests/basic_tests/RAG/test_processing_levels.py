import pytest

from lazyllm.tools.rag.doc_node import DocNode
from lazyllm.tools.rag.doc_service.base import ReparseRequest, UploadRequest, AddFileItem
from lazyllm.tools.rag.global_metadata import RAG_DOC_ID, RAG_DOC_PATH, RAG_KB_ID
from lazyllm.tools.rag.parsing_service.impl import _Processor
from lazyllm.tools.rag.store import LAZY_ROOT_NAME, MapStore
from lazyllm.tools.rag.store.document_store import _DocumentStore


class _Reader:
    def load_data(self, input_files, metadatas, split_nodes_by_type=True):
        metadata = metadatas[0]
        return {
            LAZY_ROOT_NAME: [
                DocNode(
                    uid='root-doc-1',
                    text='parsed text',
                    group=LAZY_ROOT_NAME,
                    global_metadata={
                        RAG_DOC_ID: metadata[RAG_DOC_ID],
                        RAG_DOC_PATH: input_files[0],
                        RAG_KB_ID: metadata[RAG_KB_ID],
                    },
                )
            ]
        }


def _processor():
    store = _DocumentStore(store=MapStore())
    store.activate_group(LAZY_ROOT_NAME)
    return _Processor(store)


def test_reparse_rejects_vector_rebuild_below_indexed():
    with pytest.raises(ValueError, match='reembed requires indexed'):
        ReparseRequest(doc_ids=['doc-1'], processing_level='chunked', strategy='reembed')


def test_processing_level_defaults_keep_legacy_indexed_behavior():
    request = UploadRequest(items=[AddFileItem(file_path='/tmp/doc.pdf')])
    assert request.processing_level == 'indexed'


def test_parsed_level_stores_root_without_creating_chunks():
    processor = _processor()
    processor._create_nodes_recursive = lambda *args, **kwargs: pytest.fail('must not create chunks')

    processor.add_doc(
        input_files=['/tmp/doc.pdf'], ids=['doc-1'], metadatas=[{}], kb_id='kb-1',
        node_groups={}, reader=_Reader(), processing_level='parsed',
    )

    roots = processor.store.get_nodes(group=LAZY_ROOT_NAME, doc_ids=['doc-1'], kb_id='kb-1')
    assert len(roots) == 1


def test_chunk_failure_keeps_successful_parse_root_for_retry():
    processor = _processor()

    def fail_chunks(*args, **kwargs):
        assert kwargs['skip_embedding'] is True
        raise RuntimeError('chunk failed')

    processor._create_nodes_recursive = fail_chunks

    with pytest.raises(RuntimeError, match='chunk failed'):
        processor.add_doc(
            input_files=['/tmp/doc.pdf'], ids=['doc-1'], metadatas=[{}], kb_id='kb-1',
            node_groups={}, reader=_Reader(), processing_level='chunked',
        )

    roots = processor.store.get_nodes(group=LAZY_ROOT_NAME, doc_ids=['doc-1'], kb_id='kb-1')
    assert len(roots) == 1
