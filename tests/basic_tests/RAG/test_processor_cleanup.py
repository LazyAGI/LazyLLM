from unittest.mock import MagicMock

import pytest

from lazyllm.tools.rag.doc_node import DocNode
from lazyllm.tools.rag.global_metadata import RAG_DOC_ID, RAG_KB_ID
from lazyllm.tools.rag.parsing_service import _Processor
from lazyllm.tools.rag.store import LAZY_IMAGE_GROUP, LAZY_ROOT_NAME


def _make_root_node(doc_id: str, kb_id: str) -> DocNode:
    return DocNode(
        uid=f'{doc_id}-root',
        text='root',
        group=LAZY_ROOT_NAME,
        global_metadata={RAG_DOC_ID: doc_id, RAG_KB_ID: kb_id},
    )


def test_add_doc_cleans_partial_segments_and_schema_on_failure():
    store = MagicMock()
    reader = MagicMock()
    schema_extractor = MagicMock()
    reader.load_data.return_value = {
        LAZY_ROOT_NAME: [_make_root_node('doc1', 'kb1')],
        LAZY_IMAGE_GROUP: [],
    }
    store.update_nodes.side_effect = RuntimeError('upsert failed')

    processor = _Processor(store=store, schema_extractors={'default': schema_extractor})
    try:
        with pytest.raises(RuntimeError, match='upsert failed'):
            processor.add_doc(
                input_files=['/tmp/doc1.txt'],
                node_groups={},
                reader=reader,
                ids=['doc1'],
                metadatas=[{}],
                kb_id='kb1',
            )
    finally:
        processor.close()

    store.remove_nodes.assert_called_once_with(doc_ids=['doc1'], kb_id='kb1')
    schema_extractor._delete_extract_data.assert_called_once_with(
        kb_id='kb1',
        doc_ids=['doc1'],
    )


def test_transfer_failure_cleans_target_segments_only():
    store = MagicMock()
    reader = MagicMock()
    store.get_nodes.return_value = [_make_root_node('source-doc', 'source-kb')]
    store.update_nodes.side_effect = RuntimeError('upsert failed')

    processor = _Processor(store=store)
    try:
        with pytest.raises(RuntimeError, match='upsert failed'):
            processor.add_doc(
                input_files=['/tmp/source.txt'],
                node_groups={},
                reader=reader,
                ids=['source-doc'],
                metadatas=[{}],
                kb_id='source-kb',
                transfer_mode='cp',
                target_kb_id='target-kb',
                target_doc_ids=['target-doc'],
            )
    finally:
        processor.close()

    store.remove_nodes.assert_called_once_with(doc_ids=['target-doc'], kb_id='target-kb')
    reader.load_data.assert_not_called()


def test_add_doc_runs_only_specified_extractors():
    store = MagicMock()
    reader = MagicMock()
    ext_a = MagicMock()
    ext_b = MagicMock()
    root = _make_root_node('doc1', 'kb1')
    reader.load_data.return_value = {
        LAZY_ROOT_NAME: [root],
        LAZY_IMAGE_GROUP: [],
    }
    store.update_nodes.return_value = None

    processor = _Processor(store=store, schema_extractors={'a': ext_a, 'b': ext_b})
    try:
        processor.add_doc(
            input_files=['/tmp/doc1.txt'],
            node_groups={},
            reader=reader,
            ids=['doc1'],
            metadatas=[{}],
            kb_id='kb1',
            extractor_names=['a'],
        )
    finally:
        processor.close()

    ext_a.assert_called_once()
    ext_b.assert_not_called()


def test_add_doc_runs_all_extractors_when_names_is_none():
    store = MagicMock()
    reader = MagicMock()
    ext_a = MagicMock()
    ext_b = MagicMock()
    root = _make_root_node('doc1', 'kb1')
    reader.load_data.return_value = {
        LAZY_ROOT_NAME: [root],
        LAZY_IMAGE_GROUP: [],
    }
    store.update_nodes.return_value = None

    processor = _Processor(store=store, schema_extractors={'a': ext_a, 'b': ext_b})
    try:
        processor.add_doc(
            input_files=['/tmp/doc1.txt'],
            node_groups={},
            reader=reader,
            ids=['doc1'],
            metadatas=[{}],
            kb_id='kb1',
        )
    finally:
        processor.close()

    ext_a.assert_called_once()
    ext_b.assert_called_once()


def test_cleanup_iterates_all_schema_extractors():
    store = MagicMock()
    reader = MagicMock()
    ext_a = MagicMock()
    ext_b = MagicMock()
    root = _make_root_node('doc1', 'kb1')
    reader.load_data.return_value = {
        LAZY_ROOT_NAME: [root],
        LAZY_IMAGE_GROUP: [],
    }
    store.update_nodes.side_effect = RuntimeError('fail')

    processor = _Processor(store=store, schema_extractors={'a': ext_a, 'b': ext_b})
    try:
        with pytest.raises(RuntimeError, match='fail'):
            processor.add_doc(
                input_files=['/tmp/doc1.txt'],
                node_groups={},
                reader=reader,
                ids=['doc1'],
                metadatas=[{}],
                kb_id='kb1',
            )
    finally:
        processor.close()

    ext_a._delete_extract_data.assert_called_once_with(kb_id='kb1', doc_ids=['doc1'])
    ext_b._delete_extract_data.assert_called_once_with(kb_id='kb1', doc_ids=['doc1'])
