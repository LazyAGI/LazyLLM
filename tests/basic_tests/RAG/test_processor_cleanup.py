from unittest.mock import MagicMock

import pytest

from lazyllm.tools.rag.doc_node import DocNode
from lazyllm.tools.rag.global_metadata import RAG_DOC_ID, RAG_KB_ID
from lazyllm.tools.rag.parsing_service import _Processor
from lazyllm.tools.rag.store import LAZY_IMAGE_GROUP, LAZY_ROOT_NAME
from lazyllm.tools.rag.transform import TransformArgs


def test_rebuild_does_not_replace_cached_children_with_null_nodes():
    store = MagicMock()
    store.activated_groups.return_value = ['block']
    store.get_nodes.return_value = []
    root = _make_root_node('doc1', 'kb1')
    root.children['block'] = [DocNode(text='old chunk', parent=root, group='block')]
    groups = {'block': {'parent': LAZY_ROOT_NAME, 'transform': TransformArgs(f=lambda text: [text])}}
    processor = _Processor(store=store)
    try:
        processor._reparse_group_recursive([root], 'block', groups, ['doc1'], 'kb1')
        nodes = store.update_nodes.call_args.args[0]
        assert len(nodes) == 1
        assert nodes[0].text == 'root'
        assert not nodes[0].is_null_node
    finally:
        processor.close()


def test_image_group_without_images_does_not_rebuild_text_document():
    store = MagicMock()
    store.get_nodes.return_value = []
    store.activated_groups.return_value = ['image-description']
    reader = MagicMock()
    groups = {'image-description': {'parent': LAZY_IMAGE_GROUP}}
    processor = _Processor(store=store)
    try:
        processor._reparse_docs('image-description', groups, ['doc1'], ['/tmp/doc1.txt'], [{}],
                                kb_id='kb1', reader=reader)
        reader.load_data.assert_not_called()
        store.remove_nodes.assert_called_once_with(group='image-description', kb_id='kb1', doc_ids=['doc1'])
    finally:
        processor.close()


def _make_root_node(doc_id: str, kb_id: str) -> DocNode:
    return DocNode(
        uid=f'{doc_id}-root',
        text='root',
        group=LAZY_ROOT_NAME,
        global_metadata={RAG_DOC_ID: doc_id, RAG_KB_ID: kb_id},
    )


def test_add_doc_keeps_partial_segments_and_schema_on_failure():
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

    store.remove_nodes.assert_not_called()
    schema_extractor._delete_extract_data.assert_not_called()


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


def test_add_failure_does_not_remove_successful_schema_results():
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

    ext_a._delete_extract_data.assert_not_called()
    ext_b._delete_extract_data.assert_not_called()
