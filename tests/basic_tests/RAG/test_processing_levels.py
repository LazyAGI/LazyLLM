from unittest.mock import MagicMock, patch

import pytest

from lazyllm.tools.rag.doc_node import DocNode
from lazyllm.tools.rag.doc_service.base import ReparseRequest, UploadRequest, AddFileItem
from lazyllm.tools.rag.parsing_service.base import AddDocRequest, FileInfo
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


@pytest.mark.parametrize('request_factory', [
    lambda: UploadRequest(items=[AddFileItem(file_path='/tmp/doc.pdf')], processing_level='stored'),
    lambda: ReparseRequest(doc_ids=['doc-1'], processing_level='stored'),
    lambda: AddDocRequest(file_infos=[FileInfo(file_path='/tmp/doc.pdf')], processing_level='stored'),
])
def test_stored_level_is_owned_by_caller_and_rejected_by_lazyllm(request_factory):
    with pytest.raises(ValueError, match='processing_level must be parsed, chunked, or indexed'):
        request_factory()


def test_parsed_level_stores_root_without_creating_chunks():
    processor = _processor()
    processor._create_nodes_recursive = lambda *args, **kwargs: pytest.fail('must not create chunks')

    processor.add_doc(
        input_files=['/tmp/doc.pdf'], ids=['doc-1'], metadatas=[{}], kb_id='kb-1',
        node_groups={}, reader=_Reader(), processing_level='parsed',
    )

    roots = processor.store.get_nodes(group=LAZY_ROOT_NAME, doc_ids=['doc-1'], kb_id='kb-1')
    assert len(roots) == 1


def test_per_parent_transform_assigns_document_wide_node_numbers():
    processor = _chunk_processor()
    parents = [_root('root-1'), _root('root-2')]
    processor.store.update_nodes(parents, copy=True)

    def split(parent_nodes, group_name, ref_path=None):
        parent = parent_nodes[0]
        return [
            DocNode(uid=f'{parent.uid}-a', text='a', group=group_name, parent=parent,
                    global_metadata=dict(parent.global_metadata)),
            DocNode(uid=f'{parent.uid}-b', text='b', group=group_name, parent=parent,
                    global_metadata=dict(parent.global_metadata)),
        ]

    transform = MagicMock(pattern=False)
    transform.batch_forward.side_effect = split
    processor._create_nodes_impl(
        parents,
        'chunks',
        {'chunks': {'parent': LAZY_ROOT_NAME, 'transform': transform, 'signature': 'sig-v1'}},
    )

    chunks = processor.store.get_nodes(group='chunks', doc_ids=['doc-1'], kb_id='kb-1')
    assert sorted(node.number for node in chunks) == [1, 2, 3, 4]


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


def _root(uid='root-1'):
    return DocNode(
        uid=uid, text='root', group=LAZY_ROOT_NAME,
        global_metadata={RAG_DOC_ID: 'doc-1', RAG_KB_ID: 'kb-1', RAG_DOC_PATH: '/tmp/doc.pdf'},
    )


def _chunk_processor(*, embed=None, group_embed_keys=None):
    store = _DocumentStore(store=MapStore(), embed=embed, group_embed_keys=group_embed_keys)
    store.activate_group([LAZY_ROOT_NAME, 'chunks', 'refs'])
    return _Processor(store)


def test_empty_transform_writes_hidden_null_node():
    processor = _chunk_processor()
    root = _root()
    processor.store.update_nodes([root], copy=True)
    transform = MagicMock()
    transform.batch_forward.return_value = []
    config = {'parent': LAZY_ROOT_NAME, 'transform': MagicMock(pattern=False), 'signature': 'sig-v1'}

    with patch('lazyllm.tools.rag.parsing_service.impl.make_transform', return_value=transform):
        result = processor._create_nodes_impl([root], 'chunks', {'chunks': config}, skip_embedding=True)

    assert result == []
    assert processor.store.get_nodes(group='chunks', doc_ids=['doc-1'], kb_id='kb-1') == []
    assert processor.store.get_segments(group='chunks', doc_ids=['doc-1'], kb_id='kb-1') == []
    markers = processor.store.get_nodes(
        group='chunks', doc_ids=['doc-1'], kb_id='kb-1', include_null=True)
    assert len(markers) == 1
    assert markers[0].is_null_node
    assert markers[0]._parent == root.uid


def test_missing_ref_skips_transform_without_null_node():
    processor = _chunk_processor()
    root = _root()
    processor.store.update_nodes([root], copy=True)
    transform = MagicMock()
    transform._get_ref_nodes.return_value = []
    config = {'parent': LAZY_ROOT_NAME, 'ref': 'refs', 'transform': MagicMock(pattern=False), 'signature': 'sig-v1'}

    with patch('lazyllm.tools.rag.parsing_service.impl.make_transform', return_value=transform):
        result = processor._create_nodes_impl(
            [root], 'chunks', {'chunks': config}, ref_path=['refs'], skip_embedding=True)

    assert result == []
    transform.batch_forward.assert_not_called()
    assert processor.store.get_nodes(group='chunks', include_null=True) == []


def test_fill_missing_accepts_null_node_as_completed():
    processor = _chunk_processor()
    root = _root()
    processor.store.update_nodes([root], copy=True)
    transform = MagicMock()
    transform.batch_forward.return_value = []
    config = {'parent': LAZY_ROOT_NAME, 'transform': MagicMock(pattern=False), 'signature': 'sig-v1'}

    with patch('lazyllm.tools.rag.parsing_service.impl.make_transform', return_value=transform):
        processor._create_nodes_impl([root], 'chunks', {'chunks': config}, skip_embedding=True)
        processor._create_nodes_impl(
            [root], 'chunks', {'chunks': config}, skip_embedding=True, only_missing=True)

    transform.batch_forward.assert_called_once()


def test_embedding_failure_keeps_segments_and_successful_vectors():
    failed = {'bad'}
    calls = []

    def embed(text):
        calls.append(text)
        if text in failed:
            raise ValueError('cannot embed')
        return [1.0, 2.0]

    processor = _chunk_processor(embed={'dense': embed}, group_embed_keys={'chunks': {'dense'}})
    nodes = [
        DocNode(uid='good', text='good', group='chunks', global_metadata={RAG_DOC_ID: 'doc-1', RAG_KB_ID: 'kb-1'}),
        DocNode(uid='bad', text='bad', group='chunks', global_metadata={RAG_DOC_ID: 'doc-1', RAG_KB_ID: 'kb-1'}),
    ]

    with pytest.raises(ValueError, match='cannot embed'):
        processor.store.update_nodes(nodes)

    stored = {node.uid: node for node in processor.store.get_nodes(group='chunks', doc_ids=['doc-1'], kb_id='kb-1')}
    assert set(stored) == {'good', 'bad'}
    assert stored['good'].embedding['dense'] == [1.0, 2.0]
    assert 'dense' not in stored['bad'].embedding
    failed.clear()
    processor.store.update_nodes(list(stored.values()))
    assert calls.count('good') == 1
    assert calls.count('bad') == 2
    completed = {node.uid: node for node in processor.store.get_nodes(group='chunks')}
    assert completed['bad'].embedding['dense'] == [1.0, 2.0]


def test_failed_parent_branch_is_skipped_but_successful_branch_reaches_children():
    store = _DocumentStore(store=MapStore())
    store.activate_group([LAZY_ROOT_NAME, 'parent', 'child'])
    processor = _Processor(store)
    roots = [_root('good-root'), _root('bad-root')]
    store.update_nodes(roots, copy=True)

    parent_transform = MagicMock()

    failed = {'bad-root'}

    def create_parent(nodes, group_name, ref_path=None):
        root = nodes[0]
        if root.uid in failed:
            raise RuntimeError('parent chunk failed')
        node = DocNode(
            uid=f'parent-{root.uid}', text='parent', group=group_name, parent=root,
            global_metadata=dict(root.global_metadata),
        )
        root.children[group_name] = [node]
        return [node]

    parent_transform.batch_forward.side_effect = create_parent
    child_transform = MagicMock()

    def create_child(nodes, group_name, ref_path=None):
        parent = nodes[0]
        node = DocNode(
            uid=f'child-{parent.uid}', text='child', group=group_name, parent=parent,
            global_metadata=dict(parent.global_metadata),
        )
        parent.children[group_name] = [node]
        return [node]

    child_transform.batch_forward.side_effect = create_child
    transforms = {'parent-transform': parent_transform, 'child-transform': child_transform}
    node_groups = {
        LAZY_ROOT_NAME: {'parent': None, 'transform': None},
        'parent': {'parent': LAZY_ROOT_NAME, 'transform': MagicMock(pattern=False, name='parent-transform')},
        'child': {'parent': 'parent', 'transform': MagicMock(pattern=False, name='child-transform')},
    }

    def make(transform, group_name):
        return transforms[f'{group_name}-transform']

    with patch('lazyllm.tools.rag.parsing_service.impl.make_transform', side_effect=make):
        with pytest.raises(RuntimeError, match='parent chunk failed'):
            processor._create_nodes_recursive(
                roots, LAZY_ROOT_NAME, node_groups=node_groups, skip_embedding=True)

    assert {node.uid for node in store.get_nodes(group='parent')} == {'parent-good-root'}
    assert {node.uid for node in store.get_nodes(group='child')} == {'child-parent-good-root'}
    child_transform.batch_forward.assert_called_once()

    failed.clear()
    with patch('lazyllm.tools.rag.parsing_service.impl.make_transform', side_effect=make):
        processor._fill_missing_docs('parent', node_groups, ['doc-1'], 'kb-1', 'chunked')

    assert {node.uid for node in store.get_nodes(group='parent')} == {
        'parent-good-root', 'parent-bad-root'}
    assert {node.uid for node in store.get_nodes(group='child')} == {
        'child-parent-good-root', 'child-parent-bad-root'}
    assert parent_transform.batch_forward.call_count == 3
    assert child_transform.batch_forward.call_count == 2
