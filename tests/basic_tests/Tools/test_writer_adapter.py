from copy import deepcopy

import pytest

from lazyllm.tools.writer.adapter.base import NativePatchOperation, WriterAdapterBase
from lazyllm.tools.writer.adapter.feishu import FeishuWriterAdapter
from lazyllm.tools.writer.data_models import (
    PatchBlock,
    PatchHunk,
    WriterBlock,
    WriterDocument,
    WriterSpan,
)


class _DummyAdapter(WriterAdapterBase):
    provider = 'dummy'

    def blocks_to_ir(self, blocks, **kwargs):
        raise NotImplementedError

    def ir_to_blocks(self, document):
        raise NotImplementedError

    def patch_to_operation(self, patch, document):
        raise NotImplementedError

def test_writer_adapter_ids_are_stable_and_provider_scoped():
    document_id = _DummyAdapter.make_document_id('external-document')
    node_id = _DummyAdapter.make_node_id('external-document', 'external-block')

    assert document_id == _DummyAdapter.make_document_id('external-document')
    assert node_id == _DummyAdapter.make_node_id('external-document', 'external-block')
    assert document_id.startswith('writer-doc-')
    assert node_id.startswith('writer-node-')
    assert node_id != _DummyAdapter.make_node_id('another-document', 'external-block')
    assert node_id != _DummyAdapter.make_node_id('external-document', 'another-block')


@pytest.mark.parametrize('value', ['', '   ', None])
def test_writer_adapter_rejects_empty_external_ids(value):
    with pytest.raises(ValueError, match='external_document_id must be a non-empty string'):
        _DummyAdapter.make_document_id(value)

    with pytest.raises(ValueError, match='external_block_id must be a non-empty string'):
        _DummyAdapter.make_node_id('external-document', value)


def test_writer_adapter_requires_provider_name():
    class MissingProviderAdapter(_DummyAdapter):
        provider = ''

    with pytest.raises(ValueError, match='provider must be a non-empty string'):
        MissingProviderAdapter.make_document_id('external-document')


def test_writer_adapter_base_is_abstract():
    with pytest.raises(TypeError):
        WriterAdapterBase()


def _raw_blocks():
    return [
        {
            'block_id': 'heading-1',
            'block_type': 3,
            'parent_id': 'doc-1',
            'children': ['paragraph-1'],
            'heading1': {
                'elements': [{
                    'text_run': {
                        'content': '标题',
                        'text_element_style': {'bold': True},
                    },
                }],
                'style': {'align': 1},
            },
            'future_field': {'kept': True},
        },
        {
            'block_id': 'paragraph-1',
            'block_type': 2,
            'parent_id': 'heading-1',
            'text': {
                'elements': [
                    {'text_run': {'content': '查看 ', 'text_element_style': {'italic': True}}},
                    {'mention_doc': {'token': 'doc-ref', 'title': '设计文档'}},
                ],
            },
        },
        {
            'block_id': 'future-1',
            'block_type': 999,
            'parent_id': 'doc-1',
            'future_block': {'opaque': True},
            'plain_text': '未来块',
        },
    ]


def _move_raw_blocks():
    return [
        {
            'block_id': 'heading-1',
            'block_type': 3,
            'parent_id': 'doc-1',
            'children': ['paragraph-1'],
            'heading1': {'elements': [{'text_run': {'content': '章节一'}}]},
        },
        {
            'block_id': 'paragraph-1',
            'block_type': 2,
            'parent_id': 'heading-1',
            'text': {'elements': [{'text_run': {'content': '段落一'}}]},
        },
        {
            'block_id': 'heading-2',
            'block_type': 3,
            'parent_id': 'doc-1',
            'children': ['paragraph-2'],
            'heading1': {'elements': [{'text_run': {'content': '章节二'}}]},
        },
        {
            'block_id': 'paragraph-2',
            'block_type': 2,
            'parent_id': 'heading-2',
            'text': {'elements': [{'text_run': {'content': '段落二'}}]},
        },
    ]


def test_blocks_to_ir_preserves_hierarchy_bindings_and_payload():
    raw_blocks = _raw_blocks()
    document = FeishuWriterAdapter().blocks_to_ir(
        raw_blocks,
        external_document_id='doc-1',
        title='测试文档',
        uri='feishu:/test',
        revision='12',
    )

    assert document.document_id == FeishuWriterAdapter.make_document_id('doc-1')
    assert document.provider_binding == {
        'provider': 'feishu',
        'document_id': 'doc-1',
        'uri': 'feishu:/test',
        'revision': '12',
    }
    assert [block.type for block in document.blocks] == ['heading', 'feishu_unknown']

    heading = document.blocks[0]
    paragraph = heading.children[0]
    assert heading.node_id == FeishuWriterAdapter.make_node_id('doc-1', 'heading-1')
    assert heading.numbering == {'level': 1}
    assert heading.spans == [WriterSpan(text='标题', style=['strong'])]
    assert paragraph.content == '查看 设计文档'
    assert paragraph.spans[-1].style == ['feishu:mention_doc']
    assert paragraph.provider_binding['block_id'] == 'paragraph-1'
    assert document.blocks[1].editable is False

    raw_blocks[0]['future_field']['kept'] = False
    assert heading.provider_payload['raw_block']['future_field']['kept'] is True


def test_blocks_to_ir_rejects_duplicate_ids_and_cycles():
    adapter = FeishuWriterAdapter()
    duplicate = [_raw_blocks()[0], deepcopy(_raw_blocks()[0])]
    with pytest.raises(ValueError, match='duplicate Feishu block_id'):
        adapter.blocks_to_ir(duplicate, external_document_id='doc-1')

    cycle = [
        {'block_id': 'a', 'block_type': 2, 'children': ['b'], 'text': {'elements': []}},
        {'block_id': 'b', 'block_type': 2, 'children': ['a'], 'text': {'elements': []}},
    ]
    with pytest.raises(ValueError, match='cycle detected'):
        adapter.blocks_to_ir(cycle, external_document_id='doc-1')


def test_ir_to_blocks_preserves_raw_fields_and_hierarchy():
    adapter = FeishuWriterAdapter()
    raw_blocks = _raw_blocks()
    document = adapter.blocks_to_ir(raw_blocks, external_document_id='doc-1')

    converted = adapter.ir_to_blocks(document)

    assert converted[0]['block_id'] == 'heading-1'
    assert 'children' not in converted[0]
    assert converted[0]['future_field'] == {'kept': True}
    assert converted[0]['heading1'] == raw_blocks[0]['heading1']
    assert converted[1]['parent_id'] == 'heading-1'
    assert converted[1]['text'] == raw_blocks[1]['text']
    assert converted[2]['future_block'] == {'opaque': True}


def test_ir_to_blocks_converts_new_blocks_and_styles():
    document = WriterDocument(
        document_id='internal-doc',
        stage='final',
        blocks=[
            WriterBlock(
                node_id='heading-node',
                type='heading',
                content='标题',
                spans=[WriterSpan(text='标题', style=['strong'])],
                stage='final',
                numbering={'level': 2},
                children=[
                    WriterBlock(
                        node_id='list-node',
                        type='list_item',
                        content='条目',
                        stage='final',
                        numbering={'ordered': True},
                    ),
                ],
            ),
        ],
    )

    converted = FeishuWriterAdapter().ir_to_blocks(document)

    assert converted == [
        {
            'block_type': 4,
            'heading2': {
                'elements': [{
                    'text_run': {
                        'content': '标题',
                        'text_element_style': {'bold': True},
                    },
                }],
            },
            'plain_text': '标题',
            'block_id': 'heading-node',
        },
        {
            'block_type': 13,
            'ordered': {'elements': [{'text_run': {'content': '条目'}}]},
            'plain_text': '条目',
            'block_id': 'list-node',
            'parent_id': 'heading-node',
        },
    ]


def test_fs_prepare_descendants_is_the_only_writer_of_children():
    pytest.importorskip('fsspec')
    from lazyllm.tools.fs.supplier.feishu import FeishuFSBase

    document = WriterDocument(
        document_id='internal-doc',
        stage='final',
        blocks=[
            WriterBlock(
                node_id='parent',
                type='paragraph',
                content='父块',
                stage='final',
                children=[
                    WriterBlock(
                        node_id='child',
                        type='paragraph',
                        content='子块',
                        stage='final',
                    ),
                ],
            ),
        ],
    )

    blocks = FeishuWriterAdapter().ir_to_blocks(document)
    assert 'children' not in blocks[0]
    assert blocks[1]['parent_id'] == 'parent'

    roots, descendants = FeishuFSBase._prepare_docx_descendants(blocks)
    assert roots == ['parent']
    assert descendants[0]['children'] == ['child']


def test_ir_to_blocks_rebuilds_modified_text_and_rejects_modified_unknown_block():
    adapter = FeishuWriterAdapter()
    document = adapter.blocks_to_ir(_raw_blocks(), external_document_id='doc-1')
    document.blocks[0].content = '新标题'
    document.blocks[0].spans = []

    converted = adapter.ir_to_blocks(document)
    assert converted[0]['heading1']['elements'] == [
        {'text_run': {'content': '新标题'}},
    ]
    assert converted[0]['heading1']['style'] == {'align': 1}

    document.blocks[1].content = '修改未来块'
    with pytest.raises(ValueError, match='non-editable Feishu block'):
        adapter.ir_to_blocks(document)


def test_replace_patch_resolves_internal_node_to_feishu_block():
    adapter = FeishuWriterAdapter()
    document = adapter.blocks_to_ir(_raw_blocks(), external_document_id='doc-1')
    target = document.blocks[0]
    patch = PatchHunk(
        target_node_id=target.node_id,
        modify_type='replace',
        old_text='标题',
        new_text='新标题',
    )

    expected_requests = [{
        'block_id': 'heading-1',
        'update_text_elements': {
            'elements': [{
                'text_run': {
                    'content': '新标题',
                    'text_element_style': {'bold': True},
                },
            }],
        },
    }]

    assert adapter.patch_to_operation(patch, document) == NativePatchOperation(
        operation='update',
        params={'requests': expected_requests},
    )


def test_insert_patch_builds_create_operation_after_target():
    adapter = FeishuWriterAdapter()
    document = adapter.blocks_to_ir(_raw_blocks(), external_document_id='doc-1')
    patch = PatchHunk(
        hunk_id='insert-1',
        target_node_id=document.blocks[0].node_id,
        modify_type='insert',
        position='after',
        new_blocks=[
            PatchBlock(type='paragraph', content='新增段落'),
            PatchBlock(type='heading', content='新增标题', numbering={'level': 2}),
        ],
    )

    operation = adapter.patch_to_operation(patch, document)

    assert operation.operation == 'create'
    assert operation.params['parent_block_id'] == 'doc-1'
    assert operation.params['index'] == 1
    assert operation.params['children_id'] == ['insert-1::0', 'insert-1::1']
    assert operation.params['descendants'] == [
        {
            'block_type': 2,
            'text': {'elements': [{'text_run': {'content': '新增段落'}}]},
            'block_id': 'insert-1::0',
        },
        {
            'block_type': 4,
            'heading2': {'elements': [{'text_run': {'content': '新增标题'}}]},
            'block_id': 'insert-1::1',
        },
    ]


def test_insert_patch_supports_anchor_position_and_new_text_fallback():
    adapter = FeishuWriterAdapter()
    document = adapter.blocks_to_ir(_raw_blocks(), external_document_id='doc-1')
    paragraph = document.blocks[0].children[0]
    patch = PatchHunk(
        target_node_id=document.blocks[0].node_id,
        anchor_node_id=paragraph.node_id,
        modify_type='insert',
        position='before',
        new_text='兼容新增段落',
    )

    operation = adapter.patch_to_operation(patch, document)

    assert operation.operation == 'create'
    assert operation.params['parent_block_id'] == 'heading-1'
    assert operation.params['index'] == 0
    assert operation.params['children_id'] == [operation.params['descendants'][0]['block_id']]
    assert operation.params['descendants'][0]['block_type'] == 2
    assert (
        operation.params['descendants'][0]['text']['elements'][0]['text_run']['content']
        == '兼容新增段落'
    )


def test_delete_patch_builds_delete_operation_and_validates_old_text():
    adapter = FeishuWriterAdapter()
    document = adapter.blocks_to_ir(_raw_blocks(), external_document_id='doc-1')
    paragraph = document.blocks[0].children[0]
    patch = PatchHunk(
        target_node_id=paragraph.node_id,
        modify_type='delete',
        old_text=paragraph.content,
    )

    assert adapter.patch_to_operation(patch, document) == NativePatchOperation(
        operation='delete',
        params={
            'parent_block_id': 'heading-1',
            'start_index': 0,
            'end_index': 1,
        },
    )

    stale = patch.model_copy(update={'old_text': '过期内容'})
    with pytest.raises(ValueError, match='old_text does not match'):
        adapter.patch_to_operation(stale, document)


def test_move_patch_builds_same_parent_operation_with_remapped_subtree_ids():
    adapter = FeishuWriterAdapter()
    document = adapter.blocks_to_ir(_move_raw_blocks(), external_document_id='doc-1')
    source, anchor = document.blocks
    patch = PatchHunk(
        hunk_id='move-1',
        target_node_id=source.node_id,
        modify_type='move',
        anchor_node_id=anchor.node_id,
        position='after',
        old_text=source.content,
    )

    operation = adapter.patch_to_operation(patch, document)

    assert operation.operation == 'move'
    assert operation.params['source_parent_block_id'] == 'doc-1'
    assert operation.params['source_index'] == 0
    assert operation.params['target_parent_block_id'] == 'doc-1'
    assert operation.params['target_index'] == 1
    assert operation.params['children_id'] == ['move-1::0']
    assert operation.params['descendants'] == [
        {
            'block_id': 'move-1::0',
            'block_type': 3,
            'heading1': {'elements': [{'text_run': {'content': '章节一'}}]},
            'children': ['move-1::1'],
        },
        {
            'block_id': 'move-1::1',
            'block_type': 2,
            'text': {'elements': [{'text_run': {'content': '段落一'}}]},
        },
    ]


def test_move_patch_builds_cross_parent_operation():
    adapter = FeishuWriterAdapter()
    document = adapter.blocks_to_ir(_move_raw_blocks(), external_document_id='doc-1')
    source = document.blocks[0].children[0]
    anchor = document.blocks[1].children[0]
    patch = PatchHunk(
        target_node_id=source.node_id,
        modify_type='move',
        anchor_node_id=anchor.node_id,
        position='before',
    )

    operation = adapter.patch_to_operation(patch, document)

    assert operation.operation == 'move'
    assert operation.params['source_parent_block_id'] == 'heading-1'
    assert operation.params['source_index'] == 0
    assert operation.params['target_parent_block_id'] == 'heading-2'
    assert operation.params['target_index'] == 0
    assert len(operation.params['children_id']) == 1
    assert operation.params['descendants'][0]['text']['elements'][0]['text_run']['content'] == '段落一'


def test_move_patch_rejects_anchor_in_source_subtree_and_stale_text():
    adapter = FeishuWriterAdapter()
    document = adapter.blocks_to_ir(_move_raw_blocks(), external_document_id='doc-1')
    source = document.blocks[0]
    descendant = source.children[0]

    into_self = PatchHunk(
        target_node_id=source.node_id,
        modify_type='move',
        anchor_node_id=descendant.node_id,
        position='after',
    )
    with pytest.raises(ValueError, match='anchor cannot be the source node or its descendant'):
        adapter.patch_to_operation(into_self, document)

    stale = PatchHunk(
        target_node_id=source.node_id,
        modify_type='move',
        anchor_node_id=document.blocks[1].node_id,
        position='before',
        old_text='过期内容',
    )
    with pytest.raises(ValueError, match='old_text does not match'):
        adapter.patch_to_operation(stale, document)


def test_replace_patch_supports_text_range_and_rejects_stale_patch():
    adapter = FeishuWriterAdapter()
    document = adapter.blocks_to_ir(_raw_blocks(), external_document_id='doc-1')
    target = document.blocks[0].children[0]

    ranged = PatchHunk(
        target_node_id=target.node_id,
        modify_type='replace',
        text_range=(0, 2),
        old_text='查看',
        new_text='阅读',
    )
    operation = adapter.patch_to_operation(ranged, document)
    elements = operation.params['requests'][0]['update_text_elements']['elements']
    assert elements == [
        {
            'text_run': {
                'content': '阅读 ',
                'text_element_style': {'italic': True},
            },
        },
        {'mention_doc': {'token': 'doc-ref', 'title': '设计文档'}},
    ]

    stale = PatchHunk(
        target_node_id=target.node_id,
        modify_type='replace',
        old_text='过期内容',
        new_text='新内容',
    )
    with pytest.raises(ValueError, match='old_text does not match'):
        adapter.patch_to_operation(stale, document)
