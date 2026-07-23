from copy import deepcopy

import pytest

from lazyllm.tools.writer.adapter.feishu import FeishuWriterAdapter
from lazyllm.tools.writer.data_models import PatchBlock, PatchHunk, WriterSpan
from lazyllm.tools.writer.utils.feishu_docx import prepare_docx_descendants


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


def test_feishu_round_trip_preserves_hierarchy_styles_and_unknown_payload():
    raw_blocks = _raw_blocks()
    adapter = FeishuWriterAdapter()
    document = adapter.blocks_to_ir(
        raw_blocks,
        external_document_id='doc-1',
        uri='feishu:/test',
        revision='12',
    )

    heading, unknown = document.blocks
    paragraph = heading.children[0]
    assert [heading.type, paragraph.type, unknown.type] == [
        'heading', 'paragraph', 'feishu_unknown']
    assert heading.spans == [WriterSpan(text='标题', style=['strong'])]
    assert paragraph.content == '查看 设计文档'
    assert paragraph.spans[-1].style == ['feishu:mention_doc']
    assert unknown.editable is False

    raw_blocks[0]['future_field']['kept'] = False
    converted = adapter.ir_to_blocks(document)
    assert converted[0]['future_field'] == {'kept': True}
    assert converted[1]['parent_id'] == 'heading-1'
    assert converted[2]['future_block'] == {'opaque': True}


@pytest.mark.parametrize(
    'blocks, message',
    [
        (lambda: [_raw_blocks()[0], deepcopy(_raw_blocks()[0])], 'duplicate Feishu block_id'),
        (
            lambda: [
                {'block_id': 'a', 'block_type': 2, 'children': ['b'], 'text': {}},
                {'block_id': 'b', 'block_type': 2, 'children': ['a'], 'text': {}},
            ],
            'cycle detected',
        ),
    ],
)
def test_blocks_to_ir_rejects_invalid_relations(blocks, message):
    with pytest.raises(ValueError, match=message):
        FeishuWriterAdapter().blocks_to_ir(blocks(), external_document_id='doc-1')


def test_insert_builds_descendant_payload_for_position_and_content():
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
    assert [item['block_type'] for item in operation.params['descendants']] == [2, 4]


def test_insert_new_text_uses_nested_anchor():
    adapter = FeishuWriterAdapter()
    document = adapter.blocks_to_ir(_raw_blocks(), external_document_id='doc-1')
    paragraph = document.blocks[0].children[0]
    operation = adapter.patch_to_operation(PatchHunk(
        target_node_id=document.blocks[0].node_id,
        anchor_node_id=paragraph.node_id,
        modify_type='insert',
        position='before',
        new_text='兼容新增段落',
    ), document)

    assert (operation.params['parent_block_id'], operation.params['index']) == ('heading-1', 0)
    assert operation.params['descendants'][0]['text']['elements'][0]['text_run'][
        'content'] == '兼容新增段落'


def test_delete_builds_range_and_rejects_stale_text():
    adapter = FeishuWriterAdapter()
    document = adapter.blocks_to_ir(_raw_blocks(), external_document_id='doc-1')
    paragraph = document.blocks[0].children[0]
    patch = PatchHunk(
        target_node_id=paragraph.node_id,
        modify_type='delete',
        old_text=paragraph.content,
    )

    operation = adapter.patch_to_operation(patch, document)
    assert operation.operation == 'delete'
    assert operation.params == {
        'parent_block_id': 'heading-1', 'start_index': 0, 'end_index': 1}
    with pytest.raises(ValueError, match='old_text does not match'):
        adapter.patch_to_operation(
            patch.model_copy(update={'old_text': '过期内容'}), document)


@pytest.mark.parametrize(
    ('source_path', 'anchor_path', 'position', 'expected'),
    [
        ((0,), (1,), 'after', ('doc-1', 0, 'doc-1', 1)),
        ((0, 0), (1, 0), 'before', ('heading-1', 0, 'heading-2', 0)),
    ],
)
def test_move_builds_same_and_cross_parent_plans(
    source_path, anchor_path, position, expected,
):
    adapter = FeishuWriterAdapter()
    document = adapter.blocks_to_ir(_move_raw_blocks(), external_document_id='doc-1')

    def locate(path):
        block = document.blocks[path[0]]
        return block if len(path) == 1 else block.children[path[1]]

    source, anchor = locate(source_path), locate(anchor_path)
    operation = adapter.patch_to_operation(PatchHunk(
        hunk_id='move-1',
        target_node_id=source.node_id,
        modify_type='move',
        anchor_node_id=anchor.node_id,
        position=position,
    ), document)

    params = operation.params
    assert operation.operation == 'move'
    assert params['source_block_id'] == source.provider_binding['block_id']
    assert (
        params['source_parent_block_id'], params['source_index'],
        params['target_parent_block_id'], params['target_index'],
    ) == expected
    assert len(params['children_id']) == 1
    assert params['descendants'][0]['block_type'] == source.provider_payload[
        'raw_block']['block_type']


def test_move_rejects_anchor_in_source_subtree():
    adapter = FeishuWriterAdapter()
    document = adapter.blocks_to_ir(_move_raw_blocks(), external_document_id='doc-1')
    source = document.blocks[0]
    with pytest.raises(ValueError, match='anchor cannot be the source node or its descendant'):
        adapter.patch_to_operation(PatchHunk(
            target_node_id=source.node_id,
            modify_type='move',
            anchor_node_id=source.children[0].node_id,
            position='after',
        ), document)


def test_merge_refreshed_move_restores_writer_identity_with_new_feishu_ids():
    adapter = FeishuWriterAdapter()
    previous = adapter.blocks_to_ir(_move_raw_blocks(), external_document_id='doc-1')
    source, anchor = previous.blocks
    operation = adapter.patch_to_operation(PatchHunk(
        hunk_id='move-1',
        target_node_id=source.node_id,
        modify_type='move',
        anchor_node_id=anchor.node_id,
        position='after',
    ), previous)
    refreshed_raw = deepcopy(_move_raw_blocks()[2:])
    moved_raw = deepcopy(_move_raw_blocks()[:2])
    moved_raw[0]['block_id'] = 'moved-heading'
    moved_raw[0]['children'] = ['moved-paragraph']
    moved_raw[1]['block_id'] = 'moved-paragraph'
    moved_raw[1]['parent_id'] = 'moved-heading'
    refreshed = adapter.blocks_to_ir(
        refreshed_raw + moved_raw, external_document_id='doc-1')

    merged = adapter.merge_refreshed_document(
        previous,
        refreshed,
        patch=PatchHunk(
            target_node_id=source.node_id,
            modify_type='move',
            anchor_node_id=anchor.node_id,
            position='after',
        ),
        operation=operation,
    )
    moved = merged.blocks[1]
    assert moved.node_id == source.node_id
    assert moved.children[0].node_id == source.children[0].node_id
    assert moved.provider_binding['block_id'] == 'moved-heading'


def test_public_codec_builds_hierarchy_without_fs_dependency():
    roots, descendants = prepare_docx_descendants([
        {'block_id': 'parent', 'block_type': 2, 'text': {'elements': []}},
        {
            'block_id': 'child',
            'parent_id': 'parent',
            'block_type': 2,
            'text': {'elements': []},
        },
    ])
    assert roots == ['parent']
    assert descendants[0]['children'] == ['child']
