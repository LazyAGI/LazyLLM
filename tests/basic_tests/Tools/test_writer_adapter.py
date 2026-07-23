from copy import deepcopy

import pytest

from lazyllm.tools.writer.adapter.feishu import FeishuWriterAdapter
from lazyllm.tools.writer.data_models import PatchBlock, PatchHunk


def _block(block_id, content, *, parent='doc-1', children=None, heading=False):
    field = 'heading1' if heading else 'text'
    block = {
        'block_id': block_id, 'block_type': 3 if heading else 2,
        'parent_id': parent,
        field: {'elements': [{'text_run': {'content': content}}]},
    }
    if children:
        block['children'] = children
    return block


def _move_blocks():
    return [
        _block('heading-1', '章节一', children=['paragraph-1'], heading=True),
        _block('paragraph-1', '段落一', parent='heading-1'),
        _block('heading-2', '章节二', children=['paragraph-2'], heading=True),
        _block('paragraph-2', '段落二', parent='heading-2'),
    ]


def test_insert_and_delete_build_native_operations():
    adapter = FeishuWriterAdapter()
    document = adapter.blocks_to_ir(_move_blocks()[:2], external_document_id='doc-1')
    heading, paragraph = document.blocks[0], document.blocks[0].children[0]

    create = adapter.patch_to_operation(PatchHunk(
        hunk_id='insert-1', target_node_id=heading.node_id,
        modify_type='insert', position='after',
        new_blocks=[PatchBlock(type='paragraph', content='新增段落')],
    ), document)
    assert create.operation == 'create'
    assert (create.params['parent_block_id'], create.params['index']) == ('doc-1', 1)
    assert create.params['descendants'][0]['block_type'] == 2

    delete = adapter.patch_to_operation(PatchHunk(
        target_node_id=paragraph.node_id, modify_type='delete',
        old_text=paragraph.content,
    ), document)
    assert delete.operation == 'delete'
    assert delete.params == {
        'parent_block_id': 'heading-1', 'start_index': 0, 'end_index': 1}


@pytest.mark.parametrize(
    ('source_path', 'anchor_path', 'position', 'expected'),
    [
        ((0,), (1,), 'after', ('doc-1', 0, 'doc-1', 1)),
        ((0, 0), (1, 0), 'before', ('heading-1', 0, 'heading-2', 0)),
    ],
)
def test_move_builds_same_and_cross_parent_operations(
    source_path, anchor_path, position, expected,
):
    adapter = FeishuWriterAdapter()
    document = adapter.blocks_to_ir(_move_blocks(), external_document_id='doc-1')

    def locate(path):
        root = document.blocks[path[0]]
        return root if len(path) == 1 else root.children[path[1]]

    source, anchor = locate(source_path), locate(anchor_path)
    operation = adapter.patch_to_operation(PatchHunk(
        hunk_id='move-1', target_node_id=source.node_id, modify_type='move',
        anchor_node_id=anchor.node_id, position=position,
    ), document)

    assert operation.operation == 'move'
    assert operation.params['source_block_id'] == source.provider_binding['block_id']
    assert tuple(operation.params[key] for key in (
        'source_parent_block_id', 'source_index',
        'target_parent_block_id', 'target_index',
    )) == expected
    assert len(operation.params['children_id']) == 1


def test_merge_refreshed_move_restores_writer_identity():
    adapter = FeishuWriterAdapter()
    previous = adapter.blocks_to_ir(_move_blocks(), external_document_id='doc-1')
    source, anchor = previous.blocks
    patch = PatchHunk(
        hunk_id='move-1', target_node_id=source.node_id, modify_type='move',
        anchor_node_id=anchor.node_id, position='after',
    )
    operation = adapter.patch_to_operation(patch, previous)

    refreshed_raw = deepcopy(_move_blocks()[2:] + _move_blocks()[:2])
    refreshed_raw[2]['block_id'] = 'moved-heading'
    refreshed_raw[2]['children'] = ['moved-paragraph']
    refreshed_raw[3]['block_id'] = 'moved-paragraph'
    refreshed_raw[3]['parent_id'] = 'moved-heading'
    refreshed = adapter.blocks_to_ir(refreshed_raw, external_document_id='doc-1')

    moved = adapter.merge_refreshed_document(
        previous, refreshed, patch=patch, operation=operation).blocks[1]
    assert moved.node_id == source.node_id
    assert moved.children[0].node_id == source.children[0].node_id
    assert moved.provider_binding['block_id'] == 'moved-heading'
