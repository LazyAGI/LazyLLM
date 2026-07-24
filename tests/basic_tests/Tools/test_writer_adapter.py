from copy import deepcopy

from lazyllm.tools.writer.adapter.feishu import FeishuWriterAdapter
from lazyllm.tools.writer.data_models import PatchHunk, WriterBlock, WriterSpan
from lazyllm.tools.writer.utils.feishu_docx import prepare_docx_clone_descendants


def _block(block_id, content, *, parent='doc-1', children=None, heading=False):
    field = 'heading1' if heading else 'text'
    block = {
        'block_id': block_id,
        'block_type': 3 if heading else 2,
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


def test_create_and_delete_build_native_operations():
    adapter = FeishuWriterAdapter()
    document = adapter.blocks_to_ir(_move_blocks()[:2], external_document_id='doc-1')
    paragraph = document.blocks[0].children[0]
    created = WriterBlock(
        node_id='new-block',
        type='paragraph',
        content='新增段落',
        spans=[WriterSpan(text='新增段落', style={'bold': True})],
        stage='final',
    )

    create = adapter.patch_to_operation(PatchHunk(
        target_node_id=created.node_id,
        modify_type='create',
        block=created,
        parent_node_id=None,
        index=1,
    ), document)
    assert create.operation == 'create'
    assert (create.params['parent_block_id'], create.params['index']) == ('doc-1', 1)
    assert create.params['descendants'][0]['text']['elements'][0][
        'text_run']['text_element_style'] == {'bold': True}

    delete = adapter.patch_to_operation(PatchHunk(
        target_node_id=paragraph.node_id,
        modify_type='delete',
    ), document)
    assert delete.operation == 'delete'
    assert delete.params == {
        'parent_block_id': 'heading-1',
        'start_index': 0,
        'end_index': 1,
    }


def test_update_maps_styles_and_block_type_changes():
    adapter = FeishuWriterAdapter()
    document = adapter.blocks_to_ir(
        [_block('paragraph-1', '段落一')],
        external_document_id='doc-1',
    )
    source = document.blocks[0]

    styled = source.model_copy(deep=True)
    styled.spans[0].style = {
        'bold': True,
        'text_color': 5,
        'background_color': 2,
        'font_size': 16,
    }
    update = adapter.patch_to_operation(PatchHunk(
        target_node_id=source.node_id,
        modify_type='update',
        block=styled,
    ), document)
    assert update.operation == 'update'
    assert update.params['requests'][0]['update_text_elements']['elements'][0][
        'text_run']['text_element_style'] == styled.spans[0].style

    heading = styled.model_copy(deep=True)
    heading.type = 'heading'
    heading.numbering = {'level': 4}
    replace = adapter.patch_to_operation(PatchHunk(
        target_node_id=source.node_id,
        modify_type='update',
        block=heading,
    ), document)
    assert replace.operation == 'replace'
    assert replace.params['replacement_block']['block_type'] == 6
    assert 'heading4' in replace.params['replacement_block']


def test_move_uses_parent_and_final_index():
    adapter = FeishuWriterAdapter()
    document = adapter.blocks_to_ir(_move_blocks(), external_document_id='doc-1')
    source = document.blocks[0].children[0]
    target_parent = document.blocks[1]

    operation = adapter.patch_to_operation(PatchHunk(
        target_node_id=source.node_id,
        modify_type='move',
        parent_node_id=target_parent.node_id,
        index=1,
    ), document)

    assert operation.operation == 'move'
    assert operation.params == {
        'source_parent_block_id': 'heading-1',
        'source_block_id': 'paragraph-1',
        'source_index': 0,
        'target_parent_block_id': 'heading-2',
        'target_index': 1,
    }


def test_merge_refreshed_move_restores_writer_identity():
    adapter = FeishuWriterAdapter()
    previous = adapter.blocks_to_ir(_move_blocks(), external_document_id='doc-1')
    source = previous.blocks[0]
    patch = PatchHunk(
        target_node_id=source.node_id,
        modify_type='move',
        parent_node_id=None,
        index=1,
    )
    operation = adapter.patch_to_operation(patch, previous)

    refreshed_raw = deepcopy(_move_blocks()[2:] + _move_blocks()[:2])
    refreshed_raw[2]['block_id'] = 'moved-heading'
    refreshed_raw[2]['children'] = ['moved-paragraph']
    refreshed_raw[3]['block_id'] = 'moved-paragraph'
    refreshed_raw[3]['parent_id'] = 'moved-heading'
    refreshed = adapter.blocks_to_ir(refreshed_raw, external_document_id='doc-1')

    moved = adapter.merge_refreshed_document(
        previous,
        refreshed,
        patch=patch,
        operation=operation,
        operation_result={
            'block_id_relations': {
                'heading-1': 'moved-heading',
                'paragraph-1': 'moved-paragraph',
            },
        },
    ).blocks[1]
    assert moved.node_id == source.node_id
    assert moved.children[0].node_id == source.children[0].node_id
    assert moved.provider_binding['block_id'] == 'moved-heading'


def test_move_clone_descendants_preserve_raw_format_and_children():
    blocks = _move_blocks()[:2]
    blocks[0]['heading1']['style'] = {'align': 2}
    blocks[0]['heading1']['elements'][0]['text_run']['text_element_style'] = {
        'bold': True,
        'text_color': 5,
    }

    children_id, descendants, id_map, _ = \
        prepare_docx_clone_descendants(blocks, 'heading-1')

    assert children_id == ['move-block-0']
    assert id_map == {
        'move-block-0': 'heading-1',
        'move-block-1': 'paragraph-1',
    }
    assert descendants[0]['heading1'] == blocks[0]['heading1']
    assert descendants[0]['children'] == ['move-block-1']
