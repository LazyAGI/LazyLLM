import pytest

from lazyllm.tools.writer.data_models import (
    PatchHunk,
    PatchSet,
    WriterBlock,
    WriterDocument,
    WriterSpan,
)
from lazyllm.tools.writer.tools.revision_tools import WriterRevisionTools, apply_patch_to_ir


def _block(node_id, content, *, children=None, style=None):
    return WriterBlock(
        node_id=node_id,
        type='paragraph',
        content=content,
        spans=[WriterSpan(text=content, style=style or {})],
        children=children or [],
        stage='final',
    )


def _document():
    return WriterDocument(
        document_id='doc-1',
        stage='final',
        blocks=[
            _block('update', 'old'),
            _block('delete', 'delete me'),
            _block('move', 'move me'),
            _block('anchor', 'anchor'),
        ],
    )


def test_apply_patch_supports_all_block_operations():
    document = _document()
    updated = document.block_by_id('update').model_copy(deep=True)
    updated.type = 'heading'
    updated.numbering = {'level': 4}
    updated.spans[0].style = {
        'bold': True,
        'text_color': '#ff0000',
        'background_color': '#ffff00',
        'font_size': 16,
    }
    created = _block('created', 'new block')
    patch = PatchSet(
        patch_id='patch-1',
        target_doc_id=document.document_id,
        hunks=[
            PatchHunk(
                hunk_id='update-hunk',
                target_node_id=updated.node_id,
                modify_type='update',
                block=updated,
            ),
            PatchHunk(
                hunk_id='delete-hunk',
                target_node_id='delete',
                modify_type='delete',
            ),
            PatchHunk(
                hunk_id='move-hunk',
                target_node_id='move',
                modify_type='move',
                parent_node_id=None,
                index=2,
            ),
            PatchHunk(
                hunk_id='create-hunk',
                target_node_id=created.node_id,
                modify_type='create',
                block=created,
                parent_node_id=None,
                index=1,
            ),
        ],
    )

    revised, result = apply_patch_to_ir(document, patch)

    assert [block.node_id for block in revised.blocks] == [
        'update', 'created', 'anchor', 'move',
    ]
    assert revised.blocks[0].type == 'heading'
    assert revised.blocks[0].numbering == {'level': 4}
    assert revised.blocks[0].spans[0].style['background_color'] == '#ffff00'
    assert result.applied_hunks == [
        'update-hunk', 'delete-hunk', 'move-hunk', 'create-hunk',
    ]


def test_document_diff_round_trips_arbitrary_visible_edits():
    source = _document()
    revised = source.model_copy(deep=True)
    revised.title = 'new title'
    revised.blocks[0].type = 'heading'
    revised.blocks[0].numbering = {'level': 4}
    revised.blocks[0].spans[0].style = {'bold': True, 'text_color': 'red'}
    revised.blocks.insert(1, _block('created', 'created by enter'))
    revised.blocks = [revised.blocks[2], revised.blocks[0], revised.blocks[1]]
    revised.blocks = [block for block in revised.blocks if block.node_id != 'anchor']

    patch = WriterRevisionTools()._diff_documents(source, revised)
    applied, _ = apply_patch_to_ir(source, patch)

    assert {hunk.modify_type for hunk in patch.hunks} == {
        'create', 'update', 'delete', 'move',
    }
    WriterRevisionTools._assert_revision_applied(applied, revised)


def test_document_diff_creates_nested_subtree():
    source = WriterDocument(
        document_id='doc-1',
        stage='final',
        blocks=[_block('existing', 'existing')],
    )
    revised = source.model_copy(deep=True)
    revised.blocks.append(_block(
        'new-parent',
        'new parent',
        children=[_block('new-child', 'new child')],
    ))

    patch = WriterRevisionTools()._diff_documents(source, revised)
    assert len(patch.hunks) == 1
    assert patch.hunks[0].modify_type == 'create'
    assert patch.hunks[0].block.children[0].node_id == 'new-child'


def test_document_diff_rejects_provider_field_changes():
    source = _document()
    revised = source.model_copy(deep=True)
    revised.blocks[0].provider_binding = {'provider': 'feishu', 'block_id': 'other'}

    with pytest.raises(ValueError, match='provider-managed fields'):
        WriterRevisionTools()._diff_documents(source, revised)


def test_apply_patch_rejects_move_into_descendant():
    child = _block('child', 'child')
    document = WriterDocument(
        document_id='doc-1',
        stage='final',
        blocks=[_block('parent', 'parent', children=[child])],
    )
    patch = PatchSet(
        target_doc_id='doc-1',
        hunks=[PatchHunk(
            target_node_id='parent',
            modify_type='move',
            parent_node_id='child',
            index=0,
        )],
    )

    with pytest.raises(ValueError, match='own subtree'):
        apply_patch_to_ir(document, patch)
