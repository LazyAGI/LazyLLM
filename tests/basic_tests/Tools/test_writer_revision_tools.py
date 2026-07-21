import tempfile
from unittest.mock import MagicMock, patch

import pytest

from lazyllm.tools.writer.data_models import WriterBlock, WriterDocument, WritingContext
from lazyllm.tools.writer.data_models.revision import (
    ModifyInstruction,
    ModifyPlan,
    PatchBlock,
    PatchHunk,
    PatchSet,
)
from lazyllm.tools.writer.adapter.feishu import FeishuWriterAdapter
from lazyllm.tools.writer.tools.revision_tools import WriterRevisionTools
from lazyllm.tools.writer.utils import load_artifact_json


def _block(node_id, content, *, children=None):
    return WriterBlock(
        node_id=node_id,
        type='paragraph',
        content=content,
        children=children or [],
        stage='final',
    )


def _document():
    return WriterDocument(
        document_id='doc-1',
        stage='final',
        blocks=[
            _block('replace', 'old replacement'),
            _block('insert-anchor', 'insert anchor'),
            _block('delete', 'delete me'),
            _block('move', 'move me'),
            _block('move-anchor', 'move anchor'),
        ],
    )


def _context():
    return WritingContext(context_id='context-1', doc_id='doc-1', query='revise')


def _revised_document(result):
    path = result['metadata']['artifact_paths']['revised_document']
    return load_artifact_json(path, WriterDocument)


def test_generate_patch_set_preserves_insert_and_move_fields():
    document = WriterDocument(
        document_id='doc-1',
        stage='final',
        blocks=[_block('a', 'A'), _block('b', 'B')],
    )
    plan = ModifyPlan(
        scope='block',
        instructions=[
            ModifyInstruction(
                target_node_id='a',
                modify_type='insert',
                position='before',
                instruction='Insert an introduction.',
            ),
            ModifyInstruction(
                target_node_id='b',
                modify_type='move',
                anchor_node_id='a',
                position='after',
                instruction='Move B after A.',
            ),
        ],
    )
    proposal = PatchSet(
        target_doc_id='doc-1',
        hunks=[
            PatchHunk(
                target_node_id='a',
                modify_type='insert',
                position='before',
                new_blocks=[PatchBlock(type='paragraph', content='Introduction')],
            ),
            PatchHunk(
                target_node_id='b',
                modify_type='move',
                anchor_node_id='a',
                position='after',
            ),
        ],
    )

    with tempfile.TemporaryDirectory() as directory:
        tool = WriterRevisionTools(llm=MagicMock(), artifact_store=directory)
        with patch.object(tool, '_call_llm_structured', return_value=proposal):
            result = tool.generate_patch_set(document, plan, _context())
        generated = load_artifact_json(result['artifact_path'], PatchSet)

    insert, move = generated.hunks
    assert insert.position == 'before'
    assert [block.content for block in insert.new_blocks] == ['Introduction']
    assert insert.new_text is None
    assert move.anchor_node_id == 'a'
    assert move.position == 'after'
    assert move.new_text is None


def test_apply_patch_supports_all_block_operations_atomically():
    patch_set = PatchSet(
        patch_id='patch-1',
        target_doc_id='doc-1',
        hunks=[
            PatchHunk(
                hunk_id='replace-hunk',
                target_node_id='replace',
                modify_type='replace',
                old_text='old replacement',
                new_text='new replacement',
            ),
            PatchHunk(
                hunk_id='insert-hunk',
                target_node_id='insert-anchor',
                modify_type='insert',
                position='before',
                new_blocks=[
                    PatchBlock(type='paragraph', content='inserted one'),
                    PatchBlock(type='paragraph', content='inserted two'),
                ],
            ),
            PatchHunk(
                hunk_id='delete-hunk',
                target_node_id='delete',
                modify_type='delete',
                old_text='delete me',
            ),
            PatchHunk(
                hunk_id='move-hunk',
                target_node_id='move',
                modify_type='move',
                old_text='move me',
                anchor_node_id='move-anchor',
                position='after',
            ),
        ],
    )

    with tempfile.TemporaryDirectory() as directory:
        result = WriterRevisionTools(artifact_store=directory).apply_patch(
            _document(), patch_set, _context(),
        )
        revised = _revised_document(result)

    assert [block.node_id for block in revised.blocks] == [
        'replace',
        'insert-hunk::block-1',
        'insert-hunk::block-2',
        'insert-anchor',
        'move-anchor',
        'move',
    ]
    assert revised.block_by_id('replace').content == 'new replacement'
    assert revised.block_by_id('delete') is None
    assert result['metadata']['counts'] == {'applied': 4, 'failed': 0}


def test_revision_patch_contract_with_feishu_adapter():
    adapter = FeishuWriterAdapter()
    source = adapter.blocks_to_ir(
        [{
            'block_id': 'feishu-block-1',
            'block_type': 2,
            'parent_id': 'feishu-doc-1',
            'text': {
                'elements': [{
                    'text_run': {
                        'content': 'Original text',
                        'text_element_style': {'bold': True},
                    },
                }],
            },
            'plain_text': 'Original text',
        }],
        external_document_id='feishu-doc-1',
        stage='final',
    )
    target = source.blocks[0]
    patch_set = PatchSet(
        patch_id='patch-feishu-contract',
        target_doc_id=source.document_id,
        hunks=[PatchHunk(
            hunk_id='replace-feishu-block',
            target_node_id=target.node_id,
            modify_type='replace',
            old_text='Original text',
            new_text='Revised text',
        )],
    )

    with tempfile.TemporaryDirectory() as directory:
        result = WriterRevisionTools(artifact_store=directory).apply_patch(
            source, patch_set, WritingContext(
                context_id='context-feishu-contract',
                doc_id=source.document_id,
                query='revise',
            ),
        )
        revised = _revised_document(result)

    operation = adapter.patch_to_operation(patch_set.hunks[0], source)
    request = operation.params['requests'][0]
    assert revised.block_by_id(target.node_id).content == 'Revised text'
    assert operation.operation == 'update'
    assert request['block_id'] == 'feishu-block-1'
    assert request['update_text_elements']['elements'][0]['text_run']['content'] == 'Revised text'
    assert request['update_text_elements']['elements'][0]['text_run']['text_element_style'] == {
        'bold': True,
    }


def test_apply_patch_inserts_next_to_nested_anchor():
    document = WriterDocument(
        document_id='doc-1',
        stage='final',
        blocks=[_block('parent', 'parent', children=[_block('child', 'child')])],
    )
    patch_set = PatchSet(
        target_doc_id='doc-1',
        hunks=[PatchHunk(
            hunk_id='nested-insert',
            target_node_id='child',
            modify_type='insert',
            position='after',
            new_blocks=[PatchBlock(type='paragraph', content='new child')],
        )],
    )

    with tempfile.TemporaryDirectory() as directory:
        result = WriterRevisionTools(artifact_store=directory).apply_patch(
            document, patch_set, _context(),
        )
        revised = _revised_document(result)

    assert [block.content for block in revised.blocks[0].children] == ['child', 'new child']


def test_apply_patch_rejects_conflict_before_any_mutation():
    document = _document()
    patch_set = PatchSet(
        target_doc_id='doc-1',
        hunks=[
            PatchHunk(
                hunk_id='valid-first',
                target_node_id='replace',
                modify_type='replace',
                old_text='old replacement',
                new_text='would be applied by a partial implementation',
            ),
            PatchHunk(
                hunk_id='conflict-second',
                target_node_id='delete',
                modify_type='delete',
                old_text='stale text',
            ),
        ],
    )

    with tempfile.TemporaryDirectory() as directory:
        with pytest.raises(ValueError, match='old_text conflict'):
            WriterRevisionTools(artifact_store=directory).apply_patch(
                document, patch_set, _context(),
            )

    assert document.block_by_id('replace').content == 'old replacement'
    assert document.block_by_id('delete') is not None


def test_apply_patch_rejects_move_into_descendant():
    child = _block('child', 'child')
    document = WriterDocument(
        document_id='doc-1',
        stage='final',
        blocks=[_block('parent', 'parent', children=[child])],
    )
    patch_set = PatchSet(
        target_doc_id='doc-1',
        hunks=[PatchHunk(
            target_node_id='parent',
            modify_type='move',
            old_text='parent',
            anchor_node_id='child',
            position='after',
        )],
    )

    with tempfile.TemporaryDirectory() as directory:
        with pytest.raises(ValueError, match='descendants'):
            WriterRevisionTools(artifact_store=directory).apply_patch(
                document, patch_set, _context(),
            )


def test_apply_patch_rejects_missing_move_anchor():
    patch_set = PatchSet(
        target_doc_id='doc-1',
        hunks=[PatchHunk(
            target_node_id='move',
            modify_type='move',
            old_text='move me',
            anchor_node_id='missing',
            position='after',
        )],
    )

    with tempfile.TemporaryDirectory() as directory:
        with pytest.raises(ValueError, match='move anchor'):
            WriterRevisionTools(artifact_store=directory).apply_patch(
                _document(), patch_set, _context(),
            )
