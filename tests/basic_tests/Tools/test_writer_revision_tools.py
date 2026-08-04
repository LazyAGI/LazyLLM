import tempfile
from unittest.mock import MagicMock, patch

import pytest

from lazyllm.tools.writer.data_models import (
    ContentRef,
    LocatedContent,
    LocateResult,
    ModifyInstruction,
    ModifyPlan,
    PatchHunk,
    PatchSet,
    StringReplace,
    StringReplaceSet,
    WriterBlock,
    WriterDocument,
    WriterSpan,
    WritingContext,
    WritingTask,
)
from lazyllm.tools.writer.data_models.revision import GeneratedRevision, RevisionBlockContent
from lazyllm.tools.writer.utils import load_artifact_json
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


def test_locate_revision_target_supports_ir_and_markdown():
    task = WritingTask(query='修改第一章', task_type='revise')
    context = WritingContext(context_id='ctx-1')
    ir_result = LocateResult(
        target_title=False,
        targets=[LocatedContent(content_ref=ContentRef(node_id='update'))],
    )
    markdown_result = LocateResult(
        target_title=False,
        targets=[LocatedContent(content_ref=ContentRef(heading_path=['第一章']))],
    )
    plain_text_result = LocateResult(
        target_title=False,
        targets=[LocatedContent(content_ref=ContentRef(document_root=True))],
    )

    with tempfile.TemporaryDirectory() as directory:
        tool = WriterRevisionTools(llm=MagicMock(), artifact_store=directory)
        with patch.object(
            tool,
            '_call_llm_structured',
            side_effect=[ir_result, markdown_result, plain_text_result],
        ):
            ir_output = tool.locate_revision_target(task, _document(), context)
            located_ir = load_artifact_json(ir_output['artifact_path'], LocateResult)
            markdown_output = tool.locate_revision_target(task, '# 第一章\n\n原始正文', context)
            located_markdown = load_artifact_json(markdown_output['artifact_path'], LocateResult)
            plain_text_output = tool.locate_revision_target(task, '原始纯文本', context)

        located_plain_text = load_artifact_json(plain_text_output['artifact_path'], LocateResult)
        assert located_ir.targets[0].content_ref.node_id == 'update'
        assert located_markdown.targets[0].content_ref.heading_path == ['第一章']
        assert located_plain_text.targets[0].content_ref.document_root is True


def test_normalize_move_plan_uses_destination_ref():
    tool = WriterRevisionTools()
    source_ref = ContentRef(node_id='move')
    destination_ref = ContentRef(node_id='anchor')
    plan = ModifyPlan(
        scope='block',
        instructions=[ModifyInstruction(
            content_ref=source_ref,
            destination_ref=destination_ref,
            modify_type='move',
            position='after',
            instruction='移动到锚点之后',
        )],
    )

    normalized = tool._normalize_modify_plan(
        plan,
        WritingTask(query='移动段落', task_type='revise'),
        [source_ref],
        {tool._content_ref_key(source_ref), tool._content_ref_key(destination_ref)},
    )

    assert normalized.instructions[0].destination_ref == destination_ref


def test_normalize_create_plan_requires_position():
    content_ref = ContentRef(node_id='anchor')
    plan = ModifyPlan(
        scope='block',
        instructions=[ModifyInstruction(
            content_ref=content_ref,
            modify_type='create',
            instruction='补充一段内容',
        )],
    )

    with pytest.raises(ValueError, match='create instruction requires position'):
        WriterRevisionTools()._normalize_modify_plan(
            plan,
            WritingTask(query='补充内容', task_type='revise'),
            [content_ref],
            {WriterRevisionTools._content_ref_key(content_ref)},
        )


def test_generate_patch_set_uses_instruction_content_ref():
    plan = ModifyPlan(
        scope='block',
        instructions=[ModifyInstruction(
            instruction_id='update-1',
            content_ref=ContentRef(node_id='update'),
            modify_type='update',
            instruction='更新正文',
        )],
    )
    generated = GeneratedRevision(changes={
        'update-1': [RevisionBlockContent(content='new')],
    })

    with tempfile.TemporaryDirectory() as directory:
        tool = WriterRevisionTools(llm=MagicMock(), artifact_store=directory)
        with patch.object(tool, '_call_llm_structured', return_value=generated):
            output = tool.generate_patch_set(_document(), plan, WritingContext(context_id='ctx-1'))
        patch_set = load_artifact_json(output['artifact_path'], PatchSet)

    assert patch_set.hunks[0].target_node_id == 'update'
    assert patch_set.hunks[0].block.content == 'new'


def test_generate_modify_plan_supports_markdown_content_ref():
    content_ref = ContentRef(heading_path=['第一章'])
    located = LocateResult(
        target_title=False,
        targets=[LocatedContent(content_ref=content_ref)],
    )
    plan = ModifyPlan(
        scope='section',
        instructions=[ModifyInstruction(
            content_ref=content_ref,
            modify_type='update',
            instruction='更新第一章',
        )],
    )

    with tempfile.TemporaryDirectory() as directory:
        tool = WriterRevisionTools(llm=MagicMock(), artifact_store=directory)
        with patch.object(tool, '_call_llm_structured', return_value=plan):
            output = tool.generate_modify_plan(
                WritingTask(query='修改第一章', task_type='revise'),
                '# 第一章\n\n原始正文',
                located,
                WritingContext(context_id='ctx-1'),
            )
        modified_plan = load_artifact_json(output['artifact_path'], ModifyPlan)

    assert modified_plan.instructions[0].content_ref == content_ref


def test_apply_string_replace_updates_markdown_section():
    markdown = '# 第一章\n\n原始正文\n\n# 第二章\n\n保持不变'
    replace_set = StringReplaceSet(replacements=[StringReplace(
        replacement_id='replace-1',
        content_ref=ContentRef(heading_path=['第一章']),
        old_string='原始正文',
        new_string='修改后的正文',
    )])

    with tempfile.TemporaryDirectory() as directory:
        result = WriterRevisionTools(artifact_store=directory).apply_string_replace(
            markdown,
            replace_set,
            WritingContext(context_id='ctx-1'),
        )
        with open(result['revised_document_md'], 'r', encoding='utf-8') as stream:
            revised = stream.read()

    assert '修改后的正文' in revised
    assert '# 第二章\n\n保持不变' in revised


def test_apply_string_replace_updates_plain_text_document():
    replace_set = StringReplaceSet(replacements=[StringReplace(
        content_ref=ContentRef(document_root=True),
        old_string='原始内容',
        new_string='修改后的内容',
    )])

    with tempfile.TemporaryDirectory() as directory:
        result = WriterRevisionTools(artifact_store=directory).apply_string_replace(
            '原始内容',
            replace_set,
            WritingContext(context_id='ctx-1'),
        )
        with open(result['revised_document_md'], 'r', encoding='utf-8') as stream:
            revised = stream.read()

    assert revised == '修改后的内容'
