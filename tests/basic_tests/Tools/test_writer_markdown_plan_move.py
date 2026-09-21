from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lazyllm.tools.writer.data_models import ContentRef, LocatedContent, LocateResult, WritingContext, WritingTask
from lazyllm.tools.writer.data_models.revision import MarkdownModifyPlan, StringReplaceSet
from lazyllm.tools.writer.tools.revision_tools import WriterRevisionTools
from lazyllm.tools.writer.utils import load_artifact_json


SOURCE = '# A\n\nAlpha.\n\n# B\n\nBeta.\n'


@pytest.fixture
def tool(tmp_path):
    return WriterRevisionTools(llm=MagicMock(), artifact_store=str(tmp_path))


def _move(**fields):
    return MarkdownModifyPlan(scope='document', instructions=[{
        'instruction_id': 'move-1', 'modify_type': 'move', 'instruction': 'Move the original text.',
        'content_ref': {'document_root': True}, 'destination_ref': {'document_root': True},
        'target_scope': 'fragment', 'destination_scope': 'fragment', 'position': 'after', **fields,
    }])


def _apply(tool, source, plan):
    context = WritingContext(context_id='markdown-move')
    generated = tool.generate_string_replace_set(source, plan, context)
    replacements = load_artifact_json(generated['artifact_path'], StringReplaceSet)
    result = tool.apply_string_replace(source, replacements, context)
    return Path(result['revised_document_md']).read_bytes().decode(), replacements


@pytest.mark.parametrize('source,fields,expected', [
    pytest.param(
        '# Same\n\nONE\n\n# Same\n\nTWO\n\n# Same\n\nTHREE\n\n',
        {'target_scope': 'section', 'content_ref': {'heading_path': ['Same'], 'occurrence': 1},
         'destination_scope': 'section', 'destination_ref': {'heading_path': ['Same'], 'occurrence': 2}},
        '# Same\n\nTWO\n\n# Same\n\nONE\n\n# Same\n\nTHREE\n\n', id='section-occurrence'),
    pytest.param(
        '# A\n\nAAA\n\n# B\n\nBBB\n\n# C\n\nCCC',
        {'target_scope': 'paragraph', 'content_ref': {'heading_path': ['C']}, 'locator_text': 'CCC',
         'destination_scope': 'section', 'destination_ref': {'heading_path': ['A']}},
        '# A\n\nAAA\n\nCCC\n\n# B\n\nBBB\n\n# C\n\n', id='paragraph-heading-separator'),
    pytest.param(
        'First.  Second.', {'locator_text': 'First.', 'destination_locator_text': 'Second.'},
        'Second.  First.', id='fragment-english-spaces'),
])
def test_move_preserves_original_text_and_separators(tool, source, fields, expected):
    with patch.object(tool, '_call_llm_structured', side_effect=AssertionError('Unexpected model call')):
        revised, replacements = _apply(tool, source, _move(**fields))
    assert revised == expected
    assert len(replacements.replacements) == 2
    assert all(item.meta['source'] == 'program_move' for item in replacements.replacements)


def test_fragment_cutting_markdown_format_uses_model_fallback(tool):
    source, expected = '**First.** Second.', 'Second. **First.**'
    response = {'replacements': [{'old_string': source, 'new_string': expected}]}
    model = MagicMock(return_value=response)
    with patch.object(tool, '_build_structured_llm', return_value=model):
        revised, _ = _apply(tool, source, _move(locator_text='First.', destination_locator_text='Second.'))
    assert revised == expected
    model.assert_called_once()


def _missing_locator_plan(operation):
    instruction = {
        'instruction_id': 'edit-1', 'modify_type': operation, 'instruction': 'Revise the selected paragraph.',
        'content_ref': {'heading_path': ['A']}, 'target_scope': 'paragraph', 'meta': {'keep': 'original'},
    }
    if operation == 'move':
        instruction.update(destination_ref={'heading_path': ['B']}, destination_scope='section', position='before')
    return MarkdownModifyPlan(scope='document', instructions=[instruction])


def _generate(tool):
    return tool.generate_modify_plan(
        WritingTask(task_id='task-1', query='Revise paragraph A.', task_type='revise'), SOURCE,
        LocateResult(task_id='task-1', target_title=False, targets=[
            LocatedContent(content_ref=ContentRef(heading_path=['A'])),
        ]), WritingContext(context_id='markdown-plan'),
    )


@pytest.mark.parametrize('operation,reference_retry', [('update', False), ('move', True)])
def test_generated_plan_fills_only_missing_fields_after_optional_reference_retry(tool, operation, reference_retry):
    plan = _missing_locator_plan(operation)
    original = plan.model_dump()
    responses = []
    if reference_retry:
        invalid = plan.model_copy(deep=True)
        invalid.instructions[0].content_ref.node_id = 'invalid-ir-reference'
        responses.append(invalid.model_dump())
    responses.extend([original, {'instructions': [{'instruction_index': 1, 'locator_text': 'Alpha.'}]}])
    model = MagicMock(side_effect=responses)
    with patch.object(tool, '_build_structured_llm', return_value=model):
        generated = _generate(tool)
    restored = load_artifact_json(
        generated['artifact_path'], MarkdownModifyPlan,
        expected_schema_name='lazyllm.tools.writer.data_models.revision.ModifyPlan',
    )
    expected = plan.instructions[0].model_copy(update={'locator_text': 'Alpha.'})
    assert restored.instructions == [expected]
    assert plan.model_dump() == original
    assert model.call_count == 2 + int(reference_retry)


@pytest.mark.parametrize('completion', [
    pytest.param({'locator_text': None}, id='still-missing'),
    pytest.param({'locator_text': 'Alpha.', 'position': 'after'}, id='overwrite-existing'),
])
def test_invalid_completion_stops_without_saving_a_plan(tool, tmp_path, completion):
    model = MagicMock(side_effect=[
        _missing_locator_plan('move').model_dump(),
        {'instructions': [{'instruction_index': 1, **completion}]},
    ])
    with patch.object(tool, '_build_structured_llm', return_value=model):
        with pytest.raises(ValueError, match='Markdown plan completion'):
            _generate(tool)
    assert model.call_count == 2
    assert not list(tmp_path.rglob('modify_plan*.json'))
