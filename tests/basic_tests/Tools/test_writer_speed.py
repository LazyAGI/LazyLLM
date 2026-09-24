'''Regression checks for Writer's deterministic fast paths.'''
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from lazyllm.tools.writer.data_models import (
    ContentRef, DocumentFact, WriterBlock, WriterDocument, WritingContext, WritingTask,
)
from lazyllm.tools.writer.data_models.context import StyleProfile
from lazyllm.tools.writer.data_models.planning import SectionInstruction
from lazyllm.tools.writer.data_models.revision import (
    GeneratedRevision, LocateResult, MarkdownModifyInstruction, MarkdownModifyPlan,
    ModifyInstruction, ModifyPlan, PatchSet, RevisionBlockContent, StringReplaceSet,
)
from lazyllm.tools.writer.data_models.task import Selection
from lazyllm.tools.writer.tools.drafting_tools import WriterDraftingTools
from lazyllm.tools.writer.tools.planning_tools import WriterPlanningTools
from lazyllm.tools.writer.tools.revision_tools import WriterRevisionTools, apply_patch_to_ir
from lazyllm.tools.writer.utils import load_artifact_json, parse_document_markdown


def document():
    return WriterDocument(document_id='doc', stage='final', blocks=[
        WriterBlock(node_id=key, type='paragraph', content=key) for key in ('a', 'b', 'c')
    ])


def test_explicit_selection_skips_model_but_rejects_stale_refs(tmp_path):
    tool = WriterRevisionTools(artifact_store=str(tmp_path))
    task = WritingTask(query='Polish this selection', task_type='revise', scope='selection',
                       selection=Selection(content_refs=[ContentRef(node_id='b')]))
    with patch.object(tool, '_call_llm_structured', side_effect=AssertionError('unexpected model call')):
        result = tool.locate_revision_target(task, document(), WritingContext(context_id='ctx'))
        located = load_artifact_json(result['artifact_path'], LocateResult)
        assert [item.content_ref.node_id for item in located.targets] == ['b']
        task.selection.content_refs.append(ContentRef(node_id='stale'))
        with pytest.raises(ValueError, match='stale'):
            tool.locate_revision_target(task, document(), WritingContext(context_id='ctx'))


def test_partial_selection_keeps_semantic_location(tmp_path):
    tool = WriterRevisionTools(artifact_store=str(tmp_path))
    task = WritingTask(query='Polish the quote', task_type='revise', scope='selection',
                       selection=Selection(content_refs=[ContentRef(node_id='b')], text='b'))
    with patch.object(tool, '_call_llm_structured', return_value=LocateResult(
        target_title=False, targets=[{'content_ref': {'node_id': 'b'}}],
    )) as model:
        tool.locate_revision_target(task, document(), WritingContext(context_id='ctx'))
    model.assert_called_once()


@pytest.mark.parametrize('mixed', [False, True])
def test_ir_structural_changes_are_compiled_without_generated_empty_content(tmp_path, mixed):
    tool = WriterRevisionTools(artifact_store=str(tmp_path))
    instructions = [
        ModifyInstruction(instruction_id='delete', content_ref=ContentRef(node_id='a'),
                          modify_type='delete', instruction='Delete a'),
        ModifyInstruction(instruction_id='move', content_ref=ContentRef(node_id='c'),
                          destination_ref=ContentRef(node_id='b'), position='before',
                          modify_type='move', instruction='Move c before b'),
    ]
    if mixed:
        instructions.append(ModifyInstruction(
            instruction_id='edit', content_ref=ContentRef(node_id='b'),
            modify_type='update', instruction='Capitalize b',
        ))
    response = GeneratedRevision(changes={'edit': [RevisionBlockContent(content='B')]})
    with patch.object(tool, '_call_llm_structured', return_value=response) as model:
        result = tool.generate_patch_set(document(), ModifyPlan(scope='document', instructions=instructions),
                                         WritingContext(context_id='ctx'))
    assert model.call_count == int(mixed)
    revised, _ = apply_patch_to_ir(document(), load_artifact_json(result['artifact_path'], PatchSet))
    assert [block.node_id for block in revised.blocks] == ['c', 'b']
    assert revised.blocks[-1].content == ('B' if mixed else 'b')


def test_exact_markdown_fragment_deletion_does_not_generate_source_text(tmp_path):
    source = '# Story\n\nKeep this. Remove this. Keep that.\n'
    plan = MarkdownModifyPlan(scope='span', instructions=[MarkdownModifyInstruction(
        instruction_id='delete', content_ref=ContentRef(document_root=True),
        modify_type='delete', target_scope='fragment', locator_text='Remove this. ',
        instruction='Delete the specified sentence',
    )])
    tool = WriterRevisionTools(artifact_store=str(tmp_path))
    with patch.object(tool, '_call_llm_structured', side_effect=AssertionError('unexpected model call')):
        result = tool.generate_string_replace_set(source, plan, WritingContext(context_id='ctx'))
    replacements = load_artifact_json(result['artifact_path'], StringReplaceSet)
    assert tool._execute_markdown_replacements(source, replacements) == '# Story\n\nKeep this. Keep that.\n'


@pytest.mark.parametrize('representation', ['markdown', 'ir'])
def test_generated_outline_has_description_and_budget_after_one_call(tmp_path, representation):
    tool = WriterPlanningTools(artifact_store=str(tmp_path))
    task = WritingTask(query='Write a story', task_type='write', constraints={'target_chars': 3000},
                       output={'representation': representation})
    context = WritingContext(context_id='ctx')
    outline = WriterDocument(document_id='outline', stage='outline', blocks=[WriterBlock(
        node_id='a', type='heading', content='Village', target_chars=1,
        outline_description='Introduce the village.',
    )])
    markdown = '# Story\n## Village\n<!-- writer:outline {"node_id":"a","target_chars":1,' \
               '"outline_description":"Introduce the village.","context_relations":[],"subtasks":[]} -->\n'
    with patch.object(tool, '_call_llm_text', return_value=markdown) as text_model, \
            patch.object(tool, '_call_llm_structured', return_value=outline) as ir_model:
        result = tool.generate_outline(task, context)
    assert text_model.call_count + ir_model.call_count == 1
    value = parse_document_markdown(Path(result['artifact_path']).read_text(), document_id='outline', stage='outline') \
        if representation == 'markdown' else load_artifact_json(result['artifact_path'], WriterDocument)
    assert value.blocks[0].target_chars == 3000
    assert value.blocks[0].outline_description == 'Introduce the village (about 3000 characters).'


def test_prompt_dedup_preserves_global_constraints_and_does_not_mutate_inputs():
    task = WritingTask(query='Write the story', task_type='write')
    context = WritingContext(context_id='ctx', query=task.query,
                             facts=[DocumentFact(fact_id='f', key='name', value='Lin')],
                             style_profile=StyleProfile(tone='quiet', notes=['Keep the ending open']))
    instruction = SectionInstruction(
        instruction_id='i', content_ref=ContentRef(node_id='a'), section_title='Village', section_goal='Begin',
        fact_constraints=['name: Lin', 'time: night'],
        style_constraints=['tone: quiet', 'Keep the ending open', 'Use short sentences'],
        heading_structure=[],
    )
    before = instruction.model_dump()
    payload = WriterDraftingTools._draft_prompt_inputs(task, instruction, context)
    local = json.loads(payload['section_instruction_json'])
    shared = json.loads(payload['context_json'])
    assert local['fact_constraints'] == ['time: night']
    assert local['style_constraints'] == ['Use short sentences']
    assert local['heading_structure'] == []
    assert shared['facts'][0]['value'] == 'Lin'
    assert shared['style_profile']['tone'] == 'quiet'
    assert 'query' not in shared
    assert instruction.model_dump() == before
    assert context.query == task.query


def test_constraint_dedup_keeps_explicit_override_of_other_section_fact():
    context = WritingContext(context_id='ctx', facts=[DocumentFact(
        fact_id='f', key='time', value='night', applies_to=[ContentRef(node_id='other')],
    )])
    instruction = SectionInstruction(
        instruction_id='i', content_ref=ContentRef(node_id='current'), section_title='Now',
        section_goal='Begin', fact_constraints=['time: night'],
    )
    payload = WriterDraftingTools._draft_prompt_inputs(
        WritingTask(query='Write', task_type='write'), instruction, context,
    )
    assert json.loads(payload['section_instruction_json'])['fact_constraints'] == ['time: night']


def test_whole_document_stream_keeps_chapters_and_global_facts_once(tmp_path):
    from lazyllm.tools.writer.data_models.planning import SectionInstructionList
    task = WritingTask(query='Write a three chapter story', task_type='write',
                       constraints={'target_chars': 5000}, output={'representation': 'markdown'})
    context = WritingContext(context_id='ctx', query=task.query,
                             facts=[DocumentFact(fact_id='name', key='name', value='林舟', locked=True)])
    plan = SectionInstructionList(instructions=[SectionInstruction(
        instruction_id=str(i), content_ref=ContentRef(heading_path=['Story', title]),
        section_title=title, section_goal='Develop the story', fact_constraints=['name: 林舟'],
    ) for i, title in enumerate(['A', 'B', 'C'])])
    output = '# Story\n\n## A\n\nOpening.\n\n## B\n\nConflict.\n\n## C\n\nResolution.\n'
    tool = WriterDraftingTools(artifact_store=str(tmp_path))

    def generate(prompt, stream_output=False, **kwargs):
        assert prompt.count('林舟') == 1
        assert kwargs['max_tokens'] >= 16000
        for char in output:
            stream_output['_stream_sink']({'tag': 'text', 'delta': char})
        return output

    with patch.object(tool, '_call_llm_text', side_effect=generate) as model:
        with tool.stream_whole_document(task, context, plan, idle_timeout=2) as stream:
            visible = ''.join(stream)
            result = stream.result()
    assert model.call_count == 1
    assert visible == output
    assert Path(result['artifact_path']).read_text() == output


def test_outline_conversion_preserves_nested_descriptions_without_model(tmp_path):
    from lazyllm.tools.writer.data_models.planning import SectionInstructionList
    from lazyllm.tools.writer.utils import apply_markdown_outline_instructions
    outline = WriterDocument(document_id='outline', stage='outline', title='Story', blocks=[WriterBlock(
        node_id='a', type='heading', content='Opening', numbering={'level': 2},
        outline_description='Begin at the harbor', children=[WriterBlock(
            node_id='b', type='heading', content='Letter', numbering={'level': 3},
            outline_description='The letter conceals the real sender',
        )],
    )])
    tool = WriterPlanningTools(artifact_store=str(tmp_path))
    markdown = apply_markdown_outline_instructions(
        '# Story\n\n<a id="block-a"></a>\n## Opening\n\n<a id="block-b"></a>\n### Letter\n', outline,
    )
    for source in (outline, markdown):
        with patch.object(tool, '_call_llm_structured', side_effect=AssertionError('unexpected call')):
            result = tool.generate_section_instructions(source, WritingContext(context_id='ctx'))
        plan = load_artifact_json(result['artifact_path'], SectionInstructionList)
        assert 'Begin at the harbor' in plan.instructions[0].required_points
        assert 'The letter conceals the real sender' in plan.instructions[0].required_points
        assert plan.instructions[0].heading_structure[0].title == 'Letter'


@pytest.mark.parametrize('body', [
    '# Story\n\n## A\n\nOpening without subheadings.\n\n## B\n\nEnding.\n',
    '# Story\n\n## A\n\n### Detail\n\nOpening.\n\n## B\n\nEnding.\n',
    '# Story\n\n## A rewritten heading\n\nComplete continuous prose.\n',
    '# Story\n\nComplete prose.\n\n![Atmosphere](media-placeholder://need_id)\n',
])
def test_whole_document_accepts_outline_heading_variations(tmp_path, body):
    from lazyllm.tools.writer.data_models.planning import SectionInstructionList
    plan = SectionInstructionList(instructions=[SectionInstruction(
        instruction_id='a', content_ref=ContentRef(heading_path=['Story', 'A']),
        section_title='A', section_goal='Opening',
        heading_structure=[{'title': 'Detail', 'level': 2}], meta={'representation': 'markdown'},
    )])
    tool = WriterDraftingTools(artifact_store=str(tmp_path))

    def generate(prompt, stream_output=False, **kwargs):
        stream_output['_stream_sink']({'tag': 'text', 'delta': body})
        return body

    with patch.object(tool, '_call_llm_text', side_effect=generate):
        with tool.stream_whole_document(
            WritingTask(query='Write a story', task_type='write'), WritingContext(context_id='ctx'), plan,
        ) as stream:
            assert ''.join(stream) == body
            result = stream.result()
    assert Path(result['artifact_path']).read_text() == body


@pytest.mark.parametrize('planned, emitted, expected', [
    (['IMAGE-1'], ['need_id'], ['IMAGE-1']),
    (['IMAGE-1'], ['IMAGE-1'], ['IMAGE-1']),
    (['IMAGE-1', 'IMAGE-2'], ['IMAGE-2', 'need_id'], ['IMAGE-2', 'IMAGE-1']),
    (['IMAGE-1', 'IMAGE-2'], ['need_id'], ['need_id']),
    ([], ['need_id'], ['need_id']),
])
def test_whole_document_repairs_only_unambiguous_image_reference(tmp_path, planned, emitted, expected):
    tool = WriterDraftingTools(artifact_store=str(tmp_path))
    visuals = [{'need_id': key, 'purpose': 'Harbor atmosphere'} for key in planned]
    body = '# Story\n\n## Opening\n\nProse.\n\n' + '\n'.join(
        f'![Harbor](media-placeholder://{key})' for key in emitted
    ) + '\n\n## Ending\n\nMore prose.\n'

    def generate(prompt, stream_output=False, **kwargs):
        for key in planned:
            assert f'![Harbor atmosphere](media-placeholder://{key})' in prompt
        assert 'media-placeholder://need_id' not in prompt
        stream_output['_stream_sink']({'tag': 'text', 'delta': body})
        return body

    with patch.object(tool, '_short_document_visuals', return_value=visuals), \
            patch.object(tool, '_call_llm_text', side_effect=generate) as model:
        with tool.stream_whole_document(
            WritingTask(query='Write', task_type='write'), WritingContext(context_id='ctx'),
        ) as stream:
            list(stream)
            result = stream.result()
    actual = Path(result['artifact_path']).read_text()
    assert tool._MARKDOWN_MEDIA_PLACEHOLDER_RE.findall(actual) == expected
    assert actual.split('## Ending')[0].count('![Harbor]') == len(emitted)
    assert model.call_count == 1
