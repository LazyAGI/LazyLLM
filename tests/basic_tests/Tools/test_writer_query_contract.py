import json
from unittest.mock import patch

import pytest

from lazyllm.tools.writer.data_models import ContentRef, WritingContext, WritingTask
from lazyllm.tools.writer.data_models.revision import MarkdownModifyInstruction, MarkdownModifyPlan
from lazyllm.tools.writer.tools.context_tools import WriterContextTools
from lazyllm.tools.writer.tools.quality_tools import WriterQualityTools
from lazyllm.tools.writer.tools.revision_tools import WriterRevisionTools
from lazyllm.tools.writer.utils import load_artifact_json


@pytest.mark.parametrize('response', [
    '{"summary": "Model overview"}',
    '<think>Internal reasoning</think>\n{"summary": "Model overview"}', 'invalid JSON', '{}',
    '{"summary": "   "}', '{"summary": ["invalid"]}', RuntimeError('unavailable'),
])
def test_summary_kind_tracks_actual_result_and_survives_reload(tmp_path, response):
    from unittest.mock import Mock

    model = Mock(side_effect=response) if isinstance(response, Exception) else Mock(return_value=response)
    tool = WriterContextTools(llm=model, artifact_store=str(tmp_path))
    source = '# Report\n\n' + 'Preserved source text. ' * 30
    created = tool.create_writing_context(WritingTask(query='Write', task_type='write'), document=source)
    context = WritingContext.load(created['context_path'])
    expected_kind = 'summary' if isinstance(response, str) and 'Model overview' in response else 'excerpt'
    assert context.meta['document_summary_kind'] == expected_kind
    assert context.document_summary.summary == (
        'Model overview' if expected_kind == 'summary' else tool._shorten(source)
    )
    context.document_summary.key_points = ['Keep this point']
    context.meta['version'] = 3
    context.meta['document_summary_kind'] = 'excerpt' if expected_kind == 'summary' else 'summary'
    before = context.model_dump()
    updated = tool.update_writing_context(source, context)
    restored = WritingContext.load(updated['context_path'])
    assert context.model_dump() == before
    assert restored.meta['document_summary_kind'] == expected_kind
    assert 'Internal reasoning' not in json.dumps(restored.model_dump())
    assert restored.meta['version'] == 3
    assert restored.document_summary.key_points == ['Keep this point']
    assert restored.block_summaries == context.block_summaries
    assert restored.document_summary.structure_summary == context.document_summary.structure_summary
    assert restored.prompt_data()['meta']['document_summary_kind'] == expected_kind
    assert model.call_count == 2


def test_short_content_with_model_is_an_excerpt_and_noop_preserves_kind(tmp_path):
    from unittest.mock import Mock

    model = Mock(side_effect=AssertionError('short content must not call model'))
    tool = WriterContextTools(llm=model, artifact_store=str(tmp_path))
    context = WritingContext(context_id='ctx', document_summary={'summary': 'Old summary'},
                             meta={'document_summary_kind': 'summary'})
    noop = WritingContext.load(tool.update_writing_context(context=context)['context_path'])
    assert noop.model_dump() == context.model_dump()
    updated = WritingContext.load(tool.update_writing_context('# Draft\nShort text.', context)['context_path'])
    assert updated.meta['document_summary_kind'] == 'excerpt'
    assert updated.document_summary.summary == '# Draft Short text.'
    model.assert_not_called()


def test_reference_material_does_not_become_document_summary(tmp_path):
    tool = WriterContextTools(artifact_store=str(tmp_path))
    task = WritingTask(query='CURRENT_REQUEST', task_type='write')
    result = tool.create_writing_context(task, [{
        'resource_id': 'reference', 'resource_role': 'background',
        'summary': 'Reference overview', 'key_facts': ['Supported fact'],
    }])
    data = json.loads(open(result['context_path']).read())['data']
    assert 'query' not in data
    context = load_artifact_json(result['context_path'], WritingContext)
    assert context.document_summary is None
    assert {fact.value for fact in context.facts} == {'Reference overview', 'Supported fact'}
    assert all(fact.source == ['reference'] for fact in context.facts)
    assert 'CURRENT_REQUEST' not in json.dumps(data)


@pytest.mark.parametrize('scope', ['section', 'fragment'])
def test_replacement_paths_receive_current_task_once(tmp_path, scope):
    tool = WriterRevisionTools(artifact_store=str(tmp_path))
    plan = MarkdownModifyPlan(scope='document', instructions=[MarkdownModifyInstruction(
        instruction_id='edit', modify_type='update', content_ref=ContentRef(heading_path=['Draft', 'Method']),
        target_scope=scope, locator_text='Original text.' if scope == 'fragment' else None,
        instruction='Polish the target',
    )])
    task = WritingTask(query='CURRENT_REQUEST', task_type='revise', constraints={'preserve_numbers': True})
    context = WritingContext(context_id='ctx')
    with patch.object(tool, '_call_llm_structured', side_effect=RuntimeError('captured')) as model:
        with pytest.raises(RuntimeError, match='captured'):
            tool.generate_revision_set('# Draft\n\n## Method\n\nOriginal text.', plan, context, task=task)
    prompt = model.call_args.args[0]
    assert prompt.count('CURRENT_REQUEST') == 1
    assert 'preserve_numbers' in prompt


def test_ir_patch_receives_explicit_task(tmp_path):
    from lazyllm.tools.writer.data_models import WriterBlock, WriterDocument
    from lazyllm.tools.writer.data_models.revision import ModifyInstruction, ModifyPlan

    tool = WriterRevisionTools(artifact_store=str(tmp_path))
    source = WriterDocument(document_id='doc', stage='final', blocks=[
        WriterBlock(node_id='p', type='paragraph', content='Original'),
    ])
    plan = ModifyPlan(scope='document', instructions=[ModifyInstruction(
        instruction_id='edit', content_ref=ContentRef(node_id='p'), modify_type='update', instruction='Polish',
    )])
    task = WritingTask(query='CURRENT_REQUEST', task_type='revise')
    with patch.object(tool, '_call_llm_structured', side_effect=RuntimeError('captured')) as model:
        with pytest.raises(RuntimeError, match='captured'):
            tool.generate_revision_set(source, plan, WritingContext(context_id='ctx'), task=task)
    assert model.call_args.args[0].count('CURRENT_REQUEST') == 1


def test_summary_updates_do_not_accumulate_history(tmp_path):
    tool = WriterContextTools(artifact_store=str(tmp_path))
    created = tool.create_writing_context(
        WritingTask(query='Edit it', task_type='revise'), document='# Draft\n\nOriginal text.',
    )
    context = WritingContext.load(created['context_path'])
    assert 'Original text.' in context.document_summary.summary
    for index in range(2):
        result = tool.update_writing_context('# Draft\n\nCurrent text.', context)
        updated = load_artifact_json(result['context_path'], WritingContext)
        assert 'context_updates' not in updated.meta
        assert 'query' not in updated.prompt_data()
        if index:
            assert updated.model_dump() == context.model_dump()
        context = updated
    assert context.document_summary.summary == '# Draft Current text.'


def test_context_prompt_view_preserves_evidence_but_excludes_internal_metadata():
    context = WritingContext.model_validate({
        'context_id': 'INTERNAL_CONTEXT', 'doc_id': 'INTERNAL_DOCUMENT',
        'document_summary': {'summary': 'Current manuscript'},
        'facts': [{'fact_id': 'fact-1', 'key': 'evidence', 'value': 'Observed result',
                   'source': ['source-1'], 'locked': True}],
        'style_profile': {'tone': 'formal'},
        'meta': {'source': 'INTERNAL_SOURCE', 'context_update_warning': 'INTERNAL_WARNING',
                 'timestamp': 'INTERNAL_TIME', 'document_summary_kind': 'excerpt'},
    })
    original = context.model_dump()
    prompt = context.prompt_data()
    assert 'context_id' not in prompt and 'doc_id' not in prompt
    assert prompt['meta'] == {'document_summary_kind': 'excerpt'}
    assert prompt['facts'][0]['source'] == ['source-1']
    assert prompt['facts'][0]['locked'] is True
    assert prompt['style_profile']['tone'] == 'formal'
    assert context.model_dump() == original
    rendered = WriterQualityTools()._task_context_prompt(
        context, WritingTask(query='Review', task_type='write'),
    )
    assert 'INTERNAL_' not in rendered
    assert 'Observed result' in rendered
    without_summary = context.model_copy(update={'document_summary': None})
    assert 'meta' not in without_summary.prompt_data()


def test_task_prompt_excludes_tracking_but_preserves_requirements():
    from lazyllm.tools.writer.data_models import WritingTask
    task = WritingTask(task_id='tracking', query='Revise carefully', task_type='revise',
                       scope='selection', meta={'source': 'workflow'},
                       constraints={'max_chars': None, 'tone': '', 'count': 0, 'allow': False, 'exclude': []},
                       selection={'text': 'selected text', 'anchor_start': 'anchor'},
                       inputs=[{'resource_type': 'image', 'uri': 'image.png', 'meta': {'caption': 'Evidence'}}])
    data = task.prompt_data()
    assert 'task_id' not in data and 'meta' not in data
    assert data['constraints'] == {'count': 0, 'allow': False, 'exclude': []}
    assert data['selection']['anchor_start'] == 'anchor'
    assert data['inputs'][0]['meta']['caption'] == 'Evidence'
    assert task.task_id == 'tracking' and task.meta == {'source': 'workflow'}


@pytest.mark.parametrize('representation', ['markdown', 'ir'])
def test_merging_section_context_update_preserves_document_and_all_context_fields(tmp_path, representation):
    import hashlib
    from pathlib import Path
    from unittest.mock import Mock
    from lazyllm.tools.writer.data_models import WriterBlock, WriterDocument
    from lazyllm.tools.writer.tools.drafting_tools import WriterDraftingTools

    initial = WritingContext.model_validate({
        'context_id': 'ctx', 'doc_id': 'original',
        'document_summary': {'summary': 'Outline overview', 'key_points': ['Keep this point'],
                             'structure_summary': 'Original outline structure'},
        'block_summaries': [{'content_ref': {'node_id': 'old'}, 'summary': 'Old block'}],
        'facts': [{'fact_id': 'f', 'key': 'evidence', 'value': 'Keep evidence',
                   'source': ['reference'], 'locked': True}],
        'style_profile': {'tone': 'formal'},
        'meta': {'source': 'create_writing_context', 'document_summary_kind': 'summary',
                 'custom': 'keep'},
    })
    text = 'Detailed evidence. ' * 30 + 'The trial failed; do not recommend deployment.'
    if representation == 'markdown':
        outline = '# Report\n\n## Findings\n\nExplain results.'
        block = '## Findings\n\n' + text
    else:
        outline = WriterDocument(document_id='outline', stage='outline', title='Report', blocks=[
            WriterBlock(node_id='section', type='heading', content='Findings', numbering={'level': 1}),
        ])
        block = WriterBlock(
            node_id='section', type='heading', content='Findings', stage='draft', numbering={'level': 1},
            children=[WriterBlock(node_id='paragraph', type='paragraph', content=text, stage='draft')],
        )

    outputs = []
    for intermediate in [True, False]:
        directory = tmp_path / str(intermediate)
        model = Mock(side_effect=lambda prompt: json.dumps({'summary': hashlib.sha256(prompt.encode()).hexdigest()}))
        contexts = WriterContextTools(llm=model, artifact_store=str(directory))
        drafting = WriterDraftingTools(artifact_store=str(directory))
        context = initial.model_copy(deep=True)
        if intermediate:
            result = contexts.update_writing_context(block, context)
            context = WritingContext.load(result['context_path'])
        draft = drafting.generate_draft_document([block], context, outline)
        result = contexts.update_writing_context(draft['artifact_path'], context)
        final_context = WritingContext.load(result['context_path'])
        final = drafting.generate_final_document(draft['artifact_path'], final_context)
        outputs.append((final_context.model_dump(), Path(final['output_file_path']).read_text(),
                        WriterQualityTools()._task_context_prompt(final_context), model.call_args.args[0]))
        assert model.call_count == (2 if intermediate else 1)
        assert final_context.document_summary.key_points == ['Keep this point']
        assert final_context.facts == initial.facts
        assert final_context.meta['custom'] == 'keep'
    assert outputs[0] == outputs[1]


def test_naive_write_updates_outline_and_assembled_document_only(tmp_path):
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import Mock
    from lazyllm.tools.writer.workflow.naive_writer_workflow import NaiveWriterWorkflow

    summary = Mock(return_value=json.dumps({'summary': 'Preserved overview'}))
    outline = '# Report\n\n## Findings\n\n' + 'Describe evidence. ' * 30
    outline_path = tmp_path / 'outline.md'
    outline_path.write_text(outline)
    workflow = NaiveWriterWorkflow(
        artifact_store=str(tmp_path),
        context_tools=WriterContextTools(llm=summary, artifact_store=str(tmp_path)),
        resource_tools=SimpleNamespace(profile_resources=lambda **kwargs: [],
                                       write_to_document=lambda **kwargs: {'saved': True}),
        quality_tools=SimpleNamespace(validate_section=lambda **kwargs: {},
                                      validate_draft_document=lambda **kwargs: {}),
    )
    with patch.object(workflow.planning, 'generate_outline', return_value={'artifact_path': str(outline_path)}), \
            patch.object(workflow.drafting, '_call_llm_text', return_value='Evidence. ' * 50):
        result = workflow.write(WritingTask(
            query='Write', task_type='write', output={'representation': 'markdown'},
        ))
    assert summary.call_count == 2
    assert 'Describe evidence.' in summary.call_args_list[0].args[0]
    assert 'Evidence.' in summary.call_args_list[1].args[0]
    context = WritingContext.load(result['stage_results']['writing_context']['context_path'])
    assert context.document_summary.summary == 'Preserved overview'
    assert context.block_summaries
    assert 'Evidence.' in Path(result['primary_result']['output_file_path']).read_text()
