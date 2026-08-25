from pathlib import Path
from unittest.mock import patch

import pytest

from lazyllm.tools.writer.data_models import (
    ContentRef,
    DocumentFact,
    MediaAsset,
    MediaAssetLibrary,
    ShortWritingPlan,
    TargetDocument,
    VisualInstruction,
    VisualPlan,
    WriterBlock,
    WriterDocument,
    WritingContext,
    WritingTask,
)
from lazyllm.tools.writer.tools.drafting_tools import WriterDraftingTools
from lazyllm.tools.writer.tools.planning_tools import WriterPlanningTools
from lazyllm.tools.writer.utils import load_artifact_json

_TITLE = '新能源汽车降价背后的市场变化'


def _short_inputs(representation='markdown'):
    task = WritingTask(
        task_id='task-short',
        query='写一篇连续正文，不使用小标题，约700字',
        task_type='write',
        target_document=TargetDocument(title=_TITLE),
        constraints={'structure_mode': 'flat', 'target_chars': 700, 'max_chars': 800},
        output={'representation': representation},
    )
    context = WritingContext(
        context_id='ctx-short',
        facts=[DocumentFact(
            fact_id='fact-1',
            key='竞争',
            value='市场竞争加剧',
            source=['resource-1'],
        )],
    )
    plan = ShortWritingPlan(
        instruction_id='short-plan',
        content_ref=ContentRef(document_root=True),
        section_title=_TITLE,
        section_goal='解释降价原因及其对消费者的影响。',
        core_viewpoint='降价带来机会，也伴随服务和保值率风险。',
        required_points=['市场竞争加剧', '消费者购车成本下降'],
        fact_constraints=['市场竞争加剧'],
        expected_blocks=['现象切入', '原因分析', '消费者建议'],
        meta={'representation': representation, 'target_chars': 700, 'max_chars': 800},
    )
    return task, context, plan


def test_generate_short_plans(tmp_path):
    task, context, model_plan = _short_inputs()
    model_plan.content_ref = ContentRef(node_id='wrong-node')
    model_plan.section_title = '模型标题'
    model_plan.references = [{'id': 'fact-1'}, {'id': 'invented'}]
    model_plan.visual_needs = [{'visual_type': 'image', 'purpose': '不应保留'}]
    model_visuals = VisualPlan(instructions=[VisualInstruction(
        need_id='model-id',
        content_ref=ContentRef(document_root=True),
        visual_type='image',
        purpose='  展示降价原因和消费者影响  ',
        required=True,
        meta={'placement_hint': '  分析消费者机会之后  '},
    )])

    tool = WriterPlanningTools(artifact_store=str(tmp_path))
    with patch.object(tool, '_call_llm_structured', side_effect=[model_plan, model_visuals]):
        writing_result = tool.generate_short_writing_plan(task=task, context=context)
        writing_plan = load_artifact_json(writing_result['artifact_path'], ShortWritingPlan)
        visual_result = tool.generate_short_visual_plan(task, writing_plan, context)
        visual_plan = load_artifact_json(visual_result['artifact_path'], VisualPlan)

    assert writing_plan.content_ref == ContentRef(document_root=True)
    assert writing_plan.section_title == _TITLE
    assert writing_plan.references == [{'id': 'fact-1'}]
    assert writing_plan.visual_needs == []
    assert writing_plan.meta['representation'] == 'markdown'
    assert writing_plan.meta['max_chars'] == 800
    assert visual_plan.instructions == [VisualInstruction(
        need_id='visual-document-1',
        content_ref=ContentRef(document_root=True),
        visual_type='image',
        purpose='展示降价原因和消费者影响',
        required=True,
        meta={'placement_hint': '分析消费者机会之后'},
    )]


def test_generate_short_markdown_document(tmp_path):
    task, context, plan = _short_inputs()
    visual_plan = VisualPlan(instructions=[VisualInstruction(
        need_id='visual-document-1',
        content_ref=ContentRef(document_root=True),
        visual_type='image',
        purpose='展示降价原因和消费者影响',
    )])
    media = MediaAssetLibrary(
        library_id='media-short',
        assets={'asset-1': MediaAsset(
            media_asset_id='asset-1',
            asset_type='generated_image',
            source_type='image_generation',
            local_path='/tmp/short-visual.png',
        )},
        visual_need_asset_ids={'visual-document-1': ['asset-1']},
    )
    body = '## 降价原因\n市场竞争加剧。\n\n## 消费者建议\n还需关注售后服务。'

    tool = WriterDraftingTools(artifact_store=str(tmp_path))
    with patch.object(tool, '_call_llm_text', return_value=body) as mocked:
        result = tool.generate_short_document(task, plan, context, visual_plan, media)
    markdown = Path(result['artifact_path']).read_text(encoding='utf-8')

    assert markdown.startswith(f'# {_TITLE}\n\n')
    assert '\n## ' not in markdown
    assert 'media-placeholder://visual-document-1' in markdown
    assert '/tmp/short-visual.png' not in mocked.call_args.args[0]
    assert result['metadata']['extra']['structure_mode'] == 'flat'
    with patch.object(
        tool,
        '_call_llm_text',
        return_value='正文。\n\n![图片](https://example.com/invented.png)',
    ), pytest.raises(ValueError, match='Unplanned short-document image target'):
        tool.generate_short_document(task, plan, context)


@pytest.mark.parametrize('representation', ['markdown', 'ir'])
def test_stream_short_document(representation, tmp_path):
    task, context, plan = _short_inputs(representation)
    tool = WriterDraftingTools(artifact_store=str(tmp_path))
    if representation == 'markdown':
        task.constraints['max_chars'] = 3
        plan.meta['max_chars'] = 3

        def stream_text(_prompt, stream_output=False):
            sink = stream_output['_stream_sink']
            sink({'tag': 'text', 'delta': '第一段。'})
            sink({'tag': 'text', 'delta': '\n\n第二段。'})
            return '第一段。\n\n第二段。'

        with (
            patch.object(tool, '_call_llm_text', side_effect=stream_text),
            tool.stream_short_document(task, plan, context, idle_timeout=1) as stream,
        ):
            preview = ''.join(stream)
            result = stream.result()
        assert preview == f'# {_TITLE}\n\n第一段。\n\n第二段。\n'
        assert Path(result['artifact_path']).read_text(encoding='utf-8') == preview
    else:
        model_document = WriterDocument(
            document_id='model-document',
            blocks=[WriterBlock(node_id='paragraph-1', type='paragraph', content='第一段。')],
        )
        with (
            patch.object(tool, '_call_llm_structured', return_value=model_document),
            tool.stream_short_document(task, plan, context, idle_timeout=1) as stream,
        ):
            preview = ''.join(stream)
            result = stream.result()
        document = load_artifact_json(result['artifact_path'], WriterDocument)
        assert preview == f'# {_TITLE}\n\n第一段。\n'
        assert document.title == _TITLE
        assert all(block.type != 'heading' for block in document.iter_blocks())

    assert result['metadata']['extra']['representation'] == representation


@pytest.mark.parametrize(
    ('model_has_image', 'local_path', 'expected_path'),
    [
        (True, '/tmp/short-ir-visual.png', '/tmp/short-ir-visual.png'),
        (False, None, 'https://example.com/short-ir-visual.png'),
    ],
)
def test_generate_short_ir_document_with_media(model_has_image, local_path, expected_path, tmp_path):
    task, context, plan = _short_inputs('ir')
    visual_plan = VisualPlan(instructions=[VisualInstruction(
        need_id='visual-document-1',
        content_ref=ContentRef(document_root=True),
        visual_type='image',
        purpose='展示消费者购车决策因素',
        required=True,
    )])
    media = MediaAssetLibrary(
        library_id='media-short-ir',
        assets={'asset-1': MediaAsset(
            media_asset_id='asset-1',
            asset_type='generated_image',
            source_type='image_generation',
            uri='https://example.com/short-ir-visual.png',
            local_path=local_path,
            caption='购车决策因素示意图',
        )},
        visual_need_asset_ids={'visual-document-1': ['asset-1']},
    )
    blocks = [WriterBlock(node_id='paragraph-1', type='paragraph', content='消费者需要综合比较。')]
    if model_has_image:
        blocks.append(WriterBlock(node_id='visual-document-1', type='image', content='购车决策因素'))
    model_document = WriterDocument(document_id='model-document', blocks=blocks)

    tool = WriterDraftingTools(artifact_store=str(tmp_path))
    with patch.object(tool, '_call_llm_structured', return_value=model_document) as mocked:
        result = tool.generate_short_document(task, plan, context, visual_plan, media)
    document = load_artifact_json(result['artifact_path'], WriterDocument)

    image = next(block for block in document.iter_blocks() if block.type == 'image')
    assert document.ui_editable is True
    assert all(block.type != 'heading' for block in document.iter_blocks())
    assert image.references == [{'type': 'media_asset', 'id': 'asset-1', 'path': expected_path}]
    assert 'asset-1' in mocked.call_args.args[0]
    assert '购车决策因素示意图' in mocked.call_args.args[0]
    assert '/tmp/short-ir-visual.png' not in mocked.call_args.args[0]
    assert 'https://example.com/short-ir-visual.png' not in mocked.call_args.args[0]
