import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from lazyllm.tools.writer.data_models import (
    DraftBlock,
    DraftDocument,
    DraftSection,
    DocBlock,
    DocumentSummary,
    DocIR,
    MaterialStyle,
    OutlineNode,
    ResourceProfile,
    WritingContext,
    WritingOutput,
    WritingTask,
)
from lazyllm.tools.writer.data_models.quality import AuditIssue, AuditResult, ReviewReport
from lazyllm.tools.writer.data_models.revision import (
    ModifyInstruction,
    ModifyPlan,
    PatchHunk,
    PatchSet,
)
from lazyllm.tools.writer.data_models.task import InputResource
from lazyllm.tools.writer.data_models.writing import (
    SectionInstruction,
    SectionInstructionList,
    WritingOutline,
)
from lazyllm.tools.writer.tools.context_tools import WriterContextTools
from lazyllm.tools.writer.tools.drafting_tools import WriterDraftingTools
from lazyllm.tools.writer.tools.planning_tools import WriterPlanningTools
from lazyllm.tools.writer.tools.quality_tools import WriterQualityTools
from lazyllm.tools.writer.tools.resource_tools import WriterResourceTools
from lazyllm.tools.writer.utils import load_artifact_json


def _make_doc_adapter():
    adapter = MagicMock()
    adapter.resolve_link.return_value = {
        'provider': 'feishu',
        'object_id': 'doc-1',
        'object_type': 'docx',
        'title': '飞书文档',
        'has_child': False,
    }
    adapter.read_bytes.return_value = '第一段\n第二段'.encode('utf-8')
    adapter.get_document_id.return_value = 'doc-1'
    adapter.get_doc_blocks.return_value = [
        {'block_id': 'b1', 'block_type': 'heading', 'plain_text': '标题'},
        {'block_id': 'b2', 'block_type': 'paragraph', 'plain_text': '正文'},
    ]
    return adapter


def _call_document_to_docir(adapter, artifact_store):
    with patch(
        'lazyllm.tools.fs.client.FS._parse',
        return_value=('feishu', None, '~docx/doc-1'),
    ):
        with patch('lazyllm.tools.fs.client.FS._get_or_create_fs', return_value=adapter):
            tool = WriterResourceTools(artifact_store=artifact_store)
            return tool.document_to_docir(
                target_document={
                    'uri': 'feishu://~docx/doc-1',
                    'adapter': 'feishu',
                    'title': '飞书文档',
                    'doc_id': 'doc-1',
                }
            )


def _call_write_to_document(adapter, markdown, artifact_store):
    with patch(
        'lazyllm.tools.fs.client.FS._parse',
        return_value=('feishu', None, '/write-test.md'),
    ):
        with patch('lazyllm.tools.fs.client.FS._get_or_create_fs', return_value=adapter):
            tool = WriterResourceTools(artifact_store=artifact_store)
            return tool.write_to_document(
                content=markdown,
                target_document={'uri': 'feishu:///write-test.md', 'adapter': 'feishu'},
            )


def _make_context():
    return WritingContext(
        context_id='ctx-test-001',
        doc_id='doc-test-001',
        query='写一份关于深度学习在金融时间序列预测中的应用的学术综述报告。',
    )


def _make_passing_audit():
    return AuditResult(is_passed=True, score=100, summary='All checks passed.', issues=[])


def _make_section_instruction_list():
    return SectionInstructionList(
        instruction_set_id='iset-test-001',
        instructions=[
            SectionInstruction(
                instruction_id='si-prologue',
                outline_node_id='prologue',
                section_title='楔子 · 星辰陨落',
                section_goal='建立世界观的宏大感和宿命基调。',
                required_points=['太古星辰大帝的实力层级'],
                fact_constraints=['星辰本源=太古大帝毕生修为+灵魂印记'],
                style_constraints=['全知视角，史诗歌谣的叙述节奏'],
            )
        ],
    )


def _make_section_data():
    return {
        'section_id': 'sec-prologue',
        'outline_node_id': 'prologue',
        'instruction_id': 'si-prologue',
        'title': '楔子 · 星辰陨落',
        'blocks': [
            {
                'block_id': 'blk-pro-01',
                'content': '万古之前，九州大陆之上，有一位统御星辰的大帝。',
            }
        ],
    }


def _make_draft_document():
    return DraftDocument(
        draft_id='draft-test-001',
        title='星辰大帝',
        sections=[
            DraftSection(
                section_id='sec-prologue',
                outline_node_id='prologue',
                title='楔子 · 星辰陨落',
                blocks=[
                    DraftBlock(
                        block_id='blk-pro-01',
                        section_id='sec-prologue',
                        content='万古之前，九州大陆上，曾有一位统御星辰的大帝。',
                    )
                ],
            )
        ],
    )


def test_create_writing_context_tool_result():
    task = WritingTask(task_id='task-1', query='写一份产品方案', task_type='write')
    profiles = [
        ResourceProfile(
            resource_id='res-1',
            resource_role='background',
            summary='产品背景资料',
            key_facts=['支持私有化部署'],
            style=MaterialStyle(notes=['正式']),
        )
    ]
    doc_ir = DocIR(
        doc_id='doc-1',
        blocks=[
            DocBlock(block_id='b1', block_type='heading', text='方案背景', level=1),
            DocBlock(block_id='b2', block_type='paragraph', text='这里是背景内容。'),
        ],
    )

    with tempfile.TemporaryDirectory() as d:
        result = WriterContextTools(artifact_store=d).create_writing_context(
            task=task.model_dump(),
            resource_profiles=[profile.model_dump() for profile in profiles],
            doc_ir=doc_ir.model_dump(),
        )

        assert result['artifact_path'].endswith('writing_context.json')
        assert result['context_path'] == result['artifact_path']
        assert result['metadata']['step_name'] == 'create_writing_context'

        context = load_artifact_json(result['context_path'], WritingContext)
        assert context.context_id == 'task-1'
        assert context.doc_id == 'doc-1'
        assert context.facts[0].value == '支持私有化部署'
        assert len(context.block_summaries) == 2


def test_update_writing_context_tool_result_from_paths():
    context = WritingContext(context_id='ctx-1')
    output = WritingOutput(title='最终稿', content='这是最终输出内容。')

    with tempfile.TemporaryDirectory() as d:
        context_path = os.path.join(d, 'context.json')
        output_path = os.path.join(d, 'output.json')
        context.save(context_path)
        output.save(output_path)

        tool = WriterContextTools(artifact_store=d)
        result = tool.update_writing_context(artifacts=output_path, context=context_path)

        assert result['artifact_path'].endswith('writing_context.json')
        assert result['metadata']['step_name'] == 'update_writing_context'

        updated = load_artifact_json(result['context_path'], WritingContext)
        assert result['metadata']['step_name'] == 'update_writing_context'
        assert updated.document_summary.summary == '最终稿 这是最终输出内容。'
        assert updated.meta['context_updates'][0]['summary'] == '最终稿 这是最终输出内容。'


# ---------------------------------------------------------------------------
# _build_structure_summary
# ---------------------------------------------------------------------------

def test_structure_summary_with_headings():
    doc_ir = DocIR(
        blocks=[
            DocBlock(block_id='b1', block_type='heading', text='背景', level=1),
            DocBlock(block_id='b2', block_type='heading', text='方案', level=1),
            DocBlock(block_id='b3', block_type='paragraph', text='正文'),
        ],
    )
    result = WriterContextTools(artifact_store='/tmp/test')._build_structure_summary(doc_ir)
    assert result == '文档结构: # 背景 > # 方案'


def test_structure_summary_heading_text_empty():
    doc_ir = DocIR(
        blocks=[
            DocBlock(block_id='b1', block_type='heading', text=''),
            DocBlock(block_id='b2', block_type='paragraph', text='正文'),
        ],
    )
    assert WriterContextTools(artifact_store='/tmp/test')._build_structure_summary(doc_ir) is None


def test_structure_summary_no_headings():
    doc_ir = DocIR(
        blocks=[
            DocBlock(block_id='b1', block_type='paragraph', text='正文'),
            DocBlock(block_id='b2', block_type='table', text=''),
        ],
    )
    result = WriterContextTools(artifact_store='/tmp/test')._build_structure_summary(doc_ir)
    assert result == '由 2 个顶层块组成'


def test_structure_summary_none_doc_ir():
    assert WriterContextTools(artifact_store='/tmp/test')._build_structure_summary(None) is None


def test_structure_summary_empty_blocks():
    doc_ir = DocIR(blocks=[])
    assert WriterContextTools(artifact_store='/tmp/test')._build_structure_summary(doc_ir) is None


def test_structure_summary_two_level_headings():
    doc_ir = DocIR(
        blocks=[
            DocBlock(block_id='b1', block_type='heading', text='第一章', level=1),
            DocBlock(block_id='b2', block_type='heading', text='第一节', level=2),
        ],
    )
    result = WriterContextTools(artifact_store='/tmp/test')._build_structure_summary(doc_ir)
    assert '## 第一节' in result


def test_structure_summary_nested_headings():
    '''Headings nested inside parent block.children — DFS traversal ensures they are found.'''
    doc_ir = DocIR(
        blocks=[
            DocBlock(
                block_id='b1', block_type='heading', text='背景', level=1,
                children=[
                    DocBlock(block_id='b1-1', block_type='heading', text='子背景', level=2),
                ],
            ),
            DocBlock(block_id='b2', block_type='heading', text='方案', level=1),
        ],
    )
    result = WriterContextTools(artifact_store='/tmp/test')._build_structure_summary(doc_ir)
    assert '# 背景' in result
    assert '## 子背景' in result
    assert '# 方案' in result


# ---------------------------------------------------------------------------
# _summarize_content_data
# ---------------------------------------------------------------------------

def test_summarize_content_empty():
    with tempfile.TemporaryDirectory() as d:
        tool = WriterContextTools(artifact_store=d)
        result = tool._summarize_content_data('')
        assert result == 'No content summary available.'


def test_summarize_content_no_llm():
    with tempfile.TemporaryDirectory() as d:
        tool = WriterContextTools(artifact_store=d)
        result = tool._summarize_content_data('这是草稿内容。' * 50)
        assert len(result) <= 243  # 240 + "..."
        assert '这是草稿内容' in result


def test_summarize_content_with_llm():
    llm = MagicMock()
    llm.return_value = '{"summary": "这是 LLM 生成的语义摘要。"}'

    with tempfile.TemporaryDirectory() as d:
        tool = WriterContextTools(artifact_store=d, llm=llm)
        result = tool._summarize_content_data('这是一段很长的草稿内容。' * 50)
        assert result == '这是 LLM 生成的语义摘要。'


def test_summarize_content_llm_exception():
    llm = MagicMock()
    llm.side_effect = RuntimeError('LLM down')

    with tempfile.TemporaryDirectory() as d:
        tool = WriterContextTools(artifact_store=d, llm=llm)
        result = tool._summarize_content_data('草稿内容' * 50)
        assert '草稿内容' in result


def test_summarize_content_long_no_llm():
    long_text = 'A' * 500

    with tempfile.TemporaryDirectory() as d:
        tool = WriterContextTools(artifact_store=d)
        result = tool._summarize_content_data(long_text)
        assert len(result) <= 243
        assert result.endswith('...')


# ---------------------------------------------------------------------------
# create_writing_context 边界
# ---------------------------------------------------------------------------

def test_create_context_doc_ir_none():
    task = WritingTask(task_id='t1', query='写方案', task_type='write')
    profiles = [
        ResourceProfile(resource_id='r1', resource_role='background',
                        summary='背景资料', key_facts=['fact1'], style=MaterialStyle(notes=['正式']))
    ]

    with tempfile.TemporaryDirectory() as d:
        tool = WriterContextTools(artifact_store=d)
        result = tool.create_writing_context(
            task=task.model_dump(),
            resource_profiles=[p.model_dump() for p in profiles],
            doc_ir=None,
        )

        context = load_artifact_json(result['context_path'], WritingContext)
        assert context.document_summary.structure_summary is None
        assert context.block_summaries == []


def test_create_context_doc_ir_with_headings():
    task = WritingTask(task_id='t2', query='写报告', task_type='write')
    profiles = [
        ResourceProfile(resource_id='r1', resource_role='background',
                        summary='行业数据', key_facts=['市场增长20%'], style=None)
    ]
    doc_ir = DocIR(
        doc_id='doc-2',
        blocks=[
            DocBlock(block_id='b1', block_type='heading', text='背景分析', level=1),
            DocBlock(block_id='b2', block_type='heading', text='市场趋势', level=1),
            DocBlock(block_id='b3', block_type='paragraph', text='行业正在快速增长。'),
        ],
    )

    with tempfile.TemporaryDirectory() as d:
        tool = WriterContextTools(artifact_store=d)
        result = tool.create_writing_context(
            task=task.model_dump(),
            resource_profiles=[p.model_dump() for p in profiles],
            doc_ir=doc_ir.model_dump(),
        )

        context = load_artifact_json(result['context_path'], WritingContext)
        assert '文档结构' in context.document_summary.structure_summary
        assert '# 背景分析' in context.document_summary.structure_summary
        assert len(context.block_summaries) == 3  # 2 headings + 1 paragraph all have text
        assert context.facts[0].value == '市场增长20%'


def test_create_context_multiple_profiles():
    task = WritingTask(task_id='t3', query='写方案', task_type='write')
    profiles = [
        ResourceProfile(resource_id='r1', resource_role='spec',
                        summary='需求规格', key_facts=['私有化部署', 'SaaS'], style=MaterialStyle(notes=['技术'])),
        ResourceProfile(resource_id='r2', resource_role='background',
                        summary='市场数据', key_facts=['市场增长'], style=None),
        ResourceProfile(resource_id='r3', resource_role='example',
                        summary='范文', key_facts=[], style=MaterialStyle(notes=['正式'])),
    ]

    with tempfile.TemporaryDirectory() as d:
        tool = WriterContextTools(artifact_store=d)
        result = tool.create_writing_context(
            task=task.model_dump(),
            resource_profiles=[p.model_dump() for p in profiles],
        )

        context = load_artifact_json(result['context_path'], WritingContext)
        assert context.document_summary.key_points == ['需求规格', '市场数据', '范文']
        assert len(context.facts) == 3
        assert context.style_profile.notes == ['技术', '正式']


# ---------------------------------------------------------------------------
# update_writing_context 边界
# ---------------------------------------------------------------------------

def test_update_context_first_update():
    ctx = WritingContext(context_id='ctx-first')
    # no document_summary

    with tempfile.TemporaryDirectory() as d:
        tool = WriterContextTools(artifact_store=d)
        result = tool.update_writing_context(
            artifacts={'title': '第一章', 'content': '这是第一章的内容。', 'draft_id': 'd1'},
            context=ctx,
        )

        updated = load_artifact_json(result['context_path'], WritingContext)
        assert updated.document_summary is not None
        assert updated.document_summary.summary == '第一章 这是第一章的内容。'
        assert len(updated.meta.get('context_updates', [])) >= 1


def test_update_context_second_update():
    ctx = WritingContext(
        context_id='ctx-second',
        document_summary=DocumentSummary(summary='第一次的摘要'),
    )

    with tempfile.TemporaryDirectory() as d:
        tool = WriterContextTools(artifact_store=d)
        result = tool.update_writing_context(
            artifacts={'title': '第二章', 'content': '第二次更新的内容。'},
            context=ctx,
        )

        updated = load_artifact_json(result['context_path'], WritingContext)
        assert updated.document_summary.summary == '第二章 第二次更新的内容。'
        assert len(updated.meta.get('context_updates', [])) >= 1


def test_update_context_content_as_pydantic():
    ctx = WritingContext(context_id='ctx-pydantic')

    with tempfile.TemporaryDirectory() as d:
        tool = WriterContextTools(artifact_store=d)
        result = tool.update_writing_context(
            artifacts=WritingOutput(title='终稿', content='最终输出内容'),
            context=ctx,
        )

        updated = load_artifact_json(result['context_path'], WritingContext)
        assert updated.document_summary.summary == '终稿 最终输出内容'


def test_generate_section_instructions_drops_unavailable_source_refs():
    context = WritingContext(context_id='ctx-no-refs')
    outline = WritingOutline(
        outline_id='outline-no-refs',
        title='无引用测试',
        nodes=[
            OutlineNode(
                node_id='section-1',
                title='第一节',
                level=1,
                constraints={
                    'source_refs': ['unrelated-reference'],
                    'fact_constraints': ['未在上下文中出现的事实'],
                },
            )
        ],
    )
    llm_result = SectionInstructionList(
        instructions=[
            SectionInstruction(
                instruction_id='instruction-section-1',
                outline_node_id='section-1',
                section_title='第一节',
                section_goal='写第一节',
                source_refs=['unrelated-reference'],
                fact_constraints=['未在上下文中出现的事实'],
            )
        ]
    )

    with tempfile.TemporaryDirectory() as d:
        tool = WriterPlanningTools(artifact_store=d)
        with patch.object(tool, '_call_llm_structured', return_value=llm_result):
            result = tool.generate_section_instructions(outline=outline, context=context)

        instructions = load_artifact_json(
            result['metadata']['artifact_paths']['section_instructions'],
            SectionInstructionList,
        )
        assert instructions.instructions[0].source_refs == []
        assert instructions.instructions[0].fact_constraints == []


def test_generate_writing_output_writes_markdown_file():
    context = WritingContext(context_id='ctx-output-file', doc_id='doc-output-file')
    draft_document = DraftDocument(
        draft_id='draft-output-file',
        title='测试文档',
        sections=[
            DraftSection(
                section_id='sec-1',
                title='第一章',
                blocks=[DraftBlock(block_id='block-1', content='这是第一章正文。')],
            )
        ],
    )

    with tempfile.TemporaryDirectory() as d:
        result = WriterDraftingTools(artifact_store=d).generate_writing_output(
            draft=draft_document,
            context=context,
        )

        markdown_path = result['output_file_path']
        assert result['artifact_path'].endswith('writing_output.json')
        assert markdown_path.endswith('writing_output.md')
        assert os.path.exists(markdown_path)

        with open(markdown_path, 'r', encoding='utf-8') as fh:
            markdown = fh.read()
        assert '# 测试文档' in markdown
        assert '## 第一章' in markdown
        assert '这是第一章正文。' in markdown


def test_document_to_docir():
    pytest.importorskip('fsspec')
    adapter = _make_doc_adapter()

    with tempfile.TemporaryDirectory() as d:
        result = _call_document_to_docir(adapter, d)

        assert result['artifact_path'].endswith('doc_ir.json')
        assert result['metadata']['step_name'] == 'document_to_docir'
        assert result['metadata']['counts']['blocks'] == 2

        doc_ir = load_artifact_json(result['artifact_path'], DocIR)
        assert doc_ir.doc_id == 'doc-1'
        assert doc_ir.title == '飞书文档'
        assert doc_ir.adapter == 'feishu'
        assert doc_ir.plain_text == '第一段\n第二段'
        assert [block.block_type for block in doc_ir.blocks] == ['heading', 'paragraph']

        adapter.resolve_link.assert_called_once()
        adapter.read_bytes.assert_called_once()
        adapter.get_doc_blocks.assert_called_once()


def test_write_to_document():
    pytest.importorskip('fsspec')
    adapter = _make_doc_adapter()

    with tempfile.TemporaryDirectory() as d:
        result = _call_write_to_document(adapter, {'content': '# Hello\n\nworld'}, d)

        assert result['artifact_path'].endswith('write_result.json')
        assert result['metadata']['step_name'] == 'write_to_document'
        assert result['metadata']['extra']['adapter'] == 'feishu'

        adapter.write_file.assert_called_once()
        args = adapter.write_file.call_args[0]
        assert 'write-test' in args[0]
        assert b'Hello' in args[1]


@pytest.mark.parametrize(
    ('resource', 'expected'),
    [
        (InputResource(resource_type='text', inline_text='本产品要求支持私有化部署'), '本产品要求支持私有化部署'),
        (InputResource(resource_type='image', uri='/tmp/img.png', summary='图片摘要'), '图片摘要'),
        (InputResource(resource_type='url', uri='https://example.com', summary='网页摘要'), '网页摘要'),
        (InputResource(resource_type='kb', kb_id='kb-123', summary='知识库摘要'), '知识库摘要'),
        (InputResource(resource_type='url', uri='https://example.com'), ''),
    ],
)
def test_read_resource_content_basic_fallbacks(resource, expected):
    assert WriterResourceTools()._read_resource_content(resource) == expected


def test_profile_resources_rule_based():
    task = WritingTask(query='写方案', task_type='write')
    resource = InputResource(
        resource_type='text',
        inline_text='需求文档',
        summary='用户摘要',
        resource_id='r1',
        meta={'role': 'spec', 'template': 'structure'},
    )

    with tempfile.TemporaryDirectory() as d:
        result = WriterResourceTools(artifact_store=d).profile_resources(
            task=task.model_dump(),
            input_resources=[resource.model_dump()],
        )

        profiles = load_artifact_json(result['artifact_path'], validate_schema=False)
        assert result['metadata']['step_name'] == 'profile_resources'
        assert profiles[0]['resource_id'] == 'r1'
        assert profiles[0]['resource_role'] == 'spec'
        assert profiles[0]['template_usage'] == 'structure'
        assert profiles[0]['summary'] == '用户摘要'


def test_profile_resources_with_llm():
    task = WritingTask(query='写方案', task_type='write')
    resource = InputResource(resource_type='text', inline_text='需求文档', resource_id='r1')
    llm_result = ResourceProfile(
        resource_id='r1',
        resource_role='spec',
        template_usage='both',
        summary='LLM summary',
        key_facts=['fact1', 'fact2'],
        style=MaterialStyle(notes=['formal']),
        confidence=0.9,
        extracted_constraints={'word_limit': '5000'},
    )

    with tempfile.TemporaryDirectory() as d:
        tool = WriterResourceTools(artifact_store=d, llm=MagicMock())
        with patch.object(tool, '_call_llm_structured', return_value=llm_result):
            result = tool.profile_resources(
                task=task.model_dump(),
                input_resources=[resource.model_dump()],
            )

        profiles = load_artifact_json(result['artifact_path'], validate_schema=False)
        assert profiles[0]['summary'] == 'LLM summary'
        assert profiles[0]['key_facts'] == ['fact1', 'fact2']
        assert profiles[0]['extracted_constraints'] == {'word_limit': '5000'}


def test_validate_section_happy_path():
    with tempfile.TemporaryDirectory() as d:
        tool = WriterQualityTools(llm=MagicMock(), artifact_store=d)
        with patch.object(tool, '_call_llm_structured', return_value=_make_passing_audit()):
            result = tool.validate_section(
                draft_section=_make_section_data(),
                section_instruction=_make_section_instruction_list(),
                context=_make_context(),
            )

        report = load_artifact_json(result['artifact_path'], ReviewReport)
        assert result['metadata']['step_name'] == 'validate_section'
        assert report.result.is_passed is True
        assert report.target == 'sec-prologue'
        assert report.meta['instruction_id'] == 'si-prologue'


def test_validate_draft_document_happy_path():
    with tempfile.TemporaryDirectory() as d:
        tool = WriterQualityTools(llm=MagicMock(), artifact_store=d)
        with patch.object(tool, '_call_llm_structured', return_value=_make_passing_audit()):
            result = tool.validate_draft_document(
                draft_document=_make_draft_document(),
                context=_make_context(),
            )

        report = load_artifact_json(result['artifact_path'], ReviewReport)
        assert result['metadata']['step_name'] == 'validate_draft_document'
        assert report.result.is_passed is True
        assert report.target == 'draft-test-001'
        assert report.meta['draft_section_count'] == 1


def _make_modify_plan(target_block_ids=None, modify_type='rewrite',
                      instruction_text='Rewrite the text.'):
    if target_block_ids is None:
        target_block_ids = ['blk-pro-01']
    return ModifyPlan(
        plan_id='plan-test-001',
        scope='document',
        target_block_ids=target_block_ids,
        instructions=[
            ModifyInstruction(
                instruction_id=f'mi-{i:03d}',
                target_block_id=bid,
                modify_type=modify_type,
                instruction=instruction_text,
            )
            for i, bid in enumerate(target_block_ids)
        ],
    )


def _make_patch_set(hunks=None):
    return PatchSet(
        patch_id='patch-test-001',
        target_doc_id='doc-test-001',
        hunks=hunks or [],
    )


def _make_failing_audit():
    return AuditResult(
        is_passed=False,
        score=70,
        summary='Validation failed: 1 high-severity issue.',
        issues=[AuditIssue(
            severity='high', category='evidence',
            description='new_text contradicts locked fact.',
            suggestion='Fix the factual error.',
        )],
    )


# --- Scenario 1: Empty hunks (boundary) ---

def test_validate_patch_set_empty():
    with tempfile.TemporaryDirectory() as d:
        tool = WriterQualityTools(llm=MagicMock(), artifact_store=d)
        with patch.object(tool, '_call_llm_structured', return_value=_make_passing_audit()) as mock_llm:
            result = tool.validate_patch_set(
                patch_set=_make_patch_set(),
                modify_plan=_make_modify_plan(),
                context=_make_context(),
            )

        audit = load_artifact_json(result['artifact_path'], AuditResult)
        assert mock_llm.call_count == 0
        assert audit.is_passed is True
        assert audit.score == 100
        assert result['metadata']['counts']['total_hunks'] == 0


# --- Scenario 2: Single hunk (basic path) ---

def test_validate_patch_set_single_hunk():
    with tempfile.TemporaryDirectory() as d:
        tool = WriterQualityTools(llm=MagicMock(), artifact_store=d)
        with patch.object(tool, '_call_llm_structured', return_value=_make_passing_audit()) as mock_llm:
            result = tool.validate_patch_set(
                patch_set=_make_patch_set(hunks=[
                    PatchHunk(hunk_id='h1', target_block_id='blk-pro-01',
                              old_text='万古之前...', new_text='太古之初...',
                              modify_type='rewrite'),
                ]),
                modify_plan=_make_modify_plan(),
                context=_make_context(),
            )

        audit = load_artifact_json(result['artifact_path'], AuditResult)
        assert mock_llm.call_count == 1
        assert audit.is_passed is True
        assert result['metadata']['counts']['total_hunks'] == 1


# --- Scenario 3: Multiple hunks, single LLM call ---

def test_validate_patch_set_multi_hunk():
    with tempfile.TemporaryDirectory() as d:
        tool = WriterQualityTools(llm=MagicMock(), artifact_store=d)
        with patch.object(tool, '_call_llm_structured', return_value=_make_passing_audit()) as mock_llm:
            result = tool.validate_patch_set(
                patch_set=_make_patch_set(hunks=[
                    PatchHunk(hunk_id='h1', target_block_id='blk-pro-01',
                              old_text='万古之前...', new_text='太古之初...',
                              modify_type='rewrite'),
                    PatchHunk(hunk_id='h2', target_block_id='blk-pro-02',
                              old_text='second...', new_text='rewrite...',
                              modify_type='rewrite'),
                ]),
                modify_plan=_make_modify_plan(target_block_ids=['blk-pro-01', 'blk-pro-02']),
                context=_make_context(),
            )

        audit = load_artifact_json(result['artifact_path'], AuditResult)
        assert mock_llm.call_count == 1
        assert result['metadata']['counts']['total_hunks'] == 2


# --- Scenario 4: Failing validation ---

def test_validate_patch_set_failing():
    with tempfile.TemporaryDirectory() as d:
        tool = WriterQualityTools(llm=MagicMock(), artifact_store=d)
        with patch.object(tool, '_call_llm_structured', return_value=_make_failing_audit()) as mock_llm:
            result = tool.validate_patch_set(
                patch_set=_make_patch_set(hunks=[
                    PatchHunk(hunk_id='h1', target_block_id='blk-pro-01',
                              old_text='万古之前...', new_text='星辰大帝是九州最强者。',
                              modify_type='rewrite'),
                ]),
                modify_plan=_make_modify_plan(),
                context=_make_context(),
            )

        audit = load_artifact_json(result['artifact_path'], AuditResult)
        assert mock_llm.call_count == 1
        assert audit.is_passed is False
        assert audit.score == 70
        assert len(audit.issues) == 1


# --- Scenario 5: Hunk without matching ModifyInstruction ---

def test_validate_patch_set_unmatched_instruction():
    with tempfile.TemporaryDirectory() as d:
        tool = WriterQualityTools(llm=MagicMock(), artifact_store=d)
        with patch.object(tool, '_call_llm_structured', return_value=_make_passing_audit()) as mock_llm:
            result = tool.validate_patch_set(
                patch_set=_make_patch_set(hunks=[
                    PatchHunk(hunk_id='h1', target_block_id='unknown-id',
                              old_text='xxx', new_text='yyy',
                              modify_type='rewrite'),
                ]),
                modify_plan=_make_modify_plan(),  # plan targets blk-pro-01, not unknown-id
                context=_make_context(),
            )

        audit = load_artifact_json(result['artifact_path'], AuditResult)
        assert mock_llm.call_count == 1
        # modify_type/instruction fall back to 'unknown'/''
        assert audit.is_passed is True


# ---------------------------------------------------------------------------
# write_to_document boundary tests
# ---------------------------------------------------------------------------


def test_write_to_document_no_target():
    'target_document=None → locator empty → logs warning, no local file saved.'
    with tempfile.TemporaryDirectory() as d:
        tool = WriterResourceTools(artifact_store=d)
        result = tool.write_to_document(
            content={'content': '# Local output'},
            target_document=None,
        )

        assert result['metadata']['step_name'] == 'write_to_document'
        assert result['metadata']['extra']['document_id'] == ''
        assert result['metadata']['extra']['adapter'] == ''
        assert 'markdown' not in result['metadata'].get('artifact_paths', {})


def test_write_to_document_fs_failure():
    'fs.write_file raises → doc_id empty, no local file saved.'
    adapter = _make_doc_adapter()
    adapter.write_file.side_effect = RuntimeError('network down')
    adapter.resolve_link.side_effect = RuntimeError('network down')

    with tempfile.TemporaryDirectory() as d:
        with patch('lazyllm.tools.fs.client.FS._parse', return_value=('feishu', None, '/fail.md')):
            with patch('lazyllm.tools.fs.client.FS._get_or_create_fs', return_value=adapter):
                tool = WriterResourceTools(artifact_store=d)
                result = tool.write_to_document(
                    content={'content': '# Should survive'},
                    target_document={'uri': 'feishu:///fail.md', 'adapter': 'feishu'},
                )

        assert result['metadata']['extra']['document_id'] == ''
        assert result['metadata']['extra']['adapter'] == 'feishu'
        assert 'markdown' not in result['metadata'].get('artifact_paths', {})


def test_write_to_document_empty_target_dict():
    'target_document={} → locator empty, same as None.'
    with tempfile.TemporaryDirectory() as d:
        tool = WriterResourceTools(artifact_store=d)
        result = tool.write_to_document(
            content={'content': '# Empty target'},
            target_document={},
        )

        assert result['metadata']['extra']['document_id'] == ''
        assert 'markdown' not in result['metadata'].get('artifact_paths', {})
