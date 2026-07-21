import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from lazyllm.tools.writer.data_models import (
    DocumentSummary,
    MaterialStyle,
    ResourceProfile,
    WriterAuthoring,
    WriterBlock,
    WriterConstraints,
    WriterDocument,
    WritingContext,
    WritingTask,
)
from lazyllm.tools.writer.data_models.quality import AuditIssue, AuditResult, ReviewReport
from lazyllm.tools.writer.data_models.revision import (
    PatchHunk,
    PatchSet,
)
from lazyllm.tools.writer.data_models.task import InputResource
from lazyllm.tools.writer.data_models.writing import (
    SectionInstruction,
    SectionInstructionList,
)
from lazyllm.tools.writer.tools.context_tools import WriterContextTools
from lazyllm.tools.writer.tools.drafting_tools import WriterDraftingTools
from lazyllm.tools.writer.tools.planning_tools import WriterPlanningTools
from lazyllm.tools.writer.tools.quality_tools import WriterQualityTools
from lazyllm.tools.writer.tools.resource_tools import WriterResourceTools
from lazyllm.tools.writer.utils import load_artifact_json, save_artifact_json


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
        {'block_id': 'b1', 'block_type': 'heading', 'plain_text': '标题', 'level': 1},
        {'block_id': 'b2', 'block_type': 'paragraph', 'plain_text': '正文'},
    ]
    return adapter


def _call_document_to_docir(adapter, artifact_store, stage=None):
    target_document = {
        'uri': 'feishu://~docx/doc-1',
        'adapter': 'feishu',
        'title': '飞书文档',
        'doc_id': 'doc-1',
    }
    if stage is not None:
        target_document['meta'] = {'stage': stage}

    with patch(
        'lazyllm.tools.fs.client.FS._parse',
        return_value=('feishu', None, '~docx/doc-1'),
    ):
        with patch('lazyllm.tools.fs.client.FS._get_or_create_fs', return_value=adapter):
            tool = WriterResourceTools(artifact_store=artifact_store)
            return tool.document_to_docir(target_document=target_document)


def _make_final_writer_document(content='# Local output', title=''):
    return WriterDocument(
        document_id='final-output',
        stage='final',
        title=title,
        blocks=[
            WriterBlock(
                node_id='final-output-p1',
                type='paragraph',
                content=content,
                stage='final',
            ),
        ],
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


def _make_quality_draft_block():
    return WriterBlock(
        node_id='sec-prologue',
        type='heading',
        content='楔子 · 星辰陨落',
        stage='draft',
        authoring=WriterAuthoring(
            instruction_id='si-prologue',
            origin_node_id='prologue',
        ),
        children=[
            WriterBlock(
                node_id='blk-pro-01',
                type='paragraph',
                content='万古之前，九州大陆之上，有一位统御星辰的大帝。',
                stage='draft',
            ),
        ],
    )


def test_update_writing_context_tool_result_from_paths():
    context = WritingContext(context_id='ctx-1')
    output = WriterDocument(
        document_id='doc-final',
        stage='final',
        title='最终稿',
        blocks=[
            WriterBlock(
                node_id='final-1',
                type='paragraph',
                content='这是最终输出内容。',
                stage='final',
            ),
        ],
    )

    with tempfile.TemporaryDirectory() as d:
        context_path = os.path.join(d, 'context.json')
        output_path = os.path.join(d, 'output.json')
        context.save(context_path)
        save_artifact_json(output, output_path)

        tool = WriterContextTools(artifact_store=d)
        result = tool.update_writing_context(artifacts=output_path, context=context_path)

        assert result['artifact_path'].endswith('writing_context.json')

        updated = load_artifact_json(result['context_path'], WritingContext)
        assert result['metadata']['step_name'] == 'update_writing_context'
        assert updated.document_summary.summary == '最终稿 这是最终输出内容。'
        assert updated.meta['context_updates'][0]['summary'] == '最终稿 这是最终输出内容。'
        assert updated.outline is None
        assert updated.draft_document is None


# ---------------------------------------------------------------------------
# _build_structure_summary
# ---------------------------------------------------------------------------

def test_structure_summary_with_headings():
    writer_ir = WriterDocument(
        document_id='doc-structure',
        stage='final',
        blocks=[
            WriterBlock(
                node_id='b1', type='heading', content='背景',
                stage='final', numbering={'level': 1},
            ),
            WriterBlock(
                node_id='b2', type='heading', content='方案',
                stage='final', numbering={'level': 1},
            ),
            WriterBlock(node_id='b3', type='paragraph', content='正文', stage='final'),
        ],
    )
    result = WriterContextTools(artifact_store='/tmp/test')._build_structure_summary(writer_ir)
    assert result == '文档结构: # 背景 > # 方案'


@pytest.mark.parametrize(
    'blocks',
    [
        [
            WriterBlock(node_id='b1', type='heading', content='', stage='final'),
            WriterBlock(node_id='b2', type='paragraph', content='正文', stage='final'),
        ],
        [
            WriterBlock(node_id='b1', type='paragraph', content='正文', stage='final'),
            WriterBlock(node_id='b2', type='table', content='', stage='final'),
        ],
    ],
    ids=['empty_heading', 'no_headings'],
)
def test_structure_summary_without_headings(blocks):
    writer_ir = WriterDocument(document_id='doc-no-headings', stage='final', blocks=blocks)
    result = WriterContextTools(artifact_store='/tmp/test')._build_structure_summary(writer_ir)
    assert result == '由 2 个顶层块组成'


def test_structure_summary_none_doc_ir():
    assert WriterContextTools(artifact_store='/tmp/test')._build_structure_summary(None) is None


def test_structure_summary_empty_blocks():
    writer_ir = WriterDocument(document_id='doc-empty', stage='final')
    assert WriterContextTools(artifact_store='/tmp/test')._build_structure_summary(writer_ir) is None


def test_structure_summary_two_level_headings():
    writer_ir = WriterDocument(
        document_id='doc-two-levels',
        stage='outline',
        blocks=[
            WriterBlock(
                node_id='b1', type='heading', content='第一章',
                stage='outline', numbering={'level': 1},
            ),
            WriterBlock(
                node_id='b2', type='heading', content='第一节',
                stage='outline', numbering={'level': 2},
            ),
        ],
    )
    result = WriterContextTools(artifact_store='/tmp/test')._build_structure_summary(writer_ir)
    assert '## 第一节' in result


def test_structure_summary_nested_headings():
    '''Headings nested inside parent block.children — DFS traversal ensures they are found.'''
    writer_ir = WriterDocument(
        document_id='doc-nested',
        stage='outline',
        blocks=[
            WriterBlock(
                node_id='b1', type='heading', content='背景',
                stage='outline', numbering={'level': 1},
                children=[
                    WriterBlock(
                        node_id='b1-1', type='heading', content='子背景',
                        stage='outline', numbering={'level': 2},
                    ),
                ],
            ),
            WriterBlock(
                node_id='b2', type='heading', content='方案',
                stage='outline', numbering={'level': 1},
            ),
        ],
    )
    result = WriterContextTools(artifact_store='/tmp/test')._build_structure_summary(writer_ir)
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
        assert result.endswith('...')
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


# ---------------------------------------------------------------------------
# create_writing_context 边界
# ---------------------------------------------------------------------------

def test_create_context_writer_ir_none():
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
            document=None,
        )

        context = load_artifact_json(result['context_path'], WritingContext)
        assert context.document_summary.structure_summary is None
        assert context.block_summaries == []


def test_create_writing_context_tool_result():
    task = WritingTask(task_id='t2', query='写报告', task_type='write')
    profiles = [
        ResourceProfile(resource_id='r1', resource_role='background',
                        summary='行业数据', key_facts=['市场增长20%'], style=None)
    ]
    writer_ir = WriterDocument(
        document_id='doc-2',
        stage='final',
        blocks=[
            WriterBlock(
                node_id='b1', type='heading', content='背景分析',
                stage='final', numbering={'level': 1},
            ),
            WriterBlock(
                node_id='b2', type='heading', content='市场趋势',
                stage='final', numbering={'level': 1},
            ),
            WriterBlock(
                node_id='b3', type='paragraph', content='行业正在快速增长。',
                stage='final',
            ),
        ],
    )

    with tempfile.TemporaryDirectory() as d:
        tool = WriterContextTools(artifact_store=d)
        result = tool.create_writing_context(
            task=task.model_dump(),
            resource_profiles=[p.model_dump() for p in profiles],
            document=writer_ir.model_dump(),
        )

        context = load_artifact_json(result['context_path'], WritingContext)
        assert result['artifact_path'].endswith('writing_context.json')
        assert result['context_path'] == result['artifact_path']
        assert result['metadata']['step_name'] == 'create_writing_context'
        assert context.context_id == 't2'
        assert context.doc_id == 'doc-2'
        assert '文档结构' in context.document_summary.structure_summary
        assert '# 背景分析' in context.document_summary.structure_summary
        assert len(context.block_summaries) == 3  # 2 headings + 1 paragraph all have text
        assert context.facts[0].value == '市场增长20%'
        assert context.outline is None
        assert context.draft_document is None


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
    writer_ir = WriterDocument(
        document_id='d1',
        stage='draft',
        title='第一章',
        blocks=[
            WriterBlock(
                node_id='d1-p1',
                type='paragraph',
                content='这是第一章的内容。',
                stage='draft',
            ),
        ],
    )

    with tempfile.TemporaryDirectory() as d:
        tool = WriterContextTools(artifact_store=d)
        result = tool.update_writing_context(
            artifacts=writer_ir,
            context=ctx,
        )

        updated = load_artifact_json(result['context_path'], WritingContext)
        assert updated.document_summary is not None
        assert updated.document_summary.summary == '第一章 这是第一章的内容。'
        assert len(updated.meta.get('context_updates', [])) >= 1
        assert updated.draft_document == writer_ir


def test_update_context_second_update():
    ctx = WritingContext(
        context_id='ctx-second',
        document_summary=DocumentSummary(summary='第一次的摘要'),
    )
    writer_ir = WriterDocument(
        document_id='d2',
        stage='draft',
        title='第二章',
        blocks=[
            WriterBlock(
                node_id='d2-p1',
                type='paragraph',
                content='第二次更新的内容。',
                stage='draft',
            ),
        ],
    )

    with tempfile.TemporaryDirectory() as d:
        tool = WriterContextTools(artifact_store=d)
        result = tool.update_writing_context(
            artifacts=writer_ir,
            context=ctx,
        )

        updated = load_artifact_json(result['context_path'], WritingContext)
        assert updated.document_summary.summary == '第二章 第二次更新的内容。'
        assert len(updated.meta.get('context_updates', [])) >= 1


def test_update_context_writer_ir_as_pydantic():
    ctx = WritingContext(context_id='ctx-pydantic')
    writer_ir = WriterDocument(
        document_id='final-document',
        stage='final',
        title='终稿',
        blocks=[
            WriterBlock(
                node_id='final-p1',
                type='paragraph',
                content='最终输出内容',
                stage='final',
            ),
        ],
    )

    with tempfile.TemporaryDirectory() as d:
        tool = WriterContextTools(artifact_store=d)
        result = tool.update_writing_context(
            artifacts=writer_ir,
            context=ctx,
        )

        updated = load_artifact_json(result['context_path'], WritingContext)
        assert updated.document_summary.summary == '终稿 最终输出内容'
        assert updated.draft_document is None


def test_update_context_routes_outline_writer_document():
    ctx = WritingContext(context_id='ctx-outline')
    outline = WriterDocument(
        document_id='outline-document',
        stage='outline',
        title='文档大纲',
        blocks=[
            WriterBlock(
                node_id='outline-1',
                type='heading',
                content='第一章',
                stage='outline',
                numbering={'level': 1},
            ),
        ],
    )

    with tempfile.TemporaryDirectory() as d:
        result = WriterContextTools(artifact_store=d).update_writing_context(
            artifacts=outline,
            context=ctx,
        )
        updated = load_artifact_json(result['context_path'], WritingContext)

    assert updated.outline == outline
    assert updated.draft_sections == []
    assert updated.draft_document is None
    assert updated.document_summary is None


def test_update_context_routes_draft_writer_block_from_artifact_path():
    ctx = WritingContext(context_id='ctx-draft-block')
    draft_section = WriterBlock(
        node_id='section-1',
        type='heading',
        content='第一章',
        stage='draft',
        children=[
            WriterBlock(
                node_id='section-1-p1',
                type='paragraph',
                content='章节正文。',
                stage='draft',
            ),
        ],
    )

    with tempfile.TemporaryDirectory() as d:
        block_path = os.path.join(d, 'draft_section.json')
        save_artifact_json(draft_section, block_path)
        result = WriterContextTools(artifact_store=d).update_writing_context(
            artifacts=block_path,
            context=ctx,
        )
        updated = load_artifact_json(result['context_path'], WritingContext)

    assert updated.draft_sections == [draft_section]
    assert updated.document_summary.summary == '第一章 章节正文。'
    assert updated.meta['context_updates'][0]['content_kind'] == 'WriterBlock:draft'


def test_generate_section_instructions_preserves_outline_references():
    context = WritingContext(context_id='ctx-no-refs')
    references = [{'id': 'resource-1', 'url': 'https://example.com/source'}]
    outline = WriterDocument(
        document_id='outline-no-refs',
        stage='outline',
        title='无引用测试',
        blocks=[
            WriterBlock(
                node_id='section-1',
                type='heading',
                content='第一节',
                stage='outline',
                references=references,
                authoring=WriterAuthoring(
                    constraints=WriterConstraints(
                        fact_constraints=['未在上下文中出现的事实'],
                    ),
                ),
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
                references=[{'id': 'llm-invented-reference'}],
                fact_constraints=['未在上下文中出现的事实'],
            )
        ],
    )
    original_outline = outline.model_copy(deep=True)

    with tempfile.TemporaryDirectory() as d:
        tool = WriterPlanningTools(artifact_store=d)
        with patch.object(tool, '_call_llm_structured', return_value=llm_result):
            result = tool.generate_section_instructions(outline=outline, context=context)

        instructions = load_artifact_json(
            result['metadata']['artifact_paths']['section_instructions'],
            SectionInstructionList,
        )
        instruction = instructions.instructions[0]
        assert instruction.references == references
        assert instruction.fact_constraints == []
        assert outline == original_outline


def test_generate_final_document_writes_markdown_file():
    context = WritingContext(context_id='ctx-output-file', doc_id='doc-output-file')
    draft_document = WriterDocument(
        document_id='draft-output-file',
        stage='draft',
        title='测试文档',
        blocks=[
            WriterBlock(
                node_id='sec-1',
                type='heading',
                content='第一章',
                stage='draft',
                references=[{'id': 'resource-1', 'url': 'https://example.com/source'}],
                children=[
                    WriterBlock(
                        node_id='block-1',
                        type='paragraph',
                        content='这是第一章正文。',
                        stage='draft',
                    ),
                ],
            ),
        ],
    )

    with tempfile.TemporaryDirectory() as d:
        result = WriterDraftingTools(artifact_store=d).generate_final_document(
            draft=draft_document,
            context=context,
        )

        markdown_path = result['output_file_path']
        assert result['artifact_path'].endswith('final_document.json')
        assert markdown_path.endswith('writing_output.md')
        assert os.path.exists(markdown_path)

        final_document = load_artifact_json(result['artifact_path'], WriterDocument)
        assert final_document.blocks[0].references == [
            {'id': 'resource-1', 'url': 'https://example.com/source'},
        ]
        assert final_document.provider_binding == {}

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

        assert result['artifact_path'].endswith('document.json')
        assert result['metadata']['step_name'] == 'document_to_docir'
        assert result['metadata']['counts']['blocks'] == 2

        document = load_artifact_json(result['artifact_path'], WriterDocument)
        assert document.document_id == 'doc-1'
        assert document.stage == 'final'
        assert document.title == '飞书文档'
        assert document.provider_binding['provider'] == 'feishu'
        assert [block.type for block in document.blocks] == ['heading', 'paragraph']
        assert [block.content for block in document.blocks] == ['标题', '正文']
        assert [block.node_id for block in document.blocks] == ['b1', 'b2']
        assert document.blocks[0].numbering == {'level': 1}

        adapter.resolve_link.assert_called_once()
        adapter.read_bytes.assert_called_once()
        adapter.get_doc_blocks.assert_called_once()


def test_document_to_docir_rejects_invalid_stage():
    pytest.importorskip('fsspec')
    adapter = _make_doc_adapter()

    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError, match='must be a valid WriterStage'):
            _call_document_to_docir(adapter, d, stage='published')


def test_write_to_document():
    pytest.importorskip('fsspec')
    adapter = _make_doc_adapter()

    with tempfile.TemporaryDirectory() as d:
        with patch(
            'lazyllm.tools.fs.client.FS._parse',
            return_value=('feishu', None, '/write-test.md'),
        ):
            with patch('lazyllm.tools.fs.client.FS._get_or_create_fs', return_value=adapter):
                result = WriterResourceTools(artifact_store=d).write_to_document(
                    content=_make_final_writer_document(content='world', title='Hello'),
                    target_document={'uri': 'feishu:///write-test.md', 'adapter': 'feishu'},
                )

        assert result['artifact_path'].endswith('write_result.json')
        assert result['metadata']['step_name'] == 'write_to_document'
        assert result['metadata']['extra']['adapter'] == 'feishu'

        adapter.write_file.assert_called_once()
        args = adapter.write_file.call_args[0]
        assert 'write-test' in args[0]
        assert b'Hello' in args[1]


def test_render_writer_document_markdown():
    document = WriterDocument(
        document_id='render-test',
        stage='final',
        title='测试文档',
        blocks=[
            WriterBlock(
                node_id='heading-1',
                type='heading',
                content='章节',
                stage='final',
                numbering={'level': 2},
            ),
            WriterBlock(
                node_id='list-1',
                type='list_item',
                content='第一项',
                stage='final',
                numbering={'ordered': True},
            ),
            WriterBlock(
                node_id='code-1',
                type='code',
                content='print("hello")',
                stage='final',
                provider_payload={'language': 'python'},
            ),
        ],
    )

    markdown = WriterResourceTools()._render_document_markdown(document)
    assert '# 测试文档' in markdown
    assert '## 章节' in markdown
    assert '1. 第一项' in markdown
    assert '```python\nprint("hello")\n```' in markdown


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
                draft_block=_make_quality_draft_block(),
                section_instruction=_make_section_instruction_list(),
                context=_make_context(),
            )

        report = load_artifact_json(result['artifact_path'], ReviewReport)
        assert result['metadata']['step_name'] == 'validate_section'
        assert report.result.is_passed is True
        assert report.target == 'sec-prologue'
        assert report.meta['instruction_id'] == 'si-prologue'
        assert report.meta['outline_node_id'] == 'prologue'


def test_validate_draft_document_happy_path():
    draft_document = WriterDocument(
        document_id='draft-test-001',
        stage='draft',
        title='星辰大帝',
        blocks=[_make_quality_draft_block()],
    )

    with tempfile.TemporaryDirectory() as d:
        tool = WriterQualityTools(llm=MagicMock(), artifact_store=d)
        with patch.object(tool, '_call_llm_structured', return_value=_make_passing_audit()):
            result = tool.validate_draft_document(
                draft_document=draft_document,
                context=_make_context(),
            )

        report = load_artifact_json(result['artifact_path'], ReviewReport)
        assert result['metadata']['step_name'] == 'validate_draft_document'
        assert report.result.is_passed is True
        assert report.target == 'draft-test-001'
        assert report.meta['draft_block_count'] == 2


def _make_patch_set(hunks=None):
    return PatchSet(
        patch_id='patch-test-001',
        target_doc_id='doc-test-001',
        hunks=hunks or [],
    )


def _make_task(query='Revise the document.'):
    return WritingTask(task_id='test-task', query=query, task_type='revise')


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
                context=_make_context(),
                task=_make_task(),
            )

        audit = load_artifact_json(result['artifact_path'], AuditResult)
        assert mock_llm.call_count == 1
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
                    PatchHunk(hunk_id='h1', target_node_id='blk-pro-01',
                              old_text='万古之前...', new_text='太古之初...',
                              modify_type='replace'),
                ]),
                context=_make_context(),
                task=_make_task(),
            )

        audit = load_artifact_json(result['artifact_path'], AuditResult)
        assert mock_llm.call_count == 1
        assert audit.is_passed is True
        assert result['metadata']['counts']['total_hunks'] == 1
        prompt = mock_llm.call_args.args[0]
        assert '"target_node_id": "blk-pro-01"' in prompt
        assert 'target_block_id' not in prompt


# --- Scenario 3: Multiple hunks, single LLM call ---

def test_validate_patch_set_multi_hunk():
    with tempfile.TemporaryDirectory() as d:
        tool = WriterQualityTools(llm=MagicMock(), artifact_store=d)
        with patch.object(tool, '_call_llm_structured', return_value=_make_passing_audit()) as mock_llm:
            result = tool.validate_patch_set(
                patch_set=_make_patch_set(hunks=[
                    PatchHunk(hunk_id='h1', target_node_id='blk-pro-01',
                              old_text='万古之前...', new_text='太古之初...',
                              modify_type='replace'),
                    PatchHunk(hunk_id='h2', target_node_id='blk-pro-02',
                              old_text='second...', new_text='rewrite...',
                              modify_type='replace'),
                ]),
                context=_make_context(),
                task=_make_task(),
            )

        audit = load_artifact_json(result['artifact_path'], AuditResult)
        assert mock_llm.call_count == 1
        assert audit.is_passed is True
        assert result['metadata']['counts']['total_hunks'] == 2


# --- Scenario 4: Failing validation ---

def test_validate_patch_set_failing():
    with tempfile.TemporaryDirectory() as d:
        tool = WriterQualityTools(llm=MagicMock(), artifact_store=d)
        with patch.object(tool, '_call_llm_structured', return_value=_make_failing_audit()) as mock_llm:
            result = tool.validate_patch_set(
                patch_set=_make_patch_set(hunks=[
                    PatchHunk(hunk_id='h1', target_node_id='blk-pro-01',
                              old_text='万古之前...', new_text='星辰大帝是九州最强者。',
                              modify_type='replace'),
                ]),
                context=_make_context(),
                task=_make_task(),
            )

        audit = load_artifact_json(result['artifact_path'], AuditResult)
        assert mock_llm.call_count == 1
        assert audit.is_passed is False
        assert audit.score == 70
        assert len(audit.issues) == 1


# --- Scenario 5: Hunk without matching ModifyInstruction ---

# ---------------------------------------------------------------------------
# write_to_document boundary tests
# ---------------------------------------------------------------------------


def test_write_to_document_rejects_non_final_writer_document():
    document = WriterDocument(document_id='draft-output', stage='draft')

    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError, match='stage="final"'):
            WriterResourceTools(artifact_store=d).write_to_document(
                content=document,
                target_document=None,
            )


@pytest.mark.parametrize('target_document', [None, {}], ids=['none', 'empty_dict'])
def test_write_to_document_no_target(target_document):
    # Both empty target forms normalize to an empty TargetDocument.
    with tempfile.TemporaryDirectory() as d:
        tool = WriterResourceTools(artifact_store=d)
        result = tool.write_to_document(
            content=_make_final_writer_document(),
            target_document=target_document,
        )

        assert result['metadata']['step_name'] == 'write_to_document'
        assert result['metadata']['extra']['document_id'] == ''
        assert result['metadata']['extra']['adapter'] == ''


def test_write_to_document_fs_failure():
    # fs.write_file raises -> doc_id empty, no local file saved.
    pytest.importorskip('fsspec')
    adapter = _make_doc_adapter()
    adapter.write_file.side_effect = RuntimeError('network down')

    with tempfile.TemporaryDirectory() as d:
        with patch('lazyllm.tools.fs.client.FS._parse', return_value=('feishu', None, '/fail.md')):
            with patch('lazyllm.tools.fs.client.FS._get_or_create_fs', return_value=adapter):
                tool = WriterResourceTools(artifact_store=d)
                result = tool.write_to_document(
                    content=_make_final_writer_document('# Should survive'),
                    target_document={'uri': 'feishu:///fail.md', 'adapter': 'feishu'},
                )

        assert result['metadata']['extra']['document_id'] == ''
        assert result['metadata']['extra']['adapter'] == 'feishu'
