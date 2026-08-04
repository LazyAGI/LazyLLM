from contextlib import contextmanager
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

import lazyllm
from lazyllm.tools.writer.adapter.feishu import FeishuWriterAdapter
from lazyllm.tools.writer.tools.base import WriterToolBase
from lazyllm.tools.writer.data_models.context import DocumentSummary, WritingContext
from lazyllm.tools.writer.data_models.quality import AuditResult, ReviewReport
from lazyllm.tools.writer.data_models.revision import LocateResult, ModifyPlan, PatchResult, PatchSet
from lazyllm.tools.writer.data_models.task import InputResource, Selection, TargetDocument, WritingTask
from lazyllm.tools.writer.data_models.writer_ir import ContentRef, WriterBlock, WriterDocument
from lazyllm.tools.writer.data_models.planning import SectionInstructionList
from lazyllm.tools.writer.workflow.naive_writer_workflow import NaiveWriterWorkflow
from lazyllm.tools.writer.utils import load_artifact_json, parse_markdown_sections
from ...utils import get_api_key, get_path


BASE_PATH = 'lazyllm/module/llms/onlinemodule/base/onlineChatModuleBase.py'
WRITER_BASE_PATH = 'lazyllm/tools/writer/tools/base.py'
QWEN_MODEL = 'qwen-turbo'


class WriterStructuredProbe(BaseModel):
    title: str
    section_count: int
    keywords: List[str]


@pytest.mark.ignore_cache_on_change(BASE_PATH, get_path('qwen'), WRITER_BASE_PATH)
def test_writer_call_llm_structured_with_qwen():
    llm = lazyllm.OnlineChatModule(
        source='qwen',
        model=QWEN_MODEL,
        api_key=get_api_key('qwen'),
        stream=False,
    )
    tool = WriterToolBase(llm=llm)

    result = tool._call_llm_structured(
        (
            'Generate a compact JSON object for testing WriterToolBase structured LLM output. '
            'Use title \'Writer Pipeline Structured Output Test\', section_count 3, '
            'and include the keywords planning, drafting, and review.'
        ),
        WriterStructuredProbe,
    )

    assert isinstance(result, WriterStructuredProbe)
    assert result.title
    assert result.section_count == 3
    assert {'planning', 'drafting', 'review'}.issubset(set(result.keywords))


# ============================================================================
# NaiveWriterWorkflow.write() E2E
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _stage_path(stages: dict, key: str) -> str:
    entry = stages.get(key) or {}
    if not isinstance(entry, dict):
        return ''
    return (
        entry.get('metadata', {}).get('artifact_paths', {}).get(key)
        or entry.get('artifact_path', '')
    )


def _load_stage(stages: dict, key: str, model_class=None):
    path = _stage_path(stages, key)
    if not path:
        return None
    return load_artifact_json(path, model_class)


@contextmanager
def _mock_document_write(wf, target_document, document_id):
    write_fs = MagicMock()
    with patch.object(
        wf.resource, '_resolve_document_target',
        return_value=(
            'feishu', f'/{document_id}.md', write_fs, FeishuWriterAdapter(),
            target_document.uri, document_id,
        ),
    ):
        yield write_fs


def test_write_workflow_e2e():
    '''Run NaiveWriterWorkflow.write() end-to-end and verify every stage's artifact.'''
    llm = lazyllm.OnlineChatModule(
        source='qwen', model=QWEN_MODEL,
        api_key=get_api_key('qwen'), stream=False,
    )
    store = str(REPO_ROOT / 'tests' / 'charge_tests' / 'artifacts' / 'write_workflow_e2e')
    wf = NaiveWriterWorkflow(llm=llm, artifact_store=store)

    task = WritingTask(
        task_id='wf-e2e',
        query=(
            'Write a technical overview for an AI-powered coding assistant product. '
            'Cover system architecture, supported languages, deployment model, and security.'
        ),
        task_type='write',
        target_document=TargetDocument(
            title='Technical Overview of AI-Powered Coding Assistant',
            uri='feishu:///writer-ir-e2e.md',
            adapter='feishu',
        ),
        output={'representation': 'ir'},
    )
    inputs = [
        InputResource(
            resource_type='text', resource_id='r1', title='需求规格',
            inline_text=(
                'The product is an AI coding assistant that supports Python, JavaScript, '
                'TypeScript, Go, Java, and Rust. Backend uses microservices architecture '
                'with Python and Go. Must support on-premises deployment and SaaS multi-tenant. '
                'Frontend is a VS Code extension and JetBrains plugin.'
            ),
        ),
        InputResource(
            resource_type='text', resource_id='r2', title='DeepSeek V4 技术报告',
            inline_text=(
                'DeepSeek V4 is a large language model with 685B parameters, '
                'using Mixture-of-Experts architecture. It supports long-context '
                'of 1M tokens and achieves competitive performance on coding benchmarks.'
            ),
        ),
        InputResource(
            resource_type='text', resource_id='r3', title='市场数据',
            inline_text=(
                'The AI coding assistant market reached $3.2 billion in 2024. '
                'GitHub Copilot has over 1.8 million paying users. User willingness '
                'to pay is concentrated on accuracy and latency.'
            ),
        ),
    ]

    with _mock_document_write(wf, task.target_document, 'writer-ir-e2e') as write_fs:
        result = lazyllm.enable_trace(
            wf.write,
            task=task.model_dump(),
            input_resources=[r.model_dump() for r in inputs],
        )
    stages = result.get('stage_results') or {}
    assert stages, 'stage_results must not be empty'

    # --- Step 1: resource_profiles ---
    profiles = _load_stage(stages, 'resource_profiles')
    assert isinstance(profiles, list), f'Expected list, got {type(profiles)}'
    assert len(profiles) >= 3, f'Expected >=3 profiles, got {len(profiles)}'
    for p_dict in profiles:
        # loaded as dict when model_class=None; validate expected keys
        assert isinstance(p_dict.get('resource_id'), str)
        assert p_dict.get('resource_role') in ('spec', 'background', 'example')
        assert isinstance(p_dict.get('key_facts'), list)

    # --- Step 2: writing_context ---
    ctx = _load_stage(stages, 'writing_context', WritingContext)
    assert ctx is not None
    assert ctx.context_id == 'wf-e2e'
    assert len(ctx.facts) >= 1
    assert ctx.document_summary is not None
    assert len(ctx.document_summary.key_points) >= 2

    # --- Step 3: outline ---
    outline = _load_stage(stages, 'outline', WriterDocument)
    assert outline is not None
    assert len(outline.blocks) >= 1

    # --- Step 4: section_instructions ---
    instructions = _load_stage(stages, 'section_instructions', SectionInstructionList)
    assert instructions is not None
    assert len(instructions.instructions) >= 1
    assert instructions.instructions[0].content_ref.node_id

    # --- Step 5: draft_block ---
    draft_block = _load_stage(stages, 'draft_block', WriterBlock)
    assert draft_block is not None
    assert draft_block.content, 'draft block root must carry the section title'
    assert draft_block.children
    assert draft_block.children[0].content, 'first body block must not be empty'

    # --- Step 6: section_review ---
    review = _load_stage(stages, 'section_review', ReviewReport)
    assert review is not None
    assert isinstance(review.result.is_passed, bool)
    assert 0 <= review.result.score <= 100

    # --- Step 7: writing_context (updated) ---
    ctx2 = _load_stage(stages, 'writing_context', WritingContext)
    assert ctx2 is not None
    assert ctx2.document_summary.summary
    assert len(ctx2.meta.get('context_updates', [])) >= 1

    # --- Step 8: draft_document ---
    doc = _load_stage(stages, 'draft_document', WriterDocument)
    assert doc is not None
    assert doc.title
    assert len(doc.blocks) >= 1

    # --- Step 9: final_document ---
    final = _load_stage(stages, 'final_document', WriterDocument)
    assert final is not None
    assert final.title
    rendered = final.metadata.get('rendered_content', '')
    assert len(rendered) >= 100
    assert final.metadata.get('output_format') == 'markdown'

    # --- Step 10: output_review ---
    out_review = _load_stage(stages, 'draft_document_review', ReviewReport)
    assert out_review is not None
    assert isinstance(out_review.result.is_passed, bool)
    assert 0 <= out_review.result.score <= 100

    # --- Step 11: write_result ---
    write_result = _load_stage(stages, 'write_result')
    assert write_result['doc_id'] == 'writer-ir-e2e'
    assert write_result['block_count'] > 0
    write_fs.write_doc_blocks.assert_called_once()

    # --- primary_result ---
    primary = result.get('primary_result') or {}
    primary_path = primary.get('artifact_path') if isinstance(primary, dict) else ''
    assert primary_path


def test_write_workflow_markdown_e2e():
    '''Run the single-section Markdown path through NaiveWriterWorkflow.write().'''
    llm = lazyllm.OnlineChatModule(
        source='qwen', model=QWEN_MODEL,
        api_key=get_api_key('qwen'), stream=False,
    )
    store = str(
        REPO_ROOT / 'tests' / 'charge_tests' / 'artifacts' / 'write_workflow_markdown_e2e'
    )
    wf = NaiveWriterWorkflow(llm=llm, artifact_store=store)
    task = WritingTask(
        task_id='wf-md-e2e',
        query=(
            'Write a concise technical note with exactly one section titled '
            'Architecture and Deployment. Explain how a Python API service is deployed '
            'to Kubernetes and supports both SaaS and on-premises operation.'
        ),
        task_type='write',
        target_document=TargetDocument(
            title='AI Coding Assistant Deployment Note',
            uri='feishu:///writer-markdown-e2e.md',
            adapter='feishu',
        ),
        output={'representation': 'markdown'},
    )
    inputs = [
        InputResource(
            resource_type='text',
            resource_id='deployment-spec',
            title='Deployment requirements',
            inline_text=(
                'The backend exposes a Python API service. It runs on Kubernetes. '
                'Supported deployment modes are multi-tenant SaaS and on-premises.'
            ),
        ),
    ]

    with _mock_document_write(wf, task.target_document, 'writer-markdown-e2e') as write_fs:
        result = lazyllm.enable_trace(
            wf.write,
            task=task.model_dump(),
            input_resources=[resource.model_dump() for resource in inputs],
        )
    stages = result.get('stage_results') or {}
    assert stages

    # --- Step 1: resource_profiles ---
    profiles = _load_stage(stages, 'resource_profiles')
    assert isinstance(profiles, list)
    assert len(profiles) == 1

    # --- Step 2: writing_context ---
    context = _load_stage(stages, 'writing_context', WritingContext)
    assert context.context_id == 'wf-md-e2e'
    assert context.facts

    # --- Step 3: outline ---
    outline = Path(_stage_path(stages, 'outline')).read_text(encoding='utf-8')
    outline_sections = parse_markdown_sections(outline)
    outline_h2 = [section for section in outline_sections if section[0] == 2]
    assert [section[0] for section in outline_sections if section[0] <= 2] == [1, 2]

    # --- Step 4: section_instructions ---
    instructions = _load_stage(stages, 'section_instructions', SectionInstructionList)
    assert len(instructions.instructions) == 1
    instruction_ref = instructions.instructions[0].content_ref
    assert instruction_ref.heading_path == outline_h2[0][1]

    # --- Step 5: draft_block ---
    draft_block = Path(_stage_path(stages, 'draft_block')).read_text(encoding='utf-8')
    draft_block_sections = parse_markdown_sections(draft_block)
    assert draft_block_sections[0][0] == 2
    assert draft_block_sections[0][1][-1] == outline_h2[0][1][-1]
    assert len(draft_block) >= 200

    # --- Step 6: section_review ---
    section_review = _load_stage(stages, 'section_review', ReviewReport)
    assert 0 <= section_review.result.score <= 100

    # --- Step 7: writing_context (updated) ---
    assert context.meta.get('context_updates')

    # --- Step 8: draft_document ---
    draft_document = Path(_stage_path(stages, 'draft_document')).read_text(encoding='utf-8')
    draft_sections = parse_markdown_sections(draft_document)
    assert [section[0] for section in draft_sections[:2]] == [1, 2]
    assert draft_block.strip() in draft_document

    # --- Step 9: final_document ---
    final_path = Path(_stage_path(stages, 'final_document'))
    final_document = final_path.read_text(encoding='utf-8')
    assert final_document == draft_document
    assert all(term in final_document for term in ('Python', 'Kubernetes', 'SaaS'))

    # --- Step 10: output_review ---
    document_review = _load_stage(stages, 'draft_document_review', ReviewReport)
    assert 0 <= document_review.result.score <= 100

    # --- Step 11: write_result ---
    write_result = _load_stage(stages, 'write_result')
    assert write_result['doc_id'] == 'writer-markdown-e2e'
    assert write_result['block_count'] > 0
    write_fs.write_doc_blocks.assert_called_once()

    # --- primary_result ---
    primary = result.get('primary_result') or {}
    assert Path(primary['artifact_path']) == final_path


# ============================================================================
# NaiveWriterWorkflow.revise() E2E
# ============================================================================


def test_revise_workflow_e2e():
    '''End-to-end verify NaiveWriterWorkflow.revise() against a multi-section document,
    covering cross-block and cross-section revision with Selection.'''
    llm = lazyllm.OnlineChatModule(
        source='qwen', model=QWEN_MODEL,
        api_key=get_api_key('qwen'), stream=False,
    )
    store = str(REPO_ROOT / 'tests' / 'charge_tests' / 'artifacts' / 'revise_workflow_e2e')
    wf = NaiveWriterWorkflow(llm=llm, artifact_store=store)

    section_a = WriterBlock(
        node_id='sec-overview', type='heading', content='Product Overview', stage='draft',
        children=[
            WriterBlock(
                node_id='blk-intro', type='paragraph', stage='draft',
                content='LazyCoder is an AI-powered coding assistant designed for professional developers.',
            ),
            WriterBlock(
                node_id='blk-pricing', type='paragraph', stage='draft',
                content='LazyCoder offers a free tier with basic features, a Pro tier at $12/month, '
                        'and an Enterprise tier with custom pricing.',
            ),
        ],
    )
    section_b = WriterBlock(
        node_id='sec-languages', type='heading', content='Supported Languages', stage='draft',
        children=[
            WriterBlock(
                node_id='blk-lang-list', type='paragraph', stage='draft',
                content='Currently supported languages include Python, JavaScript, TypeScript, '
                        'and Go. The team is actively working on expanding coverage.',
            ),
            WriterBlock(
                node_id='blk-lsp', type='paragraph', stage='draft',
                content='IntelliSense and LSP integration is available for all supported languages. '
                        'Code completion quality varies by language maturity.',
            ),
        ],
    )
    section_c = WriterBlock(
        node_id='sec-deployment', type='heading', content='Deployment & Security', stage='draft',
        children=[
            WriterBlock(
                node_id='blk-deploy', type='paragraph', stage='draft',
                content='Deployment modes include on-premises Kubernetes, SaaS multi-tenant cloud, '
                        'and a single-tenant dedicated option for regulated industries.',
            ),
            WriterBlock(
                node_id='blk-security', type='paragraph', stage='draft',
                content='All customer code is encrypted at rest and in transit. '
                        'The product has SOC 2 Type II certification.',
            ),
        ],
    )
    document = WriterDocument(
        document_id='draft-1',
        stage='draft',
        title='LazyCoder Product Overview',
        blocks=[section_a, section_b, section_c],
    )
    context = WritingContext(
        context_id='revise-ut',
        doc_id='draft-1',
        document_summary=DocumentSummary(summary='LazyCoder Product Overview', key_points=[]),
    )

    result = lazyllm.enable_trace(
        wf.revise,
        task=WritingTask(
            task_id='revise-ut',
            query=(
                'Add support for Rust and Java to the supported languages list '
                'in the Languages section. Also update the deployment section '
                'to mention that single-tenant is available for financial services. '
                'Do not change anything else.'
            ),
            task_type='revise',
            selection=Selection(content_refs=[
                ContentRef(node_id='blk-lang-list'),
                ContentRef(node_id='blk-deploy'),
            ]),
        ).model_dump(),
        document=document,
        context=context,
    )
    stages = result.get('stage_results') or {}
    assert stages, 'revise() stage_results must not be empty.'

    # --- locate ---
    locate = _load_stage(stages, 'locate_result', LocateResult)
    located_node_ids = [target.content_ref.node_id for target in locate.targets]
    assert located_node_ids, 'locate must select at least one block.'
    assert set(located_node_ids) <= {'blk-lang-list', 'blk-deploy'}, (
        f'locate must only pick blocks within selection, got {located_node_ids}'
    )
    for target in locate.targets:
        assert target.reason.strip(), f'missing reason for selected block {target.content_ref.node_id}.'

    # --- modify_plan ---
    plan = _load_stage(stages, 'modify_plan', ModifyPlan)
    assert {i.content_ref.node_id for i in plan.instructions} == set(located_node_ids)
    for instr in plan.instructions:
        assert instr.modify_type in {'create', 'update', 'delete', 'move'}
        assert instr.instruction.strip()

    # --- patch_set ---
    patch = _load_stage(stages, 'patch_set', PatchSet)
    assert len(patch.hunks) == len(plan.instructions)
    original_text_by_id = {}
    for block in document.iter_blocks():
        original_text_by_id[block.node_id] = block.content
    for hunk in patch.hunks:
        assert hunk.modify_type == 'update'
        assert hunk.block is not None
        assert hunk.block.content != original_text_by_id[hunk.target_node_id]
    assert {h.target_node_id for h in patch.hunks} <= {'blk-lang-list', 'blk-deploy'}
    assert any('rust' in (h.block.content if h.block else '').lower() for h in patch.hunks
               if h.target_node_id == 'blk-lang-list'), 'Rust must appear in blk-lang-list patch.'
    assert any('financial' in (h.block.content if h.block else '').lower() for h in patch.hunks
               if h.target_node_id == 'blk-deploy'), 'financial must appear in blk-deploy patch.'

    # --- patch_review ---
    review = _load_stage(stages, 'patch_review', AuditResult)
    assert review is not None
    assert isinstance(review.is_passed, bool)
    assert 0 <= review.score <= 100

    # --- apply_patch ---
    patch_result = _load_stage(stages, 'patch_result', PatchResult)
    assert patch_result is not None and patch_result.success
    assert not patch_result.failed_hunks

    # --- revised_document ---
    revised = load_artifact_json(stages['revised_document'], WriterDocument)
    revised_text_by_id = {b.node_id: b.content for b in revised.iter_blocks()}
    assert revised_text_by_id['blk-lang-list'] != original_text_by_id['blk-lang-list']
    assert revised_text_by_id['blk-deploy'] != original_text_by_id['blk-deploy']
    # Unchanged blocks
    assert revised_text_by_id['blk-intro'] == original_text_by_id['blk-intro']
    assert revised_text_by_id['blk-pricing'] == original_text_by_id['blk-pricing']
    assert revised_text_by_id['blk-lsp'] == original_text_by_id['blk-lsp']
    assert revised_text_by_id['blk-security'] == original_text_by_id['blk-security']
    assert any('rust' in t.lower() for t in revised_text_by_id.values())
    assert any('financial' in t.lower() for t in revised_text_by_id.values())

    # --- rebuild + final_document ---
    revised_context = _load_stage(stages, 'writing_context', WritingContext)
    assert revised_context is not None

    final = _load_stage(stages, 'final_document', WriterDocument)
    assert final is not None
    rendered = final.metadata.get('rendered_content', '')
    assert len(rendered) >= 100
    assert 'rust' in rendered.lower(), 'Rust must appear in the final output.'
    assert 'financial' in rendered.lower(), 'Financial services must appear in the final output.'

    primary = result.get('primary_result') or {}
    assert primary.get('artifact_path'), 'primary_result must carry an artifact_path.'
