from pathlib import Path
from typing import List

import pytest
from pydantic import BaseModel

import lazyllm
from lazyllm.tools.writer.tools.base import WriterToolBase
from lazyllm.tools.writer.data_models.context import WritingContext
from lazyllm.tools.writer.data_models.quality import ReviewReport
from lazyllm.tools.writer.data_models.task import InputResource, WritingTask
from lazyllm.tools.writer.data_models.writing import (
    DraftDocument,
    DraftSection,
    SectionInstructionList,
    WritingOutline,
    WritingOutput,
)
from lazyllm.tools.writer.workflow.naive_writer_workflow import NaiveWriterWorkflow
from lazyllm.tools.writer.utils import load_artifact_json
from ...utils import get_api_key, get_path


BASE_PATH = "lazyllm/module/llms/onlinemodule/base/onlineChatModuleBase.py"
WRITER_BASE_PATH = "lazyllm/tools/writer/tools/base.py"
QWEN_MODEL = "qwen-turbo"


class WriterStructuredProbe(BaseModel):
    title: str
    section_count: int
    keywords: List[str]


@pytest.mark.ignore_cache_on_change(BASE_PATH, get_path("qwen"), WRITER_BASE_PATH)
def test_writer_call_llm_structured_with_qwen():
    llm = lazyllm.OnlineChatModule(
        source="qwen",
        model=QWEN_MODEL,
        api_key=get_api_key("qwen"),
        stream=False,
    )
    tool = WriterToolBase(llm=llm)

    result = tool._call_llm_structured(
        (
            "Generate a compact JSON object for testing WriterToolBase structured LLM output. "
            "Use title 'Writer Pipeline Structured Output Test', section_count 3, "
            "and include the keywords planning, drafting, and review."
        ),
        WriterStructuredProbe,
    )

    assert isinstance(result, WriterStructuredProbe)
    assert result.title
    assert result.section_count == 3
    assert {"planning", "drafting", "review"}.issubset(set(result.keywords))


# ============================================================================
# NaiveWriterWorkflow.write() E2E
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _load_stage(stages: dict, key: str, model_class=None):
    entry = stages.get(key) or {}
    if not isinstance(entry, dict):
        return None
    path = (
        entry.get("metadata", {}).get("artifact_paths", {}).get(key)
        or entry.get("artifact_path", "")
    )
    if not path:
        return None
    return load_artifact_json(path, model_class)


def test_write_workflow_e2e():
    """Run NaiveWriterWorkflow.write() end-to-end and verify every stage's artifact."""
    llm = lazyllm.OnlineChatModule(
        source="qwen", model=QWEN_MODEL,
        api_key=get_api_key("qwen"), stream=False,
    )
    store = str(REPO_ROOT / "tests" / "charge_tests" / "artifacts" / "write_workflow_e2e")
    wf = NaiveWriterWorkflow(llm=llm, artifact_store=store)

    task = WritingTask(
        task_id="wf-e2e",
        query=(
            "Write a technical overview for an AI-powered coding assistant product. "
            "Cover system architecture, supported languages, deployment model, and security."
        ),
        task_type="write",
    )
    pdf_path = str(REPO_ROOT / "DeepSeek_V4.pdf")
    inputs = [
        InputResource(
            resource_type="text", resource_id="r1", title="需求规格",
            inline_text=(
                "The product is an AI coding assistant that supports Python, JavaScript, "
                "TypeScript, Go, Java, and Rust. Backend uses microservices architecture "
                "with Python and Go. Must support on-premises deployment and SaaS multi-tenant. "
                "Frontend is a VS Code extension and JetBrains plugin."
            ),
        ),
        InputResource(
            resource_type="file", resource_id="r2", title="DeepSeek V4 技术报告",
            uri=pdf_path,
        ),
        InputResource(
            resource_type="text", resource_id="r3", title="市场数据",
            inline_text=(
                "The AI coding assistant market reached $3.2 billion in 2024. "
                "GitHub Copilot has over 1.8 million paying users. User willingness "
                "to pay is concentrated on accuracy and latency."
            ),
        ),
    ]

    result = wf.write(
        task=task.model_dump(),
        input_resources=[r.model_dump() for r in inputs],
    )
    stages = result.get("stage_results") or {}
    assert stages, "stage_results must not be empty"

    # --- Step 1: resource_profiles ---
    profiles = _load_stage(stages, "resource_profiles")
    assert isinstance(profiles, list), f"Expected list, got {type(profiles)}"
    assert len(profiles) >= 3, f"Expected >=3 profiles, got {len(profiles)}"
    for p_dict in profiles:
        # loaded as dict when model_class=None; validate expected keys
        assert isinstance(p_dict.get("resource_id"), str)
        assert p_dict.get("resource_role") in ("spec", "background", "example")
        assert isinstance(p_dict.get("key_facts"), list)

    # --- Step 2: writing_context ---
    ctx = _load_stage(stages, "writing_context", WritingContext)
    assert ctx is not None
    assert ctx.context_id == "wf-e2e"
    assert len(ctx.facts) >= 1
    assert ctx.document_summary is not None
    assert len(ctx.document_summary.key_points) >= 2

    # --- Step 3: outline ---
    outline = _load_stage(stages, "outline", WritingOutline)
    assert outline is not None
    assert len(outline.nodes) >= 1

    # --- Step 4: section_instructions ---
    instructions = _load_stage(stages, "section_instructions", SectionInstructionList)
    assert instructions is not None
    assert len(instructions.instructions) >= 1
    assert instructions.instructions[0].section_title

    # --- Step 5: draft_section ---
    section = _load_stage(stages, "draft_section", DraftSection)
    assert section is not None
    assert section.title
    assert len(section.blocks) >= 2
    assert section.blocks[0].content, f"Block 0 has empty content"

    # --- Step 6: section_review ---
    review = _load_stage(stages, "section_review", ReviewReport)
    assert review is not None
    assert isinstance(review.result.is_passed, bool)
    assert 0 <= review.result.score <= 100

    # --- Step 7: writing_context (updated) ---
    ctx2 = _load_stage(stages, "writing_context", WritingContext)
    assert ctx2 is not None
    assert ctx2.document_summary.summary
    assert len(ctx2.meta.get("context_updates", [])) >= 1

    # --- Step 8: draft_document ---
    doc = _load_stage(stages, "draft_document", DraftDocument)
    assert doc is not None
    assert doc.title
    assert len(doc.sections) >= 1

    # --- Step 9: writing_output ---
    output = _load_stage(stages, "writing_output", WritingOutput)
    assert output is not None
    assert output.title
    assert len(output.content) >= 100
    assert output.output_format == "markdown"

    # --- Step 10: output_review ---
    out_review = _load_stage(stages, "draft_document_review", ReviewReport)
    assert out_review is not None
    assert isinstance(out_review.result.is_passed, bool)
    assert 0 <= out_review.result.score <= 100

    # --- primary_result ---
    primary = result.get("primary_result") or {}
    primary_path = primary.get("artifact_path") if isinstance(primary, dict) else ""
    assert primary_path
