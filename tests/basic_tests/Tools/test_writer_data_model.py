import json
import tempfile
import os

from lazyllm.tools.writer.data_models import (
    WritingTask,
    WritingContext,
    WritingOutline,
    OutlineNode,
    SectionInstruction,
    DraftSection,
    DraftBlock,
    DraftDocument,
    ReviewReport,
    WritingOutput,
)
from lazyllm.tools.writer.utils import ArtifactModel, ToolResult, save_artifact_json, load_artifact_json


def test_imports():
    assert WritingTask is not None
    assert ArtifactModel is not None
    assert ToolResult is not None


def test_writing_task_roundtrip():
    m = WritingTask(query="写一份行业报告", task_type="write")
    restored = WritingTask.model_validate_json(m.model_dump_json())
    assert restored.query == m.query
    assert restored.task_type == "write"


def test_writing_context_roundtrip():
    m = WritingContext(context_id="ctx-1")
    restored = WritingContext.model_validate_json(m.model_dump_json())
    assert restored.context_id == "ctx-1"


def test_writing_outline_roundtrip():
    m = WritingOutline(title="大纲", nodes=[
        OutlineNode(title="第一章", level=1, children=[
            OutlineNode(title="1.1 背景", level=2),
        ]),
    ])
    restored = WritingOutline.model_validate_json(m.model_dump_json())
    assert restored.title == "大纲"
    assert restored.nodes[0].children[0].title == "1.1 背景"


def test_section_instruction_roundtrip():
    m = SectionInstruction(
        instruction_id="ins-1",
        outline_node_id="n1",
        section_title="背景",
        section_goal="介绍行业背景",
    )
    restored = SectionInstruction.model_validate_json(m.model_dump_json())
    assert restored.section_goal == "介绍行业背景"


def test_draft_section_roundtrip():
    m = DraftSection(title="第一章", blocks=[DraftBlock(content="段落内容")])
    restored = DraftSection.model_validate_json(m.model_dump_json())
    assert restored.blocks[0].content == "段落内容"


def test_draft_document_roundtrip():
    m = DraftDocument(title="初稿", sections=[DraftSection(title="第一章")])
    restored = DraftDocument.model_validate_json(m.model_dump_json())
    assert restored.sections[0].title == "第一章"


def test_review_report_roundtrip():
    m = ReviewReport(result={"is_passed": True, "score": 88, "summary": "通过审核"})
    restored = ReviewReport.model_validate_json(m.model_dump_json())
    assert restored.result.score == 88
    assert restored.result.is_passed is True


def test_writing_output_roundtrip():
    m = WritingOutput(content="最终输出内容", output_format="markdown")
    restored = WritingOutput.model_validate_json(m.model_dump_json())
    assert restored.content == "最终输出内容"


def test_artifact_envelope_fields():
    outline = WritingOutline(title="企业报告")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "outline.json")
        outline.save(path, created_by="test")
        raw = json.load(open(path))
        assert "schema" in raw
        assert "schema_version" in raw
        assert "data" in raw
        assert "meta" in raw
        assert raw["meta"]["created_by"] == "test"


def test_save_and_load_outline():
    outline = WritingOutline(title="企业报告", nodes=[
        OutlineNode(title="第一章 背景", level=1),
        OutlineNode(title="第二章 方案", level=1),
    ])
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "outline.json")
        outline.save(path)
        loaded = WritingOutline.load(path)
        assert loaded.title == "企业报告"
        assert len(loaded.nodes) == 2
        assert loaded.nodes[1].title == "第二章 方案"


def test_save_and_load_draft_document():
    draft = DraftDocument(title="初稿", sections=[
        DraftSection(title="第一章", blocks=[DraftBlock(content="背景内容")]),
        DraftSection(title="第二章", blocks=[DraftBlock(content="方案内容")]),
    ])
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "draft.json")
        draft.save(path)
        loaded = DraftDocument.load(path)
        assert loaded.sections[0].blocks[0].content == "背景内容"
        assert loaded.sections[1].title == "第二章"


def test_save_and_load_review_report():
    report = ReviewReport(result={"is_passed": False, "score": 62, "summary": "有问题"})
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "review.json")
        report.save(path)
        loaded = ReviewReport.load(path)
        assert loaded.result.is_passed is False
        assert loaded.result.score == 62


def test_standalone_save_load_functions():
    outline = WritingOutline(title="测试大纲")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "outline.json")
        returned_path = save_artifact_json(outline, path, created_by="unit_test")
        assert os.path.isabs(returned_path)
        loaded = load_artifact_json(returned_path, WritingOutline)
        assert loaded.title == "测试大纲"
