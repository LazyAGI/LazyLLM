import json
import tempfile
import os
import pytest

from lazyllm.tools.writer.data_models import (
    ResourceProfile,
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
from lazyllm.tools.writer.tools.base import WriterToolBase
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


def test_load_artifact_schema_mismatch():
    outline = WritingOutline(title="测试大纲")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "outline.json")
        save_artifact_json(outline, path)
        with pytest.raises(ValueError, match="schema mismatch"):
            load_artifact_json(path, WritingContext)


def test_save_and_load_dict_list_artifacts():
    data = [{"resource_id": "r1", "resource_role": "background"}]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "profiles.json")
        returned_path = save_artifact_json(
            data,
            path,
            schema_name="lazyllm.tools.writer.artifacts.resource_profiles",
        )
        loaded = load_artifact_json(
            returned_path,
            expected_schema_name="lazyllm.tools.writer.artifacts.resource_profiles",
        )
        assert loaded == data


def test_writer_tool_base_save_artifacts_metadata():
    outline = WritingOutline(title="测试大纲", nodes=[OutlineNode(title="背景")])
    context = WritingContext(context_id="ctx-1")
    profiles = [ResourceProfile(resource_id="r1", resource_role="background")]

    with tempfile.TemporaryDirectory() as d:
        tool = WriterToolBase(artifact_store=d)
        result = tool._save_artifacts(
            {
                "outline": outline,
                "writing_context": context,
                "resource_profiles": profiles,
            },
            step_name="generate_outline",
            primary_key="outline",
            summary="Generated outline.",
            counts={"outline_nodes": 1, "resource_profiles": 1},
        )

        assert result.artifact_path.endswith("outline.json")
        assert result.context_path.endswith("writing_context.json")
        assert result.metadata["artifact_key"] == "outline"
        assert result.metadata["step_name"] == "generate_outline"
        assert result.metadata["artifact_paths"]["resource_profiles"].endswith(
            "resource_profiles.json"
        )
        assert result.metadata["schema_names"]["resource_profiles"] == (
            "lazyllm.tools.writer.artifacts.resource_profiles"
        )
        assert result.metadata["counts"]["outline_nodes"] == 1


def test_writer_tool_base_save_artifacts_defaults_primary_key():
    outline = WritingOutline(title="测试大纲")

    with tempfile.TemporaryDirectory() as d:
        tool = WriterToolBase(artifact_store=d)
        result = tool._save_artifacts(
            {"outline": outline},
            step_name="generate_outline",
            summary="Generated outline.",
        )

        assert result.artifact_path.endswith("outline.json")
        assert result.metadata["artifact_key"] == "outline"
        assert result.metadata["step_name"] == "generate_outline"
