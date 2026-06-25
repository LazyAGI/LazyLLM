import os
import tempfile

from lazyllm.tools.writer.data_models import (
    DocBlock,
    DocIR,
    ResourceProfile,
    WritingContext,
    WritingOutput,
    WritingTask,
)
from lazyllm.tools.writer.tools.context_tools import WriterContextTools
from lazyllm.tools.writer.utils import load_artifact_json


def test_create_writing_context_tool_result():
    task = WritingTask(task_id="task-1", query="写一份产品方案", task_type="write")
    profiles = [
        ResourceProfile(
            resource_id="res-1",
            resource_role="background",
            summary="产品背景资料",
            key_facts=["支持私有化部署"],
            style_notes=["正式"],
        )
    ]
    doc_ir = DocIR(
        doc_id="doc-1",
        blocks=[
            DocBlock(block_id="b1", block_type="heading", text="方案背景", level=1),
            DocBlock(block_id="b2", block_type="paragraph", text="这里是背景内容。"),
        ],
    )

    with tempfile.TemporaryDirectory() as d:
        tool = WriterContextTools(artifact_store=d)
        result = tool.create_writing_context(
            task=task.model_dump(),
            resource_profiles=[profile.model_dump() for profile in profiles],
            doc_ir=doc_ir.model_dump(),
        )

        assert result["artifact_path"].endswith("writing_context.json")
        assert result["context_path"] == result["artifact_path"]
        assert result["metadata"]["step_name"] == "create_writing_context"
        assert result["metadata"]["artifact_key"] == "writing_context"
        assert result["metadata"]["counts"]["facts"] == 1

        context = load_artifact_json(result["context_path"], WritingContext)
        assert context.context_id == "task-1"
        assert context.doc_id == "doc-1"
        assert context.facts[0].value == "支持私有化部署"
        assert len(context.block_summaries) == 2


def test_update_writing_context_tool_result_from_paths():
    context = WritingContext(context_id="ctx-1")
    output = WritingOutput(title="最终稿", content="这是最终输出内容。")

    with tempfile.TemporaryDirectory() as d:
        context_path = os.path.join(d, "context.json")
        output_path = os.path.join(d, "output.json")
        context.save(context_path)
        output.save(output_path)

        tool = WriterContextTools(artifact_store=d)
        result = tool.update_writing_context(content_artifact=output_path, context=context_path)

        assert result["artifact_path"].endswith("writing_context.json")
        assert result["metadata"]["step_name"] == "update_writing_context"
        assert result["metadata"]["counts"]["block_summaries"] == 1

        updated = load_artifact_json(result["context_path"], WritingContext)
        assert updated.document_summary.summary == "最终稿 这是最终输出内容。"
        assert updated.block_summaries[0].summary == "最终稿 这是最终输出内容。"
