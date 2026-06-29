import os
import tempfile
from unittest.mock import MagicMock, patch

from lazyllm.tools.writer.data_models import (
    DocBlock,
    DocIR,
    ResourceProfile,
    WritingContext,
    WritingOutput,
    WritingTask,
)
from lazyllm.tools.writer.tools.resource_tools import WriterResourceTools
from lazyllm.tools.writer.tools.context_tools import WriterContextTools
from lazyllm.tools.writer.utils import load_artifact_json


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_doc_adapter(blocks=None, plain_text="第一段\n第二段",
                      doc_id="doc-1", ref_title="飞书文档"):
    adapter = MagicMock()
    adapter.resolve_link.return_value = {
        "provider": "feishu", "object_id": doc_id,
        "object_type": "docx", "title": ref_title, "has_child": False,
    }
    adapter.read_bytes.return_value = plain_text.encode("utf-8")
    adapter.get_document_id.return_value = doc_id
    adapter.get_doc_blocks.return_value = blocks or [
        {"block_id": "b1", "block_type": "heading", "plain_text": "标题"},
        {"block_id": "b2", "block_type": "paragraph", "plain_text": "正文"},
    ]
    return adapter


def _call_target_to_doc_ir(adapter, artifact_store, protocol="feishu", **kwargs):
    with patch("lazyllm.tools.fs.client.FS._parse",
               return_value=(protocol, None, "~docx/doc-1")):
        with patch("lazyllm.tools.fs.client.FS._get_or_create_fs",
                   return_value=adapter):
            tool = WriterResourceTools(artifact_store=artifact_store)
            params = {"uri": "feishu://~docx/doc-1", "adapter": "feishu",
                      "title": "飞书文档", "doc_id": "doc-1",
                      "meta": {"space_id": "wikcn-demo"}, **kwargs}
            return tool.target_to_doc_ir(target_document=params)


def _call_write_to_document(adapter, markdown, artifact_store,
                            protocol="feishu", uri="feishu:///write-test.md", **kwargs):
    with patch("lazyllm.tools.fs.client.FS._parse",
               return_value=(protocol, None, "/write-test.md")):
        with patch("lazyllm.tools.fs.client.FS._get_or_create_fs",
                   return_value=adapter):
            tool = WriterResourceTools(artifact_store=artifact_store)
            params = {"uri": uri, "adapter": "feishu", **kwargs}
            return tool.write_to_document(markdown=markdown,
                                          target_document=params)


# ---------------------------------------------------------------------------
# context-tools tests (pre-existing)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# target_to_doc_ir
# ---------------------------------------------------------------------------

def test_target_to_doc_ir():
    """Normal path: produces DocIR artifact with correct fields and source."""
    adapter = _make_doc_adapter()

    with tempfile.TemporaryDirectory() as d:
        result = _call_target_to_doc_ir(adapter, d)

        assert result["artifact_path"].endswith("doc_ir.json")
        assert result["metadata"]["step_name"] == "target_to_doc_ir"
        assert result["metadata"]["counts"]["blocks"] == 2
        assert result["metadata"]["extra"]["adapter"] == "feishu"

        doc_ir = load_artifact_json(result["artifact_path"], DocIR)
        assert doc_ir.doc_id == "doc-1"
        assert doc_ir.title == "飞书文档"
        assert doc_ir.adapter == "feishu"
        assert doc_ir.plain_text == "第一段\n第二段"
        assert doc_ir.blocks[0].block_type == "heading"
        assert doc_ir.blocks[1].block_type == "paragraph"

        # source kept as-is (no back-fill)
        assert doc_ir.source.uri == "feishu://~docx/doc-1"
        assert doc_ir.source.adapter == "feishu"
        assert "source_locator" not in doc_ir.meta

        # adapter called in correct order
        adapter.resolve_link.assert_called_once()
        adapter.read_bytes.assert_called_once()
        adapter.get_document_id.assert_called_once()
        adapter.get_doc_blocks.assert_called_once()
        calls = [c[0] for c in adapter.method_calls]
        assert calls.index("resolve_link") < calls.index("read_bytes")
        assert calls.index("read_bytes") < calls.index("get_document_id")
        assert calls.index("get_document_id") < calls.index("get_doc_blocks")


def test_target_to_doc_ir_source_snapshot():
    """source keeps user value even when title comes from resolved_ref."""
    adapter = _make_doc_adapter(ref_title="Resolved Title")

    with tempfile.TemporaryDirectory() as d:
        result = _call_target_to_doc_ir(adapter, d, title="User Title")
        doc_ir = load_artifact_json(result["artifact_path"], DocIR)

        assert doc_ir.source.doc_id == "doc-1"
        assert doc_ir.title == "User Title"
        assert doc_ir.source.title == "User Title"


def test_target_to_doc_ir_title_fallback():
    """title falls back to resolved_ref when target does not supply one."""
    adapter = _make_doc_adapter(ref_title="Resolved Title")

    with tempfile.TemporaryDirectory() as d:
        result = _call_target_to_doc_ir(adapter, d, title=None)
        doc_ir = load_artifact_json(result["artifact_path"], DocIR)

    assert doc_ir.title == "Resolved Title"


def test_target_to_doc_ir_normalizes_int_block_types():
    """Feishu int block_types map to DocIR Literals with unknown→\"block\"."""
    adapter = _make_doc_adapter(blocks=[
        {"block_id": "b1", "block_type": 1,  "plain_text": ""},
        {"block_id": "b2", "block_type": 3,  "plain_text": "标题"},
        {"block_id": "b3", "block_type": 2,  "plain_text": "正文"},
        {"block_id": "b4", "block_type": 14, "plain_text": "代码"},
        {"block_id": "b5", "block_type": 31, "plain_text": ""},
        {"block_id": "b6", "block_type": 99, "plain_text": "未知"},
    ], plain_text="全文")

    with tempfile.TemporaryDirectory() as d:
        result = _call_target_to_doc_ir(adapter, d)
        doc_ir = load_artifact_json(result["artifact_path"], DocIR)

    assert [b.block_type for b in doc_ir.blocks] == [
        "document", "heading", "paragraph", "code", "table", "block",
    ]


def test_target_to_doc_ir_non_document_fs():
    """Plain FS (protocol=file) does not crash — hasattr guards work."""
    class _PlainFS:
        def read_bytes(self, *_):
            return b"plain"

    with patch("lazyllm.tools.fs.client.FS._parse",
               return_value=("file", None, "/tmp/doc.txt")):
        with patch("lazyllm.tools.fs.client.FS._get_or_create_fs",
                   return_value=_PlainFS()):
            with tempfile.TemporaryDirectory() as d:
                tool = WriterResourceTools(artifact_store=d)
                result = tool.target_to_doc_ir(target_document={
                    "uri": "/tmp/doc.txt", "adapter": "file", "title": "plain",
                })
                doc_ir = load_artifact_json(result["artifact_path"], DocIR)

    assert doc_ir.plain_text == "plain"
    assert len(doc_ir.blocks) == 0
    assert doc_ir.doc_id is None


# ---------------------------------------------------------------------------
# write_to_document
# ---------------------------------------------------------------------------

def test_write_to_document():
    """markdown is written via fs.write_file; write_result artifact saved."""
    adapter = _make_doc_adapter()

    with tempfile.TemporaryDirectory() as d:
        result = _call_write_to_document(adapter, "# Hello\n\nworld", d)

        assert result["artifact_path"].endswith("write_result.json")
        assert result["metadata"]["step_name"] == "write_to_document"
        assert result["metadata"]["extra"]["adapter"] == "feishu"
        assert result["metadata"]["extra"]["document_id"] == "doc-1"

        adapter.write_file.assert_called_once()
        args = adapter.write_file.call_args[0]
        assert "write-test" in args[0]
        assert b"Hello" in args[1]


def test_write_to_document_non_document_fs():
    """Plain FS does not crash — resolve_link guard works, doc_id is empty."""
    class _PlainWriteFS:
        def write_file(self, *_):
            pass

    with patch("lazyllm.tools.fs.client.FS._parse",
               return_value=("file", None, "/tmp/doc.md")):
        with patch("lazyllm.tools.fs.client.FS._get_or_create_fs",
                   return_value=_PlainWriteFS()):
            with tempfile.TemporaryDirectory() as d:
                tool = WriterResourceTools(artifact_store=d)
                result = tool.write_to_document(
                    markdown="# Hi",
                    target_document={"uri": "/tmp/doc.md", "adapter": "file"},
                )

    assert result["metadata"]["extra"]["document_id"] == ""


