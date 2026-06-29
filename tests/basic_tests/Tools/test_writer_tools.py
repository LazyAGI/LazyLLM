import os
import tempfile
from contextlib import contextmanager

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
from lazyllm.tools.agent.toolsManager import ToolManager
from lazyllm import locals as lazyllm_locals
from lazyllm.tools.writer.utils import load_artifact_json


# ---------------------------------------------------------------------------
# test helpers
# ---------------------------------------------------------------------------

class _FakeDocAdapter:
    """Implements the 4-method LinkDocumentFSBase contract for testing."""

    def __init__(self, blocks=None, plain_text="第一段\n第二段", doc_id="doc-1",
                 ref_title="飞书文档"):
        self.calls = []
        self._blocks = blocks or [
            {"block_id": "b1", "block_type": "heading", "plain_text": "标题"},
            {"block_id": "b2", "block_type": "paragraph", "plain_text": "正文"},
        ]
        self._plain_text = plain_text
        self._doc_id = doc_id
        self._ref_title = ref_title

    def resolve_link(self, locator):
        self.calls.append(("resolve_link", locator))
        return {"provider": "feishu", "object_id": self._doc_id,
                "object_type": "docx", "title": self._ref_title, "has_child": False}

    def read_bytes(self, locator):
        self.calls.append(("read_bytes", locator))
        return self._plain_text.encode("utf-8")

    def get_document_id(self, locator):
        self.calls.append(("get_document_id", locator))
        return self._doc_id

    def write_file(self, locator, data):
        self.calls.append(("write_file", locator))

    def get_doc_blocks(self, locator, with_descendants=True):
        self.calls.append(("get_doc_blocks", locator, with_descendants))
        return self._blocks


@contextmanager
def _mock_fs(adapter, protocol="feishu"):
    """Replace FS singleton so target_to_doc_ir routes to *adapter*."""
    class _M:
        def _parse(self, path):
            return protocol, None, path
        def _get_or_create_fs(self, protocol, space_id, real_path):
            return adapter

    import lazyllm.tools.fs.client as _fs_client
    orig = _fs_client.FS
    _fs_client.FS = _M()
    try:
        yield
    finally:
        _fs_client.FS = orig


# ---------------------------------------------------------------------------
# pre-existing context-tools tests
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

def test_target_to_doc_ir_artifact():
    """Happy path: adapter provides string block_type, full doc_ir artifact."""
    adapter = _FakeDocAdapter()

    with _mock_fs(adapter), tempfile.TemporaryDirectory() as d:
        tool = WriterResourceTools(artifact_store=d)
        result = tool.target_to_doc_ir(target_document={
            "doc_id": "doc-1",
            "uri": "feishu://~docx/doc-1",
            "adapter": "feishu",
            "title": "飞书文档",
            "meta": {"space_id": "wikcn-demo"},
        })

        assert result["artifact_path"].endswith("doc_ir.json")
        assert result["metadata"]["step_name"] == "target_to_doc_ir"
        assert result["metadata"]["counts"]["blocks"] == 2
        assert result["metadata"]["extra"]["adapter"] == "feishu"

        doc_ir = load_artifact_json(result["artifact_path"], DocIR)
        assert doc_ir.doc_id == "doc-1"
        assert doc_ir.title == "飞书文档"
        assert doc_ir.adapter == "feishu"
        assert doc_ir.plain_text == "第一段\n第二段"
        assert len(doc_ir.blocks) == 2
        assert doc_ir.blocks[0].block_id == "b1"
        assert doc_ir.blocks[0].block_type == "heading"
        assert doc_ir.blocks[1].block_id == "b2"
        assert doc_ir.blocks[1].block_type == "paragraph"

        # source kept as-is (no back-fill)
        assert doc_ir.source.doc_id == "doc-1"
        assert doc_ir.source.uri == "feishu://~docx/doc-1"
        assert doc_ir.source.adapter == "feishu"
        assert "source_locator" not in doc_ir.meta

        assert adapter.calls[0][0] == "resolve_link"
        assert adapter.calls[1][0] == "read_bytes"
        assert adapter.calls[2][0] == "get_document_id"
        assert adapter.calls[3][0] == "get_doc_blocks"


def test_target_to_doc_ir_normalizes_int_block_types():
    """Feishu int block_types are mapped to DocIR Literals."""
    adapter = _FakeDocAdapter(blocks=[
        {"block_id": "b1", "block_type": 1,  "plain_text": ""},
        {"block_id": "b2", "block_type": 3,  "plain_text": "标题"},
        {"block_id": "b3", "block_type": 2,  "plain_text": "正文"},
        {"block_id": "b4", "block_type": 14, "plain_text": "代码"},
        {"block_id": "b5", "block_type": 31, "plain_text": ""},
        {"block_id": "b6", "block_type": 99, "plain_text": "未知"},
    ], plain_text="全文")

    with _mock_fs(adapter), tempfile.TemporaryDirectory() as d:
        tool = WriterResourceTools(artifact_store=d)
        result = tool.target_to_doc_ir(target_document={
            "doc_id": "doc-1",
            "uri": "feishu://~docx/doc-1",
            "adapter": "feishu",
            "title": "飞书文档",
        })
        doc_ir = load_artifact_json(result["artifact_path"], DocIR)
        assert [b.block_type for b in doc_ir.blocks] == [
            "document", "heading", "paragraph", "code", "table", "block",
        ]


def test_target_to_doc_ir_title_fallback():
    """title falls back to resolved_ref when target does not supply one."""
    adapter = _FakeDocAdapter(ref_title="Resolved Title")

    with _mock_fs(adapter), tempfile.TemporaryDirectory() as d:
        tool = WriterResourceTools(artifact_store=d)
        result = tool.target_to_doc_ir(target_document={
            "uri": "feishu://~docx/doc-1",
            "adapter": "feishu",
        })
        doc_ir = load_artifact_json(result["artifact_path"], DocIR)
        assert doc_ir.title == "Resolved Title"


def test_target_to_doc_ir_falls_back_on_plain_fs():
    """Non-document FS (e.g. protocol=file) still produces DocIR."""
    class _PlainFS:
        def read_bytes(self, locator):
            return b"plain"
    class _M:
        def _parse(self, path): return "file", None, path
        def _get_or_create_fs(self, p, s, r): return _PlainFS()

    import lazyllm.tools.fs.client as _fs_client
    orig = _fs_client.FS
    _fs_client.FS = _M()
    try:
        with tempfile.TemporaryDirectory() as d:
            tool = WriterResourceTools(artifact_store=d)
            result = tool.target_to_doc_ir(target_document={
                "uri": "/tmp/doc.txt",
                "adapter": "file",
                "title": "plain",
            })
            doc_ir = load_artifact_json(result["artifact_path"], DocIR)
            assert doc_ir.plain_text == "plain"
            assert len(doc_ir.blocks) == 0
            assert doc_ir.doc_id is None
    finally:
        _fs_client.FS = orig


# ---------------------------------------------------------------------------
# write_to_document
# ---------------------------------------------------------------------------

def test_write_to_document_artifact():
    """markdown is written and a write_result artifact is saved."""
    adapter = _FakeDocAdapter()

    with _mock_fs(adapter), tempfile.TemporaryDirectory() as d:
        tool = WriterResourceTools(artifact_store=d)
        result = tool.write_to_document(
            markdown="# Hello\n\nworld",
            target_document={
                "uri": "feishu:///write-test.md",
                "adapter": "feishu",
            },
        )

        assert result["artifact_path"].endswith("write_result.json")
        assert result["metadata"]["step_name"] == "write_to_document"
        assert result["metadata"]["extra"]["adapter"] == "feishu"
        assert result["metadata"]["extra"]["document_id"] == "doc-1"


def test_write_to_document_falls_back_on_plain_fs():
    """Non-document FS still writes, resolve_link returns empty dict."""
    class _PlainWriteFS:
        def write_file(self, path, data):
            pass
    class _M:
        def _parse(self, path): return "file", None, path
        def _get_or_create_fs(self, p, s, r): return _PlainWriteFS()

    import lazyllm.tools.fs.client as _fs_client
    orig = _fs_client.FS
    _fs_client.FS = _M()
    try:
        with tempfile.TemporaryDirectory() as d:
            tool = WriterResourceTools(artifact_store=d)
            result = tool.write_to_document(
                markdown="# Hi",
                target_document={"uri": "/tmp/doc.md", "adapter": "file"},
            )
            assert result["metadata"]["extra"]["document_id"] == ""
    finally:
        _fs_client.FS = orig


# ---------------------------------------------------------------------------
# agent-tool exposure
# ---------------------------------------------------------------------------

def test_writer_resource_tools_is_exposed_as_agent_tools():
    lazyllm_locals["_lazyllm_agent"] = {"workspace": {}}

    manager = ToolManager([WriterResourceTools(artifact_store="/tmp")])
    names = {item["function"]["name"] for item in manager.tools_description}
    assert names == {"get_WriterResourceTools_methods"}

    manager._tool_call["get_WriterResourceTools_methods"]({})
    names = {item["function"]["name"] for item in manager.tools_description}
    assert "WriterResourceTools_target_to_doc_ir" in names
    assert "WriterResourceTools_write_to_document" in names
