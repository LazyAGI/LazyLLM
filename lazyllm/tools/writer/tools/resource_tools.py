from __future__ import annotations
from typing import Any, Dict, List

from .base import WriterToolBase
from ..data_models.docir import DocBlock, DocIR
from ..data_models.resource import ResourceProfile
from ..data_models.task import InputResource, TargetDocument, WritingTask
from ..prompts.profile_resources import RESOURCE_PROFILE_PROMPT

_BLOCK_TYPE_MAPS: Dict[str, Dict[Any, str]] = {
    "feishu": {
        1: "document", 2: "paragraph",
        3: "heading", 4: "heading", 5: "heading", 6: "heading",
        7: "heading", 8: "heading", 9: "heading", 10: "heading", 11: "heading",
        12: "list_item", 13: "list_item",
        14: "code", 15: "quote",
        27: "image", 31: "table", 32: "table_cell",
    },
}


def _read_resource_content(res: InputResource) -> str:
    """InputResource → plain text. Delegates to existing LazyLLM systems."""
    if res.resource_type == "text":
        return res.inline_text or ""

    if res.resource_type == "document":
        import lazyllm.tools.fs.client as _fs_client
        protocol, space_id, real_path = _fs_client.FS._parse(res.uri or "")
        fs = _fs_client.FS._get_or_create_fs(protocol, space_id, real_path)
        return fs.read_bytes(real_path).decode("utf-8")

    if res.resource_type in ("file", "table", "slide"):
        from pathlib import Path
        from lazyllm.tools.rag.dataReader import SimpleDirectoryReader
        reader = SimpleDirectoryReader(input_files=[str(res.uri)])
        nodes = reader._load_data()
        content = "\n".join(n.text for n in nodes if n.text)
        return content if content.strip() else ""

    if res.resource_type == "image":
        return res.summary or ""

    # url / kb — no ready gateway yet
    return res.summary or ""


class WriterResourceTools(WriterToolBase):
    __public_apis__ = [
        "profile_resources",
        "target_to_doc_ir",
        "write_to_document",
    ]

    def profile_resources(
        self,
        task: Any,
        input_resources: Any = None,
    ) -> dict:
        """Profile input resources for the writing task."""
        writing_task = self._unified_model(task, WritingTask)
        inputs = self._unified_models(input_resources, InputResource)

        profiles: List[ResourceProfile] = []
        for res in inputs:
            content = _read_resource_content(res)

            resource_role = res.meta.get("role", "background")
            template_usage = res.meta.get("template", "none")
            summary = res.summary or (content[:500] if content else "")
            key_facts: List[str] = []
            style_notes_list: List[str] = []
            confidence = 1.0
            extracted_constraints: Dict[str, Any] = {}
            extracted_outline = None

            if self.llm is not None and content.strip():
                try:
                    prompt = RESOURCE_PROFILE_PROMPT.format(
                        query=writing_task.query,
                        task_type=writing_task.task_type,
                        constraints=str(writing_task.constraints),
                        title=res.title or "",
                        summary=res.summary or "",
                        content=content,
                    )
                    llm_result = self._call_llm_structured(prompt, ResourceProfile)
                    resource_role = llm_result.resource_role or resource_role
                    template_usage = llm_result.template_usage or template_usage
                    summary = llm_result.summary or summary
                    key_facts = llm_result.key_facts or []
                    style_notes_list = llm_result.style_notes or []
                    confidence = llm_result.confidence or 1.0
                    extracted_constraints = llm_result.extracted_constraints or {}
                    extracted_outline = llm_result.extracted_outline or None
                except Exception:
                    pass

            profiles.append(ResourceProfile(
                resource_id=res.resource_id or f"res-{len(profiles)}",
                resource_role=resource_role,
                template_usage=template_usage,
                summary=summary,
                key_facts=key_facts,
                style_notes=style_notes_list,
                confidence=confidence,
                extracted_constraints=extracted_constraints,
                extracted_outline=extracted_outline,
            ))

        return self._save_artifacts(
            {"resource_profiles": profiles},
            step_name="profile_resources",
            primary_key="resource_profiles",
            context_key=None,
            summary=f"Profiled {len(profiles)} resources.",
            counts={"resource_profiles": len(profiles)},
        ).model_dump()

    def target_to_doc_ir(self, target_document: Any) -> dict:
        """Convert a target document into a DocIR artifact."""
        target = self._unified_model(target_document, TargetDocument)
        locator = target.uri or target.doc_id
        if not locator:
            raise ValueError("target_document must provide uri or doc_id.")

        import lazyllm.tools.fs.client as _fs_client
        protocol, space_id, real_path = _fs_client.FS._parse(locator)
        fs = _fs_client.FS._get_or_create_fs(protocol, space_id, real_path)

        if hasattr(fs, "resolve_link"):
            resolved_ref = fs.resolve_link(real_path) or {}
        else:
            resolved_ref = {}
        plain_text = fs.read_bytes(real_path).decode("utf-8", errors="replace")
        document_id = fs.get_document_id(real_path) if hasattr(fs, "get_document_id") else ""
        raw_blocks = fs.get_doc_blocks(real_path, with_descendants=True) if hasattr(fs, "get_doc_blocks") else []
        raw_blocks = raw_blocks or []

        bt_map = _BLOCK_TYPE_MAPS.get(protocol, {})
        blocks: List[DocBlock] = []
        for raw in raw_blocks:
            if not isinstance(raw, dict):
                continue
            bt = raw.get("block_type", "block")
            if isinstance(bt, int):
                bt = bt_map.get(bt, "block")
            if bt not in DocBlock.model_fields["block_type"].annotation.__args__:
                bt = "block"
            blocks.append(DocBlock(
                block_id=str(raw.get("block_id") or ""),
                block_type=bt,
                text=raw.get("plain_text") or "",
            ))

        doc_id = target.doc_id or document_id or (
            resolved_ref.get("object_id")
            or resolved_ref.get("obj_token")
            or resolved_ref.get("node_token")
            or None
        )
        title = target.title or resolved_ref.get("title") or None

        doc_ir = DocIR(
            doc_id=doc_id,
            source=target,
            title=title,
            blocks=blocks,
            plain_text=plain_text or None,
            adapter=protocol,
            meta={
                "block_count": len(blocks),
            },
        )

        result = self._save_artifacts(
            {"doc_ir": doc_ir},
            step_name="target_to_doc_ir",
            primary_key="doc_ir",
            summary="Loaded target document into DocIR.",
            counts={"blocks": len(blocks)},
            extra={
                "adapter": protocol,
                "document_id": doc_ir.doc_id,
            },
        )
        return result.model_dump()

    def write_to_document(self, markdown: str, target_document: Any) -> dict:
        """Write markdown content back to a target document platform."""
        target = self._unified_model(target_document, TargetDocument)
        locator = target.uri or target.doc_id
        if not locator:
            raise ValueError("target_document must provide uri or doc_id.")

        import lazyllm.tools.fs.client as _fs_client
        protocol, space_id, real_path = _fs_client.FS._parse(locator)
        fs = _fs_client.FS._get_or_create_fs(protocol, space_id, real_path)

        fs.write_file(real_path, markdown.encode("utf-8"))

        resolved_ref = fs.resolve_link(real_path) if hasattr(fs, "resolve_link") else {}
        resolved_ref = resolved_ref or {}
        doc_id = resolved_ref.get("object_id") or resolved_ref.get("obj_token") or ""

        result = self._save_artifacts(
            {"write_result": {
                "doc_id": doc_id,
                "adapter": protocol,
                "locator": locator,
                "content": markdown,
            }},
            step_name="write_to_document",
            primary_key="write_result",
            summary="Wrote content to target document.",
            extra={
                "adapter": protocol,
                "document_id": doc_id,
            },
        )
        return result.model_dump()
