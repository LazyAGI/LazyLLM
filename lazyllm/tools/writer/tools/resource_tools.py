from __future__ import annotations
from typing import Any, Dict, List

from .base import WriterToolBase
from ..data_models.docir import DocBlock, DocIR
from ..data_models.task import TargetDocument

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
        context: Any = None,
    ) -> dict:
        """Profile input resources for the writing task."""
        raise NotImplementedError("profile_resources is not implemented yet.")

    def target_to_doc_ir(self, target_document: Any) -> dict:
        """Convert a target document into a DocIR artifact."""
        target = self._unified_model(target_document, TargetDocument)
        locator = target.uri or target.doc_id
        if not locator:
            raise ValueError("target_document must provide uri or doc_id.")

        import lazyllm.tools.fs.client as _fs_client
        protocol, space_id, real_path = _fs_client.FS._parse(locator)
        fs = _fs_client.FS._get_or_create_fs(protocol, space_id, real_path)

        resolved_ref = fs.resolve_link(real_path) or {}
        plain_text = fs.read_bytes(real_path).decode("utf-8")
        document_id = fs.get_document_id(real_path)
        raw_blocks = fs.get_doc_blocks(real_path, with_descendants=True) or []

        bt_map = _BLOCK_TYPE_MAPS.get(protocol, {})
        blocks: List[DocBlock] = []
        for raw in raw_blocks:
            if not isinstance(raw, dict):
                continue
            bt = raw.get("block_type", "block")
            if isinstance(bt, int):
                bt = bt_map.get(bt, "block")
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

        resolved_ref = fs.resolve_link(real_path) or {}
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
