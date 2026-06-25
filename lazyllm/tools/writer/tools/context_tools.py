from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from .base import WriterToolBase
from ..data_models.context import (
    BlockSummary,
    DocumentFact,
    DocumentSummary,
    StyleProfile,
    WritingContext,
)
from ..data_models.docir import DocBlock, DocIR
from ..data_models.resource import ResourceProfile
from ..data_models.task import WritingTask


class WriterContextTools(WriterToolBase):
    __public_apis__ = [
        "create_writing_context",
        "update_writing_context",
    ]

    def create_writing_context(
        self,
        task: Any,
        resource_profiles: Any = None,
        doc_ir: Any = None,
    ) -> dict:
        writing_task = self._unified_model(task, WritingTask)
        profiles = self._unified_models(resource_profiles, ResourceProfile)
        source_doc = self._unified_optional_model(doc_ir, DocIR)

        context = WritingContext(
            context_id=writing_task.task_id or "writer-context",
            doc_id=self._resolve_doc_id(writing_task, source_doc),
            document_summary=self._build_document_summary(writing_task, profiles, source_doc),
            block_summaries=self._build_block_summaries(source_doc),
            facts=self._build_facts(profiles),
            style_profile=self._build_style_profile(profiles),
            meta={
                "task_type": writing_task.task_type,
                "task_id": writing_task.task_id,
                "source": "create_writing_context",
                "resource_profile_count": len(profiles),
                "has_doc_ir": source_doc is not None,
            },
        )

        result = self._save_artifacts(
            {"writing_context": context},
            filename_prefix="create_writing_context",
            primary_key="writing_context",
            summary="Created writing context.",
            tool_name="create_writing_context",
            counts={
                "resource_profiles": len(profiles),
                "facts": len(context.facts),
                "block_summaries": len(context.block_summaries),
            },
        )
        return result.model_dump()

    def update_writing_context(self, content_artifact: Any, context: Any) -> dict:
        writing_context = self._unified_model(context, WritingContext)
        content_data = self._load_content_artifact(content_artifact)
        content_summary = self._summarize_content_data(content_data)

        if writing_context.document_summary is None:
            writing_context.document_summary = DocumentSummary(
                summary=content_summary,
                key_points=[],
                structure_summary=None,
            )
        else:
            writing_context.document_summary.summary = content_summary

        block_id = f"content_update_{len(writing_context.block_summaries) + 1}"
        writing_context.block_summaries.append(
            BlockSummary(
                block_id=block_id,
                summary=content_summary,
                key_points=[],
            )
        )
        writing_context.meta.update(
            {
                "source": "update_writing_context",
                "last_updated_from": self._content_artifact_kind(content_data),
            }
        )

        result = self._save_artifacts(
            {"writing_context": writing_context},
            filename_prefix="update_writing_context",
            primary_key="writing_context",
            summary="Updated writing context.",
            tool_name="update_writing_context",
            counts={
                "facts": len(writing_context.facts),
                "block_summaries": len(writing_context.block_summaries),
            },
        )
        return result.model_dump()

    def _resolve_doc_id(self, task: WritingTask, doc_ir: Optional[DocIR]) -> Optional[str]:
        if task.target_document and task.target_document.doc_id:
            return task.target_document.doc_id
        if doc_ir and doc_ir.doc_id:
            return doc_ir.doc_id
        return None

    def _build_document_summary(
        self,
        task: WritingTask,
        profiles: List[ResourceProfile],
        doc_ir: Optional[DocIR],
    ) -> DocumentSummary:
        key_points = [profile.summary for profile in profiles if profile.summary]
        structure_summary = None
        if doc_ir:
            structure_summary = f"{len(doc_ir.blocks)} top-level blocks"
        return DocumentSummary(
            summary=task.query,
            key_points=key_points,
            structure_summary=structure_summary,
        )

    def _build_block_summaries(self, doc_ir: Optional[DocIR]) -> List[BlockSummary]:
        if not doc_ir:
            return []
        summaries: List[BlockSummary] = []
        for block in self._iter_blocks(doc_ir.blocks):
            text = block.text.strip()
            if text:
                summaries.append(
                    BlockSummary(
                        block_id=block.block_id,
                        summary=self._shorten(text),
                        key_points=[],
                    )
                )
        return summaries

    def _build_facts(self, profiles: List[ResourceProfile]) -> List[DocumentFact]:
        facts: List[DocumentFact] = []
        for profile in profiles:
            for fact in profile.key_facts:
                facts.append(
                    DocumentFact(
                        fact_id=f"fact-{len(facts) + 1}",
                        key=profile.resource_id,
                        value=fact,
                        source=[profile.resource_id],
                    )
                )
        return facts

    def _build_style_profile(self, profiles: List[ResourceProfile]) -> Optional[StyleProfile]:
        notes: List[str] = []
        for profile in profiles:
            notes.extend(profile.style_notes)
        if not notes:
            return None
        return StyleProfile(notes=notes)

    def _iter_blocks(self, blocks: List[DocBlock]) -> List[DocBlock]:
        result: List[DocBlock] = []
        for block in blocks:
            result.append(block)
            result.extend(self._iter_blocks(block.children))
        return result

    def _load_content_artifact(self, content_artifact: Any) -> Any:
        if isinstance(content_artifact, str):
            return self._load_artifact(content_artifact, validate_schema=False)
        if isinstance(content_artifact, BaseModel):
            return content_artifact.model_dump()
        return content_artifact

    def _summarize_content_data(self, content_data: Any) -> str:
        text = self._extract_text(content_data)
        return self._shorten(text or "No content summary available.")

    def _extract_text(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            parts: List[str] = []
            for key in ("title", "summary", "content", "plain_text"):
                item = value.get(key)
                if item:
                    parts.append(str(item))
            for key in ("sections", "blocks", "sub_sections"):
                item = value.get(key)
                if item:
                    parts.append(self._extract_text(item))
            return " ".join(part for part in parts if part)
        if isinstance(value, list):
            return " ".join(self._extract_text(item) for item in value)
        return str(value) if value is not None else ""

    def _content_artifact_kind(self, content_data: Any) -> str:
        if isinstance(content_data, dict):
            return str(content_data.get("draft_id") or content_data.get("output_id") or content_data.get("doc_id") or "dict")
        return type(content_data).__name__

    def _shorten(self, text: str, limit: int = 240) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3] + "..."
