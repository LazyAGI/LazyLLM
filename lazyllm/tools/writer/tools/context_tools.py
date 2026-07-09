from __future__ import annotations
from datetime import datetime
from typing import Any, List, Optional

from lazyllm import LOG

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
from ..data_models.writing import DraftDocument, DraftSection, WritingOutline
from ..prompts.context import CONTENT_SUMMARY_PROMPT


class WriterContextTools(WriterToolBase):
    __public_apis__ = [
        'create_writing_context',
        'update_writing_context',
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
            context_id=writing_task.task_id or 'writer-context',
            doc_id=self._resolve_doc_id(writing_task, source_doc),
            document_summary=self._build_document_summary(writing_task, profiles, source_doc),
            block_summaries=self._build_block_summaries(source_doc),
            facts=self._build_facts(profiles),
            style_profile=self._build_style_profile(profiles),
            query=writing_task.query,
            meta={
                'source': 'create_writing_context',
            },
        )

        result = self._save_artifacts(
            {'writing_context': context},
            step_name='create_writing_context',
            primary_key='writing_context',
            summary='Created writing context.',
            counts={
                'resource_profiles': len(profiles),
                'facts': len(context.facts),
                'block_summaries': len(context.block_summaries),
            },
            artifact_meta={
                'task_id': writing_task.task_id,
                'task_type': writing_task.task_type,
                'doc_id': context.doc_id,
                'resource_profile_count': len(profiles),
                'has_doc_ir': source_doc is not None,
            },
        )
        return result.model_dump()

    def update_writing_context(
        self,
        artifacts: Any = None,
        context: Any = None,
    ) -> dict:
        writing_context = self._unified_model(context, WritingContext)

        if artifacts is None:
            return self._save_artifacts(
                {'writing_context': writing_context},
                step_name='update_writing_context',
                primary_key='writing_context',
                summary='Updated writing context (no artifacts).',
                counts={'facts': len(writing_context.facts)},
            ).model_dump()

        if not isinstance(artifacts, list):
            artifacts = [artifacts]

        content_kind = None

        for artifact in artifacts:
            raw = self._unified_raw_data(artifact)
            kind = self._resolve_artifact_kind(artifact) or 'content'

            if kind == 'WritingOutline':
                writing_context.outline = self._unified_model(raw, WritingOutline)

            elif kind == 'DraftSection':
                summary = self._ensure_document_summary(writing_context, raw)
                self._append_context_update(writing_context, summary, kind)
                writing_context.draft_sections.append(self._unified_model(raw, DraftSection))

            elif kind == 'DraftDocument':
                summary = self._ensure_document_summary(writing_context, raw)
                self._append_context_update(writing_context, summary, kind)
                writing_context.draft_document = self._unified_model(raw, DraftDocument)

            else:
                summary = self._ensure_document_summary(writing_context, raw)
                self._append_context_update(writing_context, summary, kind)

            content_kind = content_kind or kind

        writing_context.meta.update(
            {
                'source': 'update_writing_context',
            }
        )

        result = self._save_artifacts(
            {'writing_context': writing_context},
            step_name='update_writing_context',
            primary_key='writing_context',
            summary='Updated writing context.',
            counts={
                'facts': len(writing_context.facts),
            },
            artifact_meta={
                'context_id': writing_context.context_id,
                'doc_id': writing_context.doc_id,
                'last_updated_from': content_kind or 'none',
                'has_outline': writing_context.outline is not None,
            },
        )
        return result.model_dump()

    def _ensure_document_summary(self, writing_context: WritingContext, raw: Any) -> str:
        content_summary = self._summarize_content_data(raw)
        if writing_context.document_summary is None:
            writing_context.document_summary = DocumentSummary(
                summary=content_summary, key_points=[], structure_summary=None,
            )
        else:
            writing_context.document_summary.summary = content_summary
        return content_summary

    def _append_context_update(self, writing_context: WritingContext, summary: str, kind: str) -> None:
        writing_context.meta.setdefault('context_updates', []).append({
            'summary': summary,
            'content_kind': kind,
            'timestamp': datetime.now().astimezone().isoformat(),
        })

    def _resolve_artifact_kind(self, artifact: Any) -> Optional[str]:
        if not isinstance(artifact, str):
            return None
        try:
            import json as _json
            with open(artifact, 'r', encoding='utf-8') as f:
                schema = _json.load(f).get('schema', '')
            return schema.rsplit('.', 1)[-1] if '.' in schema else None
        except Exception:
            return None

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
        structure_summary = self._build_structure_summary(doc_ir)

        summary = task.query
        if doc_ir and doc_ir.plain_text:
            summary = self._summarize_content_data(doc_ir.plain_text)

        return DocumentSummary(
            summary=summary,
            key_points=key_points,
            structure_summary=structure_summary,
        )

    def _build_structure_summary(self, doc_ir: Optional[DocIR]) -> Optional[str]:
        if not doc_ir or not doc_ir.blocks:
            return None
        headings = [b for b in self._iter_blocks(doc_ir.blocks) if b.block_type == 'heading']
        if headings:
            parts = [f'{"#" * (b.level or 1)} {b.text}' for b in headings if b.text]
            return '文档结构: ' + ' > '.join(parts) if parts else None
        return f'由 {len(doc_ir.blocks)} 个顶层块组成'

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
                        fact_id=f'fact-{len(facts) + 1}',
                        key=profile.resource_id,
                        value=fact,
                        source=[profile.resource_id],
                    )
                )
        return facts

    def _build_style_profile(self, profiles: List[ResourceProfile]) -> Optional[StyleProfile]:
        notes: List[str] = []
        tone: Optional[str] = None
        formality: Optional[str] = None
        audience: Optional[str] = None
        for profile in profiles:
            if profile.style:
                notes.extend(profile.style.notes)
                tone = tone or profile.style.tone
                formality = formality or profile.style.formality
                audience = audience or profile.style.audience
        if not notes and not tone:
            return None
        return StyleProfile(tone=tone, formality=formality, audience=audience, notes=notes)

    def _iter_blocks(self, blocks: List[DocBlock]) -> List[DocBlock]:
        result: List[DocBlock] = []
        for block in blocks:
            result.append(block)
            result.extend(self._iter_blocks(block.children))
        return result

    def _summarize_content_data(self, content_data: Any) -> str:
        text = self._extract_text(content_data)
        if not text or not text.strip():
            return 'No content summary available.'

        result = self._shorten(text)

        if self.llm is not None and len(text) > 240:
            try:
                import json as _json
                prompt = CONTENT_SUMMARY_PROMPT.format(content=text[:3000])
                response = str(self.llm(prompt))
                parsed = _json.loads(response)
                result = parsed.get('summary') or result
            except Exception:
                LOG.warning('update_writing_context: LLM summary failed, using truncation fallback')

        return result

    def _extract_text(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            parts: List[str] = []
            for key in ('title', 'summary', 'content', 'plain_text'):
                item = value.get(key)
                if item:
                    parts.append(str(item))
            for key in ('sections', 'blocks', 'sub_sections'):
                item = value.get(key)
                if item:
                    parts.append(self._extract_text(item))
            return ' '.join(part for part in parts if part)
        if isinstance(value, list):
            return ' '.join(self._extract_text(item) for item in value)
        return str(value) if value is not None else ''

    def _content_artifact_kind(self, content_data: Any) -> str:
        if isinstance(content_data, dict):
            return str(
                content_data.get('draft_id')
                or content_data.get('output_id')
                or content_data.get('doc_id')
                or 'dict'
            )
        return type(content_data).__name__

    def _shorten(self, text: str, limit: int = 240) -> str:
        normalized = ' '.join(text.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3] + '...'
