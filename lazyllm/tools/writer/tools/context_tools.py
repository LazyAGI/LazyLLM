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
from ..data_models.resource import ResourceProfile
from ..data_models.task import WritingTask
from ..data_models.writer_ir import WriterBlock, WriterDocument
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
        document: Any = None,
    ) -> dict:
        '''Create a WritingContext whose optional content document uses Writer IR.'''
        writing_task = self._unified_model(task, WritingTask)
        profiles = self._unified_models(resource_profiles, ResourceProfile)
        source_doc = self._unified_optional_model(document, WriterDocument)

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

        return self._save_artifacts(
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
                'has_writer_ir': source_doc is not None,
                'writer_stage': source_doc.stage if source_doc else None,
            },
        ).model_dump()

    def update_writing_context(
        self,
        artifacts: Any = None,
        context: Any = None,
    ) -> dict:
        '''Update a WritingContext from WriterDocument or WriterBlock artifacts.'''
        source_context = self._unified_model(context, WritingContext)
        writing_context = source_context.model_copy(deep=True)

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

        content_kind: Optional[str] = None
        for artifact in artifacts:
            raw = self._unified_raw_data(artifact)
            kind = self._resolve_artifact_kind(artifact, raw)

            if kind == 'WriterBlock':
                block = self._unified_model(raw, WriterBlock)
                if block.stage != 'draft':
                    raise ValueError('WritingContext.draft_sections only accepts draft-stage WriterBlock')
                summary = self._ensure_document_summary(writing_context, block)
                writing_context.draft_sections.append(block)
                self._append_context_update(
                    writing_context,
                    summary,
                    content_kind='WriterBlock:draft',
                )
                content_kind = 'WriterBlock:draft'
                continue

            if kind != 'WriterDocument':
                raise TypeError(
                    'artifacts must contain WriterDocument or WriterBlock values, '
                    f'got {kind or type(artifact).__name__}.'
                )

            writer_document = self._unified_model(raw, WriterDocument)
            stage_kind = f'WriterDocument:{writer_document.stage}'
            if writer_document.stage == 'outline':
                writing_context.outline = writer_document
            elif writer_document.stage == 'draft':
                summary = self._ensure_document_summary(writing_context, writer_document)
                writing_context.draft_document = writer_document
                self._append_context_update(
                    writing_context,
                    summary,
                    content_kind=stage_kind,
                    document=writer_document,
                )
            else:
                summary = self._ensure_document_summary(writing_context, writer_document)
                self._append_context_update(
                    writing_context,
                    summary,
                    content_kind=stage_kind,
                    document=writer_document,
                )

            writing_context.doc_id = writing_context.doc_id or writer_document.document_id
            content_kind = stage_kind

        writing_context.meta.update({
            'source': 'update_writing_context',
        })

        return self._save_artifacts(
            {'writing_context': writing_context},
            step_name='update_writing_context',
            primary_key='writing_context',
            summary='Updated writing context.',
            counts={
                'facts': len(writing_context.facts),
                'block_summaries': len(writing_context.block_summaries),
            },
            artifact_meta={
                'context_id': writing_context.context_id,
                'doc_id': writing_context.doc_id,
                'last_updated_from': content_kind or 'none',
                'has_outline': writing_context.outline is not None,
                'has_draft_document': writing_context.draft_document is not None,
            },
        ).model_dump()

    def _ensure_document_summary(
        self,
        writing_context: WritingContext,
        content: WriterDocument | WriterBlock,
    ) -> str:
        if isinstance(content, WriterDocument):
            text = self._document_text(content)
            structure_summary = self._build_structure_summary(content)
        else:
            text = self._block_text(content)
            structure_summary = None
        content_summary = self._summarize_content_data(text)
        if writing_context.document_summary is None:
            writing_context.document_summary = DocumentSummary(
                summary=content_summary,
                key_points=[],
                structure_summary=structure_summary,
            )
        else:
            writing_context.document_summary.summary = content_summary
            if isinstance(content, WriterDocument):
                writing_context.document_summary.structure_summary = structure_summary
        return content_summary

    def _append_context_update(
        self,
        writing_context: WritingContext,
        summary: str,
        content_kind: str,
        document: Optional[WriterDocument] = None,
    ) -> None:
        writing_context.meta.setdefault('context_updates', []).append({
            'summary': summary,
            'content_kind': content_kind,
            'document_id': document.document_id if document else None,
            'revision': document.revision if document else None,
            'timestamp': datetime.now().astimezone().isoformat(),
        })

    def _resolve_artifact_kind(self, artifact: Any, raw: Any = None) -> Optional[str]:
        if isinstance(artifact, WriterDocument):
            return 'WriterDocument'
        if isinstance(artifact, WriterBlock):
            return 'WriterBlock'

        if isinstance(artifact, str):
            try:
                import json as _json
                with open(artifact, 'r', encoding='utf-8') as file:
                    schema = _json.load(file).get('schema', '')
                kind = schema.rsplit('.', 1)[-1] if '.' in schema else schema
                return kind if kind in ('WriterDocument', 'WriterBlock') else None
            except Exception:
                return None

        candidate = raw if raw is not None else artifact
        if isinstance(candidate, dict):
            if 'document_id' in candidate and 'stage' in candidate:
                return 'WriterDocument'
            if 'node_id' in candidate and 'type' in candidate and 'stage' in candidate:
                return 'WriterBlock'
        return None

    def _resolve_doc_id(
        self,
        task: WritingTask,
        writer_document: Optional[WriterDocument],
    ) -> Optional[str]:
        if task.target_document and task.target_document.doc_id:
            return task.target_document.doc_id
        if writer_document:
            return writer_document.document_id
        return None

    def _build_document_summary(
        self,
        task: WritingTask,
        profiles: List[ResourceProfile],
        writer_document: Optional[WriterDocument],
    ) -> DocumentSummary:
        key_points = [profile.summary for profile in profiles if profile.summary]
        structure_summary = self._build_structure_summary(writer_document)

        summary = task.query
        if writer_document:
            text = self._document_text(writer_document)
            if text.strip():
                summary = self._summarize_content_data(text)

        return DocumentSummary(
            summary=summary,
            key_points=key_points,
            structure_summary=structure_summary,
        )

    def _build_structure_summary(
        self,
        writer_document: Optional[WriterDocument],
    ) -> Optional[str]:
        if not writer_document or not writer_document.blocks:
            return None

        headings = [
            block for block in writer_document.iter_blocks()
            if block.type == 'heading' and block.content.strip()
        ]
        if headings:
            parts: List[str] = []
            for block in headings:
                level = block.numbering.get('level', 1)
                if not isinstance(level, int) or isinstance(level, bool) or not 1 <= level <= 9:
                    level = 1
                parts.append(f'{"#" * level} {block.content.strip()}')
            return '文档结构: ' + ' > '.join(parts)

        return f'由 {len(writer_document.blocks)} 个顶层块组成'

    def _build_block_summaries(
        self,
        writer_document: Optional[WriterDocument],
    ) -> List[BlockSummary]:
        if not writer_document:
            return []
        return [
            BlockSummary(
                block_id=block.node_id,
                summary=self._shorten(block.content),
                key_points=[],
            )
            for block in writer_document.iter_blocks()
            if block.content.strip()
        ]

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

    def _document_text(self, document: WriterDocument) -> str:
        parts = [document.title.strip()] if document.title.strip() else []
        parts.extend(
            block.content.strip()
            for block in document.iter_blocks()
            if block.content.strip()
        )
        return '\n'.join(parts)

    def _block_text(self, block: WriterBlock) -> str:
        parts: List[str] = []

        def walk(current: WriterBlock) -> None:
            if current.content.strip():
                parts.append(current.content.strip())
            for child in current.children:
                walk(child)

        walk(block)
        return '\n'.join(parts)

    def _summarize_content_data(self, text: str) -> str:
        if not text.strip():
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

    def _shorten(self, text: str, limit: int = 240) -> str:
        normalized = ' '.join(text.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3] + '...'
