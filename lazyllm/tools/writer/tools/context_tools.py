from __future__ import annotations

from datetime import datetime
import os
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
from ..data_models.writer_ir import ContentRef, WriterBlock, WriterDocument
from ..prompts.context import CONTENT_SUMMARY_PROMPT
from ..utils import parse_markdown_sections


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
        '''Create a WritingContext from optional Writer IR or Markdown content.'''
        writing_task = self._unified_model(task, WritingTask)
        profiles = self._unified_models(resource_profiles, ResourceProfile)
        source_content = self._unified_context_content(document)
        source_doc = source_content if isinstance(source_content, WriterDocument) else None

        context = WritingContext(
            context_id=writing_task.task_id or 'writer-context',
            doc_id=source_doc.document_id if source_doc else None,
            document_summary=self._build_document_summary(writing_task, profiles, source_content),
            block_summaries=self._build_block_summaries(source_content),
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
        '''Update a WritingContext from Writer IR or Markdown artifacts.'''
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
            raw = self._unified_context_content(artifact)

            if isinstance(raw, str):
                summary = self._ensure_document_summary(writing_context, raw)
                writing_context.block_summaries = self._build_block_summaries(raw)
                self._append_context_update(
                    writing_context,
                    summary,
                    content_kind='Markdown',
                )
                content_kind = 'Markdown'
                continue

            if isinstance(raw, WriterBlock):
                block = raw
                if block.stage != 'draft':
                    raise ValueError('context updates only accept draft-stage WriterBlock')
                summary = self._ensure_document_summary(writing_context, block)
                writing_context.block_summaries = self._build_block_summaries(block)
                self._append_context_update(
                    writing_context,
                    summary,
                    content_kind='WriterBlock:draft',
                )
                content_kind = 'WriterBlock:draft'
                continue

            if not isinstance(raw, WriterDocument):
                raise TypeError(
                    'artifacts must contain WriterDocument, WriterBlock, or Markdown values, '
                    f'got {type(artifact).__name__}.'
                )

            writer_document = raw
            stage_kind = f'WriterDocument:{writer_document.stage}'
            summary = self._ensure_document_summary(writing_context, writer_document)
            writing_context.block_summaries = self._build_block_summaries(writer_document)
            self._append_context_update(
                writing_context,
                summary,
                content_kind=stage_kind,
                document=writer_document,
            )

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
            },
        ).model_dump()

    def _ensure_document_summary(
        self,
        writing_context: WritingContext,
        content: WriterDocument | WriterBlock | str,
    ) -> str:
        if isinstance(content, str):
            text = content
            structure_summary = self._build_structure_summary(content)
        elif isinstance(content, WriterDocument):
            text = self._document_text(content)
            structure_summary = self._build_structure_summary(content)
        else:
            text = '\n'.join(
                item.content.strip()
                for item in content.iter_blocks()
                if item.content.strip()
            )
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
            if isinstance(content, (WriterDocument, str)):
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

    def _unified_context_content(self, value: Any) -> Optional[WriterDocument | WriterBlock | str]:
        if value is None or isinstance(value, (WriterDocument, WriterBlock)):
            return value
        if isinstance(value, str):
            if os.path.isfile(value) and value.lower().endswith(('.md', '.markdown')):
                with open(value, 'r', encoding='utf-8') as stream:
                    return stream.read()
            if not os.path.isfile(value):
                return value
        raw = self._unified_raw_data(value)
        if isinstance(raw, dict) and 'document_id' in raw:
            return self._unified_model(raw, WriterDocument)
        if isinstance(raw, dict) and 'node_id' in raw and 'type' in raw:
            return self._unified_model(raw, WriterBlock)
        raise TypeError(
            'content must be WriterDocument, WriterBlock, Markdown text, or a Markdown path.'
        )

    def _build_document_summary(
        self,
        task: WritingTask,
        profiles: List[ResourceProfile],
        content: Optional[WriterDocument | str],
    ) -> DocumentSummary:
        key_points = [profile.summary for profile in profiles if profile.summary]
        structure_summary = self._build_structure_summary(content)

        summary = task.query
        if content:
            text = content if isinstance(content, str) else self._document_text(content)
            if text.strip():
                summary = self._summarize_content_data(text)

        return DocumentSummary(
            summary=summary,
            key_points=key_points,
            structure_summary=structure_summary,
        )

    def _build_structure_summary(
        self,
        content: Optional[WriterDocument | str],
    ) -> Optional[str]:
        if not content:
            return None

        if isinstance(content, str):
            sections = parse_markdown_sections(content)
            if not sections:
                return None
            return '文档结构: ' + ' > '.join(
                f'{"#" * level} {heading_path[-1]}'
                for level, heading_path, _, _ in sections
            )

        if not content.blocks:
            return None

        headings = [
            block for block in content.iter_blocks()
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

        return f'由 {len(content.blocks)} 个顶层块组成'

    def _build_block_summaries(
        self,
        content: Optional[WriterDocument | WriterBlock | str],
    ) -> List[BlockSummary]:
        if not content:
            return []
        if isinstance(content, str):
            return [
                BlockSummary(
                    content_ref=ContentRef(
                        heading_path=heading_path,
                        occurrence=occurrence,
                    ),
                    summary=self._shorten(body or heading_path[-1]),
                    key_points=[],
                )
                for _, heading_path, occurrence, body in parse_markdown_sections(content)
            ]
        blocks = content.iter_blocks()
        return [
            BlockSummary(
                content_ref=ContentRef(node_id=block.node_id),
                summary=self._shorten(block.content),
                key_points=[],
            )
            for block in blocks
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
