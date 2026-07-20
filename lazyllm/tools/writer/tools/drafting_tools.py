from __future__ import annotations
import os
from typing import Any, List, Optional

from .base import WriterToolBase
from ..data_models.context import WritingContext
from ..data_models.task import WritingTask
from ..data_models.writer_ir import (
    WriterAuthoring,
    WriterBlock,
    WriterDocument,
)
from ..prompts import GENERATE_DRAFT_SECTION_PROMPT
from ..utils import to_prompt_json


class WriterDraftingTools(WriterToolBase):
    __public_apis__ = [
        'generate_draft_section',
        'generate_draft_document',
        'generate_writing_output',
    ]

    def generate_draft_section(
        self,
        task: Any,
        outline_block: Any,
        context: Any,
        previous_blocks: Any = None,
    ) -> dict:
        writing_task = self._unified_model(task, WritingTask)
        section_block = self._unified_outline_block(outline_block)
        writing_context = self._unified_model(context, WritingContext)
        previous_data = self._unified_raw_data(previous_blocks)

        prompt = GENERATE_DRAFT_SECTION_PROMPT.format(
            task_json=to_prompt_json(writing_task),
            outline_block_json=to_prompt_json(section_block),
            context_json=to_prompt_json(writing_context),
            previous_blocks_json=to_prompt_json(previous_data),
        )
        draft_block = self._call_llm_structured(prompt, WriterBlock)
        draft_block = self._normalize_draft_block(draft_block, section_block)

        result = self._save_artifacts(
            {'draft_block': draft_block},
            step_name='generate_draft_section',
            primary_key='draft_block',
            context_key=None,
            summary='Generated draft section.',
            counts={
                'draft_blocks': len(draft_block.children) + 1,
            },
            artifact_meta={
                'task_id': writing_task.task_id,
                'context_id': writing_context.context_id,
                'node_id': draft_block.node_id,
                'instruction_id': section_block.authoring.instruction_id if section_block.authoring else None,
                'origin_node_id': section_block.node_id,
                'document_title': (
                    section_block.authoring.meta.get('document_title')
                    if section_block.authoring else None
                ),
            },
            artifact_filenames={
                'draft_block': f'draft_block/{draft_block.node_id}.json',
            },
        )
        return result.model_dump()

    def generate_draft_document(
        self,
        draft_blocks: Any,
        context: Any,
        outline: Any = None,
        title: Any = None,
    ) -> dict:
        blocks = self._unified_draft_blocks(draft_blocks)
        if not blocks:
            raise ValueError('draft_blocks must contain at least one WriterBlock.')

        writing_context = self._unified_model(context, WritingContext)
        writing_outline = self._unified_optional_model(outline, WriterDocument)
        normalized_blocks = [
            self._normalize_document_block(block, index)
            for index, block in enumerate(blocks, start=1)
        ]
        draft_document = WriterDocument(
            document_id=self._default_draft_document_id(writing_context),
            stage='draft',
            title=self._resolve_draft_document_title(
                title,
                writing_outline,
                writing_context,
                normalized_blocks,
            ),
            blocks=normalized_blocks,
            metadata={
                'source': 'generate_draft_document',
                'context_id': writing_context.context_id,
                'outline_id': writing_outline.document_id if writing_outline else None,
                'outline_title': writing_outline.title if writing_outline else None,
            },
        )

        result = self._save_artifacts(
            {'draft_document': draft_document},
            step_name='generate_draft_document',
            primary_key='draft_document',
            context_key=None,
            summary='Generated draft document.',
            counts={
                'draft_sections': len(draft_document.blocks),
                'draft_blocks': self._count_draft_blocks(draft_document.blocks),
            },
            artifact_meta={
                'context_id': writing_context.context_id,
                'doc_id': writing_context.doc_id,
                'outline_id': writing_outline.document_id if writing_outline else None,
                'outline_title': writing_outline.title if writing_outline else None,
                'draft_section_count': len(draft_document.blocks),
            },
        )
        return result.model_dump()

    def generate_writing_output(
        self,
        draft: Any,
        context: Any,
        output_format: str = 'markdown',
    ) -> dict:
        if output_format != 'markdown':
            raise ValueError('Only markdown output is supported for now.')

        writing_context = self._unified_model(context, WritingContext)
        draft_document = self._unified_draft_document(draft, writing_context)
        content = self._render_document_markdown(draft_document)
        final_document = WriterDocument(
            document_id=self._default_final_document_id(draft_document, writing_context),
            stage='final',
            title=draft_document.title,
            blocks=[block.model_copy(deep=True) for block in draft_document.blocks],
            metadata={
                'source': 'generate_writing_output',
                'draft_id': draft_document.document_id,
                'context_id': writing_context.context_id,
                'output_format': output_format,
                'rendered_content': content,
            },
            provider_binding={
                'references': self._collect_output_references(draft_document, writing_context),
            },
        )

        result = self._save_artifacts(
            {'final_document': final_document},
            step_name='generate_writing_output',
            primary_key='final_document',
            context_key=None,
            summary='Generated writing output.',
            counts={
                'characters': len(content),
                'draft_sections': len(draft_document.blocks),
                'draft_blocks': self._count_draft_blocks(draft_document.blocks),
            },
            artifact_meta={
                'context_id': writing_context.context_id,
                'doc_id': writing_context.doc_id,
                'draft_id': draft_document.document_id,
                'output_format': output_format,
            },
        )
        output_file_path = self._write_output_file(final_document, content)
        dumped = result.model_dump()
        dumped['output_file_path'] = output_file_path
        return dumped

    def _unified_outline_block(self, value: Any) -> WriterBlock:
        if isinstance(value, WriterBlock):
            return value
        if isinstance(value, WriterDocument):
            return self._select_section_block(list(value.blocks))
        if isinstance(value, str):
            value = self._load_artifact(value, validate_schema=False)
        if isinstance(value, dict):
            if 'blocks' in value:
                doc = WriterDocument.model_validate(value)
                return self._select_section_block(list(doc.blocks))
            return WriterBlock.model_validate(value)
        if isinstance(value, list):
            blocks = [self._unified_outline_block(item) for item in value]
            return self._select_section_block(blocks)
        raise TypeError(
            'Expected WriterBlock, WriterDocument, dict, or artifact path, '
            f'got {type(value).__name__}.'
        )

    def _select_section_block(self, blocks: List[WriterBlock]) -> WriterBlock:
        if not blocks:
            raise ValueError('outline block list is empty.')
        return blocks[0]

    def _normalize_draft_block(
        self,
        draft_block: WriterBlock,
        section_block: WriterBlock,
    ) -> WriterBlock:
        section_id = self._default_section_node_id(section_block)
        draft_block.node_id = draft_block.node_id or section_id
        draft_block.stage = 'draft'
        if not draft_block.type.strip():
            draft_block.type = 'heading'
        if not draft_block.content.strip():
            draft_block.content = section_block.content

        if not draft_block.children:
            draft_block.children.append(WriterBlock(
                node_id=f'{section_id}-block-1',
                type='paragraph',
                content='',
                stage='draft',
            ))

        for index, child in enumerate(draft_block.children, start=1):
            child.node_id = child.node_id or f'{section_id}-block-{index}'
            child.stage = 'draft'
            if not child.type.strip():
                child.type = 'paragraph'

        if draft_block.authoring is None and section_block.authoring is not None:
            draft_block.authoring = WriterAuthoring(
                instruction_id=section_block.authoring.instruction_id,
                origin_node_id=section_block.node_id,
                instruction=section_block.authoring.instruction,
            )
        elif draft_block.authoring is not None:
            draft_block.authoring.origin_node_id = section_block.node_id
            if section_block.authoring and section_block.authoring.instruction_id:
                draft_block.authoring.instruction_id = (
                    draft_block.authoring.instruction_id
                    or section_block.authoring.instruction_id
                )

        if draft_block.authoring is not None:
            draft_block.authoring.meta.update(
                {
                    'source': 'llm',
                    'instruction_id': (
                        section_block.authoring.instruction_id if section_block.authoring else None
                    ),
                    'origin_node_id': section_block.node_id,
                }
            )
            if section_block.authoring:
                self._copy_meta_value(section_block.authoring.meta, draft_block.authoring.meta, 'document_id')
                self._copy_meta_value(section_block.authoring.meta, draft_block.authoring.meta, 'document_title')
        return draft_block

    def _default_section_node_id(self, section_block: WriterBlock) -> str:
        origin = section_block.node_id
        if section_block.authoring and section_block.authoring.instruction_id:
            origin = origin or section_block.authoring.instruction_id
        return f'draft-{origin or "section"}'

    def _unified_draft_blocks(self, value: Any) -> List[WriterBlock]:
        if value is None:
            return []
        if isinstance(value, WriterBlock):
            return [value]
        if isinstance(value, WriterDocument):
            return list(value.blocks)
        if isinstance(value, str):
            value = self._load_artifact(value, validate_schema=False)
            return self._unified_draft_blocks(value)
        if isinstance(value, dict):
            if 'draft' in value:
                return self._unified_draft_blocks(value['draft'])
            if 'blocks' in value:
                return [WriterBlock.model_validate(b) for b in value['blocks']]
            return [WriterBlock.model_validate(value)]
        if isinstance(value, list):
            blocks: List[WriterBlock] = []
            for item in value:
                blocks.extend(self._unified_draft_blocks(item))
            return blocks
        raise TypeError(
            'Expected WriterBlock, WriterDocument, list, dict, or artifact path, '
            f'got {type(value).__name__}.'
        )

    def _unified_draft_document(self, value: Any, context: WritingContext) -> WriterDocument:
        if isinstance(value, WriterDocument):
            return value
        if isinstance(value, str):
            value = self._load_artifact(value, validate_schema=False)
            return self._unified_draft_document(value, context)
        if isinstance(value, dict):
            if 'data' in value:
                return self._unified_draft_document(value['data'], context)
            if 'draft' in value:
                return self._unified_draft_document(value['draft'], context)
            if 'blocks' in value:
                return WriterDocument.model_validate(value)

        blocks = self._unified_draft_blocks(value)
        if not blocks:
            raise ValueError('draft must contain at least one WriterBlock.')
        normalized_blocks = [
            self._normalize_document_block(block, index)
            for index, block in enumerate(blocks, start=1)
        ]
        return WriterDocument(
            document_id=self._default_draft_document_id(context),
            stage='draft',
            title=self._default_draft_document_title(context, normalized_blocks),
            blocks=normalized_blocks,
            metadata={
                'source': 'generate_writing_output',
                'context_id': context.context_id,
            },
        )

    def _normalize_document_block(self, block: WriterBlock, index: int) -> WriterBlock:
        node_id = block.node_id or f'draft-block-{index}'
        block.node_id = node_id
        block.stage = 'draft'
        if not block.type.strip():
            block.type = 'heading'
        for child_index, child in enumerate(block.children, start=1):
            child.node_id = child.node_id or f'{node_id}-block-{child_index}'
            child.stage = 'draft'
            if not child.type.strip():
                child.type = 'paragraph'
        return block

    def _default_draft_document_id(self, context: WritingContext) -> str:
        source_id = context.context_id or context.doc_id or 'document'
        return f'draft-document-{source_id}'

    def _default_final_document_id(
        self,
        draft_document: WriterDocument,
        context: WritingContext,
    ) -> str:
        source_id = draft_document.document_id or context.context_id or context.doc_id or 'document'
        return f'output-{source_id}'

    def _default_draft_document_title(
        self,
        context: WritingContext,
        blocks: List[WriterBlock],
    ) -> str:
        return self._resolve_draft_document_title(None, None, context, blocks)

    def _resolve_draft_document_title(
        self,
        title: Any,
        outline: Optional[WriterDocument],
        context: WritingContext,
        blocks: List[WriterBlock],
    ) -> str:
        title = self._first_non_empty(
            title,
            outline.title if outline else None,
            context.meta.get('title') if context.meta else None,
            context.meta.get('document_title') if context.meta else None,
            context.meta.get('outline_title') if context.meta else None,
            self._first_block_authoring_meta(blocks, 'document_title'),
            self._first_block_authoring_meta(blocks, 'outline_title'),
        )
        if title:
            return str(title)
        if context.doc_id:
            return context.doc_id
        if blocks and blocks[0].content:
            return blocks[0].content
        return 'Draft Document'

    def _count_draft_blocks(self, blocks: List[WriterBlock]) -> int:
        total = 0
        for block in blocks:
            total += len(block.children)
            total += self._count_draft_blocks(block.children)
        return total

    def _render_document_markdown(self, document: WriterDocument) -> str:
        parts: List[str] = []
        if document.title:
            parts.append(f'# {document.title.strip()}')
        for block in document.blocks:
            parts.extend(self._render_block_markdown(block, level=2))
        return '\n\n'.join(part for part in parts if part).strip() + '\n'

    def _render_block_markdown(self, block: WriterBlock, level: int) -> List[str]:
        parts: List[str] = []
        heading_level = min(max(level, 1), 6)
        if block.type == 'heading':
            if block.content.strip():
                parts.append(f'{"#" * heading_level} {block.content.strip()}')
        else:
            content = block.content.strip()
            if content:
                parts.append(content)
        for child in block.children:
            parts.extend(self._render_block_markdown(child, heading_level + 1))
        return parts

    def _collect_output_references(
        self,
        document: WriterDocument,
        context: WritingContext,
    ) -> List[str]:
        # References come from block.source_refs (List[Dict] with an "id" key),
        # not from document.metadata — the latter is not populated for drafts.
        references: List[str] = []
        for block in document.iter_blocks():
            for source_ref in block.source_refs:
                ref_id = source_ref.get('id') if isinstance(source_ref, dict) else None
                if ref_id:
                    self._extend_unique(references, ref_id)
        for fact in context.facts:
            self._extend_unique(references, fact.source)
        return references

    def _extend_unique(self, target: List[str], values: Any) -> None:
        if values is None:
            return
        if isinstance(values, str):
            values = [values]
        for value in values:
            if value and value not in target:
                target.append(str(value))

    def _copy_meta_value(self, source: dict, target: dict, key: str) -> None:
        value = source.get(key) if source else None
        if value and key not in target:
            target[key] = value

    def _first_block_authoring_meta(self, blocks: List[WriterBlock], key: str) -> Any:
        for block in blocks:
            if block.authoring and block.authoring.meta:
                value = block.authoring.meta.get(key)
                if value:
                    return value
            child_value = self._first_block_authoring_meta(block.children, key)
            if child_value:
                return child_value
        return None

    def _first_non_empty(self, *values: Any) -> Any:
        for value in values:
            if value:
                return value
        return None

    def _write_output_file(self, document: WriterDocument, content: str) -> str:
        if not self.artifact_store:
            raise ValueError('artifact_store is not set')
        output_format = document.metadata.get('output_format', 'markdown')
        extension = self._output_file_extension(output_format)
        path = os.path.join(self.artifact_store, f'writing_output.{extension}')
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(content)
        return os.path.abspath(path)

    def _output_file_extension(self, output_format: str) -> str:
        extensions = {
            'markdown': 'md',
            'plain_text': 'txt',
            'html': 'html',
        }
        if output_format not in extensions:
            raise ValueError(f'Unsupported output_format for file export: {output_format}')
        return extensions[output_format]
