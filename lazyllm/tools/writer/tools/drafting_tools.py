from __future__ import annotations
import os
from typing import Any, List

from .base import WriterToolBase
from ..data_models.context import WritingContext
from ..data_models.task import WritingTask
from ..data_models.writer_ir import WriterBlock, WriterDocument
from ..data_models.planning import SectionInstruction
from ..prompts import GENERATE_DRAFT_SECTION_PROMPT
from ..utils import render_document_markdown, to_prompt_json


class WriterDraftingTools(WriterToolBase):
    __public_apis__ = [
        'generate_draft_section',
        'generate_draft_document',
        'generate_final_document',
    ]

    def generate_draft_section(
        self,
        task: Any,
        section_instruction: Any,
        context: Any,
        previous_blocks: Any = None,
    ) -> dict:
        writing_task = self._unified_model(task, WritingTask)
        instruction = self._unified_model(section_instruction, SectionInstruction)
        writing_context = self._unified_model(context, WritingContext)
        previous_data = self._unified_raw_data(previous_blocks)

        prompt = GENERATE_DRAFT_SECTION_PROMPT.format(
            task_json=to_prompt_json(writing_task),
            section_instruction_json=to_prompt_json(instruction),
            context_json=to_prompt_json(writing_context),
            previous_blocks_json=to_prompt_json(previous_data),
        )
        draft_block = self._call_llm_structured(prompt, WriterBlock)
        draft_block = self._normalize_draft_block(draft_block, instruction)

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
                'instruction_id': instruction.instruction_id,
                'origin_node_id': instruction.outline_node_id,
                'outline_title': instruction.meta.get('outline_title'),
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
        for block in blocks:
            for item in block.iter_blocks():
                item.stage = 'draft'
        draft_document = WriterDocument(
            document_id=f'draft-document-{writing_context.context_id}',
            stage='draft',
            title=(
                str(title)
                if title is not None
                else writing_outline.title if writing_outline else ''
            ),
            blocks=blocks,
            ui_editable=False,
            metadata={
                'source': 'generate_draft_document',
                'context_id': writing_context.context_id,
                'outline_id': writing_outline.document_id if writing_outline else None,
                'outline_title': writing_outline.title if writing_outline else None,
            },
        )
        draft_block_count = len(list(draft_document.iter_blocks())) - len(draft_document.blocks)

        result = self._save_artifacts(
            {'draft_document': draft_document},
            step_name='generate_draft_document',
            primary_key='draft_document',
            context_key=None,
            summary='Generated draft document.',
            counts={
                'draft_sections': len(draft_document.blocks),
                'draft_blocks': draft_block_count,
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

    def generate_final_document(
        self,
        draft: Any,
        context: Any,
        output_format: str = 'markdown',
    ) -> dict:
        if output_format != 'markdown':
            raise ValueError('Only markdown output is supported for now.')

        writing_context = self._unified_model(context, WritingContext)
        draft_document = self._unified_model(draft, WriterDocument)
        content = render_document_markdown(draft_document)
        final_document = WriterDocument(
            document_id=f'output-{draft_document.document_id}',
            stage='final',
            title=draft_document.title,
            blocks=[block.model_copy(deep=True) for block in draft_document.blocks],
            ui_editable=False,
            metadata={
                'source': 'generate_final_document',
                'draft_id': draft_document.document_id,
                'context_id': writing_context.context_id,
                'output_format': output_format,
                'rendered_content': content,
            },
        )
        for block in final_document.iter_blocks():
            block.stage = 'final'

        result = self._save_artifacts(
            {'final_document': final_document},
            step_name='generate_final_document',
            primary_key='final_document',
            context_key=None,
            summary='Generated writing output.',
            counts={
                'characters': len(content),
                'draft_sections': len(draft_document.blocks),
                'draft_blocks': len(list(draft_document.iter_blocks())) - len(draft_document.blocks),
            },
            artifact_meta={
                'context_id': writing_context.context_id,
                'doc_id': writing_context.doc_id,
                'draft_id': draft_document.document_id,
                'output_format': output_format,
            },
        )
        output_file_path = self._write_output_file(content)
        dumped = result.model_dump()
        dumped['output_file_path'] = output_file_path
        return dumped

    def _normalize_draft_block(
        self,
        draft_block: WriterBlock,
        instruction: SectionInstruction,
    ) -> WriterBlock:
        section_id = instruction.outline_node_id
        draft_block.node_id = section_id
        draft_block.stage = 'draft'
        draft_block.type = 'heading'
        draft_block.content = instruction.section_title
        draft_block.numbering['level'] = 1
        for block in draft_block.iter_blocks():
            block.stage = 'draft'
        draft_block.references = [dict(reference) for reference in instruction.references]
        return draft_block

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

    def _write_output_file(self, content: str) -> str:
        if not self.artifact_store:
            raise ValueError('artifact_store is not set')
        path = os.path.join(self.artifact_store, 'writing_output.md')
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(content)
        return os.path.abspath(path)
