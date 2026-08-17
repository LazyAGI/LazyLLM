from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .base import WriterToolBase
from .stream_tools import DraftIRStream, DraftMarkdownStream
from ..data_models.context import WritingContext
from ..data_models.multimodal import MediaAssetLibrary, VisualPlan
from ..data_models.task import WritingTask
from ..data_models.writer_ir import WriterBlock, WriterDocument
from ..data_models.planning import SectionInstruction
from ..numbering import (
    build_numbering_view_from_ir,
    build_numbering_view_from_markdown,
    compute_numbering,
)
from ..prompts import (
    CONDENSE_DRAFT_SECTION_MARKDOWN_PROMPT,
    CONDENSE_DRAFT_SECTION_PROMPT,
    GENERATE_DRAFT_SECTION_MARKDOWN_PROMPT,
    GENERATE_DRAFT_SECTION_PROMPT,
)
from ..utils import (
    get_markdown_outline_targets,
    make_markdown_tool_result,
    parse_markdown_sections,
    render_document_markdown,
    strip_caption_numbering,
    strip_heading_numbering,
    to_prompt_json,
)

class WriterDraftingTools(WriterToolBase):
    __public_apis__ = [
        'generate_draft_section',
        'generate_draft_document',
        'generate_final_document',
    ]

    _MARKDOWN_INTERNAL_LINK_RE = re.compile(r'\[([^\]]*)\]\(#((?:block-)?[^)]+)\)')
    _MARKDOWN_ANCHOR_RE = re.compile(r'<a id="((?:block-)?[^"]+)"></a>')
    _MARKDOWN_MEDIA_PLACEHOLDER_RE = re.compile(
        r'!\[[^\]]*\]\(media-placeholder://([A-Za-z0-9_-]+)\)'
    )

    def generate_draft_section(
        self, task: Any, section_instruction: Any,
        context: Any, previous_blocks: Any = None, visual_plan: Any = None,
        media_assets: Any = None,
    ) -> dict:
        writing_task = self._unified_model(task, WritingTask)
        instruction = self._unified_model(section_instruction, SectionInstruction)
        writing_context = self._unified_model(context, WritingContext)
        representation = self._instruction_representation(instruction)
        result_extra = self._draft_result_extra(
            writing_task, instruction, writing_context, representation,
        )

        if representation == 'markdown':
            return self._generate_markdown_draft_section(
                writing_task,
                instruction,
                writing_context,
                previous_blocks,
                visual_plan,
                result_extra,
            )

        return self._generate_ir_draft_section(
            writing_task,
            instruction,
            writing_context,
            previous_blocks,
            visual_plan,
            media_assets,
            result_extra,
        )

    def stream_draft_section(
        self,
        task: Any,
        section_instruction: Any,
        context: Any,
        previous_blocks: Any = None,
        visual_plan: Any = None,
        *,
        idle_timeout: Optional[float] = None,
    ) -> DraftMarkdownStream:
        writing_task = self._unified_model(task, WritingTask)
        instruction = self._unified_model(section_instruction, SectionInstruction)
        writing_context = self._unified_model(context, WritingContext)
        if self._instruction_representation(instruction) != 'markdown':
            raise ValueError('stream_draft_section only supports Markdown section instructions.')

        result_extra = self._draft_result_extra(
            writing_task, instruction, writing_context, 'markdown',
        )
        prompt = self._markdown_draft_prompt(
            writing_task, instruction, writing_context, previous_blocks,
            visual_plan,
        )
        heading = self._markdown_draft_heading(instruction)
        prefix = self._markdown_draft_prefix(instruction, heading)
        timeout = self._draft_stream_idle_timeout(idle_timeout)
        return DraftMarkdownStream(
            call=lambda sink: self._call_llm_text(
                prompt,
                stream_output={'_stream_sink': sink},
            ),
            finalize=lambda body: self._finalize_markdown_draft_section(
                body, instruction, result_extra,
            ),
            prefix=prefix,
            idle_timeout=timeout,
        )

    def stream_draft_section_ir(
        self,
        task: Any,
        section_instruction: Any,
        context: Any,
        previous_blocks: Any = None,
        visual_plan: Any = None,
        media_assets: Any = None,
        *,
        idle_timeout: Optional[float] = None,
    ) -> DraftIRStream:
        writing_task = self._unified_model(task, WritingTask)
        instruction = self._unified_model(section_instruction, SectionInstruction)
        writing_context = self._unified_model(context, WritingContext)
        if self._instruction_representation(instruction) != 'ir':
            raise ValueError('stream_draft_section_ir only supports IR section instructions.')

        result_extra = self._draft_result_extra(
            writing_task, instruction, writing_context, 'ir',
        )
        prompt = self._ir_draft_prompt(
            writing_task,
            instruction,
            writing_context,
            previous_blocks,
            visual_plan,
            media_assets,
        )
        return DraftIRStream(
            call=lambda sink: self._call_llm_structured(
                prompt,
                WriterBlock,
                stream_output={'_stream_sink': sink},
            ),
            normalize=lambda block: self._normalize_draft_block(
                block, instruction, allow_deferred_create=True,
            ),
            finalize=lambda block: self._save_ir_draft_section(
                self._normalize_draft_block(
                    self._attach_section_media(
                        block, instruction, visual_plan, media_assets,
                    ),
                    instruction,
                ),
                result_extra,
                media_assets,
            ),
            instruction=instruction,
            idle_timeout=self._draft_stream_idle_timeout(idle_timeout),
        )

    def _generate_ir_draft_section(
        self,
        task: WritingTask,
        instruction: SectionInstruction,
        context: WritingContext,
        previous_blocks: Any,
        visual_plan: Any,
        media_assets: Any,
        result_extra: Dict[str, Any],
    ) -> dict:
        prompt = self._ir_draft_prompt(
            task,
            instruction,
            context,
            previous_blocks,
            visual_plan,
            media_assets,
        )
        block = self._normalize_draft_block(
            self._attach_section_media(
                self._call_llm_structured(prompt, WriterBlock),
                instruction,
                visual_plan,
                media_assets,
            ),
            instruction,
        )
        block = self._condense_ir_section_if_needed(block, instruction)
        return self._save_ir_draft_section(block, result_extra, media_assets)

    def _ir_draft_prompt(
        self,
        task: WritingTask,
        instruction: SectionInstruction,
        context: WritingContext,
        previous_blocks: Any,
        visual_plan: Any,
        media_assets: Any,
    ) -> str:
        previous_data = self._unified_raw_data(previous_blocks)
        resolved_visual_plan = self._unified_optional_model(
            visual_plan, VisualPlan,
        ) or VisualPlan()
        media_library = self._unified_optional_model(media_assets, MediaAssetLibrary)
        section_media = self._media_assets_for_section(
            instruction, resolved_visual_plan, media_library,
        )
        return GENERATE_DRAFT_SECTION_PROMPT.format(
            task_json=to_prompt_json(task),
            section_instruction_json=to_prompt_json(instruction),
            context_json=to_prompt_json(context),
            previous_blocks_json=to_prompt_json(previous_data),
            section_media_json=to_prompt_json(section_media),
        )

    def _save_ir_draft_section(
        self,
        draft_block: WriterBlock,
        result_extra: Dict[str, Any],
        media_assets: Any = None,
    ) -> dict:
        library = self._unified_optional_model(media_assets, MediaAssetLibrary)
        if library is not None:
            for image_block in draft_block.iter_blocks():
                if image_block.type == 'image':
                    for ref in image_block.references or []:
                        asset = library.assets.get(ref.get('id'))
                        if asset is not None and asset.uri:
                            ref.setdefault('path', asset.uri)
        result = self._save_artifacts(
            {'draft_block': draft_block},
            step_name='generate_draft_section',
            primary_key='draft_block',
            context_key=None,
            summary='Generated draft section.',
            counts={
                'draft_blocks': len(draft_block.children) + 1,
            },
            extra=result_extra,
            artifact_filenames={
                'draft_block': f'draft_block/{draft_block.node_id}_ir.lmd',
            },
        )
        return result.model_dump()

    def _generate_markdown_draft_section(
        self,
        task: WritingTask,
        instruction: SectionInstruction,
        context: WritingContext,
        previous_blocks: Any,
        visual_plan: Any,
        result_extra: Dict[str, Any],
    ) -> dict:
        prompt = self._markdown_draft_prompt(
            task, instruction, context, previous_blocks, visual_plan,
        )
        body = self._call_llm_text(prompt)
        body = self._condense_markdown_section_if_needed(body, instruction)
        return self._finalize_markdown_draft_section(body, instruction, result_extra)

    def _condense_ir_section_if_needed(
        self,
        block: WriterBlock,
        instruction: SectionInstruction,
    ) -> WriterBlock:
        max_chars = instruction.meta.get('max_chars')
        if not isinstance(max_chars, int) or self._ir_prose_chars(block) <= max_chars:
            return block
        prompt = CONDENSE_DRAFT_SECTION_PROMPT.format(
            max_chars=max_chars,
            section_instruction_json=to_prompt_json(instruction),
            draft_section_json=to_prompt_json(block),
        )
        condensed = self._normalize_draft_block(
            self._call_llm_structured(prompt, WriterBlock),
            instruction,
        )
        if self._ir_prose_chars(condensed) > max_chars:
            raise ValueError(f'Condensed draft section still exceeds max_chars={max_chars}.')
        return condensed

    def _attach_section_media(
        self,
        draft_block: WriterBlock,
        instruction: SectionInstruction,
        visual_plan: Any,
        media_assets: Any,
    ) -> WriterBlock:
        resolved_plan = self._unified_optional_model(visual_plan, VisualPlan) or VisualPlan()
        library = self._unified_optional_model(media_assets, MediaAssetLibrary)
        if library is None:
            return draft_block
        section_id = instruction.content_ref.node_id
        needs = [
            need for need in resolved_plan.instructions
            if need.content_ref.node_id == section_id
        ]
        needs_by_id = {need.need_id: need for need in needs}
        block_by_id = {block.node_id: block for block in draft_block.iter_blocks()}
        for item in instruction.meta.get('cross_references') or []:
            if not item.get('must_create') or item.get('kind') != 'image':
                continue
            target = str(item.get('target'))
            need = needs_by_id.get(target)
            if need is None:
                continue
            asset_ids = [
                asset_id for asset_id in library.visual_need_asset_ids.get(need.need_id, [])
                if asset_id in library.assets
            ]
            if len(asset_ids) != 1:
                continue
            image = block_by_id.get(target)
            if image is not None:
                if image.type == 'image' and not any(
                    reference.get('type') == 'media_asset' and reference.get('id')
                    for reference in image.references or []
                ):
                    image.references = [{'type': 'media_asset', 'id': asset_ids[0]}]
                continue
            image = WriterBlock(
                node_id=target,
                type='image',
                content=strip_caption_numbering(str(item.get('caption') or '图')),
                stage='draft',
                references=[{'type': 'media_asset', 'id': asset_ids[0]}],
            )
            draft_block.children.append(image)
            block_by_id[target] = image
        return draft_block

    def _condense_markdown_section_if_needed(
        self,
        body: str,
        instruction: SectionInstruction,
    ) -> str:
        max_chars = instruction.meta.get('max_chars')
        if not isinstance(max_chars, int) or self._text_chars(body) <= max_chars:
            return body
        condensed = self._call_llm_text(CONDENSE_DRAFT_SECTION_MARKDOWN_PROMPT.format(
            max_chars=max_chars,
            section_instruction_json=to_prompt_json(instruction),
            draft_body=body,
        )).strip()
        if self._text_chars(condensed) > max_chars:
            raise ValueError(f'Condensed draft section still exceeds max_chars={max_chars}.')
        return condensed

    @classmethod
    def _ir_prose_chars(cls, block: WriterBlock) -> int:
        return sum(
            cls._text_chars(item.content)
            for item in block.iter_blocks()
            if item is not block and item.type != 'heading'
        )

    @staticmethod
    def _text_chars(text: str) -> int:
        return len(re.sub(r'\s+', '', text))

    def _markdown_draft_prompt(
        self,
        task: WritingTask,
        instruction: SectionInstruction,
        context: WritingContext,
        previous_blocks: Any,
        visual_plan: Any = None,
    ) -> str:
        previous_markdown = self._unified_previous_markdown(previous_blocks)
        resolved_visual_plan = self._unified_optional_model(visual_plan, VisualPlan) or VisualPlan()
        key = (tuple(instruction.content_ref.heading_path), instruction.content_ref.occurrence)
        section_visual_needs = {
            'visual_needs': [
                {
                    'need_id': need.need_id,
                    'visual_type': need.visual_type,
                    'purpose': need.purpose,
                    'required': need.required,
                }
                for need in resolved_visual_plan.instructions
                if (tuple(need.content_ref.heading_path), need.content_ref.occurrence) == key
            ],
        }
        return GENERATE_DRAFT_SECTION_MARKDOWN_PROMPT.format(
            task_json=to_prompt_json(task),
            section_instruction_json=to_prompt_json(instruction),
            context_json=to_prompt_json(context),
            previous_markdown=previous_markdown or '(none)',
            section_visual_needs_json=to_prompt_json(section_visual_needs),
        )

    def _finalize_markdown_draft_section(
        self,
        body: str,
        instruction: SectionInstruction,
        result_extra: Dict[str, Any],
    ) -> dict:
        body = body.strip()
        if not body:
            raise ValueError('Markdown draft section body must not be empty.')
        body = self._normalize_markdown_draft_body(body)
        body = self._strip_system_section_anchors(body)
        if not body:
            raise ValueError('Markdown draft section body contains only headings, no content.')
        heading = self._markdown_draft_heading(instruction)
        body = self._normalize_markdown_cross_references(body, instruction)
        prefix = self._markdown_draft_prefix(instruction, heading)
        markdown = f'{prefix}{body}\n'
        self._validate_markdown_draft_section(markdown, instruction)

        filename = self._safe_artifact_component(instruction.instruction_id)
        path = self._write_markdown_artifact(f'draft_block/{filename}.md', markdown)
        return make_markdown_tool_result(
            path=path,
            step_name='generate_draft_section',
            artifact_key='draft_block',
            summary='Generated draft section as Markdown.',
            counts={'characters': len(markdown)},
            extra=result_extra,
        ).model_dump()

    @staticmethod
    def _markdown_draft_heading(instruction: SectionInstruction) -> str:
        heading_level = instruction.meta.get(
            'outline_heading_level',
            len(instruction.content_ref.heading_path),
        )
        if heading_level != 2:
            raise ValueError('Markdown draft sections must target an H2 outline section.')
        return f'## {instruction.section_title.strip()}'

    @staticmethod
    def _markdown_draft_prefix(instruction: SectionInstruction, heading: str) -> str:
        outline_node_id = instruction.meta.get('outline_node_id')
        if outline_node_id:
            return f'<a id="block-{outline_node_id}"></a>\n\n{heading}\n\n'
        return f'{heading}\n\n'

    def _draft_stream_idle_timeout(self, idle_timeout: Optional[float]) -> float:
        value: Any = idle_timeout
        if value is None:
            value = getattr(self.llm, '_timeout', None)
        if isinstance(value, (tuple, list)):
            value = value[-1] if value else None
        if value is None:
            value = 180.0
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError('idle_timeout must be a positive number.')
        return float(value)

    @staticmethod
    def _draft_result_extra(
        task: WritingTask,
        instruction: SectionInstruction,
        context: WritingContext,
        representation: str,
    ) -> Dict[str, Any]:
        return {
            'representation': representation,
            'task_id': task.task_id,
            'context_id': context.context_id,
            'instruction_id': instruction.instruction_id,
            'content_ref': instruction.content_ref.model_dump(
                exclude_none=True,
                exclude_defaults=True,
            ),
            'outline_title': instruction.meta.get('outline_title'),
        }

    def generate_draft_document(
        self, draft_blocks: Any, context: Any,
        outline: Any = None, title: Any = None,
    ) -> dict:
        blocks = self._unified_draft_blocks(draft_blocks)
        if not blocks:
            raise ValueError('draft_blocks must contain at least one draft section.')
        writing_context = self._unified_model(context, WritingContext)

        if all(isinstance(block, WriterBlock) for block in blocks):
            return self._generate_ir_draft_document(blocks, writing_context, outline, title)

        if not all(isinstance(block, str) for block in blocks):
            raise ValueError('draft_blocks must not mix WriterBlock and Markdown sections.')

        return self._generate_markdown_draft_document(blocks, writing_context, outline, title)

    def _generate_ir_draft_document(
        self, blocks: List[WriterBlock | str],
        context: WritingContext, outline: Any, title: Any,
    ) -> dict:
        writing_outline = None
        if outline is not None:
            writing_outline = self._unified_document(outline)
            if not isinstance(writing_outline, WriterDocument):
                raise ValueError('IR draft blocks require a WriterDocument outline.')
        writer_blocks = [block for block in blocks if isinstance(block, WriterBlock)]
        for block in writer_blocks:
            for item in block.iter_blocks():
                item.stage = 'draft'
                if item.type == 'heading':
                    item.content = strip_heading_numbering(item.content)
                elif item.type == 'image':
                    item.content = strip_caption_numbering(item.content)
        draft_document = WriterDocument(
            document_id=f'draft-document-{context.context_id}',
            stage='draft',
            title=str(title) if title is not None else writing_outline.title if writing_outline else '',
            blocks=writer_blocks,
            ui_editable=True,
            metadata={
                'source': 'generate_draft_document',
                'context_id': context.context_id,
                'outline_id': writing_outline.document_id if writing_outline else None,
                'outline_title': writing_outline.title if writing_outline else None,
            },
        )
        compute_numbering(build_numbering_view_from_ir(draft_document))
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
            extra={'representation': 'ir'},
            artifact_meta={
                'context_id': context.context_id,
                'doc_id': context.doc_id,
                'outline_id': writing_outline.document_id if writing_outline else None,
                'outline_title': writing_outline.title if writing_outline else None,
                'draft_section_count': len(draft_document.blocks),
            },
        )
        return result.model_dump()

    def _generate_markdown_draft_document(
        self, blocks: List[WriterBlock | str],
        context: WritingContext, outline: Any, title: Any,
    ) -> dict:
        section_markdown = [block for block in blocks if isinstance(block, str)]
        section_titles = [self._markdown_draft_section_title(section) for section in section_markdown]
        content_refs = []
        if outline is None:
            document_title = str(title or '').strip()
            if not document_title:
                raise ValueError('Markdown draft assembly requires an outline or document title.')
        else:
            writing_outline = self._unified_document(outline)
            if not isinstance(writing_outline, str):
                raise ValueError('Markdown draft sections require a Markdown outline.')
            outline_title, outline_targets = get_markdown_outline_targets(writing_outline)
            if title is not None and str(title).strip() != outline_title:
                raise ValueError('Markdown draft title must match the outline H1 title.')
            document_title = outline_title
            target_index = 0
            for section_title in section_titles:
                while target_index < len(outline_targets) \
                        and outline_targets[target_index][1][-1] != section_title:
                    target_index += 1
                if target_index == len(outline_targets):
                    raise ValueError(
                        'Markdown draft sections must match the outline headings in order, '
                        f'including occurrence: {section_title!r}.'
                    )
                _, heading_path, occurrence, _ = outline_targets[target_index]
                content_refs.append({
                    'heading_path': heading_path,
                    'occurrence': occurrence,
                })
                target_index += 1

        markdown = f'# {document_title}\n\n' + '\n\n'.join(
            section.strip() for section in section_markdown
        )
        markdown = markdown.rstrip() + '\n'
        markdown = self._ensure_markdown_outline_anchors(markdown)
        compute_numbering(build_numbering_view_from_markdown(markdown))
        assembled_sections = [
            section for section in parse_markdown_sections(markdown)
            if section[0] == 2
        ]
        assembled_targets = [section[1][-1] for section in assembled_sections]
        if assembled_targets != section_titles:
            raise ValueError('Assembled Markdown draft does not preserve section order.')
        if outline is None:
            content_refs = [
                {'heading_path': heading_path, 'occurrence': occurrence}
                for _, heading_path, occurrence, _ in assembled_sections
            ]

        path = self._write_markdown_artifact('draft_document.md', markdown)
        return make_markdown_tool_result(
            path=path,
            step_name='generate_draft_document',
            artifact_key='draft_document',
            summary='Generated draft document as Markdown.',
            counts={
                'draft_sections': len(section_markdown),
                'characters': len(markdown),
            },
            extra={
                'representation': 'markdown',
                'context_id': context.context_id,
                'doc_id': context.doc_id,
                'outline_title': document_title,
                'content_refs': content_refs,
            },
        ).model_dump()

    def generate_final_document(
        self, draft: Any, context: Any,
        output_format: str = 'markdown',
    ) -> dict:
        if output_format != 'markdown':
            raise ValueError('Only markdown output is supported for now.')

        writing_context = self._unified_model(context, WritingContext)
        draft_document = self._unified_document(draft)
        if isinstance(draft_document, str):
            content = draft_document.rstrip() + '\n'
            if not content.strip() or not parse_markdown_sections(content):
                raise ValueError('Markdown draft document must contain at least one heading.')
            path = self._write_markdown_artifact('final_document.md', content)
            result = make_markdown_tool_result(
                path=path,
                step_name='generate_final_document',
                artifact_key='final_document',
                summary='Generated final document as Markdown.',
                counts={
                    'characters': len(content),
                    'draft_sections': len([
                        section
                        for section in parse_markdown_sections(content)
                        if section[0] == 2
                    ]),
                },
                extra={
                    'representation': 'markdown',
                    'context_id': writing_context.context_id,
                    'doc_id': writing_context.doc_id,
                    'output_format': output_format,
                },
            ).model_dump()
            result['output_file_path'] = path
            return result

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
            extra={'representation': 'ir'},
            artifact_meta={
                'context_id': writing_context.context_id,
                'doc_id': writing_context.doc_id,
                'draft_id': draft_document.document_id,
                'output_format': output_format,
            },
        )
        output_file_path = self._write_markdown_artifact('writing_output.md', content)
        dumped = result.model_dump()
        dumped['output_file_path'] = output_file_path
        return dumped

    @staticmethod
    def _instruction_representation(
        instruction: SectionInstruction,
    ) -> str:
        ref = instruction.content_ref
        if ref.node_id and not ref.heading_path and not ref.placeholder_id:
            return 'ir'
        if ref.heading_path and not ref.node_id and not ref.placeholder_id:
            return 'markdown'
        raise ValueError('Section instruction must contain exactly one supported content locator.')

    def _unified_previous_markdown(self, value: Any) -> str:
        if value is None:
            return ''
        values = value if isinstance(value, (list, tuple)) else [value]
        sections = [self._unified_section(item) for item in values]
        if not all(isinstance(section, str) for section in sections):
            raise ValueError('Markdown previous_blocks must contain only Markdown sections.')
        return '\n\n'.join(section.strip() for section in sections if isinstance(section, str))

    @staticmethod
    def _normalize_markdown_draft_body(body: str) -> str:
        '''Strip leading H1/H2 and downgrade stray H1/H2 to H3.'''
        lines = body.split('\n')
        result: List[str] = []
        in_fence = False
        seen_content = False

        for line in lines:
            stripped = line.strip()
            if stripped[:3] in ('```', '~~~'):
                in_fence = not in_fence
                result.append(line)
                seen_content = True
                continue
            if in_fence:
                result.append(line)
                continue

            m = re.match(r'^(#{1,2})\s+(.+?)\s*$', line)
            if m:
                title = strip_heading_numbering(m.group(2))
                if seen_content:
                    result.append(f'### {title}')
                continue

            subheading = re.match(r'^(#{3,6})\s+(.+?)\s*$', line)
            if subheading:
                result.append(
                    f'{subheading.group(1)} '
                    f'{strip_heading_numbering(subheading.group(2))}'
                )
                seen_content = True
                continue

            line = re.sub(
                r'!\[([^\]]*)\]',
                lambda image: f'![{strip_caption_numbering(image.group(1))}]',
                line,
            )

            if stripped:
                seen_content = True
            result.append(line)

        return '\n'.join(result).strip()

    @staticmethod
    def _strip_system_section_anchors(body: str) -> str:
        '''Remove LLM-created section anchors; the prefix emits the only section anchor.'''
        def remove_anchor(match: re.Match[str]) -> str:
            return '' if match.group(1).startswith('block-sec-') else match.group(0)

        return WriterDraftingTools._MARKDOWN_ANCHOR_RE.sub(remove_anchor, body)

    @staticmethod
    def _validate_markdown_draft_section(
        markdown: str,
        instruction: SectionInstruction,
    ) -> None:
        sections = parse_markdown_sections(markdown)
        top_sections = [section for section in sections if section[0] <= 2]
        if len(top_sections) != 1 or top_sections[0][0] != 2:
            raise ValueError('Markdown draft section must contain exactly one H2 root heading.')
        if top_sections[0][1][-1] != strip_heading_numbering(instruction.content_ref.heading_path[-1]):
            raise ValueError('Markdown draft section heading does not match its content_ref.')

    @staticmethod
    def _markdown_draft_section_title(markdown: str) -> str:
        sections = parse_markdown_sections(markdown)
        top_sections = [section for section in sections if section[0] <= 2]
        if len(top_sections) != 1 or top_sections[0][0] != 2:
            raise ValueError('Each Markdown draft section must contain exactly one H2 root heading.')
        if not top_sections[0][3].strip() and len(sections) == 1:
            raise ValueError('Markdown draft section body must not be empty.')
        return top_sections[0][1][-1]

    def _normalize_draft_block(
        self,
        draft_block: WriterBlock,
        instruction: SectionInstruction,
        *,
        allow_deferred_create: bool = False,
    ) -> WriterBlock:
        section_id = instruction.content_ref.node_id
        if not section_id:
            raise ValueError('IR draft section requires content_ref.node_id.')
        draft_block.node_id = section_id
        draft_block.stage = 'draft'
        draft_block.type = 'heading'
        draft_block.content = strip_heading_numbering(instruction.section_title)
        draft_block.numbering['level'] = 1
        for block in draft_block.iter_blocks():
            block.stage = 'draft'
            if block.type == 'heading':
                block.content = strip_heading_numbering(block.content)
            elif block.type == 'image':
                block.content = strip_caption_numbering(block.content)
        draft_block.references = [dict(reference) for reference in instruction.references]
        self._normalize_ir_cross_references(
            draft_block, instruction, allow_deferred_create=allow_deferred_create,
        )
        return draft_block

    @staticmethod
    def _normalize_ir_cross_references(
        draft_block: WriterBlock,
        instruction: SectionInstruction,
        *,
        allow_deferred_create: bool = False,
    ) -> None:
        references = [
            item for item in instruction.meta.get('cross_references') or []
            if isinstance(item, dict)
        ]
        allowed_targets = {str(item.get('target')) for item in references}
        block_by_id = {block.node_id: block for block in draft_block.iter_blocks()}
        for item in references:
            if not item.get('must_create'):
                continue
            target = str(item.get('target'))
            target_block = block_by_id.get(target)
            if target_block is None or target_block.type != str(item.get('kind')):
                if allow_deferred_create or not item.get('required', True):
                    continue
                raise ValueError(f'Missing created cross-reference target {target!r}.')
            if target_block.type == 'image':
                media_ids = [
                    reference.get('id') for reference in target_block.references
                    if reference.get('type') == 'media_asset' and reference.get('id')
                ]
                if len(media_ids) != 1:
                    if not allow_deferred_create:
                        raise ValueError(
                            f'Created image {target!r} requires exactly one media_asset reference.'
                        )

        found_targets: set[str] = set()
        for block in draft_block.iter_blocks():
            for span in block.spans:
                link = span.style.get('link')
                if not isinstance(link, dict) or link.get('type') != 'internal_ref':
                    continue
                target = link.get('target_node_id')
                if target not in allowed_targets:
                    continue
                span.text = ''
                found_targets.add(str(target))
            if block.spans:
                block.content = ''.join(span.text for span in block.spans)

        missing = [
            str(item.get('target')) for item in references
            if item.get('required', True)
            and str(item.get('target')) not in found_targets
            and not (
                allow_deferred_create
                and item.get('must_create')
                and item.get('kind') == 'image'
            )
        ]
        if missing:
            raise ValueError(f'Missing required cross-references: {missing!r}.')

    @classmethod
    def _normalize_markdown_cross_references(
        cls,
        body: str,
        instruction: SectionInstruction,
    ) -> str:
        references = [
            item for item in instruction.meta.get('cross_references') or []
            if isinstance(item, dict)
        ]
        allowed_targets = {str(item.get('target')) for item in references}
        found_targets: set[str] = set()
        found_anchors: set[str] = set()
        created_targets = {
            str(item.get('target')): item
            for item in references
            if item.get('must_create')
        }
        media_lines: Dict[str, int] = {}
        output: List[str] = []
        fence: str | None = None
        for line in body.splitlines():
            fence_match = re.match(r'^\s*(```+|~~~+)', line)
            if fence_match:
                marker = fence_match.group(1)[0]
                if fence is None:
                    fence = marker
                elif fence == marker:
                    fence = None
                output.append(line)
                continue
            if fence is not None:
                output.append(line)
                continue
            raw_anchors = cls._MARKDOWN_ANCHOR_RE.findall(line)
            anchors = []
            for raw_target in raw_anchors:
                if raw_target.startswith('block-'):
                    target = raw_target[len('block-'):]
                    if target in allowed_targets:
                        anchors.append(target)
                elif raw_target in allowed_targets:
                    anchors.append(raw_target)
            found_anchors.update(f'block-{target}' for target in anchors)
            for target in cls._MARKDOWN_MEDIA_PLACEHOLDER_RE.findall(line):
                if target in media_lines:
                    raise ValueError(f'Duplicate media placeholder {target!r}.')
                if target not in created_targets:
                    raise ValueError(f'Unplanned media placeholder {target!r}.')
                media_lines[target] = len(output)

            def replace_link(match: re.Match[str]) -> str:
                raw_target = match.group(2)
                if raw_target.startswith('block-'):
                    target = raw_target[len('block-'):]
                    if target not in allowed_targets:
                        return match.group(0)
                    found_targets.add(target)
                    return f'[](#block-{target})'
                if raw_target not in allowed_targets:
                    return match.group(0)
                found_targets.add(raw_target)
                return f'[](#block-{raw_target})'

            line = cls._MARKDOWN_INTERNAL_LINK_RE.sub(replace_link, line)
            output.append(line)

        for item in references:
            target = str(item.get('target'))
            if item.get('must_create'):
                if target not in media_lines:
                    if item.get('required', True):
                        raise ValueError(f'Missing planned media placeholder {target!r}.')
                    continue
                if f'block-{target}' not in found_anchors:
                    pattern = re.compile(
                        r'!\[[^\]]*\]\(media-placeholder://'
                        + re.escape(target) + r'\)'
                    )
                    media = pattern.search(output[media_lines[target]])
                    if media is None:
                        raise ValueError(f'Invalid media placeholder {target!r}.')
                    start = media.start()
                    output[media_lines[target]] = (
                        output[media_lines[target]][:start]
                        + f'<a id="block-{target}"></a>'
                        + output[media_lines[target]][start:]
                    )
                    found_anchors.add(f'block-{target}')
            if item.get('required', True) and target not in found_targets:
                raise ValueError(f'Missing required cross-reference {target!r}.')
        return '\n'.join(output).strip()

    @staticmethod
    def _ensure_markdown_outline_anchors(markdown: str) -> str:
        output: List[str] = []
        pending_anchors: List[str] = []
        counters: List[int] = []
        fence: str | None = None
        for line in markdown.splitlines():
            fence_match = re.match(r'^\s*(```+|~~~+)', line)
            if fence_match:
                marker = fence_match.group(1)[0]
                if fence is None:
                    fence = marker
                elif fence == marker:
                    fence = None
                output.append(line)
                continue
            if fence is not None:
                output.append(line)
                continue

            anchors = [
                target
                for target in WriterDraftingTools._MARKDOWN_ANCHOR_RE.findall(line)
                if target.startswith('block-sec-')
            ]
            if anchors:
                pending_anchors.extend(anchors)
                output.append(line)
                continue
            heading = re.match(r'^(#{2,6})\s+(.+?)\s*$', line)
            if heading:
                depth = len(heading.group(1)) - 1
                counters = counters[:depth]
                counters.extend([0] * (depth - len(counters)))
                counters[-1] += 1
                expected = 'block-sec-' + '-'.join(
                    f'{value:03d}' for value in counters
                )
                if not pending_anchors:
                    output.append(f'<a id="{expected}"></a>')
                elif pending_anchors[0] != expected:
                    raise ValueError(
                        f'Unexpected heading anchor {pending_anchors[0]!r}; expected {expected!r}.'
                    )
                pending_anchors = []
            elif line.strip() and pending_anchors:
                pending_anchors = []
            output.append(line)
        return '\n'.join(output)

    @staticmethod
    def _media_assets_for_section(
        instruction: SectionInstruction,
        visual_plan: VisualPlan,
        library: Optional[MediaAssetLibrary],
    ) -> Dict[str, Any]:
        node_id = instruction.content_ref.node_id
        needs = [need for need in visual_plan.instructions if need.content_ref.node_id == node_id]
        asset_ids = {
            asset_id
            for need in needs
            for asset_id in (library.visual_need_asset_ids.get(need.need_id, []) if library else [])
        }
        return {
            'visual_needs': [
                {
                    'need_id': need.need_id,
                    'visual_type': need.visual_type,
                    'purpose': need.purpose,
                    'required': need.required,
                }
                for need in needs
            ],
            'assets': [
                {
                    'media_asset_id': asset.media_asset_id,
                    'asset_type': asset.asset_type,
                    'caption': asset.caption,
                    'summary': asset.summary,
                }
                for asset_id, asset in (library.assets.items() if library else [])
                if asset_id in asset_ids
            ],
            'visual_need_asset_ids': {
                need.need_id: library.visual_need_asset_ids.get(need.need_id, []) if library else []
                for need in needs
            },
        }

    def _unified_draft_blocks(self, value: Any) -> List[WriterBlock | str]:
        if value is None:
            return []
        if isinstance(value, WriterBlock):
            return [value]
        if isinstance(value, WriterDocument):
            return list(value.blocks)
        if isinstance(value, str):
            return [self._unified_section(value)]
        if isinstance(value, dict):
            if 'blocks' in value:
                return [WriterBlock.model_validate(block) for block in value['blocks']]
            return [WriterBlock.model_validate(value)]
        if isinstance(value, (list, tuple)):
            blocks: List[WriterBlock | str] = []
            for item in value:
                blocks.extend(self._unified_draft_blocks(item))
            return blocks
        raise TypeError(
            'Expected WriterBlock, WriterDocument, Markdown, list, dict, or artifact path, '
            f'got {type(value).__name__}.'
        )

    @staticmethod
    def _safe_artifact_component(value: str) -> str:
        component = re.sub(r'[^\w.-]+', '_', value, flags=re.UNICODE).strip('._')
        return component or 'section'
