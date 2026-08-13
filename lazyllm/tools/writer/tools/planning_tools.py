from __future__ import annotations
from typing import Any, Dict, List, Literal

from .base import WriterToolBase
from .stream_tools import DraftPreviewStream, build_outline_stream
from ..data_models.context import WritingContext
from ..data_models.multimodal import VisualPlan, _VISUAL_STRATEGY_ORDER
from ..data_models.resource import ResourceProfile
from ..data_models.task import WritingTask
from ..data_models.writer_ir import ContentRef, WriterBlock, WriterDocument
from ..data_models.planning import SectionInstruction, SectionInstructionList
from ..prompts import (
    GENERATE_OUTLINE_MARKDOWN_PROMPT,
    GENERATE_OUTLINE_PROMPT,
    GENERATE_REWRITE_OUTLINE_PROMPT,
    GENERATE_REWRITE_SECTION_INSTRUCTIONS_PROMPT,
    GENERATE_SECTION_INSTRUCTIONS_PROMPT,
    GENERATE_VISUAL_PLAN_PROMPT,
)
from ..utils import (
    get_markdown_outline_targets,
    make_markdown_tool_result,
    parse_markdown_sections,
    render_block_markdown,
    render_document_markdown,
    to_prompt_json,
)


class WriterPlanningTools(WriterToolBase):
    __public_apis__ = [
        'generate_outline',
        'generate_rewrite_outline',
        'generate_rewrite_section_instructions',
        'generate_section_instructions',
    ]

    def generate_outline(
        self,
        task: Any,
        context: Any,
        resource_profiles: Any = None,
        execution_results: Any = None,
        representation: Literal['ir', 'markdown'] | None = None,
    ) -> dict:
        writing_task = self._unified_model(task, WritingTask)
        writing_context = self._unified_model(context, WritingContext)
        profiles = self._unified_models(resource_profiles, ResourceProfile)
        execution_data = self._unified_raw_data(execution_results)
        resolved_representation = self._resolve_representation(writing_task, representation)

        if resolved_representation == 'markdown':
            prompt = GENERATE_OUTLINE_MARKDOWN_PROMPT.format(
                task_json=to_prompt_json(writing_task),
                context_json=to_prompt_json(writing_context),
                resource_profiles_json=to_prompt_json(profiles),
                execution_results_json=to_prompt_json(execution_data),
            )
            outline = self._call_llm_text(prompt).strip() + '\n'
            _, targets = get_markdown_outline_targets(outline)
            path = self._write_markdown_artifact('outline.md', outline)
            return make_markdown_tool_result(
                path=path,
                step_name='generate_outline',
                artifact_key='outline',
                summary='Generated writing outline as Markdown.',
                counts={
                    'top_level_sections': len(targets),
                    'outline_nodes': len(parse_markdown_sections(outline)),
                    'characters': len(outline),
                },
                extra={
                    'representation': 'markdown',
                    'task_id': writing_task.task_id,
                    'context_id': writing_context.context_id,
                    'resource_profile_count': len(profiles),
                    'has_execution_results': execution_data is not None,
                },
            ).model_dump()

        document_id = f'{writing_context.context_id}-outline'
        prompt = GENERATE_OUTLINE_PROMPT.format(
            task_json=to_prompt_json(writing_task),
            document_id=document_id,
            context_json=to_prompt_json(writing_context),
            resource_profiles_json=to_prompt_json(profiles),
            execution_results_json=to_prompt_json(execution_data),
        )
        outline = self._call_llm_structured(prompt, WriterDocument)
        outline.document_id = document_id
        outline = self._normalize_outline(outline, writing_task, writing_context, profiles)

        result = self._save_artifacts(
            {'outline': outline},
            step_name='generate_outline',
            primary_key='outline',
            context_key=None,
            summary='Generated writing outline.',
            counts={
                'top_level_sections': len(outline.blocks),
                'outline_nodes': len(list(outline.iter_blocks())),
            },
            extra={'representation': 'ir'},
            artifact_meta={
                'task_id': writing_task.task_id,
                'context_id': writing_context.context_id,
                'resource_profile_count': len(profiles),
                'has_execution_results': execution_data is not None,
            },
        )
        return result.model_dump()

    def stream_outline(
        self,
        task: Any,
        context: Any,
        resource_profiles: Any = None,
        execution_results: Any = None,
        representation: Literal['ir', 'markdown'] | None = None,
        *,
        idle_timeout: float | None = None,
    ) -> DraftPreviewStream:
        return build_outline_stream(
            self,
            task,
            context,
            resource_profiles,
            execution_results,
            representation,
            idle_timeout=idle_timeout,
        )

    def generate_rewrite_outline(
        self,
        task: Any,
        source_document: Any,
        context: Any,
    ) -> dict:
        writing_task = self._unified_model(task, WritingTask)
        source = self._unified_document(source_document)
        if not isinstance(source, WriterDocument):
            raise TypeError('generate_rewrite_outline requires a WriterDocument source.')
        writing_context = self._unified_model(context, WritingContext)
        document_id = f'{writing_context.context_id}-rewrite-outline'
        prompt = GENERATE_REWRITE_OUTLINE_PROMPT.format(
            task_json=to_prompt_json(writing_task),
            document_id=document_id,
            context_json=to_prompt_json(writing_context),
            source_document_json=render_document_markdown(source),
        )
        outline = self._call_llm_structured(prompt, WriterDocument)
        outline.document_id = document_id
        outline = self._normalize_outline(outline, writing_task, writing_context, [])
        return self._save_artifacts(
            {'outline': outline},
            step_name='generate_rewrite_outline',
            primary_key='outline',
            context_key=None,
            summary='Generated an internal outline for a complete document rewrite.',
            counts={
                'top_level_sections': len(outline.blocks),
                'outline_nodes': len(list(outline.iter_blocks())),
            },
            extra={'representation': 'ir'},
            artifact_meta={
                'task_id': writing_task.task_id,
                'context_id': writing_context.context_id,
                'source_document_id': source.document_id,
            },
        ).model_dump()

    def generate_rewrite_section_instructions(
        self,
        task: Any,
        source_document: Any,
        context: Any,
    ) -> dict:
        writing_task = self._unified_model(task, WritingTask)
        source = self._unified_document(source_document)
        writing_context = self._unified_model(context, WritingContext)
        representation, source_title, source_sections = self._rewrite_source_sections(source)
        prompt = GENERATE_REWRITE_SECTION_INSTRUCTIONS_PROMPT.format(
            representation=representation,
            task_json=to_prompt_json(writing_task),
            source_title=source_title,
            source_sections_json=to_prompt_json(source_sections),
            context_json=to_prompt_json(writing_context),
        )
        instructions = self._call_llm_structured(prompt, SectionInstructionList)
        instructions = self._normalize_rewrite_section_instructions(
            instructions,
            representation,
            source_title,
            source_sections,
            writing_context,
        )
        return self._save_artifacts(
            {'section_instructions': instructions},
            step_name='generate_rewrite_section_instructions',
            primary_key='section_instructions',
            context_key=None,
            summary='Generated section instructions for a complete document rewrite.',
            counts={'section_instructions': len(instructions.instructions)},
            extra={
                'representation': representation,
                'document_title': instructions.meta['document_title'],
            },
            artifact_meta={
                'context_id': writing_context.context_id,
                'source_document_id': source.document_id if isinstance(source, WriterDocument) else None,
            },
        ).model_dump()

    def generate_visual_plan(self, task: Any, outline: Any, context: Any) -> dict:
        writing_task = self._unified_model(task, WritingTask)
        writing_outline = self._unified_document(outline)
        writing_context = self._unified_model(context, WritingContext)
        visual_plan = VisualPlan()
        if isinstance(writing_outline, WriterDocument):
            prompt = GENERATE_VISUAL_PLAN_PROMPT.format(
                task_json=to_prompt_json(writing_task),
                context_json=to_prompt_json(writing_context),
                outline_json=to_prompt_json(writing_outline),
            )
            visual_plan = self._normalize_visual_plan(
                self._call_llm_structured(prompt, VisualPlan), writing_outline)
        return self._save_artifacts(
            {'visual_plan': visual_plan},
            step_name='generate_visual_plan',
            primary_key='visual_plan',
            context_key=None,
            summary='Generated visual plan.',
            counts={'visual_instructions': len(visual_plan.instructions)},
            extra={'representation': 'ir' if isinstance(writing_outline, WriterDocument) else 'markdown'},
        ).model_dump()

    def generate_section_instructions(
        self,
        outline: Any,
        context: Any,
        visual_plan: Any = None,
        execution_results: Any = None,
    ) -> dict:
        writing_outline = self._unified_document(outline)
        writing_context = self._unified_model(context, WritingContext)
        execution_data = self._unified_raw_data(execution_results)
        writing_visual_plan = self._unified_optional_model(visual_plan, VisualPlan) or VisualPlan()

        if isinstance(writing_outline, WriterDocument):
            target_blocks = [block for block in writing_outline.blocks if block.type == 'heading']
            target_payload = [
                {
                    'content_ref': {'node_id': block.node_id},
                    'section_title': block.content,
                    'outline_content': block.model_dump(exclude_defaults=True),
                }
                for block in target_blocks
            ]
            outline_payload = writing_outline
            representation = 'ir'
        else:
            _, targets = get_markdown_outline_targets(writing_outline)
            target_payload = [
                {
                    'content_ref': {
                        'heading_path': heading_path,
                        'occurrence': occurrence,
                    },
                    'section_title': heading_path[-1],
                    'outline_heading_level': level,
                    'outline_body': body,
                }
                for level, heading_path, occurrence, body in targets
            ]
            outline_payload = writing_outline
            representation = 'markdown'

        prompt = GENERATE_SECTION_INSTRUCTIONS_PROMPT.format(
            outline_json=to_prompt_json(outline_payload),
            target_outline_blocks_json=to_prompt_json(target_payload),
            context_json=to_prompt_json(writing_context),
            execution_results_json=to_prompt_json(execution_data),
            visual_plan_json=to_prompt_json(writing_visual_plan),
        )
        instruction_list = self._call_llm_structured(prompt, SectionInstructionList)
        if isinstance(writing_outline, WriterDocument):
            instruction_list = self._normalize_ir_section_instructions(
                instruction_list,
                writing_outline,
                target_blocks,
                writing_context,
                execution_data,
            )
            outline_id = writing_outline.document_id
        else:
            instruction_list = self._normalize_markdown_section_instructions(
                instruction_list,
                targets,
                writing_context,
                execution_data,
            )
            outline_id = instruction_list.outline_id

        result = self._save_artifacts(
            {'section_instructions': instruction_list},
            step_name='generate_section_instructions',
            primary_key='section_instructions',
            context_key=None,
            summary='Generated section writing instructions.',
            counts={
                'section_instructions': len(instruction_list.instructions),
            },
            extra={'representation': representation},
            artifact_meta={
                'outline_id': outline_id,
                'context_id': writing_context.context_id,
                'has_execution_results': execution_data is not None,
                'visual_instructions': len(writing_visual_plan.instructions),
            },
        )
        return result.model_dump()

    @staticmethod
    def _normalize_visual_plan(visual_plan: VisualPlan, outline: WriterDocument) -> VisualPlan:
        node_ids = {block.node_id for block in outline.blocks if block.type == 'heading'}
        counts: Dict[str, int] = {}
        for need in visual_plan.instructions:
            ref = need.content_ref
            node_id = ref.node_id
            if (
                not node_id or ref.heading_path or ref.placeholder_id or ref.document_root
                or node_id not in node_ids
            ):
                raise ValueError('Visual plan must target a top-level outline node_id.')
            if not need.purpose.strip():
                raise ValueError(f'Visual need for {node_id!r} has an empty purpose.')
            if need.preferred_strategy is None:
                need.preferred_strategy = _VISUAL_STRATEGY_ORDER[need.visual_type][0]
            counts[node_id] = counts.get(node_id, 0) + 1
            need.need_id = f'visual-{node_id}-{counts[node_id]}'
        return visual_plan

    @staticmethod
    def _resolve_representation(
        task: WritingTask,
        representation: Literal['ir', 'markdown'] | None,
    ) -> Literal['ir', 'markdown']:
        resolved = representation or task.output.get('representation') or 'markdown'
        if resolved not in {'ir', 'markdown'}:
            raise ValueError("representation must be 'ir' or 'markdown'.")
        return resolved

    def _normalize_outline(
        self,
        outline: WriterDocument,
        task: WritingTask,
        context: WritingContext,
        profiles: List[ResourceProfile],
    ) -> WriterDocument:
        outline.stage = 'outline'
        if task.target_document and task.target_document.title:
            outline.title = task.target_document.title
        outline.ui_editable = False

        valid_reference_ids = {profile.resource_id for profile in profiles if profile.resource_id}
        for fact in context.facts:
            if fact.fact_id:
                valid_reference_ids.add(fact.fact_id)
            valid_reference_ids.update(source for source in fact.source if source)

        pending = [(block, 1) for block in reversed(outline.blocks)]
        while pending:
            block, level = pending.pop()
            block.stage = 'outline'
            block.numbering['level'] = level
            block.references = [
                reference
                for reference in block.references
                if reference.get('id') in valid_reference_ids
            ]
            pending.extend(
                (child, level + 1)
                for child in reversed(block.children)
            )

        outline.metadata.setdefault('source', 'llm')
        return outline

    def _normalize_ir_section_instructions(
        self,
        instruction_list: SectionInstructionList,
        outline: WriterDocument,
        target_blocks: List[WriterBlock],
        context: WritingContext,
        execution_results: Any,
    ) -> SectionInstructionList:
        target_by_id = {block.node_id: block for block in target_blocks}
        instruction_by_node_id: Dict[str, SectionInstruction] = {}

        for instruction in instruction_list.instructions:
            ref = instruction.content_ref
            node_id = ref.node_id
            if not node_id or ref.heading_path or ref.placeholder_id:
                raise ValueError('IR section instructions must contain only content_ref.node_id.')
            if node_id in instruction_by_node_id:
                raise ValueError(f'Duplicate section instruction for outline node {node_id!r}.')
            if node_id not in target_by_id:
                raise ValueError(f'Section instruction references unknown outline node {node_id!r}.')
            instruction_by_node_id[node_id] = instruction

        missing_node_ids = [
            block.node_id for block in target_blocks
            if block.node_id not in instruction_by_node_id
        ]
        if missing_node_ids:
            raise ValueError(
                'Missing section instructions for outline nodes: '
                + ', '.join(missing_node_ids)
            )

        instruction_list.outline_id = outline.document_id
        instruction_list.instruction_set_id = f'{outline.document_id}-section-instructions'
        instruction_list.instructions = [
            self._normalize_ir_section_instruction(
                instruction_by_node_id[block.node_id],
                block,
                outline,
                bool(context.facts),
            )
            for block in target_blocks
        ]
        instruction_list.meta.update({
            'source': 'llm',
            'representation': 'ir',
            'outline_id': outline.document_id,
            'outline_title': outline.title,
            'context_id': context.context_id,
            'has_execution_results': execution_results is not None,
        })
        return instruction_list

    def _normalize_markdown_section_instructions(
        self,
        instruction_list: SectionInstructionList,
        targets: List[tuple[int, List[str], int, str]],
        context: WritingContext,
        execution_results: Any,
    ) -> SectionInstructionList:
        target_by_ref = {
            (tuple(heading_path), occurrence): (level, heading_path)
            for level, heading_path, occurrence, _ in targets
        }
        instruction_by_ref: Dict[tuple[tuple[str, ...], int], SectionInstruction] = {}

        for instruction in instruction_list.instructions:
            ref = instruction.content_ref
            if ref.node_id or ref.placeholder_id or not ref.heading_path:
                raise ValueError(
                    'Markdown section instructions must contain only '
                    'content_ref.heading_path and occurrence.'
                )
            key = (tuple(ref.heading_path), ref.occurrence)
            if key in instruction_by_ref:
                raise ValueError(f'Duplicate section instruction for Markdown heading {key!r}.')
            if key not in target_by_ref:
                raise ValueError(f'Section instruction references unknown Markdown heading {key!r}.')
            instruction_by_ref[key] = instruction

        missing_refs = [key for key in target_by_ref if key not in instruction_by_ref]
        if missing_refs:
            raise ValueError(f'Missing section instructions for Markdown headings: {missing_refs!r}.')

        outline_id = f'{context.context_id}-outline-markdown'
        normalized = []
        for level, heading_path, occurrence, _ in targets:
            key = (tuple(heading_path), occurrence)
            instruction = instruction_by_ref[key]
            self._validate_instruction(instruction, '/'.join(heading_path))
            instruction.section_title = heading_path[-1]
            instruction.references = []
            if not context.facts:
                instruction.fact_constraints = []
            instruction.meta.update({
                'representation': 'markdown',
                'outline_heading_level': level,
                'outline_id': outline_id,
                'outline_title': heading_path[0],
            })
            normalized.append(instruction)

        instruction_list.outline_id = outline_id
        instruction_list.instruction_set_id = f'{outline_id}-section-instructions'
        instruction_list.instructions = normalized
        instruction_list.meta.update({
            'source': 'llm',
            'representation': 'markdown',
            'outline_id': outline_id,
            'outline_title': targets[0][1][0],
            'context_id': context.context_id,
            'has_execution_results': execution_results is not None,
        })
        return instruction_list

    @staticmethod
    def _rewrite_source_sections(
        source: WriterDocument | str,
    ) -> tuple[str, str, List[Dict[str, Any]]]:
        if isinstance(source, WriterDocument):
            if not source.blocks:
                raise ValueError('WriterDocument rewrite source must contain at least one block.')
            grouped_blocks: List[List[WriterBlock]] = []
            for block in source.blocks:
                if block.type == 'heading' or not grouped_blocks:
                    grouped_blocks.append([block])
                else:
                    grouped_blocks[-1].append(block)
            return 'ir', source.title or 'Rewrite', [
                {
                    'source_ref': {'node_id': blocks[0].node_id},
                    'section_title': (
                        blocks[0].content or source.title or f'Section {index}'
                    ),
                    'content': '\n\n'.join(
                        render_block_markdown(block, level=2).strip()
                        for block in blocks
                    ),
                    'format': {
                        'type': blocks[0].type,
                        'numbering': blocks[0].numbering,
                        'top_level_types': [block.type for block in blocks],
                        'child_types': [
                            child.type for block in blocks for child in block.children
                        ],
                    },
                }
                for index, blocks in enumerate(grouped_blocks, start=1)
            ]

        parsed = parse_markdown_sections(source)
        title = next(
            (heading_path[-1] for level, heading_path, _, _ in parsed if level == 1),
            'Rewrite',
        )
        sections = [section for section in parsed if section[0] == 2]
        if not sections:
            return 'markdown', title, [{
                'source_ref': {'document_root': True},
                'section_title': title,
                'content': source,
                'format': {'heading_level': 2},
            }]
        return 'markdown', title, [
            {
                'source_ref': {
                    'heading_path': heading_path,
                    'occurrence': occurrence,
                },
                'section_title': heading_path[-1],
                'content': body,
                'format': {'heading_level': level},
            }
            for level, heading_path, occurrence, body in sections
        ]

    @classmethod
    def _normalize_rewrite_section_instructions(
        cls,
        instruction_list: SectionInstructionList,
        representation: str,
        source_title: str,
        source_sections: List[Dict[str, Any]],
        context: WritingContext,
    ) -> SectionInstructionList:
        if not instruction_list.instructions:
            raise ValueError('Rewrite section instructions must contain at least one section.')
        document_title = str(instruction_list.meta.get('document_title') or source_title).strip()
        if not document_title:
            raise ValueError('Rewrite section instructions require a document title.')
        source_by_ref = {
            cls._rewrite_source_ref_key(section['source_ref']): section
            for section in source_sections
        }
        seen_titles: Dict[str, int] = {}
        for index, instruction in enumerate(instruction_list.instructions, start=1):
            cls._validate_instruction(instruction, f'rewrite-section-{index}')
            instruction.instruction_id = instruction.instruction_id.strip() or f'rewrite-instruction-{index}'
            instruction.section_title = instruction.section_title.strip()
            seen_titles[instruction.section_title] = seen_titles.get(instruction.section_title, 0) + 1
            selected_sections = []
            selected_refs = []
            for reference in instruction.references:
                source_section = source_by_ref.get(cls._rewrite_source_ref_key(reference))
                if source_section is not None and source_section not in selected_sections:
                    selected_sections.append(source_section)
                    selected_refs.append(dict(source_section['source_ref']))
            if not selected_sections:
                source_section = source_sections[min(index - 1, len(source_sections) - 1)]
                selected_sections = [source_section]
                selected_refs = [dict(source_section['source_ref'])]
            if representation == 'ir':
                instruction.content_ref = ContentRef(node_id=f'rewrite-section-{index}')
            else:
                instruction.content_ref = ContentRef(
                    heading_path=[document_title, instruction.section_title],
                    occurrence=seen_titles[instruction.section_title],
                )
            instruction.references = []
            if not context.facts:
                instruction.fact_constraints = []
            instruction.meta.update({
                'representation': representation,
                'rewrite': True,
                'document_title': document_title,
                'outline_title': document_title,
                'source_content_refs': selected_refs,
                'source_content': '\n\n'.join(
                    str(section.get('content') or '').strip()
                    for section in selected_sections
                    if str(section.get('content') or '').strip()
                ),
                'source_format': [section.get('format') or {} for section in selected_sections],
            })
            if representation == 'markdown':
                instruction.meta['outline_heading_level'] = 2
            else:
                instruction.meta['outline_node_level'] = 1
        instruction_list.instruction_set_id = f'{context.context_id}-rewrite-section-instructions'
        instruction_list.outline_id = None
        instruction_list.meta.update({
            'source': 'llm',
            'representation': representation,
            'rewrite': True,
            'document_title': document_title,
            'context_id': context.context_id,
        })
        return instruction_list

    @staticmethod
    def _rewrite_source_ref_key(reference: Dict[str, Any]) -> tuple[Any, ...]:
        if reference.get('node_id'):
            return 'node_id', str(reference['node_id'])
        if reference.get('heading_path'):
            return (
                'heading_path',
                tuple(reference['heading_path']),
                int(reference.get('occurrence') or 1),
            )
        if reference.get('document_root'):
            return 'document_root',
        return 'invalid',

    def _normalize_ir_section_instruction(
        self,
        instruction: SectionInstruction,
        block: WriterBlock,
        outline: WriterDocument,
        has_available_facts: bool,
    ) -> SectionInstruction:
        self._validate_instruction(instruction, block.node_id)
        instruction.section_title = block.content
        instruction.references = [dict(reference) for reference in block.references]
        instruction.visual_needs = []
        if not has_available_facts:
            instruction.fact_constraints = []
        instruction.meta.update({
            'representation': 'ir',
            'outline_node_level': block.numbering.get('level'),
            'outline_id': outline.document_id,
            'outline_title': outline.title,
        })
        return instruction

    @staticmethod
    def _validate_instruction(instruction: SectionInstruction, target: str) -> None:
        if not instruction.instruction_id.strip():
            raise ValueError(f'Section instruction for {target!r} has an empty instruction_id.')
        if not instruction.section_goal.strip():
            raise ValueError(f'Section instruction for {target!r} has an empty section_goal.')
