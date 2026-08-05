from __future__ import annotations
from typing import Any, Dict, List, Literal

from .base import WriterToolBase
from ..data_models.context import WritingContext
from ..data_models.multimodal import VisualPlan
from ..data_models.resource import ResourceProfile
from ..data_models.task import WritingTask
from ..data_models.writer_ir import WriterBlock, WriterDocument
from ..data_models.planning import SectionInstruction, SectionInstructionList
from ..prompts import (
    GENERATE_OUTLINE_MARKDOWN_PROMPT,
    GENERATE_OUTLINE_PROMPT,
    GENERATE_SECTION_INSTRUCTIONS_PROMPT,
    GENERATE_VISUAL_PLAN_PROMPT,
)
from ..utils import (
    get_markdown_outline_targets,
    make_markdown_tool_result,
    parse_markdown_sections,
    to_prompt_json,
)


class WriterPlanningTools(WriterToolBase):
    __public_apis__ = [
        'generate_outline',
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
