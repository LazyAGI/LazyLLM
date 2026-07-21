from __future__ import annotations
from typing import Any, Dict, List, Optional

from .base import WriterToolBase
from ..data_models.context import WritingContext
from ..data_models.resource import ResourceProfile
from ..data_models.task import WritingTask
from ..data_models.writer_ir import (
    WriterAuthoring,
    WriterBlock,
    WriterConstraints,
    WriterDocument,
)
from ..data_models.planning import SectionInstruction, SectionInstructionList
from ..prompts import GENERATE_OUTLINE_PROMPT, GENERATE_SECTION_INSTRUCTIONS_PROMPT
from ..utils import to_prompt_json


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
    ) -> dict:
        writing_task = self._unified_model(task, WritingTask)
        writing_context = self._unified_model(context, WritingContext)
        profiles = self._unified_models(resource_profiles, ResourceProfile)
        execution_data = self._normalize_execution_results(execution_results)
        document_id_hint = self._default_outline_id(writing_task, writing_context)

        prompt = GENERATE_OUTLINE_PROMPT.format(
            task_json=to_prompt_json(writing_task),
            document_id_hint=document_id_hint,
            context_json=to_prompt_json(writing_context),
            resource_profiles_json=to_prompt_json(profiles),
            execution_results_json=to_prompt_json(execution_data),
        )
        outline = self._call_llm_structured(prompt, WriterDocument)
        outline = self._normalize_outline(outline, writing_task, writing_context, profiles, execution_data)

        result = self._save_artifacts(
            {'outline': outline},
            step_name='generate_outline',
            primary_key='outline',
            context_key=None,
            summary='Generated writing outline.',
            counts={
                'top_level_sections': len(outline.blocks),
                'outline_nodes': self._count_outline_blocks(outline.blocks),
            },
            artifact_meta={
                'task_id': writing_task.task_id,
                'context_id': writing_context.context_id,
                'resource_profile_count': len(profiles),
                'has_execution_results': execution_data is not None,
            },
        )
        return result.model_dump()

    def generate_section_instructions(
        self,
        outline: Any,
        context: Any,
        execution_results: Any = None,
    ) -> dict:
        writing_outline = self._unified_model(outline, WriterDocument)
        writing_context = self._unified_model(context, WritingContext)
        execution_data = self._normalize_execution_results(execution_results)
        target_blocks = writing_outline.blocks

        prompt = GENERATE_SECTION_INSTRUCTIONS_PROMPT.format(
            outline_json=to_prompt_json(writing_outline),
            target_outline_blocks_json=to_prompt_json(target_blocks),
            context_json=to_prompt_json(writing_context),
            execution_results_json=to_prompt_json(execution_data),
        )
        instruction_list = self._call_llm_structured(prompt, SectionInstructionList)
        instruction_list = self._normalize_section_instructions(
            instruction_list,
            writing_outline,
            writing_context,
            execution_data,
        )

        result = self._save_artifacts(
            {'section_instructions': instruction_list},
            step_name='generate_section_instructions',
            primary_key='section_instructions',
            context_key=None,
            summary='Generated section writing instructions.',
            counts={
                'section_instructions': len(instruction_list.instructions),
            },
            artifact_meta={
                'outline_id': writing_outline.document_id,
                'context_id': writing_context.context_id,
                'has_execution_results': execution_data is not None,
            },
        )
        return result.model_dump()

    def _normalize_execution_results(self, execution_results: Any) -> Any:
        return self._unified_raw_data(execution_results)

    def _normalize_outline(
        self,
        outline: WriterDocument,
        task: WritingTask,
        context: WritingContext,
        profiles: List[ResourceProfile],
        execution_results: Any,
    ) -> WriterDocument:
        if len(outline.blocks) < 3:
            raise ValueError('generate_outline must produce at least 3 top-level sections.')

        outline.stage = 'outline'
        outline.document_id = outline.document_id or self._default_outline_id(task, context)
        outline.title = outline.title or self._default_outline_title(task)
        valid_reference_ids = self._valid_reference_ids(context, profiles)
        has_available_facts = self._has_available_facts(context, profiles)
        for index, block in enumerate(outline.blocks, start=1):
            self._normalize_outline_block(
                block,
                level=1,
                fallback_id=f'section-{index}',
                valid_reference_ids=valid_reference_ids,
                has_available_facts=has_available_facts,
            )

        outline.metadata.setdefault('source', 'llm')
        return outline

    def _normalize_outline_block(
        self,
        block: WriterBlock,
        *,
        level: int,
        fallback_id: str,
        valid_reference_ids: set[str],
        has_available_facts: bool,
    ) -> None:
        block.stage = 'outline'
        if not block.type.strip():
            block.type = 'heading'
        block.node_id = block.node_id or fallback_id
        block.numbering['level'] = level
        if block.authoring is None:
            block.authoring = WriterAuthoring()

        block.references = self._filter_references(block.references, valid_reference_ids)
        if not has_available_facts:
            block.authoring.constraints.fact_constraints = []

        for index, child in enumerate(block.children, start=1):
            self._normalize_outline_block(
                child,
                level=level + 1,
                fallback_id=f'{block.node_id}-{index}',
                valid_reference_ids=valid_reference_ids,
                has_available_facts=has_available_facts,
            )

    def _default_outline_id(self, task: WritingTask, context: WritingContext) -> str:
        source_id = task.task_id or context.context_id or 'writer'
        return f'{source_id}-outline'

    def _default_outline_title(self, task: WritingTask) -> str:
        if task.target_document and task.target_document.title:
            return task.target_document.title
        query = ' '.join(task.query.split())
        return query[:80] if query else 'Writing Outline'

    def _count_outline_blocks(self, blocks: List[WriterBlock]) -> int:
        return sum(1 + self._count_outline_blocks(block.children) for block in blocks)

    def _normalize_section_instructions(
        self,
        instruction_list: SectionInstructionList,
        outline: WriterDocument,
        context: WritingContext,
        execution_results: Any,
    ) -> SectionInstructionList:
        target_blocks = outline.blocks
        target_by_id = {block.node_id: block for block in target_blocks}
        instruction_by_node_id: Dict[str, SectionInstruction] = {}

        for instruction in instruction_list.instructions:
            node_id = instruction.outline_node_id
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

        has_available_facts = self._has_available_facts(context)
        normalized = [
            self._normalize_section_instruction(
                instruction_by_node_id[block.node_id],
                block,
                outline,
                has_available_facts,
            )
            for block in target_blocks
        ]

        instruction_list.outline_id = outline.document_id
        instruction_list.instruction_set_id = (
            instruction_list.instruction_set_id
            or f'{outline.document_id}-section-instructions'
        )
        instruction_list.instructions = normalized
        instruction_list.meta.update(
            {
                'source': 'llm',
                'outline_id': outline.document_id,
                'outline_title': outline.title,
                'context_id': context.context_id,
                'has_execution_results': execution_results is not None,
            }
        )
        return instruction_list

    def _normalize_section_instruction(
        self,
        instruction: SectionInstruction,
        block: WriterBlock,
        outline: WriterDocument,
        has_available_facts: bool,
    ) -> SectionInstruction:
        block_constraints = block.authoring.constraints if block.authoring else WriterConstraints()

        if not instruction.instruction_id.strip():
            raise ValueError(f'Section instruction for {block.node_id!r} has an empty instruction_id.')
        if not instruction.section_goal.strip():
            raise ValueError(f'Section instruction for {block.node_id!r} has an empty section_goal.')

        instruction.section_title = block.content
        instruction.references = [dict(reference) for reference in block.references]
        if not instruction.required_points:
            instruction.required_points = list(block_constraints.required_points)
        if not instruction.fact_constraints:
            instruction.fact_constraints = list(block_constraints.fact_constraints)
        if not has_available_facts:
            instruction.fact_constraints = []
        if not instruction.style_constraints:
            instruction.style_constraints = list(block_constraints.style_constraints)
            if block_constraints.pov:
                instruction.style_constraints.append(f'POV: {block_constraints.pov}')
            if block_constraints.tone:
                instruction.style_constraints.append(f'Tone: {block_constraints.tone}')
        if not instruction.relation_constraints:
            instruction.relation_constraints = list(block_constraints.relation_constraints)
        if not instruction.expected_blocks:
            instruction.expected_blocks = self._default_expected_blocks(block, block_constraints)

        instruction.meta.update(
            {
                'outline_node_level': block.numbering.get('level'),
                'outline_node_instruction': block.authoring.instruction if block.authoring else None,
                'outline_id': outline.document_id,
                'outline_title': outline.title,
            }
        )
        return instruction

    def _default_expected_blocks(
        self,
        block: WriterBlock,
        constraints: WriterConstraints,
    ) -> List[str]:
        blocks = [block.content] if block.content else []
        if constraints.required_points:
            blocks.extend(constraints.required_points[:3])
        return blocks

    def _valid_reference_ids(
        self,
        context: WritingContext,
        profiles: Optional[List[ResourceProfile]] = None,
    ) -> set[str]:
        refs: set[str] = set()
        for profile in profiles or []:
            if profile.resource_id:
                refs.add(profile.resource_id)
        for fact in context.facts:
            if fact.fact_id:
                refs.add(fact.fact_id)
            refs.update(source for source in fact.source if source)
        return refs

    def _has_available_facts(
        self,
        context: WritingContext,
        profiles: Optional[List[ResourceProfile]] = None,
    ) -> bool:
        if context.facts:
            return True
        return any(profile.key_facts for profile in profiles or [])

    def _filter_references(
        self,
        references: List[Dict[str, Any]],
        valid_reference_ids: set[str],
    ) -> List[Dict[str, Any]]:
        if not valid_reference_ids:
            return []
        return [
            reference
            for reference in references
            if reference.get('id') in valid_reference_ids
        ]
