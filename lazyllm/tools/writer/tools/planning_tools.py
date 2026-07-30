from __future__ import annotations
from typing import Any, Dict, List

from .base import WriterToolBase
from ..data_models.context import WritingContext
from ..data_models.resource import ResourceProfile
from ..data_models.task import WritingTask
from ..data_models.writer_ir import WriterBlock, WriterDocument
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
        execution_data = self._unified_raw_data(execution_results)
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
        execution_data = self._unified_raw_data(execution_results)
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

        normalized = [
            self._normalize_section_instruction(
                instruction_by_node_id[block.node_id],
                block,
                outline,
                bool(context.facts),
            )
            for block in target_blocks
        ]

        instruction_list.outline_id = outline.document_id
        instruction_list.instruction_set_id = f'{outline.document_id}-section-instructions'
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
        if not instruction.instruction_id.strip():
            raise ValueError(f'Section instruction for {block.node_id!r} has an empty instruction_id.')
        if not instruction.section_goal.strip():
            raise ValueError(f'Section instruction for {block.node_id!r} has an empty section_goal.')

        instruction.section_title = block.content
        instruction.references = [dict(reference) for reference in block.references]
        if not has_available_facts:
            instruction.fact_constraints = []

        instruction.meta.update(
            {
                'outline_node_level': block.numbering.get('level'),
                'outline_id': outline.document_id,
                'outline_title': outline.title,
            }
        )
        return instruction
