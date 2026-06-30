from __future__ import annotations
from typing import Any, List

from .base import WriterToolBase
from ..data_models.context import WritingContext
from ..data_models.resource import ResourceProfile
from ..data_models.task import WritingTask
from ..data_models.writing import (
    OutlineNode,
    SectionInstruction,
    SectionInstructionList,
    WritingOutline,
)
from ..prompts import GENERATE_OUTLINE_PROMPT, GENERATE_SECTION_INSTRUCTIONS_PROMPT
from ..utils import to_prompt_json


class WriterPlanningTools(WriterToolBase):
    __public_apis__ = [
        "generate_outline",
        "generate_section_instructions",
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

        prompt = GENERATE_OUTLINE_PROMPT.format(
            task_json=to_prompt_json(writing_task),
            context_json=to_prompt_json(writing_context),
            resource_profiles_json=to_prompt_json(profiles),
            execution_results_json=to_prompt_json(execution_data),
        )
        outline = self._call_llm_structured(prompt, WritingOutline)
        outline = self._normalize_outline(outline, writing_task, writing_context, profiles, execution_data)

        result = self._save_artifacts(
            {"outline": outline},
            step_name="generate_outline",
            primary_key="outline",
            context_key=None,
            summary="Generated writing outline.",
            counts={
                "top_level_sections": len(outline.nodes),
                "outline_nodes": self._count_outline_nodes(outline.nodes),
            },
            artifact_meta={
                "task_id": writing_task.task_id,
                "context_id": writing_context.context_id,
                "resource_profile_count": len(profiles),
                "has_execution_results": execution_data is not None,
            },
        )
        return result.model_dump()

    def generate_section_instructions(
        self,
        outline: Any,
        context: Any,
        execution_results: Any = None,
    ) -> dict:
        writing_outline = self._unified_model(outline, WritingOutline)
        writing_context = self._unified_model(context, WritingContext)
        execution_data = self._normalize_execution_results(execution_results)
        target_nodes = self._instruction_target_nodes(writing_outline)

        prompt = GENERATE_SECTION_INSTRUCTIONS_PROMPT.format(
            outline_json=to_prompt_json(writing_outline),
            target_outline_nodes_json=to_prompt_json(target_nodes),
            context_json=to_prompt_json(writing_context),
            execution_results_json=to_prompt_json(execution_data),
        )
        instruction_list = self._call_llm_structured(prompt, SectionInstructionList)
        instruction_list = self._normalize_section_instructions(
            instruction_list,
            writing_outline,
            writing_context,
            target_nodes,
            execution_data,
        )

        result = self._save_artifacts(
            {"section_instructions": instruction_list},
            step_name="generate_section_instructions",
            primary_key="section_instructions",
            context_key=None,
            summary="Generated section writing instructions.",
            counts={
                "section_instructions": len(instruction_list.instructions),
            },
            artifact_meta={
                "outline_id": writing_outline.outline_id,
                "context_id": writing_context.context_id,
                "has_execution_results": execution_data is not None,
            },
        )
        return result.model_dump()

    def _normalize_execution_results(self, execution_results: Any) -> Any:
        return self._unified_raw_data(execution_results)

    def _normalize_outline(
        self,
        outline: WritingOutline,
        task: WritingTask,
        context: WritingContext,
        profiles: List[ResourceProfile],
        execution_results: Any,
    ) -> WritingOutline:
        if len(outline.nodes) < 3:
            raise ValueError("generate_outline must produce at least 3 top-level sections.")

        outline.outline_id = outline.outline_id or self._default_outline_id(task, context)
        outline.title = outline.title or self._default_outline_title(task)
        for index, node in enumerate(outline.nodes, start=1):
            self._normalize_outline_node(node, level=1, fallback_id=f"section-{index}")

        outline.meta.update(
            {
                "source": "llm",
            }
        )
        return outline

    def _normalize_outline_node(self, node: OutlineNode, *, level: int, fallback_id: str) -> None:
        node.level = level
        node.node_id = node.node_id or fallback_id
        for index, child in enumerate(node.children, start=1):
            self._normalize_outline_node(
                child,
                level=level + 1,
                fallback_id=f"{node.node_id}-{index}",
            )

    def _default_outline_id(self, task: WritingTask, context: WritingContext) -> str:
        source_id = task.task_id or context.context_id or "writer"
        return f"{source_id}-outline"

    def _default_outline_title(self, task: WritingTask) -> str:
        if task.target_document and task.target_document.title:
            return task.target_document.title
        query = " ".join(task.query.split())
        return query[:80] if query else "Writing Outline"

    def _count_outline_nodes(self, nodes: List[OutlineNode]) -> int:
        return sum(1 + self._count_outline_nodes(node.children) for node in nodes)

    def _instruction_target_nodes(self, outline: WritingOutline) -> List[OutlineNode]:
        return outline.nodes

    def _normalize_section_instructions(
        self,
        instruction_list: SectionInstructionList,
        outline: WritingOutline,
        context: WritingContext,
        target_nodes: List[OutlineNode],
        execution_results: Any,
    ) -> SectionInstructionList:
        instructions_by_node_id = {
            instruction.outline_node_id: instruction
            for instruction in instruction_list.instructions
            if instruction.outline_node_id
        }

        normalized: List[SectionInstruction] = []
        for node in target_nodes:
            instruction = instructions_by_node_id.get(node.node_id or "")
            if instruction is None:
                instruction = self._instruction_from_outline_node(node)
            normalized.append(self._normalize_section_instruction(instruction, node, outline))

        instruction_list.instruction_set_id = (
            instruction_list.instruction_set_id
            or self._default_instruction_set_id(outline)
        )
        instruction_list.outline_id = instruction_list.outline_id or outline.outline_id
        instruction_list.instructions = normalized
        instruction_list.meta.update(
            {
                "source": "llm",
                "outline_id": outline.outline_id,
                "outline_title": outline.title,
                "context_id": context.context_id,
                "has_execution_results": execution_results is not None,
            }
        )
        return instruction_list

    def _normalize_section_instruction(
        self,
        instruction: SectionInstruction,
        node: OutlineNode,
        outline: WritingOutline,
    ) -> SectionInstruction:
        constraints = node.constraints
        instruction.outline_node_id = instruction.outline_node_id or node.node_id or ""
        instruction.instruction_id = instruction.instruction_id or f"instruction-{instruction.outline_node_id}"
        instruction.section_title = instruction.section_title or node.title
        instruction.section_goal = (
            instruction.section_goal
            or constraints.section_goal
            or node.instruction
            or f"Write the section: {node.title}"
        )
        if not instruction.required_points:
            instruction.required_points = list(constraints.required_points)
        if not instruction.source_refs:
            instruction.source_refs = list(constraints.source_refs)
        if not instruction.fact_constraints:
            instruction.fact_constraints = list(constraints.fact_constraints)
        if not instruction.style_constraints:
            instruction.style_constraints = list(constraints.style_constraints)
            if constraints.pov:
                instruction.style_constraints.append(f"POV: {constraints.pov}")
            if constraints.tone:
                instruction.style_constraints.append(f"Tone: {constraints.tone}")
        if not instruction.relation_constraints:
            instruction.relation_constraints = list(constraints.relation_constraints)
        if not instruction.expected_blocks:
            instruction.expected_blocks = self._default_expected_blocks(node)
        instruction.meta.update(
            {
                "outline_node_level": node.level,
                "outline_node_instruction": node.instruction,
                "outline_id": outline.outline_id,
                "outline_title": outline.title,
            }
        )
        return instruction

    def _instruction_from_outline_node(self, node: OutlineNode) -> SectionInstruction:
        constraints = node.constraints
        return SectionInstruction(
            instruction_id=f"instruction-{node.node_id or node.title}",
            outline_node_id=node.node_id or "",
            section_title=node.title,
            section_goal=constraints.section_goal or node.instruction or f"Write the section: {node.title}",
            required_points=list(constraints.required_points),
            source_refs=list(constraints.source_refs),
            fact_constraints=list(constraints.fact_constraints),
            style_constraints=list(constraints.style_constraints),
            relation_constraints=list(constraints.relation_constraints),
            expected_blocks=self._default_expected_blocks(node),
        )

    def _default_expected_blocks(self, node: OutlineNode) -> List[str]:
        blocks = [node.title]
        if node.constraints.required_points:
            blocks.extend(node.constraints.required_points[:3])
        return blocks

    def _default_instruction_set_id(self, outline: WritingOutline) -> str:
        source_id = outline.outline_id or "outline"
        return f"{source_id}-section-instructions"
