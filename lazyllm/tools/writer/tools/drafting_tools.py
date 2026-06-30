from __future__ import annotations
from typing import Any, List

from .base import WriterToolBase
from ..data_models.context import WritingContext
from ..data_models.task import WritingTask
from ..data_models.writing import (
    DraftBlock,
    DraftDocument,
    DraftSection,
    SectionInstruction,
    SectionInstructionList,
    WritingOutline,
    WritingOutput,
)
from ..prompts import GENERATE_DRAFT_SECTION_PROMPT
from ..utils import to_prompt_json


class WriterDraftingTools(WriterToolBase):
    __public_apis__ = [
        "generate_draft_section",
        "generate_draft_document",
        "generate_writing_output",
    ]

    def generate_draft_section(
        self,
        task: Any,
        section_instruction: Any,
        context: Any,
        previous_sections: Any = None,
    ) -> dict:
        writing_task = self._unified_model(task, WritingTask)
        instruction = self._unified_section_instruction(section_instruction)
        writing_context = self._unified_model(context, WritingContext)
        previous_data = self._unified_raw_data(previous_sections)

        prompt = GENERATE_DRAFT_SECTION_PROMPT.format(
            task_json=to_prompt_json(writing_task),
            section_instruction_json=to_prompt_json(instruction),
            context_json=to_prompt_json(writing_context),
            previous_sections_json=to_prompt_json(previous_data),
        )
        draft_section = self._call_llm_structured(prompt, DraftSection)
        draft_section = self._normalize_draft_section(draft_section, instruction)

        result = self._save_artifacts(
            {"draft_section": draft_section},
            step_name="generate_draft_section",
            primary_key="draft_section",
            context_key=None,
            summary="Generated draft section.",
            counts={
                "draft_blocks": len(draft_section.blocks),
                "subtasks": len(draft_section.subtasks),
            },
            artifact_meta={
                "task_id": writing_task.task_id,
                "context_id": writing_context.context_id,
                "instance_id": draft_section.section_id,
                "instruction_id": instruction.instruction_id,
                "outline_node_id": instruction.outline_node_id,
                "outline_title": instruction.meta.get("outline_title"),
            },
            artifact_filenames={
                "draft_section": f"draft_section_{draft_section.section_id}.json",
            },
        )
        return result.model_dump()

    def generate_draft_document(
        self,
        draft_sections: Any,
        context: Any,
        outline: Any = None,
        title: Any = None,
    ) -> dict:
        sections = self._unified_draft_sections(draft_sections)
        if not sections:
            raise ValueError("draft_sections must contain at least one DraftSection.")

        writing_context = self._unified_model(context, WritingContext)
        writing_outline = self._unified_optional_model(outline, WritingOutline)
        normalized_sections = [
            self._normalize_document_section(section, index)
            for index, section in enumerate(sections, start=1)
        ]
        draft_document = DraftDocument(
            draft_id=self._default_draft_document_id(writing_context),
            title=self._resolve_draft_document_title(
                title,
                writing_outline,
                writing_context,
                normalized_sections,
            ),
            sections=normalized_sections,
            meta={
                "source": "generate_draft_document",
                "context_id": writing_context.context_id,
                "outline_id": writing_outline.outline_id if writing_outline else None,
                "outline_title": writing_outline.title if writing_outline else None,
            },
        )

        result = self._save_artifacts(
            {"draft_document": draft_document},
            step_name="generate_draft_document",
            primary_key="draft_document",
            context_key=None,
            summary="Generated draft document.",
            counts={
                "draft_sections": len(draft_document.sections),
                "draft_blocks": self._count_draft_blocks(draft_document.sections),
                "subtasks": self._count_draft_subtasks(draft_document.sections),
            },
            artifact_meta={
                "context_id": writing_context.context_id,
                "doc_id": writing_context.doc_id,
                "outline_id": writing_outline.outline_id if writing_outline else None,
                "outline_title": writing_outline.title if writing_outline else None,
                "draft_section_count": len(draft_document.sections),
            },
        )
        return result.model_dump()

    def generate_writing_output(
        self,
        draft: Any,
        context: Any,
        output_format: str = "markdown",
    ) -> dict:
        if output_format != "markdown":
            raise ValueError("Only markdown output is supported for now.")

        writing_context = self._unified_model(context, WritingContext)
        draft_document = self._unified_draft_document(draft, writing_context)
        content = self._render_draft_document_markdown(draft_document)
        writing_output = WritingOutput(
            output_id=self._default_writing_output_id(draft_document, writing_context),
            title=draft_document.title,
            content=content,
            output_format="markdown",
            references=self._collect_output_references(draft_document, writing_context),
            meta={
                "source": "generate_writing_output",
                "draft_id": draft_document.draft_id,
                "context_id": writing_context.context_id,
            },
        )

        result = self._save_artifacts(
            {"writing_output": writing_output},
            step_name="generate_writing_output",
            primary_key="writing_output",
            context_key=None,
            summary="Generated writing output.",
            counts={
                "characters": len(writing_output.content),
                "draft_sections": len(draft_document.sections),
                "draft_blocks": self._count_draft_blocks(draft_document.sections),
            },
            artifact_meta={
                "context_id": writing_context.context_id,
                "doc_id": writing_context.doc_id,
                "draft_id": draft_document.draft_id,
                "output_format": writing_output.output_format,
            },
        )
        return result.model_dump()

    def _unified_section_instruction(self, value: Any) -> SectionInstruction:
        if isinstance(value, SectionInstruction):
            return value
        if isinstance(value, SectionInstructionList):
            return self._select_section_instruction(value.instructions)
        if isinstance(value, str):
            value = self._load_artifact(value, validate_schema=False)
        if isinstance(value, dict):
            if "instructions" in value:
                instruction_list = SectionInstructionList.model_validate(value)
                return self._select_section_instruction(instruction_list.instructions)
            return SectionInstruction.model_validate(value)
        if isinstance(value, list):
            instructions = [self._unified_section_instruction(item) for item in value]
            return self._select_section_instruction(instructions)
        raise TypeError(
            "Expected SectionInstruction, SectionInstructionList, dict, or artifact path, "
            f"got {type(value).__name__}."
        )

    def _select_section_instruction(self, instructions: List[SectionInstruction]) -> SectionInstruction:
        if not instructions:
            raise ValueError("section_instruction list is empty.")
        return instructions[0]

    def _normalize_draft_section(
        self,
        draft_section: DraftSection,
        instruction: SectionInstruction,
    ) -> DraftSection:
        section_id = self._default_section_id(instruction)
        draft_section.section_id = section_id
        draft_section.outline_node_id = instruction.outline_node_id
        draft_section.title = instruction.section_title
        draft_section.instruction_id = instruction.instruction_id

        if not draft_section.blocks:
            draft_section.blocks.append(
                DraftBlock(
                    block_id=f"{section_id}-block-1",
                    outline_node_id=instruction.outline_node_id,
                    section_id=section_id,
                    content="",
                )
            )

        for index, block in enumerate(draft_section.blocks, start=1):
            block.block_id = block.block_id or f"{section_id}-block-{index}"
            block.outline_node_id = instruction.outline_node_id
            block.section_id = section_id
            for subtask in block.subtasks:
                subtask.section_id = section_id
                subtask.block_id = block.block_id

        for subtask in draft_section.subtasks:
            subtask.section_id = section_id

        draft_section.meta.update(
            {
                "source": "llm",
                "instruction_id": instruction.instruction_id,
                "outline_node_id": instruction.outline_node_id,
            }
        )
        self._copy_meta_value(instruction.meta, draft_section.meta, "outline_id")
        self._copy_meta_value(instruction.meta, draft_section.meta, "outline_title")
        return draft_section

    def _default_section_id(self, instruction: SectionInstruction) -> str:
        source_id = instruction.outline_node_id or instruction.instruction_id or "section"
        return f"draft-{source_id}"

    def _unified_draft_sections(self, value: Any) -> List[DraftSection]:
        if value is None:
            return []
        if isinstance(value, DraftSection):
            return [value]
        if isinstance(value, DraftDocument):
            return list(value.sections)
        if isinstance(value, str):
            value = self._load_artifact(value, validate_schema=False)
            return self._unified_draft_sections(value)
        if isinstance(value, dict):
            if "draft" in value:
                return self._unified_draft_sections(value["draft"])
            if "sections" in value:
                return self._unified_draft_sections(value["sections"])
            return [DraftSection.model_validate(value)]
        if isinstance(value, list):
            sections: List[DraftSection] = []
            for item in value:
                sections.extend(self._unified_draft_sections(item))
            return sections
        raise TypeError(
            "Expected DraftSection, DraftDocument, list, dict, or artifact path, "
            f"got {type(value).__name__}."
        )

    def _unified_draft_document(self, value: Any, context: WritingContext) -> DraftDocument:
        if isinstance(value, DraftDocument):
            return value
        if isinstance(value, str):
            value = self._load_artifact(value, validate_schema=False)
            return self._unified_draft_document(value, context)
        if isinstance(value, dict):
            if "data" in value:
                return self._unified_draft_document(value["data"], context)
            if "draft" in value:
                return self._unified_draft_document(value["draft"], context)
            if "sections" in value:
                return DraftDocument.model_validate(value)

        sections = self._unified_draft_sections(value)
        if not sections:
            raise ValueError("draft must contain at least one DraftSection.")
        normalized_sections = [
            self._normalize_document_section(section, index)
            for index, section in enumerate(sections, start=1)
        ]
        return DraftDocument(
            draft_id=self._default_draft_document_id(context),
            title=self._default_draft_document_title(context, normalized_sections),
            sections=normalized_sections,
            meta={
                "source": "generate_writing_output",
                "context_id": context.context_id,
            },
        )

    def _normalize_document_section(self, section: DraftSection, index: int) -> DraftSection:
        section_id = section.section_id or f"draft-section-{index}"
        section.section_id = section_id
        for block_index, block in enumerate(section.blocks, start=1):
            block.block_id = block.block_id or f"{section_id}-block-{block_index}"
            block.section_id = section_id
            block.outline_node_id = block.outline_node_id or section.outline_node_id
            for subtask in block.subtasks:
                subtask.section_id = subtask.section_id or section_id
                subtask.block_id = subtask.block_id or block.block_id
        for subtask in section.subtasks:
            subtask.section_id = subtask.section_id or section_id
        return section

    def _default_draft_document_id(self, context: WritingContext) -> str:
        source_id = context.context_id or context.doc_id or "document"
        return f"draft-document-{source_id}"

    def _default_writing_output_id(
        self,
        draft_document: DraftDocument,
        context: WritingContext,
    ) -> str:
        source_id = draft_document.draft_id or context.context_id or context.doc_id or "document"
        return f"output-{source_id}"

    def _default_draft_document_title(
        self,
        context: WritingContext,
        sections: List[DraftSection],
    ) -> str:
        return self._resolve_draft_document_title(None, None, context, sections)

    def _resolve_draft_document_title(
        self,
        title: Any,
        outline: WritingOutline | None,
        context: WritingContext,
        sections: List[DraftSection],
    ) -> str:
        title = self._first_non_empty(
            title,
            outline.title if outline else None,
            context.meta.get("title") if context.meta else None,
            context.meta.get("document_title") if context.meta else None,
            context.meta.get("outline_title") if context.meta else None,
            self._first_section_meta_value(sections, "document_title"),
            self._first_section_meta_value(sections, "outline_title"),
        )
        if title:
            return str(title)
        if context.doc_id:
            return context.doc_id
        if sections and sections[0].title:
            return sections[0].title
        return "Draft Document"

    def _count_draft_blocks(self, sections: List[DraftSection]) -> int:
        total = 0
        for section in sections:
            total += len(section.blocks)
            total += self._count_draft_blocks(section.sub_sections)
        return total

    def _count_draft_subtasks(self, sections: List[DraftSection]) -> int:
        total = 0
        for section in sections:
            total += len(section.subtasks)
            for block in section.blocks:
                total += len(block.subtasks)
            total += self._count_draft_subtasks(section.sub_sections)
        return total

    def _render_draft_document_markdown(self, draft_document: DraftDocument) -> str:
        parts: List[str] = []
        if draft_document.title:
            parts.append(f"# {draft_document.title.strip()}")
        for section in draft_document.sections:
            parts.extend(self._render_draft_section_markdown(section, level=2))
        return "\n\n".join(part for part in parts if part).strip() + "\n"

    def _render_draft_section_markdown(self, section: DraftSection, level: int) -> List[str]:
        parts: List[str] = []
        heading_level = min(max(level, 1), 6)
        if section.title:
            parts.append(f"{'#' * heading_level} {section.title.strip()}")
        for block in section.blocks:
            block_text = self._render_draft_block_markdown(block, heading_level + 1)
            if block_text:
                parts.append(block_text)
        for sub_section in section.sub_sections:
            parts.extend(self._render_draft_section_markdown(sub_section, heading_level + 1))
        return parts

    def _render_draft_block_markdown(self, block: DraftBlock, heading_level: int) -> str:
        parts: List[str] = []
        if block.heading:
            level = min(max(heading_level, 1), 6)
            parts.append(f"{'#' * level} {block.heading.strip()}")

        content = block.content.strip()
        if content:
            parts.append(content)
        elif block.subtasks:
            placeholders = [subtask.placeholder for subtask in block.subtasks if subtask.placeholder]
            parts.extend(placeholders)

        return "\n\n".join(part for part in parts if part).strip()

    def _collect_output_references(
        self,
        draft_document: DraftDocument,
        context: WritingContext,
    ) -> List[str]:
        references: List[str] = []
        self._extend_unique(references, draft_document.meta.get("references", []))
        self._extend_unique(references, draft_document.meta.get("source_refs", []))
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

    def _first_section_meta_value(self, sections: List[DraftSection], key: str) -> Any:
        for section in sections:
            value = section.meta.get(key) if section.meta else None
            if value:
                return value
            child_value = self._first_section_meta_value(section.sub_sections, key)
            if child_value:
                return child_value
        return None

    def _first_non_empty(self, *values: Any) -> Any:
        for value in values:
            if value:
                return value
        return None
