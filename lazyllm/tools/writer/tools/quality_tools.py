from __future__ import annotations
from typing import Any, Optional

from .base import WriterToolBase
from ..data_models.context import WritingContext
from ..data_models.quality import AuditResult, ReviewReport
from ..data_models.writing import DraftDocument, SectionInstruction, SectionInstructionList
from ..prompts.quality import VALIDATE_DRAFT_DOCUMENT_PROMPT, VALIDATE_SECTION_PROMPT
from ..utils import to_prompt_json


class WriterQualityTools(WriterToolBase):
    __public_apis__ = [
        "validate_section",
        "validate_draft_document",
    ]

    def validate_section(
        self,
        draft_section: Any,
        section_instruction: Any,
        context: Any,
    ) -> dict:
        section_data = self._unified_raw_data(draft_section)
        instruction_list = self._unified_model(section_instruction, SectionInstructionList)
        writing_context = self._unified_model(context, WritingContext)

        instruction = self._match_instruction(section_data or {}, instruction_list)

        if instruction is None:
            fallback = AuditResult(
                is_passed=True,
                score=100,
                summary="未找到匹配的章节指令，跳过详细校验。",
                issues=[],
            )
            report = ReviewReport(
                target=section_data.get("section_id") or section_data.get("title") if section_data else None,
                result=fallback,
            )
            result = self._save_artifacts(
                {"section_review": report},
                step_name="validate_section",
                primary_key="section_review",
                summary="Section validation skipped: no matching instruction.",
                counts={"total_issues": 0, "high_severity": 0, "medium_severity": 0, "low_severity": 0},
                artifact_meta={"is_passed": True, "score": 100, "match_found": False},
            )
            return result.model_dump()

        prompt = VALIDATE_SECTION_PROMPT.format(
            section_json=to_prompt_json(section_data),
            instruction_json=to_prompt_json(instruction),
            context_json=to_prompt_json(writing_context),
        )

        audit_result = self._call_llm_structured(prompt, AuditResult)

        section_title = section_data.get("title") if section_data else None
        section_id = section_data.get("section_id") if section_data else None

        report = ReviewReport(
            target=section_id or section_title or instruction.section_title or "unknown",
            result=audit_result,
            meta={
                "instruction_id": instruction.instruction_id,
                "outline_node_id": instruction.outline_node_id,
                "section_title": instruction.section_title,
            },
        )

        high_count = sum(1 for i in audit_result.issues if i.severity == "high")
        medium_count = sum(1 for i in audit_result.issues if i.severity == "medium")
        low_count = sum(1 for i in audit_result.issues if i.severity == "low")

        result = self._save_artifacts(
            {"section_review": report},
            step_name="validate_section",
            primary_key="section_review",
            summary=f"Section validation: {'PASSED' if audit_result.is_passed else 'FAILED'} (score: {audit_result.score}/100)",
            counts={
                "total_issues": len(audit_result.issues),
                "high_severity": high_count,
                "medium_severity": medium_count,
                "low_severity": low_count,
            },
            artifact_meta={
                "section_id": section_id,
                "section_title": section_title,
                "instruction_id": instruction.instruction_id,
                "is_passed": audit_result.is_passed,
                "score": audit_result.score,
            },
        )
        return result.model_dump()

    def validate_draft_document(
        self,
        draft_document: Any,
        context: Any,
    ) -> dict:
        document = self._unified_model(draft_document, DraftDocument)
        writing_context = self._unified_model(context, WritingContext)

        prompt = VALIDATE_DRAFT_DOCUMENT_PROMPT.format(
            draft_document_json=to_prompt_json(document),
            context_json=to_prompt_json(writing_context),
        )

        audit_result = self._call_llm_structured(prompt, AuditResult)

        report = ReviewReport(
            target=document.draft_id or document.title or "untitled",
            result=audit_result,
            meta={
                "draft_id": document.draft_id,
                "draft_title": document.title,
                "draft_section_count": len(document.sections),
                "context_id": writing_context.context_id,
            },
        )

        high_count = sum(1 for i in audit_result.issues if i.severity == "high")
        medium_count = sum(1 for i in audit_result.issues if i.severity == "medium")
        low_count = sum(1 for i in audit_result.issues if i.severity == "low")

        result = self._save_artifacts(
            {"draft_document_review": report},
            step_name="validate_draft_document",
            primary_key="draft_document_review",
            summary=f"Draft document validation: {'PASSED' if audit_result.is_passed else 'FAILED'} (score: {audit_result.score}/100)",
            counts={
                "total_issues": len(audit_result.issues),
                "high_severity": high_count,
                "medium_severity": medium_count,
                "low_severity": low_count,
            },
            artifact_meta={
                "draft_id": document.draft_id,
                "draft_title": document.title,
                "is_passed": audit_result.is_passed,
                "score": audit_result.score,
            },
        )
        return result.model_dump()

    def _match_instruction(
        self,
        section_data: dict,
        instruction_list: SectionInstructionList,
    ) -> Optional[SectionInstruction]:
        section_instruction_id = section_data.get("instruction_id") or ""
        section_node_id = section_data.get("outline_node_id") or ""
        section_title = section_data.get("title") or ""

        for inst in instruction_list.instructions:
            if section_instruction_id and inst.instruction_id == section_instruction_id:
                return inst
            if section_node_id and inst.outline_node_id == section_node_id:
                return inst

        for inst in instruction_list.instructions:
            if section_title and inst.section_title == section_title:
                return inst

        for section_block in section_data.get("blocks") or []:
            block_heading = (section_block.get("heading") or "").strip()
            if block_heading:
                for inst in instruction_list.instructions:
                    if block_heading == inst.section_title:
                        return inst

        return None
