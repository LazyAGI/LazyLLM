from __future__ import annotations

from typing import Any, Optional

from .base import WriterToolBase
from ..data_models.context import WritingContext
from ..data_models.quality import AuditResult, ReviewReport
from ..data_models.revision import PatchSet
from ..data_models.task import WritingTask
from ..data_models.writer_ir import WriterBlock, WriterDocument
from ..data_models.writing import SectionInstruction, SectionInstructionList
from ..prompts.quality import (
    VALIDATE_DRAFT_DOCUMENT_PROMPT,
    VALIDATE_PATCH_SET_PROMPT,
    VALIDATE_SECTION_PROMPT,
)
from ..utils import to_prompt_json


class WriterQualityTools(WriterToolBase):
    __public_apis__ = [
        'validate_section',
        'validate_draft_document',
        'validate_patch_set',
    ]

    def validate_section(
        self,
        draft_block: Any,
        section_instruction: Any,
        context: Any,
    ) -> dict:
        draft = self._unified_model(draft_block, WriterBlock)
        instruction_list = self._unified_model(section_instruction, SectionInstructionList)
        writing_context = self._unified_model(context, WritingContext)
        self._require_stage(draft.stage, 'draft', 'draft_block')

        instruction = self._match_instruction(draft, instruction_list)
        if instruction is None:
            raise ValueError(
                f'No section instruction matches draft block {draft.node_id!r}.'
            )

        prompt = VALIDATE_SECTION_PROMPT.format(
            section_json=to_prompt_json(draft),
            instruction_json=to_prompt_json(instruction),
            context_json=to_prompt_json(writing_context),
        )
        audit_result = self._call_llm_structured(prompt, AuditResult)

        report = ReviewReport(
            target=draft.node_id,
            result=audit_result,
            meta={
                'instruction_id': instruction.instruction_id,
                'outline_node_id': instruction.outline_node_id,
                'section_title': instruction.section_title,
            },
        )
        counts = self._issue_counts(audit_result)

        return self._save_artifacts(
            {'section_review': report},
            step_name='validate_section',
            primary_key='section_review',
            summary=f'Section validation: {"PASSED" if audit_result.is_passed else "FAILED"} '
                    f'(score: {audit_result.score}/100)',
            counts={
                'total_issues': len(audit_result.issues),
                **counts,
            },
            artifact_meta={
                'draft_node_id': draft.node_id,
                'outline_node_id': instruction.outline_node_id,
                'instruction_id': instruction.instruction_id,
                'is_passed': audit_result.is_passed,
                'score': audit_result.score,
            },
        ).model_dump()

    def validate_draft_document(
        self,
        draft_document: Any,
        context: Any,
    ) -> dict:
        document = self._unified_model(draft_document, WriterDocument)
        writing_context = self._unified_model(context, WritingContext)
        self._require_stage(document.stage, 'draft', 'draft_document')

        prompt = VALIDATE_DRAFT_DOCUMENT_PROMPT.format(
            draft_document_json=to_prompt_json(document),
            context_json=to_prompt_json(writing_context),
        )
        audit_result = self._call_llm_structured(prompt, AuditResult)
        block_count = len(list(document.iter_blocks()))

        report = ReviewReport(
            target=document.document_id,
            result=audit_result,
            meta={
                'draft_document_id': document.document_id,
                'draft_title': document.title,
                'draft_block_count': block_count,
                'context_id': writing_context.context_id,
            },
        )
        counts = self._issue_counts(audit_result)

        return self._save_artifacts(
            {'draft_document_review': report},
            step_name='validate_draft_document',
            primary_key='draft_document_review',
            summary=f'Draft document validation: {"PASSED" if audit_result.is_passed else "FAILED"} '
                    f'(score: {audit_result.score}/100)',
            counts={
                'total_issues': len(audit_result.issues),
                **counts,
            },
            artifact_meta={
                'draft_document_id': document.document_id,
                'draft_title': document.title,
                'draft_block_count': block_count,
                'is_passed': audit_result.is_passed,
                'score': audit_result.score,
            },
        ).model_dump()

    def validate_patch_set(
        self,
        patch_set: Any,
        context: Any,
        task: Any,
    ) -> dict:
        patch = self._unified_model(patch_set, PatchSet)
        writing_context = self._unified_model(context, WritingContext)
        writing_task = self._unified_model(task, WritingTask)

        hunks_json = to_prompt_json([
            hunk.model_dump(
                exclude={'anchor', 'meta', 'target_block_id'},
                exclude_none=True,
            )
            for hunk in patch.hunks
        ])
        context_json = to_prompt_json({
            'facts': [
                fact.model_dump(
                    exclude={'fact_id', 'source', 'applies_to_block_ids', 'locked'},
                )
                for fact in writing_context.facts
                if fact.locked
            ],
            'style_profile': (
                writing_context.style_profile.model_dump()
                if writing_context.style_profile
                else None
            ),
        })

        prompt = VALIDATE_PATCH_SET_PROMPT.format(
            task_query=writing_task.query,
            hunks_json=hunks_json,
            context_json=context_json,
        )
        audit_result = self._call_llm_structured(prompt, AuditResult)
        counts = self._issue_counts(audit_result)

        return self._save_artifacts(
            {'patch_set_review': audit_result},
            step_name='validate_patch_set',
            primary_key='patch_set_review',
            summary=f'PatchSet validation: {"PASSED" if audit_result.is_passed else "FAILED"} '
                    f'(score: {audit_result.score}/100)',
            counts={
                'total_hunks': len(patch.hunks),
                'total_issues': len(audit_result.issues),
                **counts,
            },
            artifact_meta={
                'patch_id': patch.patch_id,
                'target_doc_id': patch.target_doc_id,
                'is_passed': audit_result.is_passed,
                'score': audit_result.score,
            },
        ).model_dump()

    def _match_instruction(
        self,
        draft_block: WriterBlock,
        instruction_list: SectionInstructionList,
    ) -> Optional[SectionInstruction]:
        draft_authoring = draft_block.authoring
        instruction_id = draft_authoring.instruction_id if draft_authoring else None
        origin_node_id = draft_authoring.origin_node_id if draft_authoring else None

        if instruction_id:
            for instruction in instruction_list.instructions:
                if instruction.instruction_id == instruction_id:
                    return instruction

        if origin_node_id:
            for instruction in instruction_list.instructions:
                if instruction.outline_node_id == origin_node_id:
                    return instruction

        return None

    def _issue_counts(self, audit_result: AuditResult) -> dict:
        return {
            'high_severity': sum(1 for issue in audit_result.issues if issue.severity == 'high'),
            'medium_severity': sum(1 for issue in audit_result.issues if issue.severity == 'medium'),
            'low_severity': sum(1 for issue in audit_result.issues if issue.severity == 'low'),
        }

    def _require_stage(self, actual: str, expected: str, argument: str) -> None:
        if actual != expected:
            raise ValueError(f'{argument} must have stage={expected!r}, got {actual!r}')
