from __future__ import annotations

from typing import Any, Optional

from .base import WriterToolBase
from ..data_models.context import WritingContext
from ..data_models.quality import AuditResult, ReviewReport
from ..data_models.revision import PatchSet
from ..data_models.task import WritingTask
from ..data_models.writer_ir import WriterBlock, WriterDocument
from ..data_models.planning import SectionInstruction, SectionInstructionList
from ..prompts.quality import (
    VALIDATE_DRAFT_DOCUMENT_PROMPT,
    VALIDATE_PATCH_SET_PROMPT,
    VALIDATE_SECTION_PROMPT,
)
from ..utils import parse_markdown_sections, to_prompt_json


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
        draft = self._unified_section(draft_block)
        instruction_list = self._unified_model(section_instruction, SectionInstructionList)
        writing_context = self._unified_model(context, WritingContext)
        if isinstance(draft, WriterBlock):
            self._require_stage(draft.stage, 'draft', 'draft_block')
            instruction = self._match_instruction(draft, instruction_list)
            target = draft.node_id
            section_content = to_prompt_json(draft)
        else:
            instruction = self._match_markdown_instruction(draft, instruction_list)
            target = '/'.join(instruction.content_ref.heading_path) if instruction else None
            section_content = draft
        if instruction is None:
            raise ValueError('No section instruction matches the draft section.')

        prompt = VALIDATE_SECTION_PROMPT.format(
            section_content=section_content,
            instruction_json=to_prompt_json(instruction),
            context_json=to_prompt_json(writing_context),
        )
        audit_result = self._call_llm_structured(prompt, AuditResult)

        report = ReviewReport(
            target=target,
            result=audit_result,
            meta={
                'instruction_id': instruction.instruction_id,
                'content_ref': instruction.content_ref.model_dump(exclude_none=True),
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
                'draft_node_id': draft.node_id if isinstance(draft, WriterBlock) else None,
                'content_ref': instruction.content_ref.model_dump(exclude_none=True),
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
        document = self._unified_document(draft_document)
        writing_context = self._unified_model(context, WritingContext)
        if isinstance(document, WriterDocument):
            self._require_stage(document.stage, 'draft', 'draft_document')
            target = document.document_id
            title = document.title
            block_count = len(list(document.iter_blocks()))
            document_content = to_prompt_json(document)
        else:
            sections = parse_markdown_sections(document)
            target = None
            title = sections[0][1][-1] if sections and sections[0][0] == 1 else ''
            block_count = len(sections)
            document_content = document

        prompt = VALIDATE_DRAFT_DOCUMENT_PROMPT.format(
            draft_document_content=document_content,
            context_json=to_prompt_json(writing_context),
        )
        audit_result = self._call_llm_structured(prompt, AuditResult)

        report = ReviewReport(
            target=target,
            result=audit_result,
            meta={
                'draft_document_id': (
                    document.document_id if isinstance(document, WriterDocument) else None
                ),
                'draft_title': title,
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
                'draft_document_id': (
                    document.document_id if isinstance(document, WriterDocument) else None
                ),
                'draft_title': title,
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
                    exclude={'fact_id', 'source', 'applies_to', 'locked'},
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
        return next(
            (
                instruction for instruction in instruction_list.instructions
                if instruction.content_ref.node_id == draft_block.node_id
            ),
            None,
        )

    def _match_markdown_instruction(
        self,
        draft: str,
        instruction_list: SectionInstructionList,
    ) -> Optional[SectionInstruction]:
        sections = parse_markdown_sections(draft)
        if not sections:
            return None
        _, heading_path, occurrence, _ = sections[0]
        exact_match = next(
            (
                instruction for instruction in instruction_list.instructions
                if instruction.content_ref.heading_path == heading_path
                and instruction.content_ref.occurrence == occurrence
            ),
            None,
        )
        if exact_match:
            return exact_match
        return next(
            (
                instruction for instruction in instruction_list.instructions
                if instruction.content_ref.heading_path
                and instruction.content_ref.heading_path[-1] == heading_path[-1]
            ),
            None,
        )

    def _issue_counts(self, audit_result: AuditResult) -> dict:
        return {
            'high_severity': sum(1 for issue in audit_result.issues if issue.severity == 'high'),
            'medium_severity': sum(1 for issue in audit_result.issues if issue.severity == 'medium'),
            'low_severity': sum(1 for issue in audit_result.issues if issue.severity == 'low'),
        }

    def _require_stage(self, actual: str, expected: str, argument: str) -> None:
        if actual != expected:
            raise ValueError(f'{argument} must have stage={expected!r}, got {actual!r}')
