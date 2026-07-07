from __future__ import annotations
from typing import Any, Dict, List, Tuple

from .base import WriterToolBase
from ..data_models.context import WritingContext
from ..data_models.docir import Anchor, DocBlock, DocIR
from ..data_models.revision import (
    BLOCK_META_FIELDS,
    HEADING_PATH_KEY,
    SECTION_META_FIELDS,
    LocateResult,
    ModifyInstruction,
    ModifyPlan,
    PatchHunk,
    PatchResult,
    PatchSet,
)
from ..data_models.task import WritingTask
from ..data_models.writing import DraftBlock, DraftDocument, DraftSection
from ..prompts import (
    GENERATE_MODIFY_PLAN_PROMPT,
    GENERATE_PATCH_SET_PROMPT,
    LOCATE_REVISION_TARGET_PROMPT,
)
from ..utils import to_prompt_json


class WriterRevisionTools(WriterToolBase):
    __public_apis__ = [
        'locate_revision_target',
        'generate_modify_plan',
        'generate_patch_set',
        'apply_patch',
        'draft_to_doc_ir',
        'doc_ir_to_draft',
    ]

    def locate_revision_target(
        self,
        task: Any,
        doc_ir: Any,
        context: Any,
    ) -> dict:
        writing_task = self._unified_model(task, WritingTask)
        if writing_task.task_type != 'revise':
            raise ValueError(
                f'task.task_type must be \'revise\', got {writing_task.task_type!r}.'
            )
        source_doc = self._unified_model(doc_ir, DocIR)
        writing_context = self._unified_model(context, WritingContext)
        user_selection = writing_task.selection

        valid_block_ids = {b.block_id for b in self._iter_blocks(source_doc.blocks)}
        if not valid_block_ids:
            raise ValueError('doc_ir must contain at least one block.')

        candidate_block_ids = valid_block_ids
        if user_selection and user_selection.block_ids:
            selected = set(user_selection.block_ids) & valid_block_ids
            if not selected:
                raise ValueError('selection.block_ids contains no valid block_ids.')
            candidate_block_ids = selected

        prompt = LOCATE_REVISION_TARGET_PROMPT.format(
            task_json=to_prompt_json(writing_task),
            doc_ir_json=to_prompt_json(source_doc),
            context_json=to_prompt_json(writing_context),
            candidate_block_ids=to_prompt_json(sorted(candidate_block_ids)),
        )
        locate_result = self._call_llm_structured(prompt, LocateResult)

        located = [bid for bid in (locate_result.target_block_ids or []) if bid]
        if not located:
            raise ValueError('LLM returned empty target_block_ids.')
        invalid = [bid for bid in located if bid not in candidate_block_ids]
        if invalid:
            raise ValueError(
                f'locate_result contains block_ids not in candidates: {invalid}.'
            )
        locate_result.target_block_ids = list(dict.fromkeys(located))
        locate_result.target_reasons = {
            bid: locate_result.target_reasons.get(bid, '')
            for bid in locate_result.target_block_ids
        }
        locate_result.task_id = writing_task.task_id
        locate_result.doc_id = source_doc.doc_id

        result = self._save_artifacts(
            {'locate_result': locate_result},
            step_name='locate_revision_target',
            primary_key='locate_result',
            context_key=None,
            summary='Located revision target blocks.',
            counts={'target_block_count': len(locate_result.target_block_ids)},
            artifact_meta={
                'task_id': writing_task.task_id,
                'doc_id': source_doc.doc_id,
                'has_selection': user_selection is not None,
            },
            artifact_filenames={
                'locate_result': f'locate_result_{writing_task.task_id or "task"}.json',
            },
        )
        return result.model_dump()

    def generate_modify_plan(
        self,
        task: Any,
        doc_ir: Any,
        locate_result: Any,
        context: Any,
    ) -> dict:
        writing_task = self._unified_model(task, WritingTask)
        source_doc = self._unified_model(doc_ir, DocIR)
        located = self._unified_model(locate_result, LocateResult)
        writing_context = self._unified_model(context, WritingContext)

        if not located.target_block_ids:
            raise ValueError('locate_result.target_block_ids is empty.')

        block_map = {b.block_id: b for b in self._iter_blocks(source_doc.blocks)}
        missing = [bid for bid in located.target_block_ids if bid not in block_map]
        if missing:
            raise ValueError(f'locate_result has block_ids absent from doc_ir: {missing}.')

        target_blocks = [block_map[bid] for bid in located.target_block_ids]
        focused_doc = DocIR(
            doc_id=source_doc.doc_id,
            title=source_doc.title,
            blocks=[b.model_copy(deep=True) for b in target_blocks],
        )

        prompt = GENERATE_MODIFY_PLAN_PROMPT.format(
            task_json=to_prompt_json(writing_task),
            doc_ir_json=to_prompt_json(focused_doc),
            locate_result_json=to_prompt_json(located),
            context_json=to_prompt_json(writing_context),
        )
        modify_plan = self._call_llm_structured(prompt, ModifyPlan)

        modify_plan = self._normalize_modify_plan(modify_plan, writing_task, located.target_block_ids)

        result = self._save_artifacts(
            {'modify_plan': modify_plan},
            step_name='generate_modify_plan',
            primary_key='modify_plan',
            context_key=None,
            summary='Generated modify plan.',
            counts={'instruction_count': len(modify_plan.instructions)},
            artifact_meta={
                'task_id': writing_task.task_id,
                'doc_id': source_doc.doc_id,
            },
            artifact_filenames={
                'modify_plan': f'modify_plan_{writing_task.task_id or "task"}.json',
            },
        )
        return result.model_dump()

    def generate_patch_set(
        self,
        doc_ir: Any,
        modify_plan: Any,
        context: Any,
    ) -> dict:
        source_doc = self._unified_model(doc_ir, DocIR)
        plan = self._unified_model(modify_plan, ModifyPlan)
        writing_context = self._unified_model(context, WritingContext)

        if not plan.instructions:
            raise ValueError('modify_plan.instructions is empty.')

        block_map = {b.block_id: b for b in self._iter_blocks(source_doc.blocks)}
        target_blocks, missing = self._resolve_target_blocks(block_map, plan.instructions)
        if missing:
            raise ValueError(f'modify_plan targets block_ids absent from doc_ir: {missing}.')

        focused_doc = DocIR(
            doc_id=source_doc.doc_id,
            title=source_doc.title,
            blocks=[b.model_copy(deep=True) for b in target_blocks],
        )

        prompt = GENERATE_PATCH_SET_PROMPT.format(
            doc_ir_json=to_prompt_json(focused_doc),
            modify_plan_json=to_prompt_json(plan),
            context_json=to_prompt_json(writing_context),
        )
        proposal = self._call_llm_structured(prompt, PatchSet)

        proposal_map: Dict[str, PatchHunk] = {
            h.target_block_id: h for h in proposal.hunks if h.target_block_id
        }

        hunks: List[PatchHunk] = []
        for instr, block in zip(plan.instructions, target_blocks):
            proposed = proposal_map.get(instr.target_block_id)
            if proposed is None or proposed.new_text is None:
                raise ValueError(
                    f'LLM did not produce new_text for block {instr.target_block_id!r}.'
                )
            hunks.append(PatchHunk(
                hunk_id=f'hunk-{instr.target_block_id}',
                target_block_id=instr.target_block_id,
                anchor=Anchor(
                    block_id=block.block_id,
                    heading_path=list(block.meta.get('heading_path', [])),
                ),
                old_text=block.text,
                new_text=proposed.new_text,
                meta={
                    'modify_type': instr.modify_type,
                    'instruction': instr.instruction,
                },
            ))

        patch_set = PatchSet(
            patch_id=f'patch-{source_doc.doc_id or "document"}',
            target_doc_id=source_doc.doc_id or '',
            hunks=hunks,
            meta={'source': 'generate_patch_set'},
        )

        result = self._save_artifacts(
            {'patch_set': patch_set},
            step_name='generate_patch_set',
            primary_key='patch_set',
            context_key=None,
            summary='Generated patch set.',
            counts={'hunk_count': len(hunks)},
            artifact_meta={
                'doc_id': source_doc.doc_id,
                'context_id': writing_context.context_id,
            },
            artifact_filenames={
                'patch_set': f'patch_set_{source_doc.doc_id or "document"}.json',
            },
        )
        return result.model_dump()

    def apply_patch(
        self,
        doc_ir: Any,
        patch_set: Any,
        context: Any,
    ) -> dict:
        source_doc = self._unified_model(doc_ir, DocIR)
        patch = self._unified_model(patch_set, PatchSet)
        writing_context = self._unified_model(context, WritingContext)

        if not patch.hunks:
            raise ValueError('patch_set.hunks is empty.')

        revised_doc = source_doc.model_copy(deep=True)
        block_map = {b.block_id: b for b in self._iter_blocks(revised_doc.blocks)}

        applied: List[str] = []
        failed: List[str] = []
        for hunk in patch.hunks:
            target_id = hunk.target_block_id or ''
            block = block_map.get(target_id)
            if block is None:
                failed.append(target_id)
                continue
            if hunk.old_text is not None and block.text == hunk.old_text:
                block.text = hunk.new_text or ''
                applied.append(target_id)
            else:
                failed.append(target_id)

        if not applied:
            raise ValueError('apply_patch produced no applied hunks; every hunk failed to match.')

        patch_result = PatchResult(
            patch_id=patch.patch_id,
            success=not failed,
            applied_hunks=applied,
            failed_hunks=failed,
            message='Patch applied.' if not failed else f'{len(failed)} hunk(s) failed to apply.',
            meta={
                'original_doc_id': source_doc.doc_id,
                'target_block_ids': [h.target_block_id for h in patch.hunks],
            },
        )

        result = self._save_artifacts(
            {'patch_result': patch_result, 'revised_doc_ir': revised_doc},
            step_name='apply_patch',
            primary_key='patch_result',
            context_key=None,
            summary='Applied patch to doc_ir.',
            counts={'applied': len(applied), 'failed': len(failed)},
            artifact_meta={
                'doc_id': source_doc.doc_id,
                'context_id': writing_context.context_id,
            },
            artifact_filenames={
                'patch_result': f'patch_result_{patch.patch_id or "patch"}.json',
                'revised_doc_ir': f'revised_doc_ir_{source_doc.doc_id or "document"}.json',
            },
        )
        return result.model_dump()

    def draft_to_doc_ir(self, draft: Any) -> dict:
        '''Convert a DraftDocument into a DocIR artifact.'''
        source_draft = self._unified_model(draft, DraftDocument)
        blocks: List[DocBlock] = []
        _flatten_sections(source_draft.sections, 1, [], blocks)
        doc_ir = DocIR(
            doc_id=source_draft.draft_id,
            title=source_draft.title,
            blocks=blocks,
            plain_text='\n\n'.join(b.text for b in blocks if b.text.strip()) or None,
            adapter='draft_document',
        )
        return self._save_artifacts(
            {'doc_ir': doc_ir},
            step_name='draft_to_doc_ir',
            primary_key='doc_ir',
            context_key=None,
            summary='Converted DraftDocument into DocIR.',
            counts={'blocks': len(blocks)},
            artifact_meta={'draft_id': source_draft.draft_id},
            artifact_filenames={
                'doc_ir': f'doc_ir_{source_draft.draft_id or "draft"}.json',
            },
        ).model_dump()

    def doc_ir_to_draft(self, doc_ir: Any) -> dict:
        '''Convert a DocIR into a DraftDocument artifact.'''
        source_doc = self._unified_model(doc_ir, DocIR)
        revised_draft = _doc_ir_to_draft(source_doc)
        return self._save_artifacts(
            {'revised_draft': revised_draft},
            step_name='doc_ir_to_draft',
            primary_key='revised_draft',
            context_key=None,
            summary='Converted DocIR into DraftDocument.',
            artifact_meta={'doc_id': source_doc.doc_id},
            artifact_filenames={
                'revised_draft': f'revised_draft_{source_doc.doc_id or "document"}.json',
            },
        ).model_dump()

    def _normalize_modify_plan(
        self,
        plan: ModifyPlan,
        task: WritingTask,
        located_block_ids: List[str],
    ) -> ModifyPlan:
        plan.task_id = task.task_id
        plan.target_block_ids = list(located_block_ids)

        located_set = set(located_block_ids)
        seen: set = set()
        normalized: List[ModifyInstruction] = []
        for instr in plan.instructions:
            target_id = instr.target_block_id
            if target_id not in located_set:
                raise ValueError(
                    f'modify_plan instruction targets block {target_id!r} '
                    f'not in locate_result.target_block_ids.'
                )
            if target_id in seen:
                raise ValueError(f'modify_plan has duplicate instruction for block {target_id!r}.')
            seen.add(target_id)
            normalized.append(instr)

        missing_instructions = located_set - seen
        if missing_instructions:
            raise ValueError(
                f'modify_plan missing instructions for blocks: {sorted(missing_instructions)}.'
            )
        plan.instructions = normalized
        return plan

    def _resolve_target_blocks(
        self,
        block_map: Dict[str, DocBlock],
        instructions: List[ModifyInstruction],
    ) -> Tuple[List[DocBlock], List[str]]:
        blocks: List[DocBlock] = []
        missing: List[str] = []
        for instr in instructions:
            block = block_map.get(instr.target_block_id)
            if block is None:
                missing.append(instr.target_block_id)
            else:
                blocks.append(block)
        return blocks, missing

    def _iter_blocks(self, blocks: List[DocBlock]) -> List[DocBlock]:
        result: List[DocBlock] = []
        for block in blocks:
            result.append(block)
            result.extend(self._iter_blocks(block.children))
        return result


def _doc_ir_to_draft(doc_ir: DocIR) -> DraftDocument:
    '''Rebuild a DraftDocument from a DocIR; section nesting is restored from heading levels.'''
    sections: List[DraftSection] = []
    stack: List[Tuple[int, DraftSection]] = []
    for block in doc_ir.blocks:
        if block.block_type == 'heading':
            while stack and stack[-1][0] >= block.level:
                stack.pop()
            meta = block.meta or {}
            section = DraftSection(
                **{field: meta.get(field) for field in SECTION_META_FIELDS},
                title=block.text or None,
            )
            if stack:
                stack[-1][1].sub_sections.append(section)
            else:
                sections.append(section)
            stack.append((block.level, section))
        else:
            if not stack:
                raise ValueError(f'Block {block.block_id!r} has no enclosing heading.')
            parent = stack[-1][1]
            meta = block.meta or {}
            parent.blocks.append(DraftBlock(
                block_id=block.block_id,
                section_id=parent.section_id,
                **{field: meta.get(field) for field in BLOCK_META_FIELDS},
                content=block.text,
            ))

    return DraftDocument(
        draft_id=doc_ir.doc_id,
        title=doc_ir.title,
        sections=sections,
    )


def _flatten_sections(
    sections: List[DraftSection],
    depth: int,
    heading_path: List[str],
    blocks: List[DocBlock],
) -> None:
    for section in sections:
        current_path = list(heading_path)
        if section.title:
            current_path.append(section.title)
        blocks.append(_heading_doc_block(section, depth, current_path, len(blocks)))
        for idx, block in enumerate(section.blocks, start=1):
            blocks.append(_paragraph_doc_block(section, block, current_path, idx, len(blocks)))
        _flatten_sections(section.sub_sections, depth + 1, current_path, blocks)


def _heading_doc_block(
    section: DraftSection,
    depth: int,
    heading_path: List[str],
    fallback_index: int,
) -> DocBlock:
    block_id = f'{section.section_id}::heading' if section.section_id else f'block-{fallback_index + 1}'
    meta: Dict[str, Any] = {HEADING_PATH_KEY: list(heading_path)}
    for field in SECTION_META_FIELDS:
        val = getattr(section, field)
        if val:
            meta[field] = val
    return DocBlock(
        block_id=block_id,
        block_type='heading',
        text=section.title or '',
        level=depth,
        meta=meta,
    )


def _paragraph_doc_block(
    section: DraftSection,
    block: DraftBlock,
    heading_path: List[str],
    local_index: int,
    fallback_index: int,
) -> DocBlock:
    if block.block_id:
        block_id = block.block_id
    elif section.section_id:
        block_id = f'{section.section_id}::block-{local_index}'
    else:
        block_id = f'block-{fallback_index + 1}'
    meta: Dict[str, Any] = {HEADING_PATH_KEY: list(heading_path)}
    for field in BLOCK_META_FIELDS:
        val = getattr(block, field)
        if val:
            meta[field] = val
    return DocBlock(
        block_id=block_id,
        block_type='paragraph',
        text=block.content,
        meta=meta,
    )
