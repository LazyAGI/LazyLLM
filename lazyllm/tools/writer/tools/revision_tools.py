from __future__ import annotations
from typing import Any, Dict, List, Tuple

from .base import WriterToolBase
from ..data_models.context import WritingContext
from ..data_models.revision import (
    Anchor,
    LocateResult,
    ModifyInstruction,
    ModifyPlan,
    PatchHunk,
    PatchResult,
    PatchSet,
)
from ..data_models.task import WritingTask
from ..data_models.writer_ir import (
    WriterBlock,
    WriterDocument,
)
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
    ]

    def locate_revision_target(
        self,
        task: Any,
        document: Any,
        context: Any,
    ) -> dict:
        writing_task = self._unified_model(task, WritingTask)
        if writing_task.task_type != 'revise':
            raise ValueError(
                f'task.task_type must be \'revise\', got {writing_task.task_type!r}.'
            )
        source_doc = self._unified_model(document, WriterDocument)
        writing_context = self._unified_model(context, WritingContext)
        user_selection = writing_task.selection

        valid_node_ids = {b.node_id for b in source_doc.iter_blocks()}
        if not valid_node_ids:
            raise ValueError('document must contain at least one block.')

        candidate_node_ids = valid_node_ids
        if user_selection and user_selection.block_ids:
            selected = set(user_selection.block_ids) & valid_node_ids
            if not selected:
                raise ValueError('selection.block_ids contains no valid node_ids.')
            candidate_node_ids = selected

        prompt = LOCATE_REVISION_TARGET_PROMPT.format(
            task_json=to_prompt_json(writing_task),
            document_json=to_prompt_json(source_doc),
            context_json=to_prompt_json(writing_context),
            candidate_node_ids=to_prompt_json(sorted(candidate_node_ids)),
        )
        locate_result = self._call_llm_structured(prompt, LocateResult)

        located = [nid for nid in (locate_result.target_node_ids or []) if nid]
        invalid = [nid for nid in located if nid not in candidate_node_ids]
        if invalid:
            raise ValueError(
                f'locate_result contains node_ids not in candidates: {invalid}.'
            )
        locate_result.target_node_ids = list(dict.fromkeys(located))
        locate_result.target_reasons = {
            nid: locate_result.target_reasons.get(nid, '')
            for nid in locate_result.target_node_ids
        }
        locate_result.task_id = writing_task.task_id
        locate_result.doc_id = source_doc.document_id

        result = self._save_artifacts(
            {'locate_result': locate_result},
            step_name='locate_revision_target',
            primary_key='locate_result',
            context_key=None,
            summary=(
                'No revision target blocks located.'
                if not locate_result.target_node_ids
                else 'Located revision target blocks.'
            ),
            counts={'target_node_count': len(locate_result.target_node_ids)},
            artifact_meta={
                'task_id': writing_task.task_id,
                'document_id': source_doc.document_id,
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
        document: Any,
        locate_result: Any,
        context: Any,
    ) -> dict:
        writing_task = self._unified_model(task, WritingTask)
        source_doc = self._unified_model(document, WriterDocument)
        located = self._unified_model(locate_result, LocateResult)
        writing_context = self._unified_model(context, WritingContext)

        if located.target_node_ids:
            block_map = {b.node_id: b for b in source_doc.iter_blocks()}
            missing = [nid for nid in located.target_node_ids if nid not in block_map]
            if missing:
                raise ValueError(f'locate_result has node_ids absent from document: {missing}.')

            target_blocks = [block_map[nid] for nid in located.target_node_ids]
            focused_doc = WriterDocument(
                document_id=source_doc.document_id,
                stage=source_doc.stage,
                title=source_doc.title,
                blocks=[b.model_copy(deep=True) for b in target_blocks],
            )

            prompt = GENERATE_MODIFY_PLAN_PROMPT.format(
                task_json=to_prompt_json(writing_task),
                document_json=to_prompt_json(focused_doc),
                locate_result_json=to_prompt_json(located),
                context_json=to_prompt_json(writing_context),
            )
            modify_plan = self._call_llm_structured(prompt, ModifyPlan)
        else:
            modify_plan = ModifyPlan(scope='document', summary='No revision targets; nothing to plan.')

        modify_plan = self._normalize_modify_plan(modify_plan, writing_task, located.target_node_ids)

        result = self._save_artifacts(
            {'modify_plan': modify_plan},
            step_name='generate_modify_plan',
            primary_key='modify_plan',
            context_key=None,
            summary='Generated modify plan.',
            counts={'instruction_count': len(modify_plan.instructions)},
            artifact_meta={
                'task_id': writing_task.task_id,
                'document_id': source_doc.document_id,
            },
            artifact_filenames={
                'modify_plan': f'modify_plan_{writing_task.task_id or "task"}.json',
            },
        )
        return result.model_dump()

    def generate_patch_set(
        self,
        document: Any,
        modify_plan: Any,
        context: Any,
    ) -> dict:
        source_doc = self._unified_model(document, WriterDocument)
        plan = self._unified_model(modify_plan, ModifyPlan)
        writing_context = self._unified_model(context, WritingContext)

        hunks: List[PatchHunk] = []
        if plan.instructions:
            block_map = {b.node_id: b for b in source_doc.iter_blocks()}
            target_blocks, missing = self._resolve_target_blocks(block_map, plan.instructions)
            if missing:
                raise ValueError(f'modify_plan targets node_ids absent from document: {missing}.')

            focused_doc = WriterDocument(
                document_id=source_doc.document_id,
                stage=source_doc.stage,
                title=source_doc.title,
                blocks=[b.model_copy(deep=True) for b in target_blocks],
            )

            prompt = GENERATE_PATCH_SET_PROMPT.format(
                document_json=to_prompt_json(focused_doc),
                modify_plan_json=to_prompt_json(plan),
                context_json=to_prompt_json(writing_context),
            )
            proposal = self._call_llm_structured(prompt, PatchSet)

            proposal_map: Dict[str, PatchHunk] = {
                h.target_node_id: h for h in proposal.hunks if h.target_node_id
            }

            for instr, block in zip(plan.instructions, target_blocks):
                proposed = proposal_map.get(instr.target_node_id)
                if proposed is None:
                    raise ValueError(f'lack hunk for block {instr.target_node_id!r}.')
                if instr.modify_type != 'delete' and not proposed.new_text:
                    raise ValueError(f'lack new_text for block {instr.target_node_id!r}.')
                hunks.append(PatchHunk(
                    hunk_id=f'hunk-{instr.target_node_id}',
                    target_node_id=instr.target_node_id,
                    modify_type=instr.modify_type,
                    anchor=Anchor(
                        node_id=block.node_id,
                    ),
                    old_text=None if instr.modify_type == 'insert' else block.content,
                    new_text=proposed.new_text,
                    meta={'instruction': instr.instruction},
                ))

        patch_set = PatchSet(
            patch_id=f'patch-{source_doc.document_id or "document"}',
            target_doc_id=source_doc.document_id or '',
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
                'document_id': source_doc.document_id,
                'context_id': writing_context.context_id,
            },
            artifact_filenames={
                'patch_set': f'patch_set_{source_doc.document_id or "document"}.json',
            },
        )
        return result.model_dump()

    def apply_patch(
        self,
        document: Any,
        patch_set: Any,
        context: Any,
    ) -> dict:
        source_doc = self._unified_model(document, WriterDocument)
        patch = self._unified_model(patch_set, PatchSet)
        writing_context = self._unified_model(context, WritingContext)

        revised_doc = source_doc.model_copy(deep=True)

        applied: List[str] = []
        failed: List[str] = []
        if patch.hunks:
            node_map: Dict[str, WriterBlock] = {b.node_id: b for b in revised_doc.iter_blocks()}
            for hunk in patch.hunks:
                target_id = hunk.target_node_id or ''
                block = node_map.get(target_id)
                if block is None:
                    failed.append(target_id)
                    continue

                applied_ok = False
                if hunk.modify_type == 'insert':
                    self._insert_sibling_after(revised_doc, block, WriterBlock(
                        node_id=f'{target_id}::inserted',
                        type=block.type or 'paragraph',
                        content=hunk.new_text,
                        stage=revised_doc.stage,
                    ))
                    applied_ok = True
                elif hunk.old_text is not None and block.content == hunk.old_text:
                    if hunk.modify_type == 'delete':
                        self._remove_block(revised_doc, block)
                    else:  # replace
                        block.content = hunk.new_text
                    applied_ok = True

                if applied_ok:
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
                'original_doc_id': source_doc.document_id,
                'target_node_ids': [h.target_node_id for h in patch.hunks],
            },
        )

        result = self._save_artifacts(
            {'patch_result': patch_result, 'revised_document': revised_doc},
            step_name='apply_patch',
            primary_key='patch_result',
            context_key=None,
            summary='Applied patch to document.',
            counts={'applied': len(applied), 'failed': len(failed)},
            artifact_meta={
                'document_id': source_doc.document_id,
                'context_id': writing_context.context_id,
            },
            artifact_filenames={
                'patch_result': f'patch_result_{patch.patch_id or "patch"}.json',
                'revised_document': f'revised_document_{source_doc.document_id or "document"}.json',
            },
        )
        return result.model_dump()

    def _normalize_modify_plan(
        self,
        plan: ModifyPlan,
        task: WritingTask,
        located_node_ids: List[str],
    ) -> ModifyPlan:
        plan.plan_id = plan.plan_id or f'plan-{task.task_id or "task"}'
        plan.task_id = task.task_id
        plan.target_node_ids = list(located_node_ids)

        located_set = set(located_node_ids)
        seen: set = set()
        normalized: List[ModifyInstruction] = []
        for instr in plan.instructions:
            target_id = instr.target_node_id
            if target_id not in located_set:
                raise ValueError(
                    f'modify_plan instruction targets block {target_id!r} '
                    f'not in locate_result.target_node_ids.'
                )
            if target_id in seen:
                raise ValueError(f'modify_plan has duplicate instruction for block {target_id!r}.')
            seen.add(target_id)
            instr.instruction_id = instr.instruction_id or f'instr-{target_id}'
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
        block_map: Dict[str, WriterBlock],
        instructions: List[ModifyInstruction],
    ) -> Tuple[List[WriterBlock], List[str]]:
        blocks: List[WriterBlock] = []
        missing: List[str] = []
        for instr in instructions:
            block = block_map.get(instr.target_node_id)
            if block is None:
                missing.append(instr.target_node_id)
            else:
                blocks.append(block)
        return blocks, missing

    def _insert_sibling_after(
        self,
        document: WriterDocument,
        target: WriterBlock,
        new_block: WriterBlock,
    ) -> None:
        parent, siblings = self._locate_sibling_list(document, target)
        if siblings is None:
            document.blocks.insert(document.blocks.index(target) + 1, new_block)
        else:
            siblings.insert(siblings.index(target) + 1, new_block)

    def _remove_block(self, document: WriterDocument, target: WriterBlock) -> None:
        parent, siblings = self._locate_sibling_list(document, target)
        if siblings is None:
            document.blocks.remove(target)
        else:
            siblings.remove(target)

    def _locate_sibling_list(
        self,
        document: WriterDocument,
        target: WriterBlock,
    ) -> Tuple[WriterBlock, List[WriterBlock]]:
        '''Find the children list that owns target. Returns (parent, list) or (None, None) at top level.'''
        for block in document.blocks:
            if target is block:
                return None, None
            owner = self._find_parent(block, target)
            if owner is not None:
                return block, owner
        return None, None

    def _find_parent(
        self,
        candidate: WriterBlock,
        target: WriterBlock,
    ) -> List[WriterBlock]:
        for child in candidate.children:
            if child is target:
                return candidate.children
            deeper = self._find_parent(child, target)
            if deeper is not None:
                return deeper
        return None
