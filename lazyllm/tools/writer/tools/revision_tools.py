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

            prompt = GENERATE_MODIFY_PLAN_PROMPT.format(
                task_json=to_prompt_json(writing_task),
                document_json=to_prompt_json(source_doc),
                locate_result_json=to_prompt_json(located),
                context_json=to_prompt_json(writing_context),
            )
            modify_plan = self._call_llm_structured(prompt, ModifyPlan)
        else:
            modify_plan = ModifyPlan(scope='document', summary='No revision targets; nothing to plan.')

        modify_plan = self._normalize_modify_plan(
            modify_plan,
            writing_task,
            located.target_node_ids,
            {block.node_id for block in source_doc.iter_blocks()},
        )

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
            if len(proposal_map) != len(proposal.hunks):
                raise ValueError('patch proposal contains duplicate or empty target_node_id values.')
            expected_targets = {instruction.target_node_id for instruction in plan.instructions}
            unexpected_targets = set(proposal_map) - expected_targets
            if unexpected_targets:
                raise ValueError(
                    f'patch proposal contains targets absent from modify_plan: {sorted(unexpected_targets)}.'
                )

            for instr, block in zip(plan.instructions, target_blocks):
                proposed = proposal_map.get(instr.target_node_id)
                if proposed is None:
                    raise ValueError(f'lack hunk for block {instr.target_node_id!r}.')
                if proposed.modify_type != instr.modify_type:
                    raise ValueError(
                        f'patch hunk for {instr.target_node_id!r} changed modify_type '
                        f'from {instr.modify_type!r} to {proposed.modify_type!r}.'
                    )

                position = instr.position or proposed.position
                anchor_node_id = instr.anchor_node_id or proposed.anchor_node_id
                if instr.modify_type == 'replace' and proposed.new_text is None:
                    raise ValueError(f'lack new_text for block {instr.target_node_id!r}.')
                if instr.modify_type == 'insert':
                    position = position or 'after'
                    if not proposed.new_blocks:
                        raise ValueError(f'lack new_blocks for insert at {instr.target_node_id!r}.')
                if instr.modify_type == 'move' and (not anchor_node_id or not position):
                    raise ValueError(
                        f'move for {instr.target_node_id!r} requires anchor_node_id and position.'
                    )

                hunks.append(PatchHunk(
                    hunk_id=f'hunk-{instr.target_node_id}',
                    target_node_id=instr.target_node_id,
                    modify_type=instr.modify_type,
                    anchor=Anchor(
                        node_id=block.node_id,
                    ),
                    old_text=None if instr.modify_type == 'insert' else block.content,
                    new_text=proposed.new_text if instr.modify_type == 'replace' else None,
                    new_blocks=[item.model_copy(deep=True) for item in proposed.new_blocks]
                    if instr.modify_type == 'insert' else [],
                    anchor_node_id=anchor_node_id if instr.modify_type == 'move' else None,
                    position=position if instr.modify_type in {'insert', 'move'} else None,
                    meta={**proposed.meta, 'instruction': instr.instruction},
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

        self._validate_patch(source_doc, patch)
        revised_doc = source_doc.model_copy(deep=True)

        applied: List[str] = []
        for hunk in patch.hunks:
            node_map: Dict[str, WriterBlock] = {b.node_id: b for b in revised_doc.iter_blocks()}
            target = node_map[hunk.target_node_id]

            if hunk.modify_type == 'replace':
                target.content = hunk.new_text or ''
                # A block-level replacement cannot safely retain offsets/styles from old spans.
                target.spans = []
            elif hunk.modify_type == 'insert':
                new_blocks = self._build_inserted_blocks(revised_doc, hunk)
                self._insert_siblings(revised_doc, target, new_blocks, hunk.position or 'after')
            elif hunk.modify_type == 'delete':
                self._remove_block(revised_doc, target)
            elif hunk.modify_type == 'move':
                anchor = node_map[hunk.anchor_node_id or '']
                self._remove_block(revised_doc, target)
                self._insert_siblings(revised_doc, anchor, [target], hunk.position or 'after')
            else:
                raise ValueError(f'unsupported modify_type: {hunk.modify_type!r}.')

            applied.append(hunk.hunk_id or hunk.target_node_id)

        # Re-validate after mutations because pydantic assignment validation is not enabled.
        revised_doc = WriterDocument.model_validate(revised_doc.model_dump())

        patch_result = PatchResult(
            patch_id=patch.patch_id,
            success=True,
            applied_hunks=applied,
            failed_hunks=[],
            message='Patch applied.',
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
            counts={'applied': len(applied), 'failed': 0},
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
        valid_node_ids: set,
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
            if instr.modify_type == 'insert':
                instr.position = instr.position or 'after'
            if instr.modify_type == 'move':
                if instr.anchor_node_id not in valid_node_ids:
                    raise ValueError(
                        f'move instruction anchor {instr.anchor_node_id!r} is absent from document.'
                    )
                if instr.anchor_node_id == target_id:
                    raise ValueError('move instruction cannot use the target block as its own anchor.')
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

    def _validate_patch(self, document: WriterDocument, patch: PatchSet) -> None:
        if patch.target_doc_id != document.document_id:
            raise ValueError(
                f'patch target_doc_id {patch.target_doc_id!r} does not match '
                f'document_id {document.document_id!r}.'
            )

        node_map = {block.node_id: block for block in document.iter_blocks()}
        target_ids: set = set()
        hunk_ids: set = set()
        for hunk in patch.hunks:
            target_id = hunk.target_node_id
            if target_id in target_ids:
                raise ValueError(f'patch contains multiple hunks for target {target_id!r}.')
            target_ids.add(target_id)
            if hunk.hunk_id:
                if hunk.hunk_id in hunk_ids:
                    raise ValueError(f'patch contains duplicate hunk_id {hunk.hunk_id!r}.')
                hunk_ids.add(hunk.hunk_id)

            self._validate_patch_hunk(node_map, hunk)

        destructive_targets = {
            hunk.target_node_id for hunk in patch.hunks
            if hunk.modify_type in {'delete', 'move'}
        }
        for hunk in patch.hunks:
            if hunk.modify_type == 'move' and hunk.anchor_node_id in destructive_targets:
                raise ValueError(
                    f'move anchor {hunk.anchor_node_id!r} is also deleted or moved by this patch.'
                )

    def _validate_patch_hunk(
        self,
        node_map: Dict[str, WriterBlock],
        hunk: PatchHunk,
    ) -> None:
        target_id = hunk.target_node_id
        target = node_map.get(target_id)
        if target is None:
            raise ValueError(f'patch target {target_id!r} is absent from document.')
        if not target.editable:
            raise ValueError(f'patch target {target_id!r} is not editable.')
        if hunk.old_text is not None and target.content != hunk.old_text:
            raise ValueError(f'patch old_text conflict for target {target_id!r}.')

        validators = {
            'replace': self._validate_replace_hunk,
            'insert': self._validate_insert_hunk,
            'move': self._validate_move_hunk,
        }
        validator = validators.get(hunk.modify_type)
        if validator:
            validator(node_map, target, hunk)

    def _validate_replace_hunk(
        self,
        node_map: Dict[str, WriterBlock],
        target: WriterBlock,
        hunk: PatchHunk,
    ) -> None:
        if hunk.new_text is None:
            raise ValueError(f'replace hunk for {target.node_id!r} requires new_text.')

    def _validate_insert_hunk(
        self,
        node_map: Dict[str, WriterBlock],
        target: WriterBlock,
        hunk: PatchHunk,
    ) -> None:
        if not hunk.new_blocks:
            raise ValueError(f'insert hunk for {target.node_id!r} requires new_blocks.')
        if hunk.position not in {'before', 'after'}:
            raise ValueError(f'insert hunk for {target.node_id!r} requires position.')

    def _validate_move_hunk(
        self,
        node_map: Dict[str, WriterBlock],
        target: WriterBlock,
        hunk: PatchHunk,
    ) -> None:
        anchor_id = hunk.anchor_node_id or ''
        anchor = node_map.get(anchor_id)
        if anchor is None:
            raise ValueError(f'move anchor {anchor_id!r} is absent from document.')
        if anchor is target:
            raise ValueError('move target cannot be its own anchor.')
        if hunk.position not in {'before', 'after'}:
            raise ValueError(f'move hunk for {target.node_id!r} requires position.')
        if self._contains_block(target, anchor):
            raise ValueError('move target cannot be moved relative to one of its descendants.')

    def _build_inserted_blocks(
        self,
        document: WriterDocument,
        hunk: PatchHunk,
    ) -> List[WriterBlock]:
        used_ids = {block.node_id for block in document.iter_blocks()}
        base_id = hunk.hunk_id or f'hunk-{hunk.target_node_id}'
        result: List[WriterBlock] = []
        for index, patch_block in enumerate(hunk.new_blocks, start=1):
            candidate = f'{base_id}::block-{index}'
            suffix = 2
            while candidate in used_ids:
                candidate = f'{base_id}::block-{index}-{suffix}'
                suffix += 1
            used_ids.add(candidate)
            result.append(WriterBlock(
                node_id=candidate,
                type=patch_block.type,
                content=patch_block.content,
                stage=document.stage,
                authoring=patch_block.authoring.model_copy(deep=True)
                if patch_block.authoring else None,
                numbering=dict(patch_block.numbering),
            ))
        return result

    def _insert_siblings(
        self,
        document: WriterDocument,
        anchor: WriterBlock,
        new_blocks: List[WriterBlock],
        position: str,
    ) -> None:
        siblings = self._sibling_list(document, anchor)
        index = siblings.index(anchor) + (1 if position == 'after' else 0)
        siblings[index:index] = new_blocks

    def _remove_block(self, document: WriterDocument, target: WriterBlock) -> None:
        self._sibling_list(document, target).remove(target)

    def _sibling_list(
        self,
        document: WriterDocument,
        target: WriterBlock,
    ) -> List[WriterBlock]:
        for block in document.blocks:
            if target is block:
                return document.blocks
            owner = self._find_parent(block, target)
            if owner is not None:
                return owner
        raise ValueError(f'block {target.node_id!r} is detached from document.')

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

    def _contains_block(self, candidate: WriterBlock, target: WriterBlock) -> bool:
        return any(
            child is target or self._contains_block(child, target)
            for child in candidate.children
        )
