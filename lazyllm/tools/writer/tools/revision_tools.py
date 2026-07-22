from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Tuple

from .base import WriterToolBase
from ..data_models.context import WritingContext
from ..data_models.revision import (
    Anchor,
    LocateResult,
    ModifyInstruction,
    ModifyPlan,
    PatchHunk,
    PatchResult,
    PatchBlock,
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


def apply_patch_to_ir(
    document: WriterDocument,
    patch_set: PatchSet,
) -> Tuple[WriterDocument, PatchResult]:
    '''Apply a provider-neutral patch without artifact or context dependencies.'''
    tools = WriterRevisionTools()
    tools._validate_patch(document, patch_set)
    revised_doc = document.model_copy(deep=True)
    revised_doc.ui_editable = False
    if patch_set.new_title is not None:
        revised_doc.title = patch_set.new_title

    applied: List[str] = []
    for hunk in patch_set.hunks:
        node_map = {block.node_id: block for block in revised_doc.iter_blocks()}
        target = node_map[hunk.target_node_id]

        if hunk.modify_type == 'replace':
            target.content = hunk.new_text or ''
            target.spans = []
        elif hunk.modify_type == 'insert':
            new_blocks = tools._build_inserted_blocks(revised_doc, hunk)
            tools._insert_siblings(revised_doc, target, new_blocks, hunk.position or 'after')
        elif hunk.modify_type == 'delete':
            tools._remove_block(revised_doc, target)
        elif hunk.modify_type == 'move':
            anchor = node_map[hunk.anchor_node_id or '']
            tools._remove_block(revised_doc, target)
            tools._insert_siblings(revised_doc, anchor, [target], hunk.position or 'after')
        else:
            raise ValueError(f'unsupported modify_type: {hunk.modify_type!r}.')

        applied.append(hunk.hunk_id or hunk.target_node_id)

    revised_doc = WriterDocument.model_validate(revised_doc.model_dump())
    result = PatchResult(
        patch_id=patch_set.patch_id,
        success=True,
        applied_hunks=applied,
        failed_hunks=[],
        message='Patch applied.',
        meta={
            'original_doc_id': document.document_id,
            'target_node_ids': [h.target_node_id for h in patch_set.hunks],
            'title_updated': patch_set.new_title is not None,
        },
    )
    return revised_doc, result


class WriterRevisionTools(WriterToolBase):
    __public_apis__ = [
        'locate_revision_target',
        'generate_modify_plan',
        'generate_patch_set',
        'build_patch_set_from_documents',
        'apply_patch',
    ]

    def build_patch_set_from_documents(
        self,
        source_document: Any,
        revised_document: Any,
    ) -> dict:
        '''Build a deterministic PatchSet from a user-edited WriterDocument.'''
        source = self._unified_model(source_document, WriterDocument)
        revised = self._unified_model(revised_document, WriterDocument)
        patch = self._diff_documents(source, revised)
        result = self._save_artifacts(
            {'patch_set': patch},
            step_name='build_patch_set_from_documents',
            primary_key='patch_set',
            context_key=None,
            summary='Built patch set from WriterDocument revisions.',
            counts={'hunk_count': len(patch.hunks)},
            artifact_meta={
                'document_id': source.document_id,
                'title_updated': patch.new_title is not None,
            },
            artifact_filenames={
                'patch_set': f'patch_set_{source.document_id or "document"}.json',
            },
        )
        return result.model_dump()

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
        proposed_title: Optional[str] = None
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
                ui_editable=False,
            )

            prompt = GENERATE_PATCH_SET_PROMPT.format(
                document_json=to_prompt_json(focused_doc),
                modify_plan_json=to_prompt_json(plan),
                context_json=to_prompt_json(writing_context),
            )
            proposal = self._call_llm_structured(prompt, PatchSet)
            proposed_title = proposal.new_title

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
            new_title=proposed_title if proposed_title != source_doc.title else None,
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

        revised_doc, patch_result = apply_patch_to_ir(source_doc, patch)

        result = self._save_artifacts(
            {'patch_result': patch_result, 'revised_document': revised_doc},
            step_name='apply_patch',
            primary_key='patch_result',
            context_key=None,
            summary='Applied patch to document.',
            counts={'applied': len(patch_result.applied_hunks), 'failed': 0},
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

    def _diff_documents(  # noqa: C901
        self,
        source: WriterDocument,
        revised: WriterDocument,
    ) -> PatchSet:
        if source.document_id != revised.document_id:
            raise ValueError('source and revised documents must have the same document_id.')
        for field in ('stage', 'revision', 'provider_binding'):
            if getattr(source, field) != getattr(revised, field):
                raise ValueError(f'source and revised documents must have the same {field}.')

        source_map = {block.node_id: block for block in source.iter_blocks()}
        revised_map = {block.node_id: block for block in revised.iter_blocks()}
        common_ids = set(source_map) & set(revised_map)
        new_ids = set(revised_map) - set(source_map)
        deleted_ids = set(source_map) - set(revised_map)

        source_layout = self._document_layout(source)
        revised_layout = self._document_layout(revised)
        hunks: List[PatchHunk] = []

        for node_id in self._ordered_node_ids(revised):
            if node_id not in common_ids:
                continue
            old_block = source_map[node_id]
            new_block = revised_map[node_id]
            self._validate_diffable_block(old_block, new_block)
            if old_block.content != new_block.content:
                hunks.append(PatchHunk(
                    hunk_id=f'update-{node_id}',
                    target_node_id=node_id,
                    modify_type='replace',
                    old_text=old_block.content,
                    new_text=new_block.content,
                ))

        hunks.extend(self._build_move_hunks(
            source, revised, source_layout, revised_layout, common_ids))

        for group in self._new_block_groups(revised, new_ids):
            anchor_id, position = self._insert_anchor(group, revised_layout, common_ids)
            if anchor_id is None:
                raise ValueError(
                    'insert cannot be represented because the target sibling list '
                    'has no existing block anchor.'
                )
            patch_blocks: List[PatchBlock] = []
            for node_id in group:
                block = revised_map[node_id]
                if block.children:
                    raise ValueError('inserting blocks with children is not supported yet.')
                if block.spans and any(span.style for span in block.spans):
                    raise ValueError('inserting styled spans is not supported yet.')
                patch_blocks.append(PatchBlock(
                    type=block.type,
                    content=block.content,
                    numbering=dict(block.numbering),
                    authoring=block.authoring.model_copy(deep=True) if block.authoring else None,
                ))
            hunks.append(PatchHunk(
                hunk_id=f'insert-{group[0]}',
                target_node_id=anchor_id,
                modify_type='insert',
                position=position,
                new_blocks=patch_blocks,
            ))

        top_level_deletes = {
            node_id for node_id in deleted_ids
            if source_layout[node_id][0] not in deleted_ids
        }
        for node_id in self._ordered_node_ids(source):
            if node_id in top_level_deletes:
                hunks.append(PatchHunk(
                    hunk_id=f'delete-{node_id}',
                    target_node_id=node_id,
                    modify_type='delete',
                    old_text=source_map[node_id].content,
                ))

        patch = PatchSet(
            patch_id=f'patch-{source.document_id}',
            target_doc_id=source.document_id,
            new_title=revised.title if source.title != revised.title else None,
            hunks=hunks,
            meta={'source': 'document_diff'},
        )
        self._validate_patch(source, patch)
        return patch

    @staticmethod
    def _ordered_node_ids(document: WriterDocument) -> List[str]:
        return [block.node_id for block in document.iter_blocks()]

    @staticmethod
    def _document_layout(
        document: WriterDocument,
    ) -> Dict[str, Tuple[Optional[str], List[str], int]]:
        layout: Dict[str, Tuple[Optional[str], List[str], int]] = {}

        def walk(blocks: List[WriterBlock], parent_id: Optional[str]) -> None:
            sibling_ids = [block.node_id for block in blocks]
            for index, block in enumerate(blocks):
                layout[block.node_id] = (parent_id, sibling_ids, index)
                walk(block.children, block.node_id)

        walk(document.blocks, None)
        return layout

    def _build_move_hunks(
        self,
        source: WriterDocument,
        revised: WriterDocument,
        source_layout: Dict[str, Tuple[Optional[str], List[str], int]],
        revised_layout: Dict[str, Tuple[Optional[str], List[str], int]],
        common_ids: Set[str],
    ) -> List[PatchHunk]:
        current_parent = {
            node_id: source_layout[node_id][0] for node_id in common_ids
        }
        current_lists: Dict[Optional[str], List[str]] = {}
        for node_id in self._ordered_node_ids(source):
            if node_id in common_ids:
                current_lists.setdefault(current_parent[node_id], []).append(node_id)

        target_parents: List[Optional[str]] = [None]
        target_parents.extend(
            node_id for node_id in self._ordered_node_ids(revised)
            if any(
                revised_layout[child_id][0] == node_id
                for child_id in common_ids
            )
        )

        hunks: List[PatchHunk] = []
        for parent_id in target_parents:
            target_order = [
                node_id for node_id in common_ids
                if revised_layout[node_id][0] == parent_id
            ]
            target_order.sort(key=lambda node_id: revised_layout[node_id][2])
            current_lists.setdefault(parent_id, [])

            for index, node_id in enumerate(target_order):
                current = current_lists[parent_id]
                previous_id = target_order[index - 1] if index else None
                correctly_placed = (
                    current_parent[node_id] == parent_id
                    and node_id in current
                    and (
                        (previous_id is None and current.index(node_id) == 0)
                        or (
                            previous_id is not None
                            and current.index(node_id) == current.index(previous_id) + 1
                        )
                    )
                )
                if correctly_placed:
                    continue

                if previous_id is not None:
                    anchor_id, position = previous_id, 'after'
                else:
                    anchor_id = next(
                        (
                            sibling_id for sibling_id in target_order[index + 1:]
                            if current_parent[sibling_id] == parent_id
                        ),
                        None,
                    )
                    position = 'before'
                if anchor_id is None:
                    raise ValueError(
                        f'move for {node_id!r} cannot be represented because its target '
                        'parent has no stable sibling anchor.'
                    )

                old_parent = current_parent[node_id]
                current_lists[old_parent].remove(node_id)
                target_list = current_lists[parent_id]
                anchor_index = target_list.index(anchor_id)
                insert_index = anchor_index + (1 if position == 'after' else 0)
                target_list.insert(insert_index, node_id)
                current_parent[node_id] = parent_id
                hunks.append(PatchHunk(
                    hunk_id=f'move-{node_id}',
                    target_node_id=node_id,
                    modify_type='move',
                    anchor_node_id=anchor_id,
                    position=position,
                ))
        return hunks

    def _new_block_groups(
        self,
        document: WriterDocument,
        new_ids: Set[str],
    ) -> List[List[str]]:
        groups: List[List[str]] = []

        def walk(blocks: List[WriterBlock]) -> None:
            current: List[str] = []
            for block in blocks:
                if block.node_id in new_ids:
                    current.append(block.node_id)
                else:
                    if current:
                        groups.append(current)
                        current = []
                    walk(block.children)
            if current:
                groups.append(current)

        walk(document.blocks)
        return groups

    @staticmethod
    def _insert_anchor(
        group: List[str],
        layout: Dict[str, Tuple[Optional[str], List[str], int]],
        common_ids: Set[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        _, siblings, first_index = layout[group[0]]
        last_index = layout[group[-1]][2]
        for sibling_id in reversed(siblings[:first_index]):
            if sibling_id in common_ids:
                return sibling_id, 'after'
        for sibling_id in siblings[last_index + 1:]:
            if sibling_id in common_ids:
                return sibling_id, 'before'
        return None, None

    @staticmethod
    def _validate_diffable_block(source: WriterBlock, revised: WriterBlock) -> None:
        fields = (
            'type', 'stage', 'status', 'authoring', 'numbering', 'references',
            'provider_binding', 'provider_payload', 'editable',
        )
        changed = [name for name in fields if getattr(source, name) != getattr(revised, name)]
        if changed:
            raise ValueError(
                f'block {source.node_id!r} changes unsupported fields: {changed}.'
            )
        if source.content == revised.content and source.spans != revised.spans:
            raise ValueError(
                f'block {source.node_id!r} changes spans without changing text; '
                'span-only patches are not supported yet.'
            )
        if source.content != revised.content and any(span.style for span in revised.spans):
            raise ValueError(
                f'block {source.node_id!r} contains edited styled spans; '
                'styled span patches are not supported yet.'
            )

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
        hunk_ids: set = set()
        for hunk in patch.hunks:
            if hunk.hunk_id:
                if hunk.hunk_id in hunk_ids:
                    raise ValueError(f'patch contains duplicate hunk_id {hunk.hunk_id!r}.')
                hunk_ids.add(hunk.hunk_id)

            self._validate_patch_hunk(node_map, hunk)

        destructive_targets = {
            hunk.target_node_id for hunk in patch.hunks
            if hunk.modify_type == 'delete'
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
