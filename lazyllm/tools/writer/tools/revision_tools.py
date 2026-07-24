from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple

from .base import WriterToolBase
from ..data_models.context import WritingContext
from ..data_models.revision import (
    LocateResult,
    ModifyInstruction,
    ModifyPlan,
    PatchHunk,
    PatchResult,
    PatchSet,
)
from ..data_models.task import WritingTask
from ..data_models.writer_ir import (
    WRITER_BLOCK_MUTABLE_FIELDS,
    WRITER_BLOCK_PROVIDER_MANAGED_FIELDS,
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
        tools._apply_patch_hunk(revised_doc, hunk)
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

    def _apply_patch_hunk(
        self,
        document: WriterDocument,
        hunk: PatchHunk,
    ) -> None:
        target = document.block_by_id(hunk.target_node_id)
        if hunk.modify_type == 'update':
            if target is None or hunk.block is None:
                raise ValueError(
                    f'update target {hunk.target_node_id!r} is absent from document.')
            self._apply_block_update(target, hunk.block)
            return
        if hunk.modify_type == 'create':
            self._apply_create_hunk(document, hunk)
            return
        if target is None:
            raise ValueError(
                f'{hunk.modify_type} target {hunk.target_node_id!r} is absent from document.')
        self._remove_block(document, target)
        if hunk.modify_type == 'delete':
            return
        if hunk.modify_type != 'move':
            raise ValueError(f'unsupported modify_type: {hunk.modify_type!r}.')
        if hunk.parent_node_id and self._subtree_has_id(target, hunk.parent_node_id):
            raise ValueError('move target cannot be moved into its own subtree.')
        siblings = self._children_for_parent(document, hunk.parent_node_id)
        if hunk.index is None or hunk.index > len(siblings):
            raise ValueError(
                f'move index for {hunk.target_node_id!r} is outside its parent.')
        siblings.insert(hunk.index, target)

    def _apply_create_hunk(
        self,
        document: WriterDocument,
        hunk: PatchHunk,
    ) -> None:
        if document.block_by_id(hunk.target_node_id) is not None:
            raise ValueError(
                f'create target {hunk.target_node_id!r} already exists in document.')
        if hunk.block is None:
            raise ValueError(f'create hunk {hunk.target_node_id!r} lacks block.')
        siblings = self._children_for_parent(document, hunk.parent_node_id)
        if hunk.index is None or hunk.index > len(siblings):
            raise ValueError(
                f'create index for {hunk.target_node_id!r} is outside its parent.')
        new_ids = [block.node_id for block in hunk.block.iter_blocks()]
        existing_ids = {block.node_id for block in document.iter_blocks()}
        if len(new_ids) != len(set(new_ids)) or existing_ids.intersection(new_ids):
            raise ValueError('create block subtree contains duplicate node_ids.')
        siblings.insert(hunk.index, hunk.block.model_copy(deep=True))

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

        valid_node_ids = {
            block.node_id for block in source_doc.iter_blocks() if block.editable
        }

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
                'No revision targets located.'
                if not locate_result.target_node_ids and not locate_result.target_title
                else 'Located revision targets.'
            ),
            counts={
                'target_node_count': len(locate_result.target_node_ids),
                'target_title': int(locate_result.target_title),
            },
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

        if located.target_node_ids or located.target_title:
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
            target_title=located.target_title,
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

        patch_set = PatchSet(
            patch_id=f'patch-{source_doc.document_id or "document"}',
            target_doc_id=source_doc.document_id or '',
            meta={'source': 'generate_patch_set'},
        )
        if plan.instructions or plan.title_instruction:
            prompt = GENERATE_PATCH_SET_PROMPT.format(
                document_json=to_prompt_json(source_doc),
                modify_plan_json=to_prompt_json(plan),
                context_json=to_prompt_json(writing_context),
            )
            revised = self._call_llm_structured(prompt, WriterDocument)
            self._validate_model_revision(source_doc, revised, plan)
            patch_set = self._diff_documents(source_doc, revised)
            patch_set.meta['source'] = 'generate_patch_set'

        result = self._save_artifacts(
            {'patch_set': patch_set},
            step_name='generate_patch_set',
            primary_key='patch_set',
            context_key=None,
            summary='Generated patch set.',
            counts={'hunk_count': len(patch_set.hunks)},
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
        hunks: List[PatchHunk] = []

        for node_id in self._ordered_node_ids(revised):
            if node_id not in common_ids:
                continue
            old_block = source_map[node_id]
            new_block = revised_map[node_id]
            self._validate_preserved_block_fields(old_block, new_block)
            if not self._same_mutable_block_fields(old_block, new_block):
                hunks.append(PatchHunk(
                    hunk_id=f'update-{node_id}',
                    target_node_id=node_id,
                    modify_type='update',
                    block=new_block.model_copy(deep=True),
                ))

        source_layout = self._document_layout(source)
        top_level_deletes = {
            node_id for node_id in deleted_ids
            if source_layout[node_id][0] not in deleted_ids
        }
        working = source.model_copy(deep=True)
        for node_id in reversed(self._ordered_node_ids(source)):
            if node_id in top_level_deletes:
                hunk = PatchHunk(
                    hunk_id=f'delete-{node_id}',
                    target_node_id=node_id,
                    modify_type='delete',
                )
                hunks.append(hunk)
                target = working.block_by_id(node_id)
                if target is not None:
                    self._remove_block(working, target)

        target_parents: List[Optional[str]] = [None]
        target_parents.extend(self._ordered_node_ids(revised))
        for parent_id in target_parents:
            target_children = revised.blocks if parent_id is None \
                else revised_map[parent_id].children
            if parent_id is not None and working.block_by_id(parent_id) is None:
                continue
            for index, desired_child in enumerate(target_children):
                current = working.block_by_id(desired_child.node_id)
                if current is None:
                    created = self._copy_new_subtree(desired_child, new_ids)
                    self._validate_new_subtree(created)
                    hunk = PatchHunk(
                        hunk_id=f'create-{desired_child.node_id}',
                        target_node_id=desired_child.node_id,
                        modify_type='create',
                        block=created,
                        parent_node_id=parent_id,
                        index=index,
                    )
                    hunks.append(hunk)
                    self._children_for_parent(working, parent_id).insert(
                        index, created.model_copy(deep=True))
                    continue
                if desired_child.node_id not in common_ids:
                    continue
                current_parent, current_index = self._block_parent_index(
                    working, desired_child.node_id)
                if current_parent == parent_id and current_index == index:
                    continue
                hunk = PatchHunk(
                    hunk_id=f'move-{desired_child.node_id}',
                    target_node_id=desired_child.node_id,
                    modify_type='move',
                    parent_node_id=parent_id,
                    index=index,
                )
                hunks.append(hunk)
                self._remove_block(working, current)
                self._children_for_parent(working, parent_id).insert(index, current)

        patch = PatchSet(
            patch_id=f'patch-{source.document_id}',
            target_doc_id=source.document_id,
            new_title=revised.title if source.title != revised.title else None,
            hunks=hunks,
            meta={'source': 'document_diff'},
        )
        applied, _ = apply_patch_to_ir(source, patch)
        self._assert_revision_applied(applied, revised)
        return patch

    @staticmethod
    def _ordered_node_ids(document: WriterDocument) -> List[str]:
        return [block.node_id for block in document.iter_blocks()]

    @staticmethod
    def _document_layout(
        document: WriterDocument,
    ) -> Dict[str, Tuple[Optional[str], int]]:
        layout: Dict[str, Tuple[Optional[str], int]] = {}

        def walk(blocks: List[WriterBlock], parent_id: Optional[str]) -> None:
            for index, block in enumerate(blocks):
                layout[block.node_id] = (parent_id, index)
                walk(block.children, block.node_id)

        walk(document.blocks, None)
        return layout

    @staticmethod
    def _same_mutable_block_fields(source: WriterBlock, revised: WriterBlock) -> bool:
        return all(
            getattr(source, field) == getattr(revised, field)
            for field in WRITER_BLOCK_MUTABLE_FIELDS
        )

    @staticmethod
    def _validate_preserved_block_fields(
        source: WriterBlock,
        revised: WriterBlock,
    ) -> None:
        changed = [
            field for field in WRITER_BLOCK_PROVIDER_MANAGED_FIELDS
            if getattr(source, field) != getattr(revised, field)
        ]
        if changed:
            raise ValueError(
                f'block {source.node_id!r} changes provider-managed fields: {changed}.')

    @classmethod
    def _copy_new_subtree(
        cls,
        block: WriterBlock,
        new_ids: Set[str],
    ) -> WriterBlock:
        copied = block.model_copy(deep=True)
        copied.children = [
            cls._copy_new_subtree(child, new_ids)
            for child in block.children
            if child.node_id in new_ids
        ]
        return copied

    @staticmethod
    def _validate_new_subtree(block: WriterBlock) -> None:
        for item in block.iter_blocks():
            if item.provider_binding or item.provider_payload:
                raise ValueError(
                    f'new block {item.node_id!r} must not contain provider-managed fields.')

    def _block_parent_index(
        self,
        document: WriterDocument,
        node_id: str,
    ) -> Tuple[Optional[str], int]:
        layout = self._document_layout(document)
        if node_id not in layout:
            raise ValueError(f'block {node_id!r} is absent from document.')
        return layout[node_id]

    @classmethod
    def _assert_revision_applied(
        cls,
        applied: WriterDocument,
        revised: WriterDocument,
    ) -> None:
        def visible(document: WriterDocument) -> Dict[str, Any]:
            return {
                'document_id': document.document_id,
                'stage': document.stage,
                'title': document.title,
                'blocks': [
                    cls._visible_block(block) for block in document.blocks
                ],
            }

        if visible(applied) != visible(revised):
            raise ValueError('generated patch does not reproduce the revised WriterDocument.')

    @classmethod
    def _visible_block(cls, block: WriterBlock) -> Dict[str, Any]:
        visible = block.model_dump(include=set(WRITER_BLOCK_MUTABLE_FIELDS))
        visible['node_id'] = block.node_id
        visible['children'] = [cls._visible_block(child) for child in block.children]
        return visible

    def _normalize_modify_plan(
        self,
        plan: ModifyPlan,
        task: WritingTask,
        located_node_ids: List[str],
        valid_node_ids: set,
        *,
        target_title: bool = False,
    ) -> ModifyPlan:
        plan.plan_id = plan.plan_id or f'plan-{task.task_id or "task"}'
        plan.task_id = task.task_id
        plan.target_node_ids = list(located_node_ids)
        if target_title:
            if not plan.title_instruction or not plan.title_instruction.strip():
                raise ValueError('modify_plan requires title_instruction for a title target.')
            plan.title_instruction = plan.title_instruction.strip()
        elif plan.title_instruction is not None:
            raise ValueError('modify_plan has title_instruction without a title target.')

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
            if instr.modify_type == 'create':
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

    @staticmethod
    def _validate_model_revision(
        source: WriterDocument,
        revised: WriterDocument,
        plan: ModifyPlan,
    ) -> None:
        if source.document_id != revised.document_id:
            raise ValueError('model revision changed document_id.')
        if source.stage != revised.stage:
            raise ValueError('model revision changed document stage.')
        if source.revision != revised.revision:
            raise ValueError('model revision changed document revision.')
        if source.provider_binding != revised.provider_binding:
            raise ValueError('model revision changed document provider_binding.')
        if plan.title_instruction is None and source.title != revised.title:
            raise ValueError('model revision changed title without a title instruction.')
        if plan.title_instruction is not None and not revised.title.strip():
            raise ValueError('model revision produced an empty title.')

    def _validate_patch(self, document: WriterDocument, patch: PatchSet) -> None:
        if patch.target_doc_id != document.document_id:
            raise ValueError(
                f'patch target_doc_id {patch.target_doc_id!r} does not match '
                f'document_id {document.document_id!r}.'
            )

        hunk_ids: set = set()
        for hunk in patch.hunks:
            if hunk.hunk_id:
                if hunk.hunk_id in hunk_ids:
                    raise ValueError(f'patch contains duplicate hunk_id {hunk.hunk_id!r}.')
                hunk_ids.add(hunk.hunk_id)

    @staticmethod
    def _apply_block_update(target: WriterBlock, revised: WriterBlock) -> None:
        if not target.editable:
            raise ValueError(f'patch target {target.node_id!r} is not editable.')
        if target.node_id != revised.node_id:
            raise ValueError('update cannot change block node_id.')
        WriterRevisionTools._validate_preserved_block_fields(target, revised)
        for field in WRITER_BLOCK_MUTABLE_FIELDS:
            setattr(target, field, deepcopy(getattr(revised, field)))

    @staticmethod
    def _children_for_parent(
        document: WriterDocument,
        parent_node_id: Optional[str],
    ) -> List[WriterBlock]:
        if parent_node_id is None:
            return document.blocks
        parent = document.block_by_id(parent_node_id)
        if parent is None:
            raise ValueError(f'parent block {parent_node_id!r} is absent from document.')
        return parent.children

    @staticmethod
    def _subtree_has_id(block: WriterBlock, node_id: str) -> bool:
        return any(item.node_id == node_id for item in block.iter_blocks())

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
