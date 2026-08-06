from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from .base import WriterToolBase
from ..data_models.context import WritingContext
from ..data_models.multimodal import MediaAssetLibrary
from ..data_models.revision import (
    GeneratedRevision,
    LocatedContent,
    LocateResult,
    ModifyInstruction,
    ModifyPlan,
    PatchHunk,
    PatchResult,
    PatchSet,
    RevisionBlockContent,
    StringReplace,
    StringReplaceResult,
    StringReplaceSet,
)
from ..data_models.task import WritingTask
from ..data_models.writer_ir import (
    WRITER_BLOCK_MUTABLE_FIELDS,
    WRITER_BLOCK_PROVIDER_MANAGED_FIELDS,
    ContentRef,
    WriterBlock,
    WriterDocument,
    WriterStage,
)
from ..prompts import (
    GENERATE_MODIFY_PLAN_PROMPT,
    GENERATE_PATCH_SET_PROMPT,
    GENERATE_STRING_REPLACE_SET_PROMPT,
    LOCATE_REVISION_TARGET_PROMPT,
)
from ..utils import parse_markdown_sections, to_prompt_json


def apply_patch_to_ir(
    document: WriterDocument,
    patch_set: PatchSet,
    media_assets: Optional[MediaAssetLibrary] = None,
) -> Tuple[WriterDocument, PatchResult]:
    '''Apply a provider-neutral patch without artifact or context dependencies.'''
    tools = WriterRevisionTools()
    tools._validate_patch(document, patch_set, media_assets=media_assets)
    if not patch_set.hunks \
            and (patch_set.new_title is None or patch_set.new_title == document.title):
        raise ValueError('patch contains no document operations.')
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
        'generate_revision_set',
        'generate_patch_set',
        'generate_string_replace_set',
        'build_patch_set_from_documents',
        'apply_revision',
        'apply_patch',
        'apply_string_replace',
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
        source_doc = self._unified_document(document)
        user_selection = writing_task.selection
        candidates = self._revision_candidates(source_doc)
        if user_selection and user_selection.content_refs:
            selected_refs = {
                self._content_ref_key(content_ref)
                for content_ref in user_selection.content_refs
            }
            candidates = [
                candidate for candidate in candidates
                if self._content_ref_key(candidate['content_ref']) in selected_refs
            ]
            if not candidates:
                raise ValueError('selection.content_refs contains no valid content references.')

        prompt = LOCATE_REVISION_TARGET_PROMPT.format(
            task_json=to_prompt_json(writing_task),
            document_content=to_prompt_json(candidates),
        )
        locate_result = self._call_llm_structured(prompt, LocateResult)
        candidate_refs = {
            self._content_ref_key(candidate['content_ref'])
            for candidate in candidates
        }
        invalid = [
            target.content_ref.model_dump(exclude_none=True)
            for target in locate_result.targets
            if self._content_ref_key(target.content_ref) not in candidate_refs
        ]
        if invalid:
            raise ValueError(f'locate_result contains references not in candidates: {invalid}.')
        located: List[LocatedContent] = []
        seen_refs = set()
        for target in locate_result.targets:
            key = self._content_ref_key(target.content_ref)
            if key not in seen_refs:
                seen_refs.add(key)
                located.append(target)
        locate_result.targets = located
        if not locate_result.targets and not locate_result.target_title:
            raise ValueError('locate_result contains no revision targets.')
        locate_result.task_id = writing_task.task_id
        locate_result.doc_id = (
            source_doc.document_id if isinstance(source_doc, WriterDocument) else None
        )

        result = self._save_artifacts(
            {'locate_result': locate_result},
            step_name='locate_revision_target',
            primary_key='locate_result',
            context_key=None,
            summary='Located revision targets.',
            counts={
                'target_count': len(locate_result.targets),
                'target_title': int(locate_result.target_title),
            },
            artifact_meta={
                'task_id': writing_task.task_id,
                'document_id': (
                    source_doc.document_id if isinstance(source_doc, WriterDocument) else None
                ),
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
        source_doc = self._unified_document(document)
        located = self._unified_model(locate_result, LocateResult)
        writing_context = self._unified_model(context, WritingContext)

        document_id = source_doc.document_id if isinstance(source_doc, WriterDocument) else None
        if located.doc_id != document_id:
            raise ValueError('locate_result does not belong to the current document.')
        if writing_task.task_id is not None and located.task_id != writing_task.task_id:
            raise ValueError('locate_result does not belong to the current task.')

        valid_refs = {
            self._content_ref_key(candidate['content_ref'])
            for candidate in self._revision_candidates(source_doc)
        }
        located_refs = [target.content_ref for target in located.targets]
        if located_refs or located.target_title:
            missing = [
                content_ref.model_dump(exclude_none=True)
                for content_ref in located_refs
                if self._content_ref_key(content_ref) not in valid_refs
            ]
            if missing:
                raise ValueError(f'locate_result has references absent from document: {missing}.')

            prompt = GENERATE_MODIFY_PLAN_PROMPT.format(
                task_json=to_prompt_json(writing_task),
                document_content=(
                    to_prompt_json(self._visible_document(source_doc))
                    if isinstance(source_doc, WriterDocument)
                    else source_doc
                ),
                locate_result_json=to_prompt_json(located),
                context_json=to_prompt_json(writing_context),
            )
            modify_plan = self._call_llm_structured(prompt, ModifyPlan)
        else:
            modify_plan = ModifyPlan(scope='document', summary='No revision targets; nothing to plan.')

        modify_plan = self._normalize_modify_plan(
            modify_plan,
            writing_task,
            located_refs,
            valid_refs,
            document=source_doc if isinstance(source_doc, WriterDocument) else None,
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
                'document_id': document_id,
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
        media_assets: Any = None,
    ) -> dict:
        source_doc = self._unified_model(document, WriterDocument)
        plan = self._unified_model(modify_plan, ModifyPlan)
        writing_context = self._unified_model(context, WritingContext)
        media_library = self._unified_optional_model(media_assets, MediaAssetLibrary)

        patch_set = PatchSet(
            patch_id=f'patch-{source_doc.document_id or "document"}',
            target_doc_id=source_doc.document_id or '',
            meta={'source': 'generate_patch_set'},
        )
        if plan.instructions or plan.title_instruction:
            prompt = GENERATE_PATCH_SET_PROMPT.format(
                document_json=to_prompt_json(self._visible_document(source_doc)),
                modify_plan_json=to_prompt_json(plan),
                context_json=to_prompt_json(writing_context),
            )
            generated = self._call_llm_structured(prompt, GeneratedRevision)
            patch_set = self._compile_generated_revision(
                source_doc,
                plan,
                generated,
                media_assets=media_library,
            )
            apply_patch_to_ir(source_doc, patch_set, media_assets=media_library)

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

    def generate_revision_set(
        self,
        document: Any,
        modify_plan: Any,
        context: Any,
        media_assets: Any = None,
    ) -> dict:
        source = self._unified_document(document)
        if isinstance(source, WriterDocument):
            return self.generate_patch_set(
                source, modify_plan, context, media_assets=media_assets)
        return self.generate_string_replace_set(source, modify_plan, context)

    def generate_string_replace_set(
        self,
        document: Any,
        modify_plan: Any,
        context: Any,
    ) -> dict:
        source = self._unified_document(document)
        if isinstance(source, WriterDocument):
            raise TypeError('generate_string_replace_set requires Markdown input.')
        plan = self._unified_model(modify_plan, ModifyPlan)
        writing_context = self._unified_model(context, WritingContext)

        replace_set = StringReplaceSet(
            replace_set_id=f'replace-{writing_context.context_id}',
            meta={'source': 'generate_string_replace_set'},
        )
        if plan.instructions or plan.title_instruction:
            prompt = GENERATE_STRING_REPLACE_SET_PROMPT.format(
                document_content=source,
                modify_plan_json=to_prompt_json(plan),
                context_json=to_prompt_json(writing_context),
            )
            replace_set = self._call_llm_structured(prompt, StringReplaceSet)
            replace_set.replace_set_id = (
                replace_set.replace_set_id or f'replace-{writing_context.context_id}'
            )
            for index, replacement in enumerate(replace_set.replacements, start=1):
                replacement.replacement_id = replacement.replacement_id or f'replace-{index}'

        return self._save_artifacts(
            {'string_replace_set': replace_set},
            step_name='generate_string_replace_set',
            primary_key='string_replace_set',
            context_key=None,
            summary='Generated Markdown string replacements.',
            counts={'replacement_count': len(replace_set.replacements)},
            artifact_meta={'context_id': writing_context.context_id},
            artifact_filenames={'string_replace_set': 'string_replace_set.json'},
        ).model_dump()

    def _compile_generated_revision(
        self,
        document: WriterDocument,
        plan: ModifyPlan,
        generated: GeneratedRevision,
        media_assets: Optional[MediaAssetLibrary] = None,
    ) -> PatchSet:
        if plan.title_instruction is not None and not (generated.new_title or '').strip():
            raise ValueError('generated revision omits the requested title change.')

        revised = document.model_copy(deep=True)
        if plan.title_instruction is not None:
            revised.title = generated.new_title
        for instruction in plan.instructions:
            self._apply_generated_instruction(
                revised,
                instruction,
                generated.changes[instruction.instruction_id],
                media_assets=media_assets,
            )
        patch = self._diff_documents(document, revised, media_assets=media_assets)
        patch.patch_id = f'patch-{document.document_id or "document"}'
        patch.meta['source'] = 'generate_patch_set'
        return patch

    def _apply_generated_instruction(
        self,
        revised: WriterDocument,
        instruction: ModifyInstruction,
        blocks: List[RevisionBlockContent],
        media_assets: Optional[MediaAssetLibrary] = None,
    ) -> None:
        if instruction.modify_type == 'update':
            content, = blocks
            target = revised.block_by_id(self._node_id(instruction.content_ref, 'content_ref'))
            self._apply_block_content(target, content)
            return

        if instruction.modify_type == 'create':
            if not blocks:
                raise ValueError(
                    f'create instruction {instruction.instruction_id!r} requires new content blocks.')
            blocks = self._prepare_generated_blocks(
                instruction,
                blocks,
                media_assets=media_assets,
            )
            parent_id, index = self._insertion_point(
                revised,
                self._node_id(instruction.content_ref, 'content_ref'),
                instruction.position,
            )
            siblings = self._children_for_parent(revised, parent_id)
            for offset, content in enumerate(blocks):
                siblings.insert(
                    index + offset,
                    self._new_block(revised.stage, content),
                )
            return

        target = revised.block_by_id(self._node_id(instruction.content_ref, 'content_ref'))
        self._remove_block(revised, target)
        if instruction.modify_type == 'move':
            parent_id, index = self._insertion_point(
                revised,
                self._node_id(instruction.destination_ref, 'destination_ref'),
                instruction.position,
            )
            self._children_for_parent(revised, parent_id).insert(index, target)

    def _apply_block_content(
        self,
        target: WriterBlock,
        content: RevisionBlockContent,
    ) -> None:
        if content.children:
            raise ValueError('update content cannot replace a block subtree.')
        if content.type is not None:
            target.type = content.type
        if content.content is not None:
            target.content = content.content
            if content.spans is None:
                target.spans = []
        if content.spans is not None:
            target.spans = deepcopy(content.spans)
        if content.numbering is not None:
            target.numbering = deepcopy(content.numbering)
        if content.references is not None:
            target.references = deepcopy(content.references)
        WriterBlock.model_validate(target.model_dump())

    def _new_block(
        self,
        stage: WriterStage,
        content: RevisionBlockContent,
    ) -> WriterBlock:
        return WriterBlock(
            node_id=f'writer-new-{uuid4()}',
            type=content.type,
            content=content.content,
            spans=deepcopy(content.spans or []),
            children=[
                self._new_block(stage, child) for child in content.children
            ],
            stage=stage,
            numbering=deepcopy(content.numbering or {}),
            references=deepcopy(content.references or []),
        )

    def _prepare_generated_blocks(
        self,
        instruction: ModifyInstruction,
        blocks: List[RevisionBlockContent],
        *,
        media_assets: Optional[MediaAssetLibrary],
    ) -> List[RevisionBlockContent]:
        visual = instruction.visual_instruction
        if visual is None:
            return blocks
        if len(blocks) != 1:
            raise ValueError(
                f'image create instruction {instruction.instruction_id!r} '
                'requires exactly one block.'
            )
        asset_id = self._require_visual_asset(visual, media_assets)
        content = blocks[0].model_copy(deep=True)
        if content.type not in {None, 'image'}:
            raise ValueError(
                f'image create instruction {instruction.instruction_id!r} '
                'must produce type="image".'
            )
        if content.children:
            raise ValueError('image create content must not contain child blocks.')
        content.type = 'image'
        content.references = [{'type': 'media_asset', 'id': asset_id}]
        return [content]

    @staticmethod
    def _require_visual_asset(
        visual: Any,
        media_assets: Optional[MediaAssetLibrary],
    ) -> str:
        if media_assets is None:
            raise ValueError(
                f'visual instruction {visual.need_id!r} requires resolved media_assets.'
            )
        asset_ids = media_assets.visual_need_asset_ids.get(visual.need_id, [])
        if len(asset_ids) != 1:
            raise ValueError(
                f'visual instruction {visual.need_id!r} must resolve to exactly one media asset.'
            )
        asset_id = asset_ids[0]
        asset = media_assets.assets.get(asset_id)
        if asset is None or not asset.local_path or not Path(asset.local_path).is_file():
            raise ValueError(f'Image media asset {asset_id!r} is unavailable.')
        return asset_id

    def _insertion_point(
        self,
        document: WriterDocument,
        destination_node_id: str,
        position: str,
    ) -> Tuple[Optional[str], int]:
        parent_id, index = self._block_parent_index(document, destination_node_id)
        return parent_id, index + int(position == 'after')

    def apply_patch(
        self,
        document: Any,
        patch_set: Any,
        context: Any,
        media_assets: Any = None,
    ) -> dict:
        source_doc = self._unified_model(document, WriterDocument)
        patch = self._unified_model(patch_set, PatchSet)
        writing_context = self._unified_model(context, WritingContext)
        media_library = self._unified_optional_model(media_assets, MediaAssetLibrary)

        revised_doc, patch_result = apply_patch_to_ir(
            source_doc,
            patch,
            media_assets=media_library,
        )

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
                'revised_document': f'revised_document_{source_doc.document_id or "document"}_ir.lmd',
            },
        )
        return result.model_dump()

    def apply_revision(
        self,
        document: Any,
        revision_set: Any,
        context: Any,
        media_assets: Any = None,
    ) -> dict:
        source = self._unified_document(document)
        if isinstance(source, WriterDocument):
            return self.apply_patch(
                source, revision_set, context, media_assets=media_assets)
        result = self.apply_string_replace(source, revision_set, context)
        metadata = result['metadata']
        metadata['artifact_paths']['revised_document'] = metadata['artifact_paths'][
            'revised_document_md'
        ]
        metadata['schema_names']['revised_document'] = 'text/markdown'
        return result

    def apply_string_replace(
        self,
        document: Any,
        replace_set: Any,
        context: Any,
    ) -> dict:
        source = self._unified_document(document)
        if isinstance(source, WriterDocument):
            raise TypeError('apply_string_replace requires Markdown input.')
        replacements = self._unified_model(replace_set, StringReplaceSet)
        writing_context = self._unified_model(context, WritingContext)

        revised = source
        applied: List[str] = []
        for index, replacement in enumerate(replacements.replacements, start=1):
            revised = self._apply_markdown_replacement(revised, replacement)
            applied.append(replacement.replacement_id or f'replace-{index}')

        replace_result = StringReplaceResult(
            replace_set_id=replacements.replace_set_id,
            success=True,
            applied_replacements=applied,
            message='String replacements applied.',
        )
        markdown_path = self._write_markdown_artifact('revised_document.md', revised)
        result = self._save_artifacts(
            {'string_replace_result': replace_result},
            step_name='apply_string_replace',
            primary_key='string_replace_result',
            context_key=None,
            summary='Applied Markdown string replacements.',
            counts={'applied': len(applied), 'failed': 0},
            artifact_meta={'context_id': writing_context.context_id},
            artifact_filenames={'string_replace_result': 'string_replace_result.json'},
        )
        dumped = result.model_dump()
        dumped['revised_document_md'] = markdown_path
        dumped['metadata']['artifact_paths']['revised_document_md'] = markdown_path
        dumped['metadata']['schema_names']['revised_document_md'] = 'text/markdown'
        return dumped

    def _diff_documents(  # noqa: C901
        self,
        source: WriterDocument,
        revised: WriterDocument,
        media_assets: Optional[MediaAssetLibrary] = None,
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
        applied, _ = apply_patch_to_ir(source, patch, media_assets=media_assets)
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
        if cls._visible_document(applied) != cls._visible_document(revised):
            raise ValueError('generated patch does not reproduce the revised WriterDocument.')

    @classmethod
    def _visible_document(cls, document: WriterDocument) -> Dict[str, Any]:
        return {
            'document_id': document.document_id,
            'stage': document.stage,
            'title': document.title,
            'blocks': [cls._visible_block(block) for block in document.blocks],
        }

    @classmethod
    def _visible_block(cls, block: WriterBlock) -> Dict[str, Any]:
        visible = block.model_dump(include=set(WRITER_BLOCK_MUTABLE_FIELDS))
        visible['node_id'] = block.node_id
        visible['children'] = [cls._visible_block(child) for child in block.children]
        return visible

    def _normalize_modify_plan(  # noqa: C901
        self,
        plan: ModifyPlan,
        task: WritingTask,
        located_refs: List[ContentRef],
        valid_refs: set,
        *,
        document: Optional[WriterDocument] = None,
        target_title: bool = False,
    ) -> ModifyPlan:
        plan.plan_id = plan.plan_id or f'plan-{task.task_id or "task"}'
        plan.task_id = task.task_id
        if target_title:
            if not plan.title_instruction or not plan.title_instruction.strip():
                raise ValueError('modify_plan requires title_instruction for a title target.')
            plan.title_instruction = plan.title_instruction.strip()
        elif plan.title_instruction is not None:
            plan.title_instruction = None

        located_set = {self._content_ref_key(content_ref) for content_ref in located_refs}
        def canonical_ref(reference: ContentRef, candidates: List[ContentRef]) -> Optional[ContentRef]:
            exact_key = self._content_ref_key(reference)
            exact = [candidate for candidate in candidates
                     if self._content_ref_key(candidate) == exact_key]
            if exact:
                return exact[0]
            if reference.node_id or not reference.heading_path:
                return None
            valid_at_location = [
                key for key in valid_refs
                if not key[0] and key[1] == tuple(reference.heading_path) and not key[3]
            ]
            if len(valid_at_location) != 1:
                return None
            same_location = [candidate for candidate in candidates
                             if not candidate.node_id
                             and candidate.heading_path == reference.heading_path
                             and not candidate.document_root]
            return same_location[0] if len(same_location) == 1 else None

        instruction_ids: set = set()
        normalized: List[ModifyInstruction] = []
        for index, instr in enumerate(plan.instructions, start=1):
            resolved_content_ref = canonical_ref(instr.content_ref, located_refs)
            if resolved_content_ref is not None:
                instr.content_ref = resolved_content_ref.model_copy(deep=True)
            content_key = self._content_ref_key(instr.content_ref)
            if instr.modify_type == 'create' and content_key not in valid_refs:
                raise ValueError(
                    'create instruction content_ref is absent from document.'
                )
            if instr.modify_type != 'create' and content_key not in located_set:
                raise ValueError(
                    'modify_plan instruction content_ref is not present in locate_result.targets.'
                )
            instr.instruction_id = (
                instr.instruction_id or f'instr-{index}'
            ).strip()
            if instr.instruction_id in instruction_ids:
                raise ValueError(
                    f'modify_plan has duplicate instruction_id {instr.instruction_id!r}.')
            instruction_ids.add(instr.instruction_id)
            if instr.modify_type in {'create', 'move'} and instr.position is None:
                raise ValueError(f'{instr.modify_type} instruction requires position.')
            self._validate_modify_instruction(instr, document=document)
            if instr.modify_type == 'create' and normalized \
                    and normalized[-1].modify_type == 'create' \
                    and normalized[-1].content_ref == instr.content_ref \
                    and normalized[-1].position == instr.position \
                    and normalized[-1].visual_instruction is None \
                    and instr.visual_instruction is None:
                normalized[-1].instruction += f'\n{instr.instruction}'
                continue
            if instr.modify_type == 'move':
                if instr.destination_ref is None:
                    raise ValueError('move instruction requires destination_ref.')
                destination_key = self._content_ref_key(instr.destination_ref)
                if destination_key not in valid_refs:
                    raise ValueError('move instruction destination_ref is absent from document.')
                if destination_key == content_key:
                    raise ValueError('move instruction cannot use its source as the destination.')
            normalized.append(instr)

        if located_set and not normalized and not target_title:
            raise ValueError('modify_plan contains no operations for the located revision scope.')
        plan.instructions = normalized
        return plan

    @staticmethod
    def _validate_modify_instruction(
        instruction: ModifyInstruction,
        *,
        document: Optional[WriterDocument] = None,
    ) -> None:
        visual = instruction.visual_instruction
        if visual is not None:
            if instruction.modify_type != 'create':
                raise ValueError('visual_instruction is only valid for create instructions.')
            if visual.visual_type != 'image':
                raise ValueError('revision visual_instruction.visual_type must be "image".')
            if not visual.purpose.strip():
                raise ValueError('revision visual_instruction.purpose must not be empty.')
            if not visual.required:
                raise ValueError('revision image additions must be required.')
            if visual.need_id != instruction.instruction_id:
                raise ValueError(
                    'visual_instruction.need_id must equal instruction_id.'
                )
            if visual.content_ref != instruction.content_ref:
                raise ValueError(
                    'visual_instruction.content_ref must equal instruction.content_ref.'
                )
            if visual.preferred_strategy not in {None, 'image_generation'}:
                raise ValueError(
                    'revision image preferred_strategy must be null or image_generation.'
                )

        if document is None or instruction.modify_type not in {'update', 'move'}:
            return
        node_id = instruction.content_ref.node_id
        target = document.block_by_id(node_id) if node_id else None
        if target is not None and target.type == 'image':
            raise ValueError('existing image blocks cannot be updated or moved.')

    def _revision_candidates(self, document: WriterDocument | str) -> List[Dict[str, Any]]:
        if isinstance(document, WriterDocument):
            return [
                {
                    'content_ref': ContentRef(node_id=block.node_id),
                    'type': block.type,
                    'content': block.content,
                }
                for block in document.iter_blocks()
                if block.editable or block.type == 'image'
            ]
        sections = parse_markdown_sections(document)
        if not sections:
            return [{
                'content_ref': ContentRef(document_root=True),
                'type': 'document',
                'content': document,
            }]
        return [
            {
                'content_ref': ContentRef(
                    heading_path=heading_path,
                    occurrence=occurrence,
                ),
                'type': 'section',
                'content': body,
            }
            for _, heading_path, occurrence, body in sections
        ]

    @staticmethod
    def _content_ref_key(content_ref: ContentRef) -> Tuple[Any, ...]:
        return (
            content_ref.node_id,
            tuple(content_ref.heading_path),
            content_ref.placeholder_id,
            content_ref.document_root,
            content_ref.occurrence,
        )

    @staticmethod
    def _node_id(content_ref: Optional[ContentRef], field: str) -> str:
        if content_ref is None or not content_ref.node_id:
            raise ValueError(f'{field}.node_id is required for IR revision.')
        return content_ref.node_id

    def _apply_markdown_replacement(
        self,
        markdown: str,
        replacement: StringReplace,
    ) -> str:
        if replacement.content_ref is None or replacement.content_ref.document_root:
            if replacement.old_string not in markdown:
                raise ValueError(f'old_string is absent for {replacement.replacement_id!r}.')
            return markdown.replace(replacement.old_string, replacement.new_string, 1)

        if not replacement.content_ref.heading_path:
            raise ValueError('Markdown content_ref requires heading_path or document_root.')

        start, end = self._markdown_section_range(markdown, replacement.content_ref)
        section = markdown[start:end]
        if replacement.old_string in section:
            revised_section = section.replace(replacement.old_string, replacement.new_string, 1)
            return markdown[:start] + revised_section + markdown[end:]

        match_starts: List[int] = []
        search_from = 0
        while True:
            match_start = markdown.find(replacement.old_string, search_from)
            if match_start < 0:
                break
            match_end = match_start + len(replacement.old_string)
            if start <= match_start < end or match_start < start < match_end:
                match_starts.append(match_start)
            search_from = match_start + 1
        if len(match_starts) != 1:
            raise ValueError(f'old_string is absent for {replacement.replacement_id!r}.')
        match_start = match_starts[0]
        match_end = match_start + len(replacement.old_string)
        return markdown[:match_start] + replacement.new_string + markdown[match_end:]

    @staticmethod
    def _markdown_section_range(markdown: str, content_ref: ContentRef) -> Tuple[int, int]:
        headings: List[Tuple[int, int, List[str], int]] = []
        heading_path: List[str] = []
        occurrences: Dict[Tuple[str, ...], int] = {}
        pattern = re.compile(r'^(#{1,6})\s+(.+?)\s*$', re.MULTILINE)
        for match in pattern.finditer(markdown):
            level = len(match.group(1))
            heading_path = heading_path[:level - 1]
            heading_path.append(match.group(2))
            path_key = tuple(heading_path)
            occurrence = occurrences.get(path_key, 0) + 1
            occurrences[path_key] = occurrence
            headings.append((match.start(), level, list(heading_path), occurrence))

        for index, (start, level, path, occurrence) in enumerate(headings):
            if path != content_ref.heading_path or occurrence != content_ref.occurrence:
                continue
            end = len(markdown)
            for next_start, next_level, _, _ in headings[index + 1:]:
                if next_level <= level:
                    end = next_start
                    break
            return start, end
        raise ValueError('content_ref is absent from Markdown document.')

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

    def _validate_patch(
        self,
        document: WriterDocument,
        patch: PatchSet,
        *,
        media_assets: Optional[MediaAssetLibrary] = None,
    ) -> None:
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
            target = document.block_by_id(hunk.target_node_id)
            if hunk.modify_type == 'create':
                if hunk.block is None:
                    continue
                for block in hunk.block.iter_blocks():
                    if block.type == 'image':
                        self._validate_image_block(block, media_assets)
            elif hunk.modify_type in {'update', 'move'} \
                    and target is not None \
                    and target.type == 'image':
                raise ValueError('existing image blocks cannot be updated or moved.')
            if hunk.modify_type == 'update' \
                    and hunk.block is not None \
                    and hunk.block.type == 'image':
                raise ValueError('image blocks cannot be created through an update.')

    @classmethod
    def _validate_image_block(
        cls,
        block: WriterBlock,
        media_assets: Optional[MediaAssetLibrary],
    ) -> None:
        references = [
            reference for reference in block.references
            if reference.get('type') == 'media_asset' and reference.get('id')
        ]
        if len(references) != 1:
            raise ValueError(
                f'new image block {block.node_id!r} requires exactly one media_asset reference.'
            )
        if media_assets is None:
            raise ValueError(
                f'new image block {block.node_id!r} requires resolved media_assets.'
            )
        asset_id = str(references[0]['id'])
        asset = media_assets.assets.get(asset_id)
        if asset is None or not asset.local_path or not Path(asset.local_path).is_file():
            raise ValueError(f'Image media asset {asset_id!r} is unavailable.')

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
