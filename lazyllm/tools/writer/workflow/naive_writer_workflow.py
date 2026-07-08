from __future__ import annotations
from typing import Any, Dict, Optional

from ..tools.context_tools import WriterContextTools
from ..tools.drafting_tools import WriterDraftingTools
from ..tools.planning_tools import WriterPlanningTools
from ..tools.quality_tools import WriterQualityTools
from ..tools.resource_tools import WriterResourceTools
from ..tools.revision_tools import WriterRevisionTools
from ..data_models.writing import DraftDocument


class NaiveWriterWorkflow:
    def __init__(
        self,
        llm=None,
        artifact_store: Optional[str] = None,
        adapters: Optional[Dict[str, Any]] = None,
        *,
        resource_tools: Optional[WriterResourceTools] = None,
        context_tools: Optional[WriterContextTools] = None,
        planning_tools: Optional[WriterPlanningTools] = None,
        drafting_tools: Optional[WriterDraftingTools] = None,
        quality_tools: Optional[WriterQualityTools] = None,
        revision_tools: Optional[WriterRevisionTools] = None,
    ):
        self.resource = resource_tools or WriterResourceTools(
            llm=llm,
            artifact_store=artifact_store,
            adapters=adapters,
        )
        self.context = context_tools or WriterContextTools(
            llm=llm,
            artifact_store=artifact_store,
            adapters=adapters,
        )
        self.planning = planning_tools or WriterPlanningTools(
            llm=llm,
            artifact_store=artifact_store,
            adapters=adapters,
        )
        self.drafting = drafting_tools or WriterDraftingTools(
            llm=llm,
            artifact_store=artifact_store,
            adapters=adapters,
        )
        self.quality = quality_tools or WriterQualityTools(
            llm=llm,
            artifact_store=artifact_store,
            adapters=adapters,
        )
        self.revision = revision_tools or WriterRevisionTools(
            llm=llm,
            artifact_store=artifact_store,
            adapters=adapters,
        )

    def write(self, task: Any, input_resources: Any = None) -> dict:
        resource_profiles = self.resource.profile_resources(
            task=task,
            input_resources=input_resources,
        )
        writing_context = self.context.create_writing_context(
            task=task,
            resource_profiles=self._artifact_ref(resource_profiles, 'resource_profiles'),
        )
        outline = self.planning.generate_outline(
            task=task,
            context=self._artifact_ref(writing_context, 'writing_context'),
            resource_profiles=self._artifact_ref(resource_profiles, 'resource_profiles'),
        )
        writing_context = self.context.update_writing_context(
            artifacts=self._artifact_ref(outline, 'outline'),
            context=self._artifact_ref(writing_context, 'writing_context'),
        )
        section_instructions = self.planning.generate_section_instructions(
            outline=self._artifact_ref(outline, 'outline'),
            context=self._artifact_ref(writing_context, 'writing_context'),
        )
        draft_section = self.drafting.generate_draft_section(
            task=task,
            section_instruction=self._artifact_ref(section_instructions, 'section_instructions'),
            context=self._artifact_ref(writing_context, 'writing_context'),
        )
        section_review = self.quality.validate_section(
            draft_section=self._artifact_ref(draft_section, 'draft_section'),
            section_instruction=self._artifact_ref(section_instructions, 'section_instructions'),
            context=self._artifact_ref(writing_context, 'writing_context'),
        )
        writing_context = self.context.update_writing_context(
            artifacts=self._artifact_ref(draft_section, 'draft_section'),
            context=self._artifact_ref(writing_context, 'writing_context'),
        )
        draft_document = self.drafting.generate_draft_document(
            draft_sections=self._artifact_ref(draft_section, 'draft_section'),
            context=self._artifact_ref(writing_context, 'writing_context'),
            outline=self._artifact_ref(outline, 'outline'),
        )
        writing_context = self.context.update_writing_context(
            artifacts=self._artifact_ref(draft_document, 'draft_document'),
            context=self._artifact_ref(writing_context, 'writing_context'),
        )
        draft_document_review = self.quality.validate_draft_document(
            draft_document=self._artifact_ref(draft_document, 'draft_document'),
            context=self._artifact_ref(writing_context, 'writing_context'),
        )
        writing_output = self.drafting.generate_writing_output(
            draft=self._artifact_ref(draft_document, 'draft_document'),
            context=self._artifact_ref(writing_context, 'writing_context'),
        )
        target_doc = task.get('target_document') if isinstance(task, dict) else getattr(task, 'target_document', None)
        write_result = self.resource.write_to_document(
            content=self._artifact_ref(writing_output, 'writing_output'),
            target_document=target_doc,
        )

        return {
            'primary_result': writing_output,
            'stage_results': {
                'resource_profiles': resource_profiles,
                'writing_context': writing_context,
                'outline': outline,
                'section_instructions': section_instructions,
                'draft_section': draft_section,
                'section_review': section_review,
                'draft_document': draft_document,
                'draft_document_review': draft_document_review,
                'writing_output': writing_output,
                'write_result': write_result,
            },
        }

    def revise(
        self,
        task: Any,
        document: Any,
        context: Any,
    ) -> dict:
        context_ref = self._artifact_ref(context, 'writing_context')
        if isinstance(document, DraftDocument):
            doc_ir_result = self.revision.draft_to_doc_ir(draft=document)
            doc_ir = self._artifact_ref(doc_ir_result, 'doc_ir')
        else:
            raise TypeError(
                f'Unsupported document type {type(document).__name__!r}.'
            )

        locate_result = self.revision.locate_revision_target(
            task=task,
            doc_ir=doc_ir,
            context=context_ref,
        )
        modify_plan = self.revision.generate_modify_plan(
            task=task,
            doc_ir=doc_ir,
            locate_result=self._artifact_ref(locate_result, 'locate_result'),
            context=context_ref,
        )
        patch_set = self.revision.generate_patch_set(
            doc_ir=doc_ir,
            modify_plan=self._artifact_ref(modify_plan, 'modify_plan'),
            context=context_ref,
        )
        patch_review = self.quality.validate_patch_set(
            patch_set=self._artifact_ref(patch_set, 'patch_set'),
            modify_plan=self._artifact_ref(modify_plan, 'modify_plan'),
            context=context_ref,
        )
        patch_result = self.revision.apply_patch(
            doc_ir=doc_ir,
            patch_set=self._artifact_ref(patch_set, 'patch_set'),
            context=context_ref,
        )

        revised_draft = self.revision.doc_ir_to_draft(
            doc_ir=self._artifact_ref(patch_result, 'revised_doc_ir'),
        )
        revised_draft_ref = self._artifact_ref(revised_draft, 'revised_draft')

        writing_context = self.context.update_writing_context(
            artifacts=revised_draft_ref,
            context=context_ref,
        )
        writing_output = self.drafting.generate_writing_output(
            draft=revised_draft_ref,
            context=self._artifact_ref(writing_context, 'writing_context'),
        )

        return {
            'primary_result': writing_output,
            'stage_results': {
                'task': task,
                'locate_result': locate_result,
                'modify_plan': modify_plan,
                'patch_set': patch_set,
                'patch_review': patch_review,
                'patch_result': patch_result,
                'revised_doc_ir': self._artifact_ref(patch_result, 'revised_doc_ir'),
                'revised_draft': revised_draft,
                'writing_context': writing_context,
                'writing_output': writing_output,
            },
        }

    def _artifact_ref(self, result: Any, artifact_key: Optional[str] = None) -> Any:
        if not isinstance(result, dict):
            return result
        metadata = result.get('metadata') or {}
        artifact_paths = metadata.get('artifact_paths') or {}
        if artifact_key and artifact_key in artifact_paths:
            return artifact_paths[artifact_key]
        return result.get('artifact_path') or result
