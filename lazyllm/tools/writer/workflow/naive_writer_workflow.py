from __future__ import annotations
from typing import Any, Dict, Optional

from ..tools.context_tools import WriterContextTools
from ..tools.drafting_tools import WriterDraftingTools
from ..tools.planning_tools import WriterPlanningTools
from ..tools.quality_tools import WriterQualityTools
from ..tools.resource_tools import WriterResourceTools


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

    def write(self, task: Any, input_resources: Any = None) -> dict:
        resource_profiles = self.resource.profile_resources(
            task=task,
            input_resources=input_resources,
        )
        writing_context = self.context.create_writing_context(
            task=task,
            resource_profiles=self._artifact_ref(resource_profiles, "resource_profiles"),
        )
        outline = self.planning.generate_outline(
            task=task,
            context=self._artifact_ref(writing_context, "writing_context"),
            resource_profiles=self._artifact_ref(resource_profiles, "resource_profiles"),
        )
        section_instructions = self.planning.generate_section_instructions(
            outline=self._artifact_ref(outline, "outline"),
            context=self._artifact_ref(writing_context, "writing_context"),
        )
        draft_section = self.drafting.generate_draft_section(
            task=task,
            section_instruction=self._artifact_ref(section_instructions, "section_instructions"),
            context=self._artifact_ref(writing_context, "writing_context"),
        )
        section_review = self.quality.validate_section(
            draft_section=self._artifact_ref(draft_section, "draft_section"),
            section_instruction=self._artifact_ref(section_instructions, "section_instructions"),
            context=self._artifact_ref(writing_context, "writing_context"),
        )
        writing_context = self.context.update_writing_context(
            content_artifact=self._artifact_ref(draft_section, "draft_section"),
            context=self._artifact_ref(writing_context, "writing_context"),
        )
        draft_document = self.drafting.generate_draft_document(
            draft_sections=self._artifact_ref(draft_section, "draft_section"),
            context=self._artifact_ref(writing_context, "writing_context"),
            outline=self._artifact_ref(outline, "outline"),
        )
        writing_output = self.drafting.generate_writing_output(
            draft=self._artifact_ref(draft_document, "draft_document"),
            context=self._artifact_ref(writing_context, "writing_context"),
        )
        output_review = self.quality.validate_output(
            output=self._artifact_ref(writing_output, "writing_output"),
            context=self._artifact_ref(writing_context, "writing_context"),
        )

        return {
            "primary_result": writing_output,
            "stage_results": {
                "resource_profiles": resource_profiles,
                "writing_context": writing_context,
                "outline": outline,
                "section_instructions": section_instructions,
                "draft_section": draft_section,
                "section_review": section_review,
                "draft_document": draft_document,
                "writing_output": writing_output,
                "output_review": output_review,
            },
        }

    def revise(self, *args, **kwargs) -> dict:
        raise NotImplementedError("NaiveWriterWorkflow.revise is not implemented yet.")

    def _artifact_ref(self, result: Any, artifact_key: Optional[str] = None) -> Any:
        if not isinstance(result, dict):
            return result
        metadata = result.get("metadata") or {}
        artifact_paths = metadata.get("artifact_paths") or {}
        if artifact_key and artifact_key in artifact_paths:
            return artifact_paths[artifact_key]
        return result.get("artifact_path") or result
