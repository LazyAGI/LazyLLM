from .tools.resource_tools import WriterResourceTools
from .tools.context_tools import WriterContextTools
from .tools.planning_tools import WriterPlanningTools
from .tools.drafting_tools import WriterDraftingTools
from .tools.quality_tools import WriterQualityTools
from .tools.revision_tools import WriterRevisionTools


class WriterToolKit:
    def __init__(
        self,
        llm=None,
        artifact_store=None,
        adapters=None,
        tools=None,
        lazy: bool = True,
    ):
        self.llm = llm
        self.artifact_store = artifact_store
        self.adapters = adapters or {}
        self.tools = tools or []
        self.lazy = lazy

        self.resource = WriterResourceTools(llm=llm, artifact_store=artifact_store, adapters=self.adapters)
        self.context = WriterContextTools(llm=llm, artifact_store=artifact_store, adapters=self.adapters)
        self.planning = WriterPlanningTools(llm=llm, artifact_store=artifact_store, adapters=self.adapters)
        self.drafting = WriterDraftingTools(llm=llm, artifact_store=artifact_store, adapters=self.adapters)
        self.quality = WriterQualityTools(llm=llm, artifact_store=artifact_store, adapters=self.adapters)
        self.revision = WriterRevisionTools(llm=llm, artifact_store=artifact_store, adapters=self.adapters)

    def as_tool_groups(self):
        groups = [
            dict(
                name='writer_resource',
                desc='Input resource profiling and document IO tools.',
                tools=[self.resource],
                lazy=self.lazy,
            ),
            dict(
                name='writer_context',
                desc='Writing context creation and update tools.',
                tools=[self.context],
                lazy=self.lazy,
            ),
            dict(
                name='writer_planning',
                desc='Outline and section instruction generation tools.',
                tools=[self.planning],
                lazy=self.lazy,
            ),
            dict(
                name='writer_drafting',
                desc='Draft section, draft document, and writing output generation tools.',
                tools=[self.drafting],
                lazy=self.lazy,
            ),
            dict(
                name='writer_quality',
                desc='Section and draft-document quality validation tools.',
                tools=[self.quality],
                lazy=self.lazy,
            ),
            dict(
                name='writer_revision',
                desc='Section and draft-document revision tools.',
                tools=[self.revision],
                lazy=self.lazy,
            ),
        ]

        if not self.tools:
            return groups

        enabled = set(self.tools)
        return [group for group in groups if group['name'] in enabled]
