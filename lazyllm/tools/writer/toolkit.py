from .tools.resource_tools import WriterResourceTools
from .tools.context_tools import WriterContextTools


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

        self.resource = WriterResourceTools(llm=llm, artifact_store=artifact_store)
        self.context = WriterContextTools(llm=llm, artifact_store=artifact_store)

    def as_tool_groups(self):
        return [
            dict(name="writer_resource", desc="Input resource profiling tools.", tools=[self.resource], lazy=self.lazy),
            dict(name="writer_context", desc="Writing context tools.", tools=[self.context], lazy=self.lazy),
        ]
