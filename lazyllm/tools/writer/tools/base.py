from lazyllm.module import ModuleBase


class WriterToolBase(ModuleBase):
    def __init__(
        self,
        llm=None,
        artifact_store=None,
        adapters=None,
        tools=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm = llm
        self.artifact_store = artifact_store
        self.adapters = adapters or {}
        self.tools = tools or []

    def _load_artifact(self, artifact_or_path):
        ...

    def _save_artifact(self, artifact):
        ...

    def _call_llm_structured(self, prompt, schema):
        ...

    def _validate_schema(self, data, schema):
        ...
