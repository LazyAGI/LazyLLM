from .base import WriterToolBase


class WriterContextTools(WriterToolBase):
    __public_apis__ = [
        "create_writing_context",
        "update_writing_context",
    ]

    def create_writing_context(self, task: dict, resource_profiles: list = None, doc_ir: dict = None) -> dict:
        ...

    def update_writing_context(self, content_artifact: dict, context: dict) -> dict:
        ...
