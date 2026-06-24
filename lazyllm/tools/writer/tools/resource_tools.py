from .base import WriterToolBase


class WriterResourceTools(WriterToolBase):
    __public_apis__ = [
        "profile_resources",
    ]

    def profile_resources(self, task: dict, input_resources: list, context: dict = None) -> dict:
        ...
