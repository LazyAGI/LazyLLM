from __future__ import annotations
from typing import Any

from .base import WriterToolBase


class WriterResourceTools(WriterToolBase):
    __public_apis__ = [
        "profile_resources",
    ]

    def profile_resources(
        self,
        task: Any,
        input_resources: Any = None,
        context: Any = None,
    ) -> dict:
        raise NotImplementedError("profile_resources is not implemented yet.")
