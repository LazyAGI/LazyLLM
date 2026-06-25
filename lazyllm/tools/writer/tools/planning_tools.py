from __future__ import annotations
from typing import Any

from .base import WriterToolBase


class WriterPlanningTools(WriterToolBase):
    __public_apis__ = [
        "generate_outline",
        "generate_section_instructions",
    ]

    def generate_outline(
        self,
        task: Any,
        context: Any,
        resource_profiles: Any = None,
        execution_results: Any = None,
    ) -> dict:
        raise NotImplementedError("generate_outline is not implemented yet.")

    def generate_section_instructions(
        self,
        outline: Any,
        context: Any,
        execution_results: Any = None,
    ) -> dict:
        raise NotImplementedError("generate_section_instructions is not implemented yet.")
