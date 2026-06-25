from __future__ import annotations
from typing import Any

from .base import WriterToolBase


class WriterQualityTools(WriterToolBase):
    __public_apis__ = [
        "validate_section",
        "validate_output",
    ]

    def validate_section(
        self,
        draft_section: Any,
        section_instruction: Any,
        context: Any,
    ) -> dict:
        raise NotImplementedError("validate_section is not implemented yet.")

    def validate_output(
        self,
        output: Any,
        context: Any,
    ) -> dict:
        raise NotImplementedError("validate_output is not implemented yet.")
