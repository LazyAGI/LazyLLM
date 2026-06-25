from __future__ import annotations
from typing import Any

from .base import WriterToolBase


class WriterDraftingTools(WriterToolBase):
    __public_apis__ = [
        "generate_draft_section",
        "generate_draft_document",
        "generate_writing_output",
    ]

    def generate_draft_section(
        self,
        task: Any,
        section_instruction: Any,
        context: Any,
        previous_sections: Any = None,
    ) -> dict:
        raise NotImplementedError("generate_draft_section is not implemented yet.")

    def generate_draft_document(
        self,
        draft_sections: Any,
        context: Any,
    ) -> dict:
        raise NotImplementedError("generate_draft_document is not implemented yet.")

    def generate_writing_output(
        self,
        draft: Any,
        context: Any,
    ) -> dict:
        raise NotImplementedError("generate_writing_output is not implemented yet.")
