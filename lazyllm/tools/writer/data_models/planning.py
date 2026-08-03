from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from ..utils.artifact import ArtifactModel
from .document import ContentRef, DocumentFormat


class SectionInstruction(BaseModel):
    instruction_id: str
    content_ref: Optional[ContentRef] = None
    # Compatibility field for existing Writer IR callers. New code should use content_ref.
    outline_node_id: Optional[str] = None
    section_title: str
    section_goal: str
    required_points: List[str] = Field(default_factory=list)
    references: List[Dict[str, Any]] = Field(default_factory=list)
    fact_constraints: List[str] = Field(default_factory=list)
    style_constraints: List[str] = Field(default_factory=list)
    relation_constraints: List[str] = Field(default_factory=list)
    visual_needs: List[Dict[str, Any]] = Field(default_factory=list)
    expected_blocks: List[str] = Field(default_factory=list)
    pending_subtasks: List[str] = Field(default_factory=list)
    revision_notes: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def normalize_content_ref(self) -> 'SectionInstruction':
        if self.content_ref is None:
            if not self.outline_node_id:
                raise ValueError('content_ref or outline_node_id is required')
        elif self.content_ref.node_id:
            if self.outline_node_id and self.outline_node_id != self.content_ref.node_id:
                raise ValueError('outline_node_id must match content_ref.node_id')
            self.outline_node_id = self.content_ref.node_id
        return self

    @property
    def resolved_content_ref(self) -> ContentRef:
        return self.content_ref or ContentRef(node_id=self.outline_node_id)


class SectionInstructionList(ArtifactModel):
    instruction_set_id: Optional[str] = None
    outline_id: Optional[str] = None
    document_format: DocumentFormat = 'writer_ir'
    instructions: List[SectionInstruction] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


__all__ = ['SectionInstruction', 'SectionInstructionList']
