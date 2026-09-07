from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..utils.artifact import ArtifactModel
from .writer_ir import ContentRef


class WritingInstructionBase(BaseModel):
    instruction_id: str
    content_ref: ContentRef
    section_title: str
    section_goal: str
    required_points: List[str] = Field(default_factory=list)
    references: List[Dict[str, Any]] = Field(default_factory=list)
    fact_constraints: List[str] = Field(default_factory=list)
    style_constraints: List[str] = Field(default_factory=list)
    visual_needs: List[Dict[str, Any]] = Field(default_factory=list)
    expected_blocks: List[str] = Field(default_factory=list)
    pending_subtasks: List[str] = Field(default_factory=list)
    revision_notes: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class HeadingStructureItem(BaseModel):
    node_id: Optional[str] = None
    level: int = Field(ge=2, le=5)
    title: str = Field(min_length=1)
    target_chars: Optional[int] = Field(default=None, gt=0)


class SectionInstruction(WritingInstructionBase):
    relation_constraints: List[str] = Field(default_factory=list)
    heading_structure: Optional[List[HeadingStructureItem]] = None


class ShortWritingPlan(WritingInstructionBase):
    core_viewpoint: str


class SectionInstructionList(ArtifactModel):
    instruction_set_id: Optional[str] = None
    outline_id: Optional[str] = None
    instructions: List[SectionInstruction] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


__all__ = [
    'HeadingStructureItem', 'SectionInstruction', 'SectionInstructionList', 'ShortWritingPlan',
]
