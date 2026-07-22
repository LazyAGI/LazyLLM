from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..utils.artifact import ArtifactModel


class SectionInstruction(BaseModel):
    instruction_id: str
    outline_node_id: str
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


class SectionInstructionList(ArtifactModel):
    instruction_set_id: Optional[str] = None
    outline_id: Optional[str] = None
    instructions: List[SectionInstruction] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


__all__ = ['SectionInstruction', 'SectionInstructionList']
