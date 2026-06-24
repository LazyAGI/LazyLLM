from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from ..utils.artifact import ArtifactModel


class OutlineNode(BaseModel):
    node_id: Optional[str] = None
    title: str
    level: int = 1
    instruction: Optional[str] = None
    constraints: Dict[str, Any] = Field(default_factory=dict)
    children: List[OutlineNode] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


OutlineNode.model_rebuild()


class WritingOutline(ArtifactModel):
    outline_id: Optional[str] = None
    title: Optional[str] = None
    nodes: List[OutlineNode] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class SectionInstruction(BaseModel):
    instruction_id: str
    outline_node_id: str
    section_title: str
    section_goal: str
    required_points: List[str] = Field(default_factory=list)
    source_refs: List[str] = Field(default_factory=list)
    fact_constraints: List[str] = Field(default_factory=list)
    style_constraints: List[str] = Field(default_factory=list)
    relation_constraints: List[str] = Field(default_factory=list)
    visual_needs: List[Dict[str, Any]] = Field(default_factory=list)
    expected_blocks: List[str] = Field(default_factory=list)
    pending_subtasks: List[str] = Field(default_factory=list)
    revision_notes: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class WritingSubTask(BaseModel):
    subtask_id: str
    subtask_type: Literal["research", "extract", "write", "table", "visual", "verify"]
    section_id: Optional[str] = None
    block_id: Optional[str] = None
    description: str
    placeholder: Optional[str] = None
    blocking: bool = False
    status: Literal["pending", "resolved", "failed"] = "pending"
    meta: Dict[str, Any] = Field(default_factory=dict)


class DraftBlock(BaseModel):
    block_id: Optional[str] = None
    outline_node_id: Optional[str] = None
    section_id: Optional[str] = None
    heading: Optional[str] = None
    content: str = ""
    subtasks: List[WritingSubTask] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class DraftSection(BaseModel):
    section_id: Optional[str] = None
    outline_node_id: Optional[str] = None
    title: Optional[str] = None
    instruction_id: Optional[str] = None
    sub_sections: List[DraftSection] = Field(default_factory=list)
    blocks: List[DraftBlock] = Field(default_factory=list)
    subtasks: List[WritingSubTask] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


DraftSection.model_rebuild()


class DraftDocument(ArtifactModel):
    draft_id: Optional[str] = None
    title: Optional[str] = None
    sections: List[DraftSection] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)
