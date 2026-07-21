from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field
from ..utils.artifact import ArtifactModel


class OutlineNodeConstraints(BaseModel):
    model_config = ConfigDict(extra='forbid')

    section_goal: Optional[str] = None
    required_points: List[str] = Field(default_factory=list)
    fact_constraints: List[str] = Field(default_factory=list)
    style_constraints: List[str] = Field(default_factory=list)
    relation_constraints: List[str] = Field(default_factory=list)
    references: List[Dict[str, Any]] = Field(default_factory=list)
    min_words: Optional[int] = None
    max_words: Optional[int] = None
    pov: Optional[str] = None
    tone: Optional[str] = None
    must_include: List[str] = Field(default_factory=list)
    must_avoid: List[str] = Field(default_factory=list)


class OutlineNode(BaseModel):
    node_id: Optional[str] = None
    title: str
    level: int = 1
    instruction: Optional[str] = None
    constraints: OutlineNodeConstraints = Field(default_factory=OutlineNodeConstraints)
    children: List[OutlineNode] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


OutlineNode.model_rebuild()


class WritingOutline(ArtifactModel):
    outline_id: Optional[str] = None
    title: Optional[str] = None
    nodes: List[OutlineNode] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class SectionInstruction(BaseModel):
    model_config = ConfigDict(extra='forbid')

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


class WritingSubTask(BaseModel):
    subtask_id: str
    subtask_type: Literal['research', 'extract', 'write', 'table', 'visual', 'verify']
    section_id: Optional[str] = None
    block_id: Optional[str] = None
    description: str
    placeholder: Optional[str] = None
    blocking: bool = False
    status: Literal['pending', 'resolved', 'failed'] = 'pending'
    meta: Dict[str, Any] = Field(default_factory=dict)


class DraftBlock(BaseModel):
    block_id: Optional[str] = None
    outline_node_id: Optional[str] = None
    section_id: Optional[str] = None
    heading: Optional[str] = None
    content: str = ''
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


class WritingOutput(ArtifactModel):
    output_id: Optional[str] = None
    title: Optional[str] = None
    content: str
    output_format: Literal['markdown', 'plain_text', 'html', 'docx'] = 'markdown'
    references: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)
