from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field
from .docir import Anchor
from ..utils.artifact import ArtifactModel


class LocateResult(ArtifactModel):
    task_id: Optional[str] = None
    doc_id: Optional[str] = None
    target_block_ids: List[str] = Field(default_factory=list)
    target_reasons: Dict[str, str] = Field(default_factory=dict)
    summary: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class ModifyInstruction(BaseModel):
    instruction_id: Optional[str] = None
    target_block_id: str
    modify_type: Literal['insert', 'replace', 'delete']
    instruction: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class ModifyPlan(BaseModel):
    plan_id: Optional[str] = None
    task_id: Optional[str] = None
    scope: Literal['document', 'section', 'block', 'span']
    target_block_ids: List[str] = Field(default_factory=list)
    instructions: List[ModifyInstruction] = Field(default_factory=list)
    summary: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class PatchHunk(BaseModel):
    hunk_id: Optional[str] = None
    target_block_id: Optional[str] = None
    modify_type: Literal['insert', 'replace', 'delete']
    anchor: Optional[Anchor] = None
    old_text: Optional[str] = None
    new_text: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class PatchSet(ArtifactModel):
    patch_id: Optional[str] = None
    target_doc_id: str
    hunks: List[PatchHunk] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class PatchResult(BaseModel):
    patch_id: Optional[str] = None
    success: bool
    applied_hunks: List[str] = Field(default_factory=list)
    failed_hunks: List[str] = Field(default_factory=list)
    message: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


# Fields side-channelled through DocBlock.meta during DraftDocument <-> DocIR conversion.
SECTION_META_FIELDS: Tuple[str, ...] = ('section_id', 'outline_node_id', 'instruction_id')
BLOCK_META_FIELDS: Tuple[str, ...] = ('heading', 'outline_node_id')
