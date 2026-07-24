from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator
from .writer_ir import WriterBlock
from ..utils.artifact import ArtifactModel


class LocateResult(ArtifactModel):
    task_id: Optional[str] = None
    doc_id: Optional[str] = None
    target_title: bool = False
    target_node_ids: List[str] = Field(default_factory=list)
    target_reasons: Dict[str, str] = Field(default_factory=dict)
    summary: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


ModifyType = Literal['create', 'update', 'delete', 'move']
PatchPosition = Literal['before', 'after']


class ModifyInstruction(BaseModel):
    instruction_id: Optional[str] = None
    target_node_id: str
    modify_type: ModifyType
    anchor_node_id: Optional[str] = None
    position: Optional[PatchPosition] = None
    instruction: str
    meta: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_move(self) -> 'ModifyInstruction':
        if self.modify_type == 'move' and (not self.anchor_node_id or not self.position):
            raise ValueError('move requires anchor_node_id and position')
        return self


class ModifyPlan(BaseModel):
    plan_id: Optional[str] = None
    task_id: Optional[str] = None
    scope: Literal['document', 'section', 'block', 'span']
    title_instruction: Optional[str] = None
    target_node_ids: List[str] = Field(default_factory=list)
    instructions: List[ModifyInstruction] = Field(default_factory=list)
    summary: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class PatchHunk(BaseModel):
    hunk_id: Optional[str] = None
    target_node_id: str
    modify_type: ModifyType
    block: Optional[WriterBlock] = None
    parent_node_id: Optional[str] = None
    index: Optional[int] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_operation(self) -> 'PatchHunk':
        if not self.target_node_id.strip():
            raise ValueError('target_node_id must not be empty')
        if self.modify_type in {'create', 'update'}:
            if self.block is None:
                raise ValueError(f'{self.modify_type} requires block')
            if self.block.node_id != self.target_node_id:
                raise ValueError(
                    f'{self.modify_type} block.node_id must equal target_node_id')
        elif self.block is not None:
            raise ValueError(f'{self.modify_type} must not provide block')
        if self.modify_type in {'create', 'move'}:
            if self.index is None or self.index < 0:
                raise ValueError(f'{self.modify_type} requires a non-negative index')
        elif self.parent_node_id is not None or self.index is not None:
            raise ValueError(
                f'{self.modify_type} must not provide parent_node_id or index')
        return self


class PatchSet(ArtifactModel):
    patch_id: Optional[str] = None
    target_doc_id: str
    new_title: Optional[str] = None
    hunks: List[PatchHunk] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class PatchResult(BaseModel):
    patch_id: Optional[str] = None
    success: bool
    applied_hunks: List[str] = Field(default_factory=list)
    failed_hunks: List[str] = Field(default_factory=list)
    message: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
