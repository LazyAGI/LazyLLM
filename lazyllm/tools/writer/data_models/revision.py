from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, model_validator
from .docir import Anchor
from .writer_ir import WriterAuthoring
from ..utils.artifact import ArtifactModel


class LocateResult(ArtifactModel):
    task_id: Optional[str] = None
    doc_id: Optional[str] = None
    target_node_ids: List[str] = Field(default_factory=list)
    target_block_ids: List[str] = Field(default_factory=list)
    target_reasons: Dict[str, str] = Field(default_factory=dict)
    summary: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def normalize_target_ids(self) -> 'LocateResult':
        if self.target_node_ids and self.target_block_ids and self.target_node_ids != self.target_block_ids:
            raise ValueError('target_node_ids and target_block_ids must match during migration')
        target_ids = self.target_node_ids or self.target_block_ids
        self.target_node_ids = list(target_ids)
        self.target_block_ids = list(target_ids)
        return self


ModifyType = Literal['insert', 'replace', 'delete', 'move']
PatchPosition = Literal['before', 'after']
PatchBlockType = Literal['paragraph', 'heading', 'list_item', 'code', 'quote']


class ModifyInstruction(BaseModel):
    instruction_id: Optional[str] = None
    # target_node_id is the canonical Writer IR field. target_block_id remains
    # during migration for callers that still use DocIR/DraftDocument.
    target_node_id: Optional[str] = None
    target_block_id: Optional[str] = None
    modify_type: ModifyType
    anchor_node_id: Optional[str] = None
    position: Optional[PatchPosition] = None
    instruction: str
    meta: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def normalize_target_id(self) -> 'ModifyInstruction':
        if self.target_node_id and self.target_block_id and self.target_node_id != self.target_block_id:
            raise ValueError('target_node_id and target_block_id must identify the same internal node')
        target_id = self.target_node_id or self.target_block_id
        if not target_id:
            raise ValueError('modify instruction requires target_node_id')
        self.target_node_id = target_id
        self.target_block_id = target_id
        if self.modify_type == 'move' and (not self.anchor_node_id or not self.position):
            raise ValueError('move requires anchor_node_id and position')
        return self


class ModifyPlan(BaseModel):
    plan_id: Optional[str] = None
    task_id: Optional[str] = None
    scope: Literal['document', 'section', 'block', 'span']
    target_node_ids: List[str] = Field(default_factory=list)
    target_block_ids: List[str] = Field(default_factory=list)
    instructions: List[ModifyInstruction] = Field(default_factory=list)
    summary: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def normalize_target_ids(self) -> 'ModifyPlan':
        if self.target_node_ids and self.target_block_ids and self.target_node_ids != self.target_block_ids:
            raise ValueError('target_node_ids and target_block_ids must match during migration')
        target_ids = self.target_node_ids or self.target_block_ids
        self.target_node_ids = list(target_ids)
        self.target_block_ids = list(target_ids)
        return self


class PatchBlock(BaseModel):
    '''Provider-neutral content for blocks created by an insert patch.'''

    type: PatchBlockType
    content: str = ''
    numbering: Dict[str, Any] = Field(default_factory=dict)
    authoring: Optional[WriterAuthoring] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_numbering(self) -> 'PatchBlock':
        if self.type == 'heading':
            level = self.numbering.get('level')
            if not isinstance(level, int) or isinstance(level, bool) or not 1 <= level <= 9:
                raise ValueError('heading requires numbering.level from 1 to 9')
        elif self.type == 'list_item':
            if not isinstance(self.numbering.get('ordered'), bool):
                raise ValueError('list_item requires boolean numbering.ordered')
        return self


class PatchHunk(BaseModel):
    hunk_id: Optional[str] = None
    # target_node_id is canonical; target_block_id is the temporary legacy alias.
    target_node_id: Optional[str] = None
    target_block_id: Optional[str] = None
    modify_type: ModifyType
    anchor: Optional[Anchor] = None
    anchor_node_id: Optional[str] = None
    text_range: Optional[Tuple[int, int]] = None
    old_text: Optional[str] = None
    new_text: Optional[str] = None
    new_blocks: List[PatchBlock] = Field(default_factory=list)
    position: Optional[PatchPosition] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def normalize_target_id(self) -> 'PatchHunk':
        if self.target_node_id and self.target_block_id and self.target_node_id != self.target_block_id:
            raise ValueError('target_node_id and target_block_id must identify the same internal node')
        target_id = self.target_node_id or self.target_block_id
        if not target_id:
            raise ValueError('patch hunk requires target_node_id')
        self.target_node_id = target_id
        self.target_block_id = target_id
        if self.modify_type == 'move' and (not self.anchor_node_id or not self.position):
            raise ValueError('move requires anchor_node_id and position')
        return self


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
