from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, model_validator
from .writer_ir import WriterAuthoring
from ..utils.artifact import ArtifactModel


class LocateResult(ArtifactModel):
    task_id: Optional[str] = None
    doc_id: Optional[str] = None
    target_node_ids: List[str] = Field(default_factory=list)
    target_reasons: Dict[str, str] = Field(default_factory=dict)
    summary: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


ModifyType = Literal['insert', 'replace', 'delete', 'move']
PatchPosition = Literal['before', 'after']
PatchBlockType = Literal['paragraph', 'heading', 'list_item', 'code', 'quote']


class Anchor(BaseModel):
    node_id: str
    text_offset: Optional[int] = None
    text_end: Optional[int] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


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
    target_node_ids: List[str] = Field(default_factory=list)
    instructions: List[ModifyInstruction] = Field(default_factory=list)
    summary: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


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
    target_node_id: str
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
    def validate_move(self) -> 'PatchHunk':
        if self.modify_type == 'move' and (not self.anchor_node_id or not self.position):
            raise ValueError('move requires anchor_node_id and position')
        return self


class PatchSet(ArtifactModel):
    patch_id: Optional[str] = None
    target_doc_id: str
    new_title: Optional[str] = None
    hunks: List[PatchHunk] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)

    def execution_hunks(self) -> List[PatchHunk]:
        '''Return hunks in an order that preserves repeated after-move order.

        Moving several sibling blocks after the same anchor in their document
        order would stack every later block directly behind the anchor and
        reverse the group. Execute each consecutive group from tail to head so
        the persisted document keeps the PatchSet's natural block order.
        '''
        ordered: List[PatchHunk] = []
        index = 0
        while index < len(self.hunks):
            hunk = self.hunks[index]
            if (
                hunk.modify_type == 'move'
                and hunk.position == 'after'
                and hunk.anchor_node_id
            ):
                end = index + 1
                while end < len(self.hunks):
                    candidate = self.hunks[end]
                    if not (
                        candidate.modify_type == 'move'
                        and candidate.position == 'after'
                        and candidate.anchor_node_id == hunk.anchor_node_id
                    ):
                        break
                    end += 1
                ordered.extend(reversed(self.hunks[index:end]))
                index = end
                continue
            ordered.append(hunk)
            index += 1
        return ordered


class PatchResult(BaseModel):
    patch_id: Optional[str] = None
    success: bool
    applied_hunks: List[str] = Field(default_factory=list)
    failed_hunks: List[str] = Field(default_factory=list)
    message: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
