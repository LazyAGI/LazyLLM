from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from .writer_ir import ContentRef, WriterBlock, WriterSpan
from ..utils.artifact import ArtifactModel


class LocatedContent(BaseModel):
    content_ref: ContentRef
    reason: str = ''


class LocateResult(ArtifactModel):
    task_id: Optional[str] = None
    doc_id: Optional[str] = None
    target_title: bool
    targets: List[LocatedContent] = Field(default_factory=list)
    summary: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


ModifyType = Literal['create', 'update', 'delete', 'move']
PatchPosition = Literal['before', 'after']


class ModifyInstruction(BaseModel):
    instruction_id: Optional[str] = None
    content_ref: ContentRef
    modify_type: ModifyType
    position: Optional[PatchPosition] = None
    instruction: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class ModifyPlan(BaseModel):
    plan_id: Optional[str] = None
    task_id: Optional[str] = None
    scope: Literal['document', 'section', 'block', 'span']
    title_instruction: Optional[str] = None
    instructions: List[ModifyInstruction] = Field(default_factory=list)
    summary: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class RevisionBlockContent(BaseModel):
    type: Optional[str] = None
    content: Optional[str] = None
    spans: Optional[List[WriterSpan]] = None
    numbering: Optional[Dict[str, Any]] = None
    references: Optional[List[Dict[str, Any]]] = None
    children: List['RevisionBlockContent'] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_spans(self) -> 'RevisionBlockContent':
        if self.spans:
            span_content = ''.join(span.text for span in self.spans)
            if self.content is None:
                self.content = span_content
            elif self.content != span_content:
                raise ValueError('content must equal the concatenated span text')
        return self


RevisionBlockContent.model_rebuild()


class GeneratedRevision(BaseModel):
    new_title: Optional[str] = None
    changes: Dict[str, List[RevisionBlockContent]] = Field(default_factory=dict)

    @field_validator('changes', mode='before')
    @classmethod
    def _normalize_changes(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        return {
            instruction_id: [change] if isinstance(change, dict) else change
            for instruction_id, change in value.items()
        }


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


class StringReplace(BaseModel):
    replacement_id: Optional[str] = None
    old_string: str
    new_string: str
    content_ref: Optional[ContentRef] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_replacement(self) -> 'StringReplace':
        if not self.old_string:
            raise ValueError('old_string must not be empty')
        if self.old_string == self.new_string:
            raise ValueError('old_string and new_string must differ')
        return self


class StringReplaceSet(ArtifactModel):
    replace_set_id: Optional[str] = None
    replacements: List[StringReplace] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class StringReplaceResult(BaseModel):
    replace_set_id: Optional[str] = None
    success: bool
    applied_replacements: List[str] = Field(default_factory=list)
    failed_replacements: List[str] = Field(default_factory=list)
    message: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
