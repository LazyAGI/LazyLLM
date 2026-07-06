from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from ..utils.artifact import ArtifactModel


class InputResource(BaseModel):
    resource_id: Optional[str] = None
    resource_type: Literal['file', 'url', 'kb', 'text', 'document', 'image', 'table', 'slide']
    uri: Optional[str] = None
    file_id: Optional[str] = None
    kb_id: Optional[str] = None
    inline_text: Optional[str] = None
    mime_type: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class TargetDocument(BaseModel):
    doc_id: Optional[str] = None
    uri: Optional[str] = None
    adapter: Optional[str] = None
    title: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class Selection(BaseModel):
    block_ids: List[str] = Field(default_factory=list)
    text: Optional[str] = None
    anchor_start: Optional[str] = None
    anchor_end: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class WritingTask(ArtifactModel):
    task_id: Optional[str] = None
    query: str
    task_type: Literal['write', 'revise']
    scope: Optional[Literal['document', 'selection', 'auto']] = None
    mode: Optional[Literal['polish', 'rewrite']] = None
    inputs: List[InputResource] = Field(default_factory=list)
    target_document: Optional[TargetDocument] = None
    selection: Optional[Selection] = None
    constraints: Dict[str, Any] = Field(default_factory=dict)
    output: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)
