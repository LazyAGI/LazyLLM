from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from .task import TargetDocument
from ..utils.artifact import ArtifactModel


class DocSpan(BaseModel):
    text: str = ''
    stype: List[str] = Field(default_factory=list)


class DocBlock(BaseModel):
    block_id: str
    block_type: Literal[
        'document', 'heading', 'paragraph', 'list', 'list_item',
        'table', 'table_row', 'table_cell', 'code', 'quote',
        'image', 'divider', 'callout', 'block',
    ]
    text: str = ''
    spans: List[DocSpan] = Field(default_factory=list)
    level: Optional[int] = None
    children: List[DocBlock] = Field(default_factory=list)
    attrs: Dict[str, Any] = Field(default_factory=dict)
    style: Dict[str, Any] = Field(default_factory=dict)
    editable: bool = True
    meta: Dict[str, Any] = Field(default_factory=dict)


DocBlock.model_rebuild()


class Anchor(BaseModel):
    node_id: str
    text_offset: Optional[int] = None
    text_end: Optional[int] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class DocIR(ArtifactModel):
    doc_id: Optional[str] = None
    source: Optional[TargetDocument] = None
    title: Optional[str] = None
    blocks: List[DocBlock] = Field(default_factory=list)
    plain_text: Optional[str] = None
    revision: Optional[str] = None
    adapter: str = ''
    meta: Dict[str, Any] = Field(default_factory=dict)
