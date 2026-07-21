from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from ..utils.artifact import ArtifactModel
from .writer_ir import WriterBlock, WriterDocument


class DocumentSummary(BaseModel):
    summary: str
    key_points: List[str] = Field(default_factory=list)
    structure_summary: Optional[str] = None


class BlockSummary(BaseModel):
    block_id: str
    summary: str
    key_points: List[str] = Field(default_factory=list)


class DocumentFact(BaseModel):
    fact_id: str
    key: str
    value: str
    source: List[str] = Field(default_factory=list)
    applies_to_block_ids: List[str] = Field(default_factory=list)
    locked: bool = False


class StyleProfile(BaseModel):
    tone: Optional[str] = None
    formality: Optional[str] = None
    audience: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


class BlockRelationGraph(BaseModel):
    relations: List[Dict[str, Any]] = Field(default_factory=list)


class WritingContext(ArtifactModel):
    context_id: str
    doc_id: Optional[str] = None
    document_summary: Optional[DocumentSummary] = None
    block_summaries: List[BlockSummary] = Field(default_factory=list)
    facts: List[DocumentFact] = Field(default_factory=list)
    style_profile: Optional[StyleProfile] = None
    relation_graph: Optional[BlockRelationGraph] = None
    query: Optional[str] = None
    outline: Optional['WriterDocument'] = None
    draft_sections: List['WriterBlock'] = Field(default_factory=list)
    draft_document: Optional['WriterDocument'] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


WritingContext.model_rebuild()
