from __future__ import annotations

from typing import Any, Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


WriterStage = Literal['outline', 'draft', 'final']


class WriterConstraints(BaseModel):
    '''Authoring requirements attached to a document block across all stages.'''

    model_config = ConfigDict(extra='forbid')

    section_goal: Optional[str] = None
    required_points: List[str] = Field(default_factory=list)
    fact_constraints: List[str] = Field(default_factory=list)
    style_constraints: List[str] = Field(default_factory=list)
    relation_constraints: List[str] = Field(default_factory=list)
    min_words: Optional[int] = None
    max_words: Optional[int] = None
    pov: Optional[str] = None
    tone: Optional[str] = None
    must_include: List[str] = Field(default_factory=list)
    must_avoid: List[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_word_range(self) -> 'WriterConstraints':
        if self.min_words is not None and self.min_words < 0:
            raise ValueError('min_words must be non-negative')
        if self.max_words is not None and self.max_words < 0:
            raise ValueError('max_words must be non-negative')
        if (
            self.min_words is not None and self.max_words is not None
            and self.min_words > self.max_words
        ):
            raise ValueError('min_words cannot exceed max_words')
        return self


class WriterAuthoring(BaseModel):
    '''Planning and execution metadata that does not belong to visible content.'''

    # Provider/plugin extensions may add namespaced fields while the common contract
    # above remains validated.
    model_config = ConfigDict(extra='allow')

    instruction: Optional[str] = None
    instruction_id: Optional[str] = None
    origin_node_id: Optional[str] = None
    constraints: WriterConstraints = Field(default_factory=WriterConstraints)
    expected_blocks: List[str] = Field(default_factory=list)
    pending_subtasks: List[str] = Field(default_factory=list)
    revision_notes: List[str] = Field(default_factory=list)
    visual_needs: List[Dict[str, Any]] = Field(default_factory=list)
    source: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class WriterSpan(BaseModel):
    text: str = ''
    style: List[str] = Field(default_factory=list)


class WriterBlock(BaseModel):
    '''A provider-neutral document node identified by an internal Writer IR ID.'''

    # node_id is always an internal, stable Writer IR identifier. External block
    # identifiers belong in provider_binding (for example provider_binding.block_id).
    node_id: str
    type: str
    content: str = ''
    spans: List[WriterSpan] = Field(default_factory=list)
    children: List['WriterBlock'] = Field(default_factory=list)
    stage: WriterStage
    status: str = ''
    authoring: Optional[WriterAuthoring] = None
    numbering: Dict[str, Any] = Field(default_factory=dict)
    references: List[Dict[str, Any]] = Field(default_factory=list)
    # Provider-neutral binding contract. Common keys are provider, uri, document_id,
    # block_id, parent_block_id and revision. IDs here belong to the external system.
    provider_binding: Dict[str, Any] = Field(default_factory=dict)
    # Lossless provider data used by an adapter when a round trip requires more than
    # the normalized Writer fields. It is never part of the visible document body.
    provider_payload: Dict[str, Any] = Field(default_factory=dict)
    editable: bool = True

    @model_validator(mode='after')
    def validate_block(self) -> 'WriterBlock':
        if not self.node_id.strip():
            raise ValueError('node_id must be a non-empty internal Writer IR identifier')
        if not self.type.strip():
            raise ValueError('type must be non-empty')
        if self.spans and ''.join(span.text for span in self.spans) != self.content:
            raise ValueError('content must equal the concatenated span text when spans are present')
        return self


WriterBlock.model_rebuild()


class WriterDocument(BaseModel):
    '''The single document representation shared by outline, draft and final stages.'''

    # document_id is always the internal Writer IR document identifier. External
    # document IDs belong in provider_binding.document_id.
    document_id: str
    stage: WriterStage
    title: str = ''
    blocks: List[WriterBlock] = Field(default_factory=list)
    revision: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    provider_binding: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_document(self) -> 'WriterDocument':
        if not self.document_id.strip():
            raise ValueError('document_id must be a non-empty internal Writer IR identifier')
        node_ids = [block.node_id for block in self.iter_blocks()]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError('WriterDocument node_id values must be unique')
        return self

    def iter_blocks(self) -> Iterable[WriterBlock]:
        def walk(blocks: List[WriterBlock]) -> Iterable[WriterBlock]:
            for block in blocks:
                yield block
                yield from walk(block.children)

        return walk(self.blocks)

    def block_by_id(self, node_id: str) -> Optional[WriterBlock]:
        return next((block for block in self.iter_blocks() if block.node_id == node_id), None)


WriterDocument.model_rebuild()


__all__ = [
    'WriterDocument', 'WriterBlock', 'WriterSpan', 'WriterStage',
    'WriterConstraints', 'WriterAuthoring',
]
