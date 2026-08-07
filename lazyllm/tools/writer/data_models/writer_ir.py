from __future__ import annotations

from typing import Any, Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, Field


WriterStage = Literal['outline', 'draft', 'final']

WRITER_IR_FILE_EXTENSION = '.lmd'
WRITER_IR_CONTENT_TYPE = 'application/vnd.lazymind.writer+json'
WRITER_BLOCK_MUTABLE_FIELDS = ('type', 'content', 'spans', 'stage', 'numbering', 'references')
WRITER_BLOCK_PROVIDER_MANAGED_FIELDS = ('provider_binding', 'provider_payload', 'editable')


class ContentRef(BaseModel):
    '''A content locator for either Writer IR or Markdown.'''

    node_id: Optional[str] = None
    heading_path: List[str] = Field(default_factory=list)
    placeholder_id: Optional[str] = None
    document_root: bool = False
    occurrence: int = Field(default=1, ge=1)


class WriterSpan(BaseModel):
    text: str = ''
    style: Dict[str, Any] = Field(default_factory=dict)


class WriterBlock(BaseModel):
    '''A provider-neutral document node identified by an internal Writer IR ID.'''

    # node_id is always an internal, stable Writer IR identifier. External block
    # identifiers belong in provider_binding (for example provider_binding.block_id).
    node_id: str
    type: str
    numbering: Dict[str, Any] = Field(default_factory=dict)
    references: List[Dict[str, Any]] = Field(default_factory=list)
    content: str = ''
    spans: List[WriterSpan] = Field(default_factory=list)
    children: List['WriterBlock'] = Field(default_factory=list)
    stage: WriterStage = 'draft'
    # Provider-neutral binding contract. Common keys are provider, uri, document_id,
    # block_id, parent_block_id and revision. IDs here belong to the external system.
    provider_binding: Dict[str, Any] = Field(default_factory=dict)
    # Lossless provider data used by an adapter when a round trip requires more than
    # the normalized Writer fields. It is never part of the visible document body.
    provider_payload: Dict[str, Any] = Field(default_factory=dict)
    editable: bool = True

    def iter_blocks(self) -> Iterable['WriterBlock']:
        yield self
        for child in self.children:
            yield from child.iter_blocks()


WriterBlock.model_rebuild()


class WriterDocument(BaseModel):
    '''The single document representation shared by outline, draft and final stages.'''

    # document_id is always the internal Writer IR document identifier. External
    # document IDs belong in provider_binding.document_id.
    document_id: str
    stage: WriterStage = 'draft'
    title: str = ''
    blocks: List[WriterBlock] = Field(default_factory=list)
    revision: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    provider_binding: Dict[str, Any] = Field(default_factory=dict)
    # UI capability hint only. Backend patch/write permissions are enforced separately.
    ui_editable: bool = False

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
    'ContentRef', 'WriterDocument', 'WriterBlock', 'WriterSpan', 'WriterStage',
    'WRITER_IR_FILE_EXTENSION', 'WRITER_IR_CONTENT_TYPE',
    'WRITER_BLOCK_MUTABLE_FIELDS', 'WRITER_BLOCK_PROVIDER_MANAGED_FIELDS',
]
