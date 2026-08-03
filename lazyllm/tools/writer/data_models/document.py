from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from .writer_ir import WriterDocument


DocumentFormat = Literal['writer_ir', 'markdown']
MARKDOWN_FILE_EXTENSIONS = ('.md', '.markdown')
MARKDOWN_CONTENT_TYPE = 'text/markdown'


class DocumentPayload(BaseModel):
    '''An in-memory writer document whose representation is explicit.'''

    document_format: DocumentFormat
    content: Union[WriterDocument, str]
    meta: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_content_format(self) -> 'DocumentPayload':
        if self.document_format == 'writer_ir' and not isinstance(self.content, WriterDocument):
            raise ValueError('writer_ir content must be a WriterDocument')
        if self.document_format == 'markdown' and not isinstance(self.content, str):
            raise ValueError('markdown content must be a string')
        return self


class ContentRef(BaseModel):
    '''A content locator for either Writer IR or Markdown.'''

    node_id: Optional[str] = None
    heading_path: List[str] = Field(default_factory=list)
    occurrence: int = Field(default=1, ge=1)

    @model_validator(mode='after')
    def validate_locator(self) -> 'ContentRef':
        self.node_id = self.node_id.strip() if self.node_id else None
        self.heading_path = [part.strip() for part in self.heading_path if part.strip()]
        if bool(self.node_id) == bool(self.heading_path):
            raise ValueError('ContentRef requires exactly one of node_id or heading_path')
        return self

    @property
    def document_format(self) -> DocumentFormat:
        return 'writer_ir' if self.node_id else 'markdown'


__all__ = [
    'ContentRef', 'DocumentFormat', 'DocumentPayload',
    'MARKDOWN_FILE_EXTENSIONS', 'MARKDOWN_CONTENT_TYPE',
]
