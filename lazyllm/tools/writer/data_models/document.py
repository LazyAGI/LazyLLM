from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from .writer_ir import WriterDocument


DocumentFormat = Literal['writer_ir', 'markdown']
MARKDOWN_FILE_EXTENSIONS = ('.md', '.markdown')
MARKDOWN_CONTENT_TYPE = 'text/markdown'


class DocumentPayload(BaseModel):
    '''An in-memory writer document whose representation is explicit.'''

    document_format: DocumentFormat
    content: Union[WriterDocument, str]
    meta: Dict[str, Any] = Field(default_factory=dict)


class ContentRef(BaseModel):
    '''A content locator for either Writer IR or Markdown.'''

    node_id: Optional[str] = None
    heading_path: List[str] = Field(default_factory=list)
    occurrence: int = Field(default=1, ge=1)


__all__ = [
    'ContentRef', 'DocumentFormat', 'DocumentPayload',
    'MARKDOWN_FILE_EXTENSIONS', 'MARKDOWN_CONTENT_TYPE',
]
