from __future__ import annotations
import json
from typing import Any, List

from ..data_models.writer_ir import WriterBlock, WriterDocument


def to_prompt_json(value: Any) -> str:
    def default(obj: Any) -> Any:
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        return str(obj)

    return json.dumps(value, ensure_ascii=False, indent=2, default=default)


def render_document_markdown(document: WriterDocument) -> str:
    parts: List[str] = []
    if document.title:
        parts.append(f'# {document.title.strip()}')
    for block in document.blocks:
        parts.extend(_render_block_markdown(block, level=2))
    return '\n\n'.join(part for part in parts if part).strip() + '\n'


def _render_block_markdown(block: WriterBlock, level: int) -> List[str]:
    parts: List[str] = []
    heading_level = min(max(level, 1), 6)
    if block.type == 'heading':
        if block.content.strip():
            parts.append(f'{"#" * heading_level} {block.content.strip()}')
    else:
        content = block.content.strip()
        if content:
            parts.append(content)
    for child in block.children:
        parts.extend(_render_block_markdown(child, heading_level + 1))
    return parts
