from __future__ import annotations

from typing import Any

from ..data_models.writer_ir import WriterBlock, WriterDocument, WriterStage


def set_document_editable(
    value: Any,
    *,
    stage: WriterStage | None = None,
) -> WriterDocument:
    '''Prepare Writer IR for editing without exposing unsupported opaque blocks.'''
    document = WriterDocument.model_validate(value)
    if stage is not None:
        document.stage = stage
    document.ui_editable = True

    def update_blocks(blocks: list[WriterBlock], level: int = 1) -> None:
        for block in blocks:
            block.editable = block.type != 'wechat_opaque'
            if stage is not None:
                block.stage = stage
            heading_level = block.numbering.get('level')
            if block.type == 'heading' and (
                not isinstance(heading_level, int)
                or isinstance(heading_level, bool)
                or not 1 <= heading_level <= 9
            ):
                block.numbering['level'] = min(level, 9)
            update_blocks(block.children, level + 1)

    update_blocks(document.blocks)
    return document


__all__ = ['set_document_editable']
