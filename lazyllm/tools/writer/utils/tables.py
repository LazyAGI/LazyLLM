from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..data_models.writer_ir import WriterBlock, WriterDocument


def _span(cell: WriterBlock, field: str) -> int:
    value = cell.numbering.get(field, 1)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f'table cell {cell.node_id!r} requires a positive integer {field}.')
    return value


def table_grid(table: WriterBlock) -> List[List[Optional[WriterBlock]]]:  # noqa: C901
    if table.type != 'table':
        raise ValueError(f'expected table block, got {table.type!r}.')
    if not table.children:
        raise ValueError(f'table {table.node_id!r} requires at least one table_row child.')
    if any(row.type != 'table_row' for row in table.children):
        raise ValueError(f'table {table.node_id!r} may contain only table_row children.')

    occupied: Dict[Tuple[int, int], Optional[WriterBlock]] = {}
    width = 0
    for row_index, row in enumerate(table.children):
        if row.content or row.spans:
            raise ValueError(f'table row {row.node_id!r} must store content in table_cell children.')
        if not row.children:
            raise ValueError(f'table row {row.node_id!r} requires at least one table_cell child.')
        if any(cell.type != 'table_cell' for cell in row.children):
            raise ValueError(f'table row {row.node_id!r} may contain only table_cell children.')
        column = 0
        for cell in row.children:
            if cell.children:
                raise ValueError(f'table cell {cell.node_id!r} cannot contain child blocks.')
            if 'header' in cell.numbering and not isinstance(cell.numbering['header'], bool):
                raise ValueError(f'table cell {cell.node_id!r} requires a boolean header value.')
            if cell.numbering.get('align') not in {None, '', 'left', 'center', 'right'}:
                raise ValueError(f'table cell {cell.node_id!r} has an invalid alignment.')
            if cell.spans and ''.join(span.text for span in cell.spans) != cell.content:
                raise ValueError(f'table cell {cell.node_id!r} content must equal its spans.')
            while (row_index, column) in occupied:
                column += 1
            row_span = _span(cell, 'row_span')
            column_span = _span(cell, 'column_span')
            if row_index + row_span > len(table.children):
                raise ValueError(f'table cell {cell.node_id!r} row_span is outside the table.')
            positions = {
                (row, col)
                for row in range(row_index, row_index + row_span)
                for col in range(column, column + column_span)
            }
            if set(occupied).intersection(positions):
                raise ValueError(f'table cell {cell.node_id!r} overlaps another merged cell.')
            for position in positions:
                occupied[position] = cell if position == (row_index, column) else None
            column += column_span
            width = max(width, column)

    for row_index in range(len(table.children)):
        if any((row_index, column) not in occupied for column in range(width)):
            raise ValueError(f'table {table.node_id!r} has an incomplete row {row_index + 1}.')
    return [
        [occupied[(row_index, column)] for column in range(width)]
        for row_index in range(len(table.children))
    ]


def validate_table(table: WriterBlock) -> None:
    table_grid(table)


def validate_writer_tables(document: WriterDocument) -> None:
    ids: List[str] = [block.node_id for block in document.iter_blocks()]
    if len(ids) != len(set(ids)):
        raise ValueError('document contains duplicate Writer node_id values.')

    def walk(blocks: List[WriterBlock], parent_type: str = '') -> None:
        for block in blocks:
            if block.type == 'table':
                validate_table(block)
            elif block.type == 'table_row' and parent_type != 'table':
                raise ValueError(f'table_row block {block.node_id!r} must be nested in a table.')
            elif block.type == 'table_cell' and parent_type != 'table_row':
                raise ValueError(f'table_cell block {block.node_id!r} must be nested in a table_row.')
            walk(block.children, block.type)

    walk(document.blocks)


__all__ = ['table_grid', 'validate_table', 'validate_writer_tables']
