import pytest

from lazyllm.tools.writer.data_models import WriterBlock, WriterDocument, WriterSpan
from lazyllm.tools.writer.utils import table_grid, validate_writer_tables


def _cell(node_id, content='', **numbering):
    return WriterBlock(
        node_id=node_id, type='table_cell', content=content,
        spans=[WriterSpan(text=content)] if content else [], numbering=numbering,
    )


def _table(*rows):
    return WriterBlock(
        node_id='table', type='table',
        children=[WriterBlock(node_id=f'row-{index}', type='table_row', children=list(cells))
                  for index, cells in enumerate(rows, start=1)],
    )


def test_table_protocol_accepts_blank_cells_and_merged_grid():
    table = _table(
        [_cell('a', 'A'), _cell('b', 'B')],
        [_cell('c', '', row_span=2), _cell('d', 'D')],
        [_cell('e', 'E')],
    )

    validate_writer_tables(WriterDocument(document_id='doc', blocks=[table]))

    grid = table_grid(table)
    assert [[cell.content if cell else '' for cell in row] for row in grid] == [
        ['A', 'B'], ['', 'D'], ['', 'E'],
    ]


@pytest.mark.parametrize(('table', 'message'), [
    (_table([_cell('a')], [_cell('b', column_span=2)]), 'incomplete row'),
    (_table([_cell('a', row_span=2), _cell('b')]), 'outside the table'),
    (_table(
        [_cell('a'), _cell('b', row_span=3)],
        [_cell('c', column_span=2)],
        [_cell('d')],
    ), 'overlaps'),
    (WriterBlock(
        node_id='table', type='table',
        children=[WriterBlock(node_id='row', type='paragraph')],
    ), 'only table_row'),
])
def test_table_protocol_rejects_invalid_grids(table, message):
    with pytest.raises(ValueError, match=message):
        validate_writer_tables(WriterDocument(document_id='doc', blocks=[table]))


def test_table_protocol_rejects_duplicate_ids_and_legacy_content_only_table():
    duplicate = _table([_cell('same')], [_cell('same')])
    with pytest.raises(ValueError, match='duplicate Writer node_id'):
        validate_writer_tables(WriterDocument(document_id='doc', blocks=[duplicate]))

    legacy = WriterBlock(node_id='legacy', type='table', content='| A |\n| --- |\n| B |')
    with pytest.raises(ValueError, match='requires at least one table_row'):
        validate_writer_tables(WriterDocument(document_id='doc', blocks=[legacy]))
