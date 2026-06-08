'''HEAD incremental columns.

Adds columns that were introduced after v1.0.0:
- lazyllm_doc_parse_state: idempotency_key, queued_at, started_at, finished_at
- lazyllm_doc_node_group_status: error_msg

All ADD COLUMN statements are guarded by column-existence checks (idempotent).
'''
from __future__ import annotations

import sqlalchemy
from sqlalchemy import text

from lazyllm import LOG

_PARSE_STATE_TABLE = 'lazyllm_doc_parse_state'
_NG_STATUS_TABLE = 'lazyllm_doc_node_group_status'

_PARSE_STATE_COLS = [
    ('idempotency_key', 'TEXT'),
    ('queued_at', 'TIMESTAMP'),
    ('started_at', 'TIMESTAMP'),
    ('finished_at', 'TIMESTAMP'),
]

_NG_STATUS_COLS = [
    ('error_msg', 'TEXT'),
]


def _existing_columns(inspector, table_name: str):
    try:
        return {c['name'] for c in inspector.get_columns(table_name)}
    except Exception:
        return set()


def _add_columns_if_missing(conn, inspector, table_name: str, columns: list):
    existing = _existing_columns(inspector, table_name)
    for col_name, col_type in columns:
        if col_name not in existing:
            conn.execute(text(f'ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}'))
            LOG.info(f'[0003_head_add_columns] added {table_name}.{col_name}')


def up(engine) -> None:
    inspector = sqlalchemy.inspect(engine)
    existing_tables = set(inspector.get_table_names())

    with engine.connect() as conn:
        if _PARSE_STATE_TABLE in existing_tables:
            _add_columns_if_missing(conn, inspector, _PARSE_STATE_TABLE, _PARSE_STATE_COLS)
        else:
            LOG.warning(f'[0003_head_add_columns] {_PARSE_STATE_TABLE} not found, skipping')

        if _NG_STATUS_TABLE in existing_tables:
            _add_columns_if_missing(conn, inspector, _NG_STATUS_TABLE, _NG_STATUS_COLS)
        else:
            LOG.warning(f'[0003_head_add_columns] {_NG_STATUS_TABLE} not found, skipping')

        conn.commit()

    LOG.info('[0003_head_add_columns] HEAD column additions complete')
