'''v1.0.0 new tables.

Creates all 10 new doc_service tables using CREATE TABLE IF NOT EXISTS (idempotent).
Also performs an optional data migration from the old v0.7.6 `documents` table
into `lazyllm_documents` when the legacy table is present.
'''
from __future__ import annotations

from sqlalchemy import text
from lazyllm import LOG


# ------------------------------------------------------------------
# DDL for 10 new tables (without the HEAD-only columns that 0003 adds)
# ------------------------------------------------------------------

_DDL = [
    '''CREATE TABLE IF NOT EXISTS lazyllm_documents (
        doc_id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        path TEXT NOT NULL,
        meta TEXT,
        upload_status TEXT NOT NULL,
        source_type TEXT NOT NULL DEFAULT 'API',
        file_type TEXT,
        content_hash TEXT,
        size_bytes INTEGER,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    )''',
    '''CREATE TABLE IF NOT EXISTS lazyllm_knowledge_bases (
        kb_id TEXT PRIMARY KEY,
        display_name TEXT,
        description TEXT,
        doc_count INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'ACTIVE',
        owner_id TEXT,
        meta TEXT,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    )''',
    '''CREATE TABLE IF NOT EXISTS lazyllm_kb_documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        kb_id TEXT NOT NULL,
        doc_id TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    )''',
    '''CREATE TABLE IF NOT EXISTS lazyllm_kb_algorithm (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        kb_id TEXT NOT NULL,
        algo_id TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    )''',
    '''CREATE TABLE IF NOT EXISTS lazyllm_doc_parse_state (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id TEXT NOT NULL,
        kb_id TEXT NOT NULL,
        status TEXT NOT NULL,
        current_task_id TEXT,
        task_type TEXT,
        priority INTEGER NOT NULL DEFAULT 0,
        task_score INTEGER,
        retry_count INTEGER NOT NULL DEFAULT 0,
        max_retry INTEGER NOT NULL DEFAULT 3,
        lease_owner TEXT,
        lease_until TIMESTAMP,
        last_error_code TEXT,
        last_error_msg TEXT,
        failed_stage TEXT,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    )''',
    '''CREATE TABLE IF NOT EXISTS lazyllm_doc_node_group_status (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id TEXT NOT NULL,
        kb_id TEXT NOT NULL,
        node_group_id TEXT NOT NULL,
        file_path TEXT,
        status TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    )''',
    '''CREATE TABLE IF NOT EXISTS lazyllm_doc_service_tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL,
        task_type TEXT NOT NULL,
        doc_id TEXT NOT NULL,
        kb_id TEXT NOT NULL,
        status TEXT NOT NULL,
        message TEXT,
        error_code TEXT,
        error_msg TEXT,
        started_at TIMESTAMP,
        finished_at TIMESTAMP,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    )''',
    '''CREATE TABLE IF NOT EXISTS lazyllm_idempotency_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        endpoint TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        req_hash TEXT NOT NULL,
        status TEXT NOT NULL,
        response_json TEXT,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    )''',
    '''CREATE TABLE IF NOT EXISTS lazyllm_callback_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        callback_id TEXT NOT NULL,
        task_id TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    )''',
    '''CREATE TABLE IF NOT EXISTS lazyllm_doc_path_locks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT NOT NULL,
        owner TEXT NOT NULL,
        expires_at TIMESTAMP NOT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    )''',
]

# v0.7.6 status → DocStatus mapping
_STATUS_MAP = {
    'WAITING': 'WAITING',
    'RUNNING': 'WORKING',
    'SUCCESS': 'SUCCESS',
    'FAILED': 'FAILED',
}


def up(engine) -> None:
    with engine.connect() as conn:
        for ddl in _DDL:
            conn.execute(text(ddl))
        conn.commit()
    LOG.info('[0002_v100_new_tables] all 10 new tables created (if not exists)')

    _migrate_legacy_documents(engine)


def _migrate_legacy_documents(engine) -> None:
    import sqlalchemy
    inspector = sqlalchemy.inspect(engine)
    existing = set(inspector.get_table_names())
    if 'documents' not in existing:
        return

    with engine.connect() as conn:
        rows = conn.execute(
            text('SELECT doc_id, filename, path, meta, status FROM documents')
        ).fetchall()

    if not rows:
        return

    LOG.info(f'[0002_v100_new_tables] migrating {len(rows)} rows from legacy `documents`')
    with engine.connect() as conn:
        for row in rows:
            doc_id, filename, path, meta, status = row
            new_status = _STATUS_MAP.get(status, 'WAITING')
            conn.execute(
                text(
                    'INSERT OR IGNORE INTO lazyllm_documents '
                    '(doc_id, filename, path, meta, upload_status, source_type) '
                    'VALUES (:doc_id, :filename, :path, :meta, :status, :src)'
                ),
                {
                    'doc_id': doc_id,
                    'filename': filename,
                    'path': path,
                    'meta': meta,
                    'status': new_status,
                    'src': 'API',
                },
            )
        conn.commit()
    LOG.info('[0002_v100_new_tables] legacy document migration complete')
