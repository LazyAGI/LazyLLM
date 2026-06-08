'''Unit tests for lazyllm.tools.rag.db_migrate.

All tests use real SQLite via tmp_path; no mocking of the engine so DDL
correctness is verified against an actual database.
'''
from __future__ import annotations

import os
import textwrap

import sqlalchemy
from sqlalchemy import create_engine, text

from lazyllm.tools.rag.db_migrate import run_migrations

_MIGRATIONS_TABLE = 'lazyllm_schema_migrations'

_ALL_NEW_TABLES = {
    'lazyllm_documents',
    'lazyllm_knowledge_bases',
    'lazyllm_kb_documents',
    'lazyllm_kb_algorithm',
    'lazyllm_doc_parse_state',
    'lazyllm_doc_node_group_status',
    'lazyllm_doc_service_tasks',
    'lazyllm_idempotency_records',
    'lazyllm_callback_records',
    'lazyllm_doc_path_locks',
}


def _tables(engine):
    return set(sqlalchemy.inspect(engine).get_table_names())


def _applied(engine):
    with engine.connect() as conn:
        rows = conn.execute(
            text(f'SELECT version FROM {_MIGRATIONS_TABLE}')
        ).fetchall()
    return {r[0] for r in rows}


def _columns(engine, table):
    return {c['name'] for c in sqlalchemy.inspect(engine).get_columns(table)}


# ------------------------------------------------------------------
# Scenario 1: fresh install
# ------------------------------------------------------------------

def test_fresh_install(tmp_path):
    engine = create_engine(f'sqlite:///{tmp_path}/test.db')
    run_migrations(engine)

    tables = _tables(engine)
    for t in _ALL_NEW_TABLES:
        assert t in tables, f'Expected table {t} to exist after fresh install'

    parse_cols = _columns(engine, 'lazyllm_doc_parse_state')
    for col in ('idempotency_key', 'queued_at', 'started_at', 'finished_at'):
        assert col in parse_cols, f'Expected column lazyllm_doc_parse_state.{col}'

    ng_cols = _columns(engine, 'lazyllm_doc_node_group_status')
    assert 'error_msg' in ng_cols

    assert _applied(engine) == {'0001', '0002', '0003'}


# ------------------------------------------------------------------
# Scenario 2: upgrade from v0.7.6
# ------------------------------------------------------------------

def test_upgrade_from_v076(tmp_path):
    engine = create_engine(f'sqlite:///{tmp_path}/test.db')
    with engine.connect() as conn:
        conn.execute(text('CREATE TABLE documents (doc_id TEXT PRIMARY KEY, filename TEXT, '
                          'path TEXT, meta TEXT, status TEXT)'))
        conn.execute(text('INSERT INTO documents VALUES ("doc1", "a.txt", "/a.txt", NULL, "SUCCESS")'))
        conn.execute(text('CREATE TABLE lazyllm_waiting_task_queue (id INTEGER PRIMARY KEY)'))
        conn.commit()

    run_migrations(engine)

    tables = _tables(engine)
    # legacy tables preserved
    assert 'documents' in tables
    # all new tables created
    for t in _ALL_NEW_TABLES:
        assert t in tables, f'Expected new table {t} after v0.7.6 upgrade'

    # data migrated
    with engine.connect() as conn:
        rows = conn.execute(
            text('SELECT doc_id, upload_status FROM lazyllm_documents')
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 'doc1'
    assert rows[0][1] == 'SUCCESS'

    versions = _applied(engine)
    assert {'0001', '0002', '0003'} == versions


# ------------------------------------------------------------------
# Scenario 3: upgrade from v1.0.0 (missing HEAD columns)
# ------------------------------------------------------------------

def test_upgrade_from_v100(tmp_path):
    engine = create_engine(f'sqlite:///{tmp_path}/test.db')
    with engine.connect() as conn:
        conn.execute(text('CREATE TABLE lazyllm_documents (doc_id TEXT PRIMARY KEY)'))
        conn.execute(text(
            'CREATE TABLE lazyllm_doc_parse_state ('
            '  id INTEGER PRIMARY KEY, doc_id TEXT, kb_id TEXT, status TEXT'
            ')'
        ))
        conn.execute(text(
            'CREATE TABLE lazyllm_doc_node_group_status ('
            '  id INTEGER PRIMARY KEY, doc_id TEXT, kb_id TEXT, node_group_id TEXT, status TEXT'
            ')'
        ))
        conn.commit()

    run_migrations(engine)

    parse_cols = _columns(engine, 'lazyllm_doc_parse_state')
    for col in ('idempotency_key', 'queued_at', 'started_at', 'finished_at'):
        assert col in parse_cols, f'Expected column lazyllm_doc_parse_state.{col}'

    ng_cols = _columns(engine, 'lazyllm_doc_node_group_status')
    assert 'error_msg' in ng_cols

    versions = _applied(engine)
    assert '0003' in versions
    # 0001 and 0002 were bootstrapped
    assert '0001' in versions and '0002' in versions


# ------------------------------------------------------------------
# Scenario 4: idempotency — running twice must not error or duplicate
# ------------------------------------------------------------------

def test_idempotent(tmp_path):
    engine = create_engine(f'sqlite:///{tmp_path}/test.db')
    run_migrations(engine)
    run_migrations(engine)

    rows = _applied(engine)
    assert len(rows) == 3, f'Expected exactly 3 version records, got {len(rows)}: {rows}'


# ------------------------------------------------------------------
# Scenario 5: dev_ file merge
# ------------------------------------------------------------------

def test_dev_file_merge(tmp_path, monkeypatch):
    from lazyllm.tools.rag.db_migrate import merge as merge_module

    versions_dir = tmp_path / 'versions'
    versions_dir.mkdir()
    (versions_dir / '__init__.py').write_text('')

    dev1 = versions_dir / 'dev_0004_add_foo.py'
    dev1.write_text(textwrap.dedent('''\
        from sqlalchemy import text

        def up(engine):
            with engine.connect() as conn:
                conn.execute(text("CREATE TABLE IF NOT EXISTS foo (id INTEGER PRIMARY KEY)"))
                conn.commit()
    '''))

    dev2 = versions_dir / 'dev_0005_add_bar.py'
    dev2.write_text(textwrap.dedent('''\
        from sqlalchemy import text

        def up(engine):
            with engine.connect() as conn:
                conn.execute(text("CREATE TABLE IF NOT EXISTS bar (id INTEGER PRIMARY KEY)"))
                conn.commit()
    '''))

    monkeypatch.setattr(merge_module, '_VERSIONS_DIR', str(versions_dir))

    merge_module.merge('v110_test')

    remaining = os.listdir(str(versions_dir))
    assert not any(f.startswith('dev_') for f in remaining), \
        f'dev_ files should be removed after merge, found: {remaining}'

    release_files = [f for f in remaining if re.match(r'^\d{4}_v110_test\.py$', f)]
    assert len(release_files) == 1, f'Expected one release file, found: {release_files}'


import re  # noqa: E402 — used inside test_dev_file_merge
