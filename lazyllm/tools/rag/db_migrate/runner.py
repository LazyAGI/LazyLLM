from __future__ import annotations

import importlib
import os
import re
from datetime import datetime

import sqlalchemy
from sqlalchemy import text

from lazyllm import LOG

_VERSIONS_DIR = os.path.join(os.path.dirname(__file__), 'versions')
_MIGRATIONS_TABLE = 'lazyllm_schema_migrations'


class MigrationRunner:
    def __init__(self, engine):
        self._engine = engine

    def run_up(self):
        self._ensure_version_table()
        self._detect_and_bootstrap()
        applied = self._applied_versions()
        pending = self._pending_versions(applied)
        for version, name, module_path in pending:
            LOG.info(f'[db_migrate] applying {version} ({name})')
            mod = importlib.import_module(module_path)
            mod.up(self._engine)
            self._record_version(version, name)
            LOG.info(f'[db_migrate] applied  {version}')

    # ------------------------------------------------------------------
    # Version table helpers
    # ------------------------------------------------------------------

    def _ensure_version_table(self):
        with self._engine.connect() as conn:
            conn.execute(text(
                f'CREATE TABLE IF NOT EXISTS {_MIGRATIONS_TABLE} ('
                '  version TEXT PRIMARY KEY,'
                '  name TEXT NOT NULL,'
                '  applied_at TIMESTAMP NOT NULL'
                ')'
            ))
            conn.commit()

    def _applied_versions(self):
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(f'SELECT version FROM {_MIGRATIONS_TABLE}')
            ).fetchall()
        return {r[0] for r in rows}

    def _record_version(self, version: str, name: str):
        with self._engine.connect() as conn:
            conn.execute(
                text(
                    f'INSERT OR IGNORE INTO {_MIGRATIONS_TABLE} (version, name, applied_at) '
                    'VALUES (:v, :n, :t)'
                ),
                {'v': version, 'n': name, 't': datetime.now()},
            )
            conn.commit()

    def _record_versions_bulk(self, pairs: list):
        with self._engine.connect() as conn:
            for version, name in pairs:
                conn.execute(
                    text(
                        f'INSERT OR IGNORE INTO {_MIGRATIONS_TABLE} (version, name, applied_at) '
                        'VALUES (:v, :n, :t)'
                    ),
                    {'v': version, 'n': name, 't': datetime.now()},
                )
            conn.commit()

    # ------------------------------------------------------------------
    # Bootstrap: detect legacy DB version when migrations table is new
    # ------------------------------------------------------------------

    def _detect_and_bootstrap(self):
        applied = self._applied_versions()
        if applied:
            return

        inspector = sqlalchemy.inspect(self._engine)
        existing = set(inspector.get_table_names())

        if 'documents' in existing or 'lazyllm_waiting_task_queue' in existing:
            # v0.7.6 legacy DB — all three migration scripts need to run
            LOG.info('[db_migrate] detected v0.7.6 legacy database')
            return

        if 'lazyllm_documents' in existing:
            parse_state_cols = {
                c['name'] for c in inspector.get_columns('lazyllm_doc_parse_state')
            } if 'lazyllm_doc_parse_state' in existing else set()
            if 'idempotency_key' not in parse_state_cols:
                # v1.0.0 legacy DB — 0001 and 0002 already applied logically
                LOG.info('[db_migrate] detected v1.0.0 legacy database')
                all_versions = self._sorted_version_files()
                bootstrap = [(v, n) for v, n, _ in all_versions if v in ('0001', '0002')]
                self._record_versions_bulk(bootstrap)
                return

            # HEAD state already — mark all three as applied
            LOG.info('[db_migrate] detected HEAD legacy database, marking all versions applied')
            all_versions = self._sorted_version_files()
            bootstrap = [(v, n) for v, n, _ in all_versions if v in ('0001', '0002', '0003')]
            self._record_versions_bulk(bootstrap)

    # ------------------------------------------------------------------
    # Version file discovery
    # ------------------------------------------------------------------

    def _sorted_version_files(self):
        entries = []
        for fname in os.listdir(_VERSIONS_DIR):
            if not fname.endswith('.py') or fname == '__init__.py':
                continue
            version, name = _parse_filename(fname)
            if version is None:
                continue
            module_path = f'lazyllm.tools.rag.db_migrate.versions.{fname[:-3]}'
            entries.append((version, name, module_path))

        release = [(v, n, m) for v, n, m in entries if not v.startswith('dev_')]
        dev = [(v, n, m) for v, n, m in entries if v.startswith('dev_')]
        release.sort(key=lambda x: x[0])
        dev.sort(key=lambda x: x[0])
        return release + dev

    def _pending_versions(self, applied: set):
        return [
            (v, n, m)
            for v, n, m in self._sorted_version_files()
            if v not in applied
        ]


def _parse_filename(fname: str):
    stem = fname[:-3]
    # dev_ prefix files: dev_NNNN_xxx  → version='dev_NNNN', name=stem
    m = re.match(r'^(dev_\d+)_(.+)$', stem)
    if m:
        return m.group(1), stem
    # release files: NNNN_xxx → version='NNNN', name=stem
    m = re.match(r'^(\d{4})_(.+)$', stem)
    if m:
        return m.group(1), stem
    return None, None
