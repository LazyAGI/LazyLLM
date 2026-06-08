'''v0.7.6 baseline marker.

This migration does not create or destroy any tables.  It merely serves as the
sentinel that proves a database has been processed by the migration runner,
starting from the v0.7.6 schema baseline.

The bootstrap detection in runner.py handles the case where this file is
reached on a genuine v0.7.6 legacy DB — the old tables are already in place
and the new tables will be created by 0002.
'''
from __future__ import annotations

import sqlalchemy
from lazyllm import LOG


def up(engine) -> None:
    inspector = sqlalchemy.inspect(engine)
    existing = set(inspector.get_table_names())
    old_tables = {'documents', 'document_groups', 'kb_group_documents',
                  'operation_logs', 'lazyllm_waiting_task_queue',
                  'lazyllm_finished_task_queue', 'lazyllm_algorithm'}
    found = old_tables & existing
    if found:
        LOG.info(f'[0001_v076_baseline] detected legacy v0.7.6 tables: {found}')
    else:
        LOG.info('[0001_v076_baseline] no legacy tables found (fresh install)')
