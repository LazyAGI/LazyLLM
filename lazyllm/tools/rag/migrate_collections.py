'''
Collection name migration script.

Before v0.x, collection names were: col_{algo_name}_{group}
After  v0.x, collection names are:  col_{group}

This script migrates all data (SQL metadata, segment store, vector store)
from the old naming scheme to the new one.

Usage:
    python -m lazyllm.tools.rag.migrate_collections \\
        --algo-name my_algo \\
        --store-type milvus \\
        --store-uri /path/to/milvus.db \\
        --groups sentences paragraphs \\
        [--segment-uri /path/to/segments.db] \\
        [--dry-run]

    # Or migrate all groups automatically (reads from SQL):
    python -m lazyllm.tools.rag.migrate_collections \\
        --algo-name my_algo \\
        --store-type milvus \\
        --store-uri /path/to/milvus.db \\
        --all-groups \\
        [--sql-db /path/to/lazyllm_parsing_service.db]
'''

import argparse
import hashlib
import re
import sqlite3
import sys
import traceback
from typing import List

from lazyllm import LOG

_COLLECTION_NAME_PATTERN = re.compile(r'[^a-z0-9_]+')
_COLLECTION_NAME_MAX_LEN = 255


# ---------------------------------------------------------------------------
# Collection name helpers (mirrors _DocumentStore logic)
# ---------------------------------------------------------------------------

def _normalize(raw_name: str) -> str:
    normalized = _COLLECTION_NAME_PATTERN.sub('_', raw_name).strip('_')
    if not normalized:
        normalized = 'col'
    if normalized[0].isdigit():
        normalized = f'col_{normalized}'
    if normalized == raw_name.lower() and len(normalized) <= _COLLECTION_NAME_MAX_LEN:
        return normalized
    digest = hashlib.sha1(raw_name.encode()).hexdigest()[:12]
    max_prefix_len = _COLLECTION_NAME_MAX_LEN - len(digest) - 1
    prefix = normalized[:max_prefix_len].rstrip('_') or 'col'
    return f'{prefix}_{digest}'


def old_collection_name(algo_name: str, group: str) -> str:
    return _normalize(f'col_{algo_name}_{group}'.lower())


def new_collection_name(group: str) -> str:
    return _normalize(f'col_{group}'.lower())


# ---------------------------------------------------------------------------
# SQLite (MapStore segment store) migration
# ---------------------------------------------------------------------------

def migrate_sqlite(db_path: str, algo_name: str, groups: List[str], dry_run: bool) -> int:
    '''Rename tables in a MapStore SQLite file. Returns number of tables renamed.'''
    LOG.info(f'[SQLite] Migrating {db_path}')
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
    except Exception as e:
        LOG.error(f'[SQLite] Cannot open {db_path}: {e}')
        return 0

    cur = conn.cursor()
    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = {row[0] for row in cur.fetchall()}

        renamed = 0
        for group in groups:
            old_name = old_collection_name(algo_name, group)
            new_name = new_collection_name(group)
            if old_name == new_name:
                LOG.info(f'[SQLite] SKIP {old_name!r} (names are identical)')
                continue
            if old_name not in existing_tables:
                LOG.info(f'[SQLite] SKIP {old_name!r} (table not found)')
                continue
            if new_name in existing_tables:
                LOG.warning(f'[SQLite] {new_name!r} already exists — skipping rename of {old_name!r}')
                continue
            LOG.info(f'[SQLite] {"DRY " if dry_run else ""}RENAME table {old_name!r} -> {new_name!r}')
            if not dry_run:
                cur.execute(f'ALTER TABLE "{old_name}" RENAME TO "{new_name}"')
                # SQLite doesn't support RENAME INDEX; drop old and recreate standard ones
                cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?",
                    (new_name,)
                )
                for (idx_name,) in cur.fetchall():
                    cur.execute(f'DROP INDEX IF EXISTS "{idx_name}"')
                for col, suffix in [
                    ('parent', 'parent'), ('doc_id', 'docid'),
                    ('kb_id', 'kbid'), ('number', 'number'),
                ]:
                    cur.execute(
                        f'CREATE INDEX IF NOT EXISTS "idx_{new_name}_{suffix}" '
                        f'ON "{new_name}"({col})'
                    )
                renamed += 1

        if not dry_run and renamed:
            conn.commit()
    finally:
        conn.close()
    LOG.info(f'[SQLite] Done: {renamed} table(s) renamed.')
    return renamed


# ---------------------------------------------------------------------------
# Milvus migration
# ---------------------------------------------------------------------------

def migrate_milvus(uri: str, algo_name: str, groups: List[str], dry_run: bool) -> int:
    '''Copy Milvus collections from old name to new name, then drop old. Returns count.'''
    LOG.info(f'[Milvus] Migrating {uri}')
    try:
        from lazyllm.thirdparty import pymilvus
        client = pymilvus.MilvusClient(uri=uri)
    except Exception as e:
        LOG.error(f'[Milvus] Cannot connect to {uri}: {e}')
        return 0

    try:
        existing = set(client.list_collections())
    except Exception as e:
        LOG.error(f'[Milvus] list_collections failed: {e}')
        return 0

    renamed = 0
    for group in groups:
        old_name = old_collection_name(algo_name, group)
        new_name = new_collection_name(group)
        if old_name == new_name:
            LOG.info(f'[Milvus] SKIP {old_name!r} (names are identical)')
            continue
        if old_name not in existing:
            LOG.info(f'[Milvus] SKIP {old_name!r} (collection not found)')
            continue
        if new_name in existing:
            LOG.warning(f'[Milvus] {new_name!r} already exists — skipping rename of {old_name!r}')
            continue
        LOG.info(f'[Milvus] {"DRY " if dry_run else ""}RENAME collection {old_name!r} -> {new_name!r}')
        if not dry_run:
            renamed += _milvus_rename_collection(client, pymilvus, old_name, new_name)

    LOG.info(f'[Milvus] Done: {renamed} collection(s) renamed.')
    return renamed


def _milvus_rename_collection(client, pymilvus, old_name: str, new_name: str) -> int:
    try:
        col_desc = client.describe_collection(old_name)
        fields = col_desc.get('fields', [])
        schema = pymilvus.MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=col_desc.get('enable_dynamic_field', False),
        )
        _FIELD_KEYS = ('field_name', 'datatype', 'is_primary', 'max_length',
                       'dim', 'is_partition_key', 'element_type', 'max_capacity')
        for f in fields:
            schema.add_field(**{k: v for k, v in f.items() if k in _FIELD_KEYS and v is not None})
        index_params = pymilvus.MilvusClient.prepare_index_params()
        for idx in col_desc.get('indexes', []):
            index_params.add_index(
                field_name=idx.get('field_name', ''),
                index_type=idx.get('index_type', 'AUTOINDEX'),
                metric_type=idx.get('metric_type', 'COSINE'),
                params=idx.get('params', {}),
            )
        client.create_collection(new_name, schema=schema, index_params=index_params)
        output_fields = [f['field_name'] for f in fields]
        offset, batch = 0, 1000
        while True:
            rows = client.query(
                collection_name=old_name, filter='',
                output_fields=output_fields, limit=batch, offset=offset,
            )
            if not rows:
                break
            client.upsert(collection_name=new_name, data=rows)
            offset += len(rows)
            if len(rows) < batch:
                break
        client.drop_collection(old_name)
        return 1
    except Exception as e:
        LOG.error(f'[Milvus] Error renaming {old_name!r}: {e}')
        LOG.error(traceback.format_exc())
        return 0


# ---------------------------------------------------------------------------
# Chroma migration
# ---------------------------------------------------------------------------

def migrate_chroma(uri: str, algo_name: str, groups: List[str], dry_run: bool) -> int:
    '''Rename Chroma collections (each group has sub-collections per embed key).'''
    LOG.info(f'[Chroma] Migrating {uri}')
    try:
        from lazyllm.thirdparty import chromadb
        if '://' in uri and not uri.startswith('file://'):
            from urllib.parse import urlparse
            p = urlparse(uri)
            client = chromadb.HttpClient(host=p.hostname, port=p.port or 80)
        else:
            path = uri.replace('file://', '')
            client = chromadb.PersistentClient(path=path)
    except Exception as e:
        LOG.error(f'[Chroma] Cannot connect to {uri}: {e}')
        return 0

    try:
        existing = {c.name for c in client.list_collections()}
    except Exception as e:
        LOG.error(f'[Chroma] list_collections failed: {e}')
        return 0

    renamed = 0
    for group in groups:
        old_base = old_collection_name(algo_name, group)
        new_base = new_collection_name(group)
        if old_base == new_base:
            LOG.info(f'[Chroma] SKIP {old_base!r} (names are identical)')
            continue
        old_subs = [n for n in existing if n.startswith(old_base + '_') and n.endswith('_embed')]
        if not old_subs:
            LOG.info(f'[Chroma] SKIP {old_base!r}* (no sub-collections found)')
            continue
        for old_sub in old_subs:
            suffix = old_sub[len(old_base):]
            new_sub = new_base + suffix
            if new_sub in existing:
                LOG.warning(f'[Chroma] {new_sub!r} already exists — skipping {old_sub!r}')
                continue
            LOG.info(f'[Chroma] {"DRY " if dry_run else ""}RENAME {old_sub!r} -> {new_sub!r}')
            if not dry_run:
                renamed += _chroma_rename_collection(client, old_sub, new_sub)

    LOG.info(f'[Chroma] Done: {renamed} sub-collection(s) renamed.')
    return renamed


def _chroma_rename_collection(client, old_sub: str, new_sub: str) -> int:
    try:
        old_col = client.get_collection(old_sub)
        result = old_col.get(include=['embeddings', 'documents', 'metadatas'])
        new_col = client.get_or_create_collection(new_sub)
        ids = result.get('ids', [])
        if ids:
            new_col.upsert(
                ids=ids,
                embeddings=result.get('embeddings'),
                documents=result.get('documents'),
                metadatas=result.get('metadatas'),
            )
        client.delete_collection(old_sub)
        return 1
    except Exception as e:
        LOG.error(f'[Chroma] Error renaming {old_sub!r}: {e}')
        LOG.error(traceback.format_exc())
        return 0


# ---------------------------------------------------------------------------
# SQL metadata check
# ---------------------------------------------------------------------------

def check_sql_metadata(sql_db: str, algo_name: str) -> None:
    LOG.info(f'[SQL] Checking {sql_db}')
    try:
        conn = sqlite3.connect(sql_db, timeout=10.0)
    except Exception as e:
        LOG.error(f'[SQL] Error reading SQL DB: {e}')
        return
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}
        if 'lazyllm_algorithm' not in tables:
            LOG.info('[SQL] lazyllm_algorithm table not found — nothing to check.')
            return
        cur.execute('SELECT id, display_name FROM lazyllm_algorithm WHERE id=?', (algo_name,))
        row = cur.fetchone()
        if row:
            LOG.info(f'[SQL] Found algorithm: id={row[0]!r}, display_name={row[1]!r}')
        else:
            LOG.warning(f'[SQL] Algorithm {algo_name!r} not found in lazyllm_algorithm.')
        if 'lazyllm_node_group' in tables:
            cur.execute('SELECT COUNT(*) FROM lazyllm_node_group')
            count = cur.fetchone()[0]
            LOG.info(f'[SQL] lazyllm_node_group: {count} row(s) — already on new schema.')
    except Exception as e:
        LOG.error(f'[SQL] Error reading SQL DB: {e}')
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Auto-discover groups from SQL
# ---------------------------------------------------------------------------

def discover_groups_from_sql(sql_db: str, algo_name: str) -> List[str]:
    groups: List[str] = []
    try:
        conn = sqlite3.connect(sql_db, timeout=10.0)
    except Exception as e:
        LOG.error(f'[SQL] Error discovering groups: {e}')
        return groups
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}

        if 'lazyllm_node_group' in tables:
            cur.execute('SELECT name FROM lazyllm_node_group')
            groups = [row[0] for row in cur.fetchall()]
            LOG.info(f'[SQL] Discovered {len(groups)} group(s) from lazyllm_node_group: {groups}')
        elif 'lazyllm_algorithm' in tables:
            cur.execute('SELECT info_pickle FROM lazyllm_algorithm WHERE id=?', (algo_name,))
            row = cur.fetchone()
            if row:
                import base64
                import pickle
                try:
                    info = pickle.loads(base64.b64decode(row[0]))
                    groups = list(info.get('node_groups', {}).keys())
                    LOG.info(f'[SQL] Discovered {len(groups)} group(s) from legacy info_pickle: {groups}')
                except Exception as e:
                    LOG.warning(f'[SQL] Could not decode info_pickle: {e}')
    except Exception as e:
        LOG.error(f'[SQL] Error discovering groups: {e}')
    finally:
        conn.close()
    return groups


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_args(args):
    '''Resolve sql_db path and groups list from parsed args.'''
    if args.sql_db:
        sql_db = args.sql_db
    else:
        import os
        from lazyllm import config
        sql_db = os.path.join(
            os.path.expanduser(os.path.join(config['home'], '.dbs')),
            'lazyllm_parsing_service.db',
        )
    LOG.info(f'SQL DB: {sql_db}')

    groups = list(args.groups or [])
    if args.all_groups or not groups:
        discovered = discover_groups_from_sql(sql_db, args.algo_name)
        seen = set(groups)
        for g in discovered:
            if g not in seen:
                groups.append(g)
                seen.add(g)
    return sql_db, groups


def _migrate_stores(args, groups: List[str]) -> int:
    '''Run segment-store and vector-store migrations. Returns total renamed count.'''
    import os
    from pathlib import Path

    total = 0
    segment_uri = args.segment_uri
    if segment_uri is None and args.store_type == 'milvus' and args.store_uri:
        p = Path(args.store_uri)
        candidate = str(p.parent / f'lazyllm_{p.stem}_segments.db')
        if os.path.exists(candidate):
            segment_uri = candidate
            LOG.info(f'[Auto] Inferred segment store: {segment_uri}')

    if segment_uri:
        total += migrate_sqlite(segment_uri, args.algo_name, groups, args.dry_run)

    if args.store_type == 'milvus' and args.store_uri:
        total += migrate_milvus(args.store_uri, args.algo_name, groups, args.dry_run)
    elif args.store_type == 'chroma' and args.store_uri:
        total += migrate_chroma(args.store_uri, args.algo_name, groups, args.dry_run)
    elif args.store_type == 'mapstore' and args.store_uri:
        total += migrate_sqlite(args.store_uri, args.algo_name, groups, args.dry_run)
    else:
        LOG.info('[Vector store] Skipped (--store-type none or no --store-uri).')
    return total


def main():
    parser = argparse.ArgumentParser(
        description='Migrate LazyLLM RAG collection names from col_{algo}_{group} to col_{group}.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--algo-name', required=True,
                        help='Algorithm name (the algo_id used when registering)')
    parser.add_argument('--store-type', choices=['milvus', 'chroma', 'mapstore', 'none'],
                        default='none', help='Vector/segment store type')
    parser.add_argument('--store-uri', default=None,
                        help='URI or path to the vector store (Milvus .db file, Chroma dir, etc.)')
    parser.add_argument('--segment-uri', default=None,
                        help='Path to MapStore SQLite .db (segment store). '
                             'If omitted and store-type is milvus, inferred from store-uri.')
    parser.add_argument('--groups', nargs='*', default=None,
                        help='Node group names to migrate. If omitted, use --all-groups.')
    parser.add_argument('--all-groups', action='store_true',
                        help='Auto-discover all groups from SQL DB.')
    parser.add_argument('--sql-db', default=None,
                        help='Path to the LazyLLM SQL DB '
                             '(default: ~/.lazyllm/.dbs/lazyllm_parsing_service.db)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be done without making any changes.')
    args = parser.parse_args()

    if args.dry_run:
        LOG.warning('*** DRY RUN — no changes will be made ***')

    sql_db, groups = _resolve_args(args)
    if not groups:
        LOG.error('No groups specified and none discovered. Use --groups or --all-groups.')
        sys.exit(1)
    LOG.info(f'Groups to migrate: {groups}')

    check_sql_metadata(sql_db, args.algo_name)
    total = _migrate_stores(args, groups)

    suffix = 'would be' if args.dry_run else 'were'
    LOG.info(f'{"[DRY RUN] " if args.dry_run else ""}Migration complete. '
             f'{total} collection(s)/table(s) {suffix} renamed.')


if __name__ == '__main__':
    main()
