import json
import os
import sqlite3
import threading

from lazyllm import LOG
from lazyllm.common import override

from ..store_base import LazyLLMStoreBase, StoreCapability, INSERT_BATCH_SIZE, DEFAULT_KB_ID
from ...global_metadata import RAG_DOC_ID, RAG_KB_ID


class SQLiteStore(LazyLLMStoreBase):
    capability = StoreCapability.SEGMENT
    need_embedding = False
    supports_index_registration = False

    # ── Column schema (row layout: 1:1 with _serialize_row / _deserialize_row order) ──
    _COLS = [('uid', 'TEXT PRIMARY KEY'),
             ('doc_id', 'TEXT NOT NULL'),
             ('"group"', 'TEXT'),
             ('content', 'TEXT'),
             ('meta', 'TEXT'),
             ('global_meta', 'TEXT'),
             ('type', 'INTEGER'),
             ('number', 'INTEGER'),
             ('kb_id', 'TEXT'),
             ('excluded_embed_metadata_keys', 'TEXT'),
             ('excluded_llm_metadata_keys', 'TEXT'),
             ('parent', 'TEXT'),
             ('answer', 'TEXT'),
             ('image_keys', 'TEXT')]
    _COL_NAMES = [c[0] for c in _COLS]
    _COL_NAMES_SQL = ', '.join(_COL_NAMES)
    _COL_NAMES_FTS = ', '.join(f'm.{c}' for c in _COL_NAMES)
    _COL_PLACEHOLDERS = ', '.join(['?'] * len(_COLS))
    _INDEX_COLS = ['doc_id', 'kb_id', '"group"', 'parent', 'number']
    _STD_KEYS = frozenset({c[0].strip('"') for c in _COLS} | {'embedding', 'copy_source'})

    def __init__(self, db_path: str, **kwargs):
        self._db_path = db_path
        self._conn = None
        self._primary_key = 'uid'

    @property
    def dir(self):
        return os.path.dirname(os.path.abspath(self._db_path)) + os.sep

    def _open_conn(self):
        if self._conn: return self._conn
        os.makedirs(os.path.dirname(self._db_path) or '.', exist_ok=True)
        conn = sqlite3.connect(self._db_path, timeout=5.0, check_same_thread=False)
        conn.execute('PRAGMA journal_mode = WAL;')
        conn.execute('PRAGMA synchronous = NORMAL;')
        conn.execute('PRAGMA busy_timeout = 5000;')
        conn.commit()
        self._conn = conn
        return conn

    @override
    def connect(self, global_metadata_desc=None, **kwargs):
        self._global_metadata_desc = global_metadata_desc or {}
        self._ddl_lock = threading.Lock()
        self._open_conn()

    def _ensure_table(self, cursor, table):
        col_defs = ', '.join(f'{name} {typ}' for name, typ in self._COLS)
        cursor.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({col_defs})')
        for idx_col in self._INDEX_COLS:
            col_name = idx_col.strip('"')
            cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{table}_{col_name} ON "{table}"({idx_col})')

    def _ensure_fts(self, cursor, table):
        cursor.execute(f'CREATE VIRTUAL TABLE IF NOT EXISTS "{table}_fts" '
                       f"USING fts5(uid UNINDEXED, content, tokenize='unicode61')")

    @override
    def upsert(self, collection_name, data):
        if not data: return True
        try:
            with self._ddl_lock:
                conn = self._open_conn()
                cur = conn.cursor()
                self._ensure_table(cur, collection_name)
                self._ensure_fts(cur, collection_name)
                sql = (f'INSERT OR REPLACE INTO "{collection_name}" ({self._COL_NAMES_SQL}) '
                       f'VALUES ({self._COL_PLACEHOLDERS})')

                for i in range(0, len(data), INSERT_BATCH_SIZE):
                    batch = data[i:i + INSERT_BATCH_SIZE]
                    cur.executemany(sql, [self._serialize_row(item) for item in batch])
                    cur.executemany(f'DELETE FROM "{collection_name}_fts" WHERE uid = ?',
                                    [(item['uid'],) for item in batch])
                    cur.executemany(f'INSERT INTO "{collection_name}_fts"(uid, content) VALUES (?, ?)',
                                    [(item['uid'], self._fts_content(item)) for item in batch])
                conn.commit()
            return True
        except Exception as e:
            LOG.error(f'[SQLiteStore - upsert] Error: {e}')
            return False

    @override
    def delete(self, collection_name, criteria=None, **kwargs):
        try:
            conn = self._open_conn()
            cur = conn.cursor()
            if not criteria:
                with self._ddl_lock:
                    cur.execute(f'DROP TABLE IF EXISTS "{collection_name}"')
                    cur.execute(f'DROP TABLE IF EXISTS "{collection_name}_fts"')
                    conn.commit()
                return True
            with self._ddl_lock:
                where, args = self._build_where(criteria)
                cur.execute(f'SELECT uid FROM "{collection_name}" {where}', args)
                uids = [row[0] for row in cur.fetchall()]
                if uids:
                    ph = ','.join('?' for _ in uids)
                    cur.execute(f'DELETE FROM "{collection_name}" WHERE uid IN ({ph})', uids)
                    cur.execute(f'DELETE FROM "{collection_name}_fts" WHERE uid IN ({ph})', uids)
                conn.commit()
            return True
        except Exception as e:
            LOG.error(f'[SQLiteStore - delete] Error: {e}')
            return False

    @override
    def get(self, collection_name, criteria=None, **kwargs):
        try:
            conn = self._open_conn()
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (collection_name,))
            if not cur.fetchone():
                return ([], 0) if kwargs.get('return_total') else []

            limit = kwargs.get('limit')
            offset = max(kwargs.get('offset', 0) or 0, 0)
            return_total = kwargs.get('return_total', False)
            sort_by_number = kwargs.get('sort_by_number', False)

            where, args = self._build_where(criteria)
            order = ' ORDER BY number ASC, uid ASC' if sort_by_number else ''
            page = ' LIMIT ? OFFSET ?' if limit is not None else (' LIMIT -1 OFFSET ?' if offset > 0 else '')
            page_args = (limit, offset) if limit is not None else ((offset,) if offset > 0 else ())

            total = None
            if return_total:
                cur.execute(f'SELECT COUNT(*) FROM "{collection_name}"{where}', args)
                total = cur.fetchone()[0]

            cur.execute(f'SELECT {self._COL_NAMES_SQL} FROM "{collection_name}"{where}{order}{page}', args + page_args)
            return (([self._deserialize_row(r) for r in cur.fetchall()], total) if return_total
                    else [self._deserialize_row(r) for r in cur.fetchall()])
        except Exception as e:
            LOG.error(f'[SQLiteStore - get] Error: {e}')
            return ([], 0) if kwargs.get('return_total') else []

    @override
    def search(self, collection_name, query=None, topk=10, filters=None, **kwargs):
        if not query: return []
        try:
            conn = self._open_conn()
            if not conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                                (collection_name,)).fetchone():
                return []

            cmatch = f'"{collection_name}_fts" MATCH ?'
            q = [query if query.endswith('*') else query + '*']
            if filters:
                fw, fa = self._build_fts_filter_where(filters)
                where = f'{cmatch} AND {fw}'
                q.extend(fa)
            else:
                where = cmatch

            sql = f'SELECT {self._COL_NAMES_FTS}, -rank AS score FROM "{collection_name}_fts" f ' \
                  f'JOIN "{collection_name}" m ON f.uid = m.uid WHERE {where} ORDER BY rank LIMIT ?'
            return [dict(self._deserialize_row(r[:-1]), score=r[-1])
                    for r in conn.execute(sql, q + [topk]).fetchall()]
        except Exception as e:
            LOG.error(f'[SQLiteStore - search] Error: {e}')
            return []

    # ── Row serialization (order must match _COLS) ──
    def _serialize_row(self, item):
        gm = dict(item.get('global_meta', {}))
        for k, v in item.items():
            if k not in self._STD_KEYS and v is not None:
                gm[k] = v
        return (item['uid'], item.get('doc_id', ''), item.get('group', ''),
                item.get('content', ''), json.dumps(item.get('meta', {})),
                json.dumps(gm), item.get('type', 0),
                item.get('number', 0), item.get('kb_id', item.get(RAG_KB_ID, DEFAULT_KB_ID)),
                json.dumps(item.get('excluded_embed_metadata_keys', [])),
                json.dumps(item.get('excluded_llm_metadata_keys', [])),
                item.get('parent'), item.get('answer', ''),
                json.dumps(item.get('image_keys', [])))

    def _deserialize_row(self, row):
        gm = json.loads(row[5]) if row[5] else {}
        result = {'uid': row[0], 'doc_id': row[1], 'group': row[2], 'content': row[3],
                  'meta': json.loads(row[4]) if row[4] else {},
                  'global_meta': gm, 'type': row[6], 'number': row[7], 'kb_id': row[8],
                  'excluded_embed_metadata_keys': json.loads(row[9]) if row[9] else [],
                  'excluded_llm_metadata_keys': json.loads(row[10]) if row[10] else [],
                  'parent': row[11], 'answer': row[12],
                  'image_keys': json.loads(row[13]) if row[13] else []}
        for k in self._global_metadata_desc:
            if k not in self._STD_KEYS and k in gm:
                result[k] = gm[k]
        return result

    def _fts_content(self, item):
        extra = ' '.join(str(v) for k, v in item.items()
                         if k not in self._STD_KEYS and v is not None and isinstance(v, str))
        return f"{item.get('content', '')} {extra}".strip()

    _STD_WHERE = {RAG_DOC_ID: 'doc_id', RAG_KB_ID: 'kb_id', 'parent': 'parent',
                  'number': 'number', 'uid': 'uid', 'doc_id': 'doc_id', 'kb_id': 'kb_id'}

    def _build_where(self, criteria):
        if not criteria: return '', ()
        clauses, args = [], []
        for field, vals in criteria.items():
            if not vals: continue
            if not isinstance(vals, (list, set, tuple)): vals = [vals]
            if lookup := self._STD_WHERE.get(field):
                clauses.append(f'"{lookup}" IN ({",".join("?" * len(vals))})')
                args.extend(vals)
            else:
                clauses.append(f'({" OR ".join(["json_extract(global_meta,?) = ?"] * len(vals))})')
                for v in vals: args.extend([f'$.{field}', v])
        return (f' WHERE {" AND ".join(clauses)}', tuple(args)) if clauses else ('', ())

    _FTS_FIELDS = {RAG_DOC_ID: 'm.doc_id', RAG_KB_ID: 'm.kb_id', 'kb_id': 'm.kb_id',
                   'group': 'm."group"', 'parent': 'm.parent', 'number': 'm.number'}

    def _build_fts_filter_where(self, filters):
        clauses, args = [], []
        for field, vals in filters.items():
            if not vals: continue
            if not isinstance(vals, (list, set, tuple)): vals = [vals]
            if lookup := self._FTS_FIELDS.get(field):
                clauses.append(f'{lookup} IN ({",".join("?" * len(vals))})')
                args.extend(vals)
            else:
                clauses.append(f'({" OR ".join(["json_extract(m.global_meta,?) = ?"] * len(vals))})')
                for v in vals: args.extend([f'$.{field}', v])
        return (' AND '.join(clauses), tuple(args)) if clauses else ('', ())
