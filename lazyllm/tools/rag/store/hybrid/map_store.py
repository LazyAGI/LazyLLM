import json
import sqlite3
import os
import threading

from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Union, Set

from lazyllm import LOG
from lazyllm.common import override

from ..store_base import LazyLLMStoreBase, StoreCapability, DEFAULT_KB_ID
from ...global_metadata import RAG_DOC_ID, RAG_KB_ID
from ...doc_node import DocNode
from ...component.bm25 import BM25


class MapStore(LazyLLMStoreBase):
    capability = StoreCapability.ALL
    need_embedding = True
    supports_index_registration = True

    def __init__(self, uri: Optional[str] = None, **kwargs):
        self._uri = uri  # filepath to SQLite .db for persistence
        self._sqlite_first = bool(uri)
        self._conn = None

    def _open_conn(self):
        if not self._uri: return None
        if self._conn: return self._conn

        conn = sqlite3.connect(self._uri, timeout=5.0, check_same_thread=False)
        cur = conn.cursor()
        cur.execute('PRAGMA journal_mode = WAL;')
        cur.execute('PRAGMA synchronous = NORMAL;')
        cur.execute('PRAGMA busy_timeout = 5000;')
        conn.commit()
        self._conn = conn
        return conn

    @property
    def dir(self):
        if not self._uri:
            return ''
        path = os.path.dirname(self._uri)
        return path if path.endswith(os.sep) else path + os.sep

    def _ensure_table(self, cursor: sqlite3.Cursor, table: str):
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {table} (
            uid TEXT PRIMARY KEY,
            doc_id TEXT,
            'group' TEXT,
            content TEXT,
            meta TEXT,
            global_meta TEXT,
            type INTEGER,
            number INTEGER,
            kb_id TEXT,
            excluded_embed_metadata_keys TEXT,
            excluded_llm_metadata_keys TEXT,
            parent TEXT,
            answer TEXT,
            image_keys TEXT
        )''')
        cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{table}_parent ON {table}(parent)')
        cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{table}_docid ON {table}(doc_id)')
        cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{table}_kbid ON {table}(kb_id)')

    def _save_to_uri(self, collection_name: str, data: List[dict]):
        with self._lock:
            conn = self._open_conn()
            cur = conn.cursor()
            self._ensure_table(cur, collection_name)
            sql = f'''INSERT OR REPLACE INTO {collection_name} (
                    uid, doc_id, \'group\', content,
                    meta, global_meta, type, number, kb_id,\
                    excluded_embed_metadata_keys, excluded_llm_metadata_keys,\
                    parent, answer, image_keys)\
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
            params = []
            for item in data:
                params.append(self._serialize_data(item))
            cur.executemany(sql, params)
            conn.commit()
            affected_rows = cur.rowcount
            LOG.debug(f'[MapStore - _save_to_uri] Inserted {affected_rows} rows into {collection_name}')

    @override
    def connect(self, collections: Optional[List[str]] = None, **kwargs):
        self._uid2data: Dict[str, dict] = {}
        self._collection2uids: Dict[str, Set[str]] = defaultdict(set)
        self._col_doc_uids: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self._col_kb_doc_uids: Dict[str, Dict[str, Dict[str, Set[str]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(set)))
        self._col_parent_uids: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self._lock = threading.Lock()
        self._bm25_cache: Dict[tuple, BM25] = {}
        self._bm25_cache_lock = threading.Lock()
        if self._uri:
            db_path = Path(self._uri)
            if not db_path.exists():
                LOG.info(f'[MapStore] SQLite DB {self._uri} does not exist, creating...')
                db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                conn = self._open_conn()
                if collections:
                    cur = conn.cursor()
                    for c in collections:
                        self._ensure_table(cur, c)
                    conn.commit()
        return

    @override
    def upsert(self, collection_name: str, data: List[dict]) -> bool:
        try:
            if self._sqlite_first:
                self._save_to_uri(collection_name, data)
            for item in data:
                uid = item.get('uid')
                doc_id = item.get('doc_id')
                kb_id = item.get(RAG_KB_ID, DEFAULT_KB_ID)
                item['kb_id'] = kb_id
                assert uid and doc_id, '[MapStore - upsert] uid and doc_id are required'
                self._uid2data[uid] = item
                self._collection2uids[collection_name].add(uid)
                self._col_kb_doc_uids[collection_name][kb_id][doc_id].add(uid)
                self._col_doc_uids[collection_name][doc_id].add(uid)
                self._col_parent_uids[collection_name][item.get('parent')].add(uid)
            self._clear_bm25_cache(collection_name)
            return True
        except Exception as e:
            LOG.error(f'[MapStore - upsert] Error upserting data: {e}')
            return False

    @override
    def delete(self, collection_name: str, criteria: Optional[dict] = None, **kwargs) -> bool:
        try:
            if self._sqlite_first:
                with self._lock:
                    conn = self._open_conn()
                    cur = conn.cursor()
                    where, args = self._build_where(collection_name, criteria)
                    cur.execute(f'DELETE FROM {collection_name} {where}', args)
                    conn.commit()
                    affected_rows = cur.rowcount
                    LOG.debug(f'[MapStore - delete] Deleted {affected_rows} rows from {collection_name}')
                    uids = self._get_uids_by_criteria(collection_name, criteria)
                    for uid in uids:
                        data = self._uid2data.pop(uid, None)
                        if not data: continue
                        kb_id = data.get(RAG_KB_ID, DEFAULT_KB_ID)
                        doc_id = data.get('doc_id'); parent = data.get('parent')
                        self._collection2uids[collection_name].discard(uid)
                        self._col_kb_doc_uids[collection_name][kb_id][doc_id].discard(uid)
                        self._col_doc_uids[collection_name][doc_id].discard(uid)
                        self._col_parent_uids[collection_name][parent].discard(uid)
                self._clear_bm25_cache(collection_name)
                return True
            else:
                need_delete = self._get_uids_by_criteria(collection_name, criteria)
                if not need_delete:
                    return False
                for uid in need_delete:
                    data = self._uid2data.pop(uid, None)
                    if not data:
                        continue
                    kb_id = data.get(RAG_KB_ID, DEFAULT_KB_ID)
                    doc_id = data.get('doc_id')
                    parent = data.get('parent')
                    self._collection2uids[collection_name].remove(uid)
                    self._col_kb_doc_uids[collection_name][kb_id][doc_id].remove(uid)
                    self._col_doc_uids[collection_name][doc_id].remove(uid)
                    self._col_parent_uids[collection_name][parent].remove(uid)
            self._clear_bm25_cache(collection_name)
            return True
        except Exception as e:
            LOG.error(f'[MapStore - delete] Error deleting data: {e}')
            return False

    @override
    def get(self, collection_name: str, criteria: Optional[dict] = None, **kwargs) -> List[dict]:
        if self._sqlite_first:
            with self._lock:
                conn = self._open_conn()
                cur = conn.cursor()
                self._ensure_table(cur, collection_name)
                where, args = self._build_where(collection_name, criteria)
                cur.execute(f'''SELECT uid, doc_id, "group", content, meta, global_meta, type, number, kb_id,
                                excluded_embed_metadata_keys, excluded_llm_metadata_keys, parent, answer, image_keys
                                FROM {collection_name}{where}''', args)
                rows = cur.fetchall()
                res = []
                for r in rows:
                    item = self._deserialize_data(r)
                    res.append(item)
                    self._uid2data[item['uid']] = item
                    self._collection2uids[collection_name].add(item['uid'])
                    self._col_doc_uids[collection_name][item['doc_id']].add(item['uid'])
                    self._col_kb_doc_uids[collection_name][item['kb_id']][item['doc_id']].add(item['uid'])
                    self._col_parent_uids[collection_name][item['parent']].add(item['uid'])
            return res
        else:
            uids = self._get_uids_by_criteria(collection_name, criteria)
            return [self._uid2data[uid] for uid in uids if uid in self._uid2data]

    def _build_where(self, collection_name: str, criteria: dict):
        if not criteria:
            return '', ()
        clauses, args = [], []
        uids = criteria.get('uid')
        kb_id = criteria.get(RAG_KB_ID)
        doc_ids = criteria.get(RAG_DOC_ID)
        parents = criteria.get('parent')
        if uids:
            placeholders = ','.join('?' for _ in uids)
            clauses.append(f'uid IN ({placeholders})')
            args.extend(uids)
        if kb_id:
            clauses.append('kb_id = ?'); args.append(kb_id)
        if doc_ids:
            placeholders = ','.join('?' for _ in doc_ids)
            clauses.append(f'doc_id IN ({placeholders})')
            args.extend(doc_ids)
        if parents:
            placeholders = ','.join('?' for _ in parents)
            clauses.append(f'parent IN ({placeholders})')
            args.extend(parents)
        where = (' WHERE ' + ' AND '.join(clauses)) if clauses else ''
        return where, tuple(args)

    def _get_uids_by_criteria(self, collection_name: str, criteria: dict) -> List[str]:
        if not criteria:
            return list(self._collection2uids.get(collection_name, set()))
        else:
            uids = criteria.get('uid', [])
            kb_id = criteria.get(RAG_KB_ID)
            doc_ids = criteria.get(RAG_DOC_ID, [])
            parents = criteria.get('parent', [])
            if uids:
                return [uid for uid in uids if uid in self._collection2uids.get(collection_name, set())]
            elif kb_id and doc_ids:
                return [uid for doc_id in doc_ids
                        for uid in self._col_kb_doc_uids.get(collection_name, {}).get(kb_id, {}).get(doc_id, ())]
            elif kb_id:
                doc_ids = self._col_kb_doc_uids.get(collection_name, {}).get(kb_id, {}).keys()
                return [uid for doc_id in doc_ids
                        for uid in self._col_kb_doc_uids.get(collection_name, {}).get(kb_id, {}).get(doc_id, ())]
            elif doc_ids:
                return [uid for doc_id in doc_ids for uid in self._col_doc_uids.get(collection_name, {}).get(doc_id, ())]
            elif parents:
                return [uid for parent in parents for uid in
                        self._col_parent_uids.get(collection_name, {}).get(parent, ())]
            else:
                raise ValueError(f'[MapStore - get] Invalid criteria: {criteria}')

    def _serialize_data(self, item: dict) -> tuple:
        kb_id = item.get(RAG_KB_ID, DEFAULT_KB_ID)
        return (item['uid'], item['doc_id'], item['group'], item.get('content', ''),
                json.dumps(item.get('meta', {})), json.dumps(item.get('global_meta', {})),
                item['type'], item['number'], kb_id,
                json.dumps(item.get('excluded_embed_metadata_keys', [])),
                json.dumps(item.get('excluded_llm_metadata_keys', [])),
                item.get('parent'), item.get('answer', ''), json.dumps(item.get('image_keys', [])))

    def _deserialize_data(self, row: tuple) -> dict:
        (uid, doc_id, group, content, meta_str, global_meta_str, type_, number, kb_id,
         excl_emb_str, excl_llm_str, parent, answer, image_keys_str) = row
        return {
            'uid': uid, 'doc_id': doc_id, 'group': group, 'content': content,
            'meta': json.loads(meta_str) if meta_str else {},
            'global_meta': json.loads(global_meta_str) if global_meta_str else {},
            'type': type_, 'number': number, 'kb_id': kb_id, 'parent': parent, 'answer': answer,
            'excluded_embed_metadata_keys': json.loads(excl_emb_str) if excl_emb_str else [],
            'excluded_llm_metadata_keys': json.loads(excl_llm_str) if excl_llm_str else [],
            'image_keys': json.loads(image_keys_str) if image_keys_str else []
        }

    @override
    def search(self, collection_name: str, query: Optional[str] = None,
               query_embedding: Optional[Union[dict, List[float]]] = None, topk: int = 10,
               filters: Optional[Dict[str, Union[str, int, List, Set]]] = None,
               embed_key: Optional[str] = None, **kwargs) -> List[dict]:
        if query_embedding is not None:
            raise ValueError(f'MapStore only supports BM25 text search, query_embedding is not supported')
        if embed_key is not None:
            raise ValueError(f'MapStore only supports BM25 text search, embed_key is not supported')
        segments = self.get(collection_name=collection_name, criteria=None)
        segments = self._apply_filters(segments, filters)
        if not query:
            return []
        language = kwargs.get('language', 'en')
        return self._search_by_text(collection_name, segments, query, topk, language, filters)

    def _apply_filters(self, segments: List[dict],
                       filters: Optional[Dict[str, Union[str, int, List, Set]]]) -> List[dict]:
        if not filters:
            return segments
        filtered = []
        for seg in segments:
            global_meta = seg.get('global_meta', {})
            for name, candidates in filters.items():
                value = global_meta.get(name)
                if isinstance(candidates, (list, set)):
                    if value not in candidates:
                        break
                else:
                    if value != candidates:
                        break
            else:
                filtered.append(seg)
        return filtered

    def _search_by_text(self, collection_name: str, segments: List[dict], query: str, topk: int,
                        language: str, filters: Optional[Dict[str, Union[str, int, List, Set]]]) -> List[dict]:
        nodes = []
        uid2segment = {}
        for seg in segments:
            content = seg.get('content', '')
            if not content:
                continue
            uid = seg.get('uid')
            node = DocNode(uid=uid, content=content, metadata=seg.get('meta', {}),
                           global_metadata=seg.get('global_meta', {}))
            nodes.append(node)
            if uid:
                uid2segment[uid] = seg
        if not nodes:
            return []
        topk = len(nodes) if topk is None else min(topk, len(nodes))
        filter_key = self._make_filter_key(filters)
        cache_key = (collection_name, language, filter_key, topk)
        bm25 = self._get_bm25(cache_key, nodes, language, topk)
        results = bm25.retrieve(query)
        scored = []
        for node, score in results:
            seg = uid2segment.get(node.uid)
            if not seg:
                continue
            item = dict(seg)
            item['score'] = float(score)
            scored.append(item)
        return scored

    def _get_bm25(self, cache_key: tuple, nodes: List[DocNode], language: str, topk: int) -> BM25:
        with self._bm25_cache_lock:
            cached = self._bm25_cache.get(cache_key)
            if cached:
                return cached
            bm25 = BM25(nodes, language=language, topk=topk)
            self._bm25_cache[cache_key] = bm25
            return bm25

    def _clear_bm25_cache(self, collection_name: str) -> None:
        with self._bm25_cache_lock:
            for key in list(self._bm25_cache.keys()):
                if key[0] == collection_name:
                    del self._bm25_cache[key]

    def _make_filter_key(self, filters: Optional[Dict[str, Union[str, int, List, Set]]]) -> Optional[tuple]:
        if not filters:
            return None
        items = []
        for key in sorted(filters.keys()):
            value = filters[key]
            if isinstance(value, (list, set)):
                value = tuple(sorted(value, key=repr))
            items.append((key, value))
        return tuple(items)
