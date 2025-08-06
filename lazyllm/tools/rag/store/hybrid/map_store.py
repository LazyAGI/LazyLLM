import json
import sqlite3
import os

from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Union, Set

from lazyllm import LOG
from lazyllm.common import override

from ..store_base import LazyLLMStoreBase, StoreCapability, DEFAULT_KB_ID
from ...global_metadata import RAG_DOC_ID, RAG_KB_ID


class MapStore(LazyLLMStoreBase):
    capability = StoreCapability.ALL
    need_embedding = True
    supports_index_registration = True

    def __init__(self, uri: Optional[str] = None, **kwargs):
        self._uri = uri  # filepath to SQLite .db for persistence

    @property
    def dir(self):
        path = os.path.dirname(self._uri)
        return path if path.endswith(os.sep) else path + os.sep

    def _ensure_table(self, cursor: sqlite3.Cursor, table: str):
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            uid TEXT PRIMARY KEY,
            doc_id TEXT,
            "group" TEXT,
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
        )
        """)
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_parent ON {table}(parent)")

    def _load_from_uri(self, collection_name: str, uri: str):
        conn = sqlite3.connect(uri)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (collection_name,))
        if not cursor.fetchone():
            LOG.warning(f"[MapStore] Table '{collection_name}' does not exist in SQLite DB {uri}, skipping.")
            conn.close()
            return

        res = []
        for row in cursor.execute(
            f"SELECT uid, doc_id, \"group\", content, meta, global_meta,"
            f" type, number, kb_id, excluded_embed_metadata_keys, excluded_llm_metadata_keys,"
            f" parent, answer, image_keys FROM {collection_name}"
        ):
            (uid, doc_id, group, content, meta_str, global_meta_str, type_, number,
             kb_id, excl_emb_str, excl_llm_str, parent, answer, image_keys_str) = row
            item = {'uid': uid, 'doc_id': doc_id, 'group': group, 'content': content,
                    'meta': json.loads(meta_str) if meta_str else {},
                    'global_meta': json.loads(global_meta_str) if global_meta_str else {},
                    'type': type_, 'number': number, 'kb_id': kb_id,
                    'excluded_embed_metadata_keys': json.loads(excl_emb_str) if excl_emb_str else [],
                    'excluded_llm_metadata_keys': json.loads(excl_llm_str) if excl_llm_str else [],
                    'parent': parent, 'answer': answer,
                    'image_keys': json.loads(image_keys_str) if image_keys_str else []}
            res.append(item)
        conn.close()
        for item in res:
            self._uid2data[item['uid']] = item
            self._collection2uids[collection_name].add(item['uid'])
            self._col_doc_uids[collection_name][item['doc_id']].add(item['uid'])
            self._col_kb_doc_uids[collection_name][item['kb_id']][item['doc_id']].add(item['uid'])
            self._col_parent_uids[collection_name][item['parent']].add(item['uid'])

    def _save_to_uri(self, collection_name: str, uri: str, data: List[dict]):
        conn = sqlite3.connect(uri)
        cursor = conn.cursor()
        self._ensure_table(cursor, collection_name)
        sql = f"INSERT OR REPLACE INTO {collection_name} (\
                uid, doc_id, \"group\", content,\
                meta, global_meta, type, number, kb_id,\
                excluded_embed_metadata_keys, excluded_llm_metadata_keys,\
                parent, answer, image_keys)\
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        params = []
        for item in data:
            params.append((item['uid'], item['doc_id'], item['group'], item['content'],
                           json.dumps(item['meta']), json.dumps(item['global_meta']),
                           item['type'], item['number'], item['kb_id'],
                           json.dumps(item['excluded_embed_metadata_keys']),
                           json.dumps(item['excluded_llm_metadata_keys']),
                           item['parent'], item['answer'], json.dumps(item['image_keys'])))
        LOG.info(f"executemany {sql} with {len(params)} params")
        cursor.executemany(sql, params)
        conn.commit()
        conn.close()

    @override
    def connect(self, collections: Optional[List[str]] = None, **kwargs):
        self._uid2data: Dict[str, dict] = {}
        self._collection2uids: Dict[str, Set[str]] = defaultdict(set)
        self._col_doc_uids: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self._col_kb_doc_uids: Dict[str, Dict[str, Dict[str, Set[str]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(set)))
        self._col_parent_uids: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        if self._uri:
            if not os.path.exists(self._uri):
                LOG.info(f"[MapStore] SQLite DB {self._uri} does not exist, creating...")
                db_path = Path(self._uri)
                db_path.parent.mkdir(parents=True, exist_ok=True)
                db_path.touch(exist_ok=True)
                self._uri = str(db_path)
            LOG.info(f"[MapStore] Loading data from {self._uri}")
            for collection_name in collections:
                self._load_from_uri(collection_name, self._uri)
        return

    @override
    def upsert(self, collection_name: str, data: List[dict]) -> None:
        try:
            for item in data:
                uid = item.get('uid')
                doc_id = item.get('doc_id')
                assert uid and doc_id, "[MapStore - upsert] uid and doc_id are required"
                self._uid2data[uid] = item
                self._collection2uids[collection_name].add(uid)
                self._col_kb_doc_uids[collection_name][item.get(RAG_KB_ID, DEFAULT_KB_ID)][doc_id].add(uid)
                self._col_doc_uids[collection_name][doc_id].add(uid)
                self._col_parent_uids[collection_name][item.get('parent')].add(uid)
            if self._uri:
                self._save_to_uri(collection_name, self._uri, data)
        except Exception as e:
            LOG.error(f"[MapStore - upsert] Error upserting data: {e}")
            return False
        return True

    @override
    def delete(self, collection_name: str, criteria: Optional[dict] = None, **kwargs) -> bool:
        try:
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
            if self._uri:
                conn = sqlite3.connect(self._uri)
                cursor = conn.cursor()
                sql_del = f"DELETE FROM {collection_name} WHERE uid = ?"
                cursor.executemany(sql_del, [(uid,) for uid in need_delete])
                conn.commit()
                conn.close()
            return True
        except Exception as e:
            LOG.error(f"[MapStore - delete] Error deleting data: {e}")
            return False

    @override
    def get(self, collection_name: str, criteria: Optional[dict] = None, **kwargs) -> List[dict]:
        uids = self._get_uids_by_criteria(collection_name, criteria)
        data = []
        for uid in uids:
            if uid in self._uid2data:
                data.append(self._uid2data[uid])
            else:
                LOG.warning(f"[MapStore - get] uid {uid} not found in data")
        return data

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
                raise ValueError(f"[MapStore - get] Invalid criteria: {criteria}")

    @override
    def search(self, collection_name: str, query: str, topk: int,
               filters: Optional[Dict[str, Union[str, int, List, Set]]] = None, **kwargs) -> List[dict]:
        # TODO(chenjiahao): implement search in map store, using default index to search data in map store
        raise NotImplementedError(
            "[MapStore - search] Not implemented, please use default index to search data in map store...")
