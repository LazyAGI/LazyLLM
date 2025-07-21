import json
import sqlite3

from collections import defaultdict
from typing import Dict, List, Optional, Union, Set

from lazyllm import LOG
from lazyllm.common import override

from ..store_base import LazyLLMStoreBase, StoreCapability, DEFAULT_KB_ID


class MapStore(LazyLLMStoreBase, capability=StoreCapability.ALL):
    def __init__(self, uri: Optional[str] = None, **kwargs):
        self._uri = uri  # filepath to SQLite .db for persistence

    def _ensure_table(self, cursor: sqlite3.Cursor, table: str):
        # Create table for a specific collection
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            uid TEXT PRIMARY KEY,
            doc_id TEXT,
            group TEXT,
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

    def _load_from_uri(self, collection_name: str, uri: str):
        conn = sqlite3.connect(uri)
        cursor = conn.cursor()
        self._ensure_table(cursor, collection_name)
        res = []
        for row in cursor.execute(
            f"SELECT uid, doc_id, group, content, meta, global_meta,"
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
            self._col_kb_doc_uids[collection_name][item['kb_id']][item['doc_id']].add(item['uid'])
            self._docid2uids[item['doc_id']].add(item['uid'])

    def _save_to_uri(self, collection_name: str, uri: str, data: List[dict]):
        conn = sqlite3.connect(uri)
        cursor = conn.cursor()
        self._ensure_table(cursor, collection_name)
        sql = f"INSERT OR REPLACE INTO {collection_name} (\
                uid, doc_id, group, content,\
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
    def lazy_init(self, collections: Optional[List[str]] = None, **kwargs):
        self._uid2data: Dict[str, dict] = {}
        self._collection2uids: Dict[str, Set[str]] = defaultdict(set)
        self._docid2uids: Dict[str, Set[str]] = defaultdict(set)
        self._col_kb_doc_uids: Dict[str, Dict[str, Dict[str, Set[str]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(set)))

        if self._uri:
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
                self._col_kb_doc_uids[collection_name][item.get('kb_id', DEFAULT_KB_ID)][doc_id].add(uid)
                self._docid2uids[doc_id].add(uid)
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
                self._collection2uids[collection_name].remove(uid)
                self._col_kb_doc_uids[collection_name][data.get('kb_id', DEFAULT_KB_ID)][data.get('doc_id')].remove(uid)
                self._docid2uids[data.get('doc_id')].remove(uid)
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
        return [self._uid2data[uid] for uid in uids]

    def _get_uids_by_criteria(self, collection_name: str, criteria: dict) -> List[str]:
        if not criteria:
            return list(self._collection2uids.get(collection_name, set()))
        else:
            uids = criteria.get('uid', [])
            kb_id = criteria.get('kb_id')
            doc_ids = criteria.get('doc_ids', [])
            if uids:
                return uids
            elif kb_id and doc_ids:
                return [uid for doc_id in doc_ids
                        for uid in self._col_kb_doc_uids.get(collection_name, {}).get(kb_id, {}).get(doc_id, ())]
            elif kb_id:
                doc_ids = self._col_kb_doc_uids.get(collection_name, {}).get(kb_id, {}).keys()
                return [uid for doc_id in doc_ids
                        for uid in self._col_kb_doc_uids.get(collection_name, {}).get(kb_id, {}).get(doc_id, ())]
            elif doc_ids:
                return [uid for doc_id in doc_ids for uid in self._docid2uids.get(doc_id, ())]
            else:
                raise ValueError(f"[MapStore - get] Invalid criteria: {criteria}")

    @override
    def search(self, collection_name: str, query: str, topk: int,
               filters: Optional[Dict[str, Union[str, int, List, Set]]] = None, **kwargs) -> List[dict]:
        raise NotImplementedError(
            "[MapStore - search] Not implemented, please use default index to search data in map store...")
