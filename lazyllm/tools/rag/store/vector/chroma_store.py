import os
import re
import traceback

from typing import Dict, List, Optional, Set, Union, Any
from collections import defaultdict
from urllib.parse import urlparse
from pathlib import Path

from ..store_base import (LazyLLMStoreBase, StoreCapability, GLOBAL_META_KEY_PREFIX)
from ...data_type import DataType
from ...global_metadata import GlobalMetadataDesc

from lazyllm import LOG
from lazyllm.common import override
from lazyllm.thirdparty import chromadb

INSERT_BATCH_SIZE = 1000

DEFAULT_INDEX_CONFIG = {
    'hnsw': {
        'space': 'cosine',
        'ef_construction': 200,
    }
}


class ChromadbStore(LazyLLMStoreBase):
    capability = StoreCapability.VECTOR
    need_embedding = True
    supports_index_registration = False

    def __init__(self, uri: Optional[str] = None, dir: Optional[str] = None,
                 index_kwargs: Optional[Union[Dict, List]] = None, client_kwargs: Optional[Dict] = None,
                 **kwargs) -> None:
        assert uri or (dir), "uri or dir must be provided"
        self._index_kwargs = index_kwargs or DEFAULT_INDEX_CONFIG
        self._client_kwargs = client_kwargs or {}
        if dir:
            self._dir = dir
        else:
            self._dir, self._host, self._port = self._parse_uri(uri)
        self._primary_key = 'uid'

    @property
    def dir(self):
        if not self._dir: return None
        p = Path(self._dir)
        p = p if p.suffix else (p / "chroma.sqlite3")
        return str(p.resolve(strict=False))

    def _parse_uri(self, uri: str):
        windows_drive = re.match(r"^[a-zA-Z]:[\\/]", uri or "")
        if ("://" not in uri) and (windows_drive or os.path.isabs(uri)):
            return os.path.abspath(uri), None, None

        p = urlparse(uri)

        if p.scheme == "":
            return os.path.abspath(uri), None, None

        if p.scheme == "file":
            path = p.path
            if os.name == "nt" and path.startswith("/") and re.match(r"^/[a-zA-Z]:", path):
                path = path.lstrip("/")  # file:///C:/... -> C:/...
            return os.path.abspath(path), None, None

        scheme = p.scheme
        if scheme.startswith("chroma+"):
            scheme = scheme.split("+", 1)[1]  # http or https

        if scheme in ("http", "https"):
            host = p.hostname or "127.0.0.1"
            port = p.port or (443 if scheme == "https" else 80)
            return None, host, port

        raise ValueError(f"Unsupported URI scheme in '{uri}'. "
                         "Use file:///path or plain path for local; http(s)://host:port for remote.")

    @override
    def connect(self, embed_dims: Optional[Dict[str, int]] = None,
                embed_datatypes: Optional[Dict[str, DataType]] = None,
                global_metadata_desc: Optional[Dict[str, GlobalMetadataDesc]] = None, **kwargs):
        self._global_metadata_desc = global_metadata_desc or {}
        self._embed_dims = embed_dims or {}
        self._embed_datatypes = embed_datatypes or {}
        for k, v in self._global_metadata_desc.items():
            if v.data_type not in [DataType.VARCHAR, DataType.INT32, DataType.FLOAT, DataType.BOOLEAN]:
                raise ValueError(f"[Chromadb Store] Unsupported data type {v.data_type} for global metadata {k}"
                                 " (only string, int, float, bool are supported)")
        for k, v in self._embed_datatypes.items():
            if v not in [DataType.FLOAT_VECTOR, DataType.SPARSE_FLOAT_VECTOR]:
                raise ValueError(f"[Chromadb Store] Unsupported data type {v} for embed key {k}"
                                 " (only float vector and sparse float vector are supported)")
        if self._dir:
            self._client = chromadb.PersistentClient(path=self._dir, **self._client_kwargs)
            LOG.success(f"Initialzed chromadb in path: {self._dir}")
        else:
            self._client = chromadb.HttpClient(host=self._host, port=self._port, **self._client_kwargs)
            LOG.success(f"Initialzed chromadb in host: {self._host}, port: {self._port}")

    @override
    def upsert(self, collection_name: str, data: List[dict]) -> bool:
        try:
            # NOTE chromadb only support single embedding for each collection
            if not data: return
            data_embeddings = data[0].get('embedding', {})
            if not data_embeddings: return
            embed_keys = list(data_embeddings.keys())
            for embed_key in embed_keys:
                if embed_key not in self._embed_datatypes:
                    raise ValueError(f"Embed key {embed_key} not found in embed_datatypes")
                collection = self._client.get_or_create_collection(
                    name=self._gen_collection_name(collection_name, embed_key), configuration=self._index_kwargs)
                for i in range(0, len(data), INSERT_BATCH_SIZE):
                    collection.upsert(**self._serialize_data(data[i: i + INSERT_BATCH_SIZE], embed_key))
            return True
        except Exception as e:
            LOG.error(f"[Chromadb Store - upsert] Failed to create collection {collection_name}: {e}")
            LOG.error(traceback.format_exc())
            return False

    def _serialize_data(self, data: List[dict], embed_key: str) -> List[dict]:
        res = {'ids': [], 'embeddings': [], 'metadatas': []}
        for d in data:
            res['ids'].append(d.get('uid'))
            res['embeddings'].append(d.get('embedding', {}).get(embed_key))
            res['metadatas'].append({self._gen_global_meta_key(k): v for k, v in d.get('global_meta', {}).items()
                                     if k in self._global_metadata_desc})
        return res

    @override
    def delete(self, collection_name: str, criteria: Optional[dict] = None, **kwargs) -> bool:
        try:
            if not criteria:
                for embed_key in self._embed_datatypes.keys():
                    try:
                        self._client.delete_collection(name=self._gen_collection_name(collection_name, embed_key))
                    except Exception:
                        continue
                return True
            else:
                filters = self._construct_criteria(criteria)
                for embed_key in self._embed_datatypes.keys():
                    collection = self._client.get_collection(name=self._gen_collection_name(collection_name, embed_key))
                    collection.delete(**filters)
                return True
        except Exception as e:
            LOG.error(f"[Chromadb Store - delete] Failed to delete collection {collection_name}: {e}")
            LOG.error(traceback.format_exc())
            return False

    @override
    def get(self, collection_name: str, criteria: Optional[dict] = None, **kwargs) -> List[dict]:
        try:
            filters = self._construct_criteria(criteria) if criteria else {}
            all_data = []
            for key in self._embed_datatypes:
                try:
                    coll = self._client.get_collection(
                        name=self._gen_collection_name(collection_name, key)
                    )
                    data = coll.get(include=['metadatas', 'embeddings'], **filters)
                    all_data.append((key, data))
                except Exception:
                    continue

            res: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
                'uid': None, 'global_meta': {}, 'embedding': {}})
            for embed_key, data in all_data:
                ids = data['ids']
                metas = data['metadatas']
                embs = data['embeddings']

                for uid, meta, emb in zip(ids, metas, embs):
                    entry = res[uid]
                    entry['uid'] = uid
                    if not entry['global_meta']:
                        entry['global_meta'] = {
                            k[len(GLOBAL_META_KEY_PREFIX):]: v
                            for k, v in meta.items()
                        }
                    entry['embedding'][embed_key] = list(emb)
            return list(res.values())
        except Exception as e:
            LOG.error(f"[ChromadbStore - get] task fail: {e}")
            LOG.error(traceback.format_exc())

    @override
    def search(self, collection_name: str, query_embedding: List[float], embed_key: str, topk: Optional[int] = 10,
               filters: Optional[Dict[str, Union[str, int, List, Set]]] = None,
               **kwargs) -> List[dict]:
        try:
            collection = self._client.get_collection(name=self._gen_collection_name(collection_name, embed_key))

            filters = self._construct_filter_expr(filters) if filters else {}
            query_results = collection.query(query_embeddings=[query_embedding], n_results=topk, **filters)
            res = []
            for i, r_list in enumerate(query_results['ids']):
                for j, uid in enumerate(r_list):
                    dis = query_results['distances'][i][j]
                    res.append({'uid': uid, 'score': 1 - dis})
            return res
        except Exception as e:
            LOG.error(f"[ChromadbStore - search] task fail: {e}")
            LOG.error(traceback.format_exc())

    def _construct_criteria(self, criteria: dict) -> dict:
        res = {}
        if self._primary_key in criteria:
            res['ids'] = criteria[self._primary_key]
        else:
            res['where'] = {}
            for key, vaule in criteria.items():
                if key not in self._global_metadata_desc:
                    continue
                field_key = self._gen_global_meta_key(key)
                if isinstance(vaule, list):
                    res['where'][field_key] = {'$in': vaule}
                elif isinstance(vaule, str):
                    res['where'][field_key] = {'$eq': vaule}
                else:
                    raise ValueError(f'invalid criteria type: {type(vaule)}')
        return res

    def _construct_filter_expr(self, filters: Dict[str, Union[str, int, List, Set]]) -> str:
        ret = {}
        for name, candidates in filters.items():
            desc = self._global_metadata_desc.get(name)
            if not desc:
                raise ValueError(f'cannot find desc of field [{name}]')
            key = self._gen_global_meta_key(name)
            if isinstance(candidates, str):
                candidates = [candidates]
            elif (not isinstance(candidates, List)) and (not isinstance(candidates, Set)):
                candidates = list(candidates)
            ret[key] = {'$in': candidates}
        return {'where': ret}

    def _gen_global_meta_key(self, k: str) -> str:
        return GLOBAL_META_KEY_PREFIX + k

    def _gen_collection_name(self, collection_name: str, embed_key: str) -> str:
        return collection_name + '_' + embed_key + "_embed"
