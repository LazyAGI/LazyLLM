from typing import Dict, List, Optional, Set, Union

from ..store_base import (LazyLLMStoreBase, StoreCapability, EMBED_PREFIX,
                          GLOBAL_META_KEY_PREFIX, DataType, GlobalMetadataDesc, BUILDIN_GLOBAL_META_DESC)

from lazyllm import LOG
from lazyllm.common import override
from lazyllm.thirdparty import chromadb

INSERT_BATCH_SIZE = 1000

DEFAULT_INDEX_CONFIG = {
    "hnsw": {
        "space": "cosine",
        "ef_construction": 200,
    }
}


class ChromadbStore(LazyLLMStoreBase, capability=StoreCapability.VECTOR):
    def __init__(self, dir: Optional[str] = None, host: Optional[str] = None, port: Optional[int] = None,
                 index_kwargs: Optional[Union[Dict, List]] = None, client_kwargs: Optional[Dict] = {},
                 embed_dims: Optional[Dict[str, int]] = {}, embed_datatypes: Optional[Dict[str, DataType]] = {},
                 global_metadata_desc: Optional[Dict[str, GlobalMetadataDesc]] = None, **kwargs) -> None:
        assert dir or (host and port), "dir or (host and port) must be provided"
        for embed_key, datatype in embed_datatypes.items():
            if datatype == DataType.SPARSE_FLOAT_VECTOR:
                raise ValueError("[Chromadb Store] Sparse float vector is not supported for chromadb")
        if len(embed_dims) > 1:
            LOG.warning("[Chromadb Store] Chromadb only support single embedding for each collection")
        self._index_kwargs = index_kwargs or DEFAULT_INDEX_CONFIG
        self._client_kwargs = client_kwargs
        self._dir = dir
        self._host = host
        self._port = port
        self._embed_dims = embed_dims
        self._embed_datatypes = embed_datatypes
        self._primary_key = 'uid'
        if global_metadata_desc:
            self._global_metadata_desc = global_metadata_desc | BUILDIN_GLOBAL_META_DESC
        else:
            self._global_metadata_desc = BUILDIN_GLOBAL_META_DESC
        for k, v in self._global_metadata_desc.items():
            if v.data_type not in [DataType.VARCHAR, DataType.INT32, DataType.FLOAT, DataType.BOOLEAN]:
                raise ValueError(f"[Chromadb Store] Unsupported data type {v.data_type} for global metadata {k}"
                                 " (only string, int, float, bool are supported)")

    @override
    def lazy_init(self):
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
            data_embeddings = data[0].get("embedding", {})
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
            return False

    def _serialize_data(self, data: List[dict], embed_key: str) -> List[dict]:
        res = {"ids": [], "embeddings": [], "metadatas": []}
        for d in data:
            res["ids"].append(d.get("uid"))
            res["embeddings"].append(d.get("embedding", {}).get(embed_key))
            res["metadatas"].append({})
            global_meta = d.get("global_meta", {})
            for k, v in global_meta.items():
                if k in self._global_metadata_desc:
                    res["metadatas"][-1][self._gen_global_meta_key(k)] = v
        return res

    @override
    def delete(self, collection_name: str, criteria: dict, **kwargs) -> bool:
        try:
            filters = self._construct_criteria(criteria)
            for embed_key in self._embed_datatypes.keys():
                collection = self._client.get_collection(name=self._gen_collection_name(collection_name, embed_key))
                collection.delete(**filters)
            return True
        except Exception as e:
            LOG.error(f"[Chromadb Store - delete] Failed to delete collection {collection_name}: {e}")
            return False

    @override
    def get(self, collection_name: str, criteria: dict, **kwargs) -> List[dict]:
        try:
            filters = self._construct_criteria(criteria)
            for embed_key in self._embed_datatypes.keys():
                collection = self._client.get_collection(name=self._gen_collection_name(collection_name, embed_key))
                res = collection.get(**filters)
            return res
        except Exception as e:
            LOG.error(f"[Chromadb Store - get] Failed to get collection {collection_name}: {e}")
            return []

    @override
    def search(self, collection_name: str, query: str, topk: int,
               filters: Optional[Dict[str, Union[str, int, List, Set]]] = None,
               embed_key: Optional[str] = None, **kwargs) -> List[dict]:
        pass

    def _construct_criteria(self, criteria: dict) -> dict:
        """ construct criteria for delete """
        res = {}
        if self._primary_key in criteria:
            res["ids"] = criteria[self._primary_key]
        else:
            res["where"] = {}
            for key, vaule in criteria.items():
                if isinstance(vaule, list):
                    res["where"][key] = {"$in": vaule}
                elif isinstance(vaule, str):
                    res["where"][key] = {"$eq": vaule}
                else:
                    raise ValueError(f'invalid criteria type: {type(vaule)}')
        return res

    def _gen_embed_key(self, k: str) -> str:
        return EMBED_PREFIX + k

    def _gen_global_meta_key(self, k: str) -> str:
        return GLOBAL_META_KEY_PREFIX + k

    def _gen_collection_name(self, collection_name: str, embed_key: str) -> str:
        return collection_name + '_' + embed_key
