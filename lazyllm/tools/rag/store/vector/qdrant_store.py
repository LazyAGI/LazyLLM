import traceback
import uuid

from typing import Dict, List, Optional, Union

from lazyllm import LOG
from lazyllm.thirdparty import qdrant_client
from lazyllm.common import override

from ..store_base import (LazyLLMStoreBase, StoreCapability, GLOBAL_META_KEY_PREFIX,
                          EMBED_PREFIX, EmbedResolveMixin, is_empty_embedding_value)
from ...data_type import DataType


class QdrantStore(EmbedResolveMixin, LazyLLMStoreBase):
    capability = StoreCapability.VECTOR
    need_embedding = True
    supports_index_registration = False

    def __init__(self, uri: str, api_key: Optional[str] = None,
                 index_kwargs: Optional[Union[Dict, List]] = None,
                 client_kwargs: Optional[Dict] = None):
        if not uri:
            raise ValueError('[Qdrant Store] uri must be provided')
        self._uri = uri
        self._api_key = api_key
        self._index_kwargs = index_kwargs
        self._client_kwargs = client_kwargs or {}
        self._primary_key = 'uid'

    @classmethod
    def rebuild(cls, uri, api_key, index_kwargs, client_kwargs):
        return cls(uri=uri, api_key=api_key, index_kwargs=index_kwargs, client_kwargs=client_kwargs)

    @override
    def __reduce__(self):
        return self.rebuild, (self._uri, self._api_key, self._index_kwargs, self._client_kwargs)

    @property
    def dir(self):
        return None

    @override
    def connect(self, embed_dims=None, embed_datatypes=None, embed=None,
                global_metadata_desc=None, collections=None, **kwargs):
        self._embed_dims = embed_dims or {}
        self._embed_datatypes = embed_datatypes or {}
        self._embed = embed or {}
        self._global_metadata_desc = global_metadata_desc or {}
        self._validate_embed_specs()
        self._validate_global_metadata()
        self._index_config = self._parse_index_kwargs()
        kw = dict(self._client_kwargs)
        if self._api_key and 'api_key' not in kw:
            kw['api_key'] = self._api_key
        self._client = qdrant_client.QdrantClient(url=self._uri, **kw)
        self._ingest_schema(collections or [])
        LOG.info(f'[Qdrant Store] init success in {self._uri}')

    def _validate_embed_specs(self):
        for k, v in self._embed_datatypes.items():
            if v not in (DataType.FLOAT_VECTOR, DataType.SPARSE_FLOAT_VECTOR):
                raise ValueError(f'[Qdrant Store] Unsupported data type {v} for embed key {k}')
            if v == DataType.FLOAT_VECTOR and k not in self._embed_dims:
                if not self._embed or k not in self._embed:
                    raise ValueError(f'[Qdrant Store] embed_dims for FLOAT_VECTOR key {k!r} is required')
                sample = self._embed[k]('a')
                if isinstance(sample, dict):
                    raise ValueError(f'[Qdrant Store] embed callable for {k!r} produced a sparse vector'
                                     f' but embed_datatypes says FLOAT_VECTOR')
                self._embed_dims[k] = len(sample)

    def _validate_global_metadata(self):
        for k, v in self._global_metadata_desc.items():
            if v.data_type not in (DataType.VARCHAR, DataType.INT32, DataType.INT64,
                                   DataType.FLOAT, DataType.BOOLEAN, DataType.ARRAY):
                raise ValueError(f'[Qdrant Store] Unsupported data type {v.data_type} for global metadata {k}')
            if v.data_type == DataType.ARRAY and v.element_type not in (
                    DataType.VARCHAR, DataType.INT32, DataType.INT64,
                    DataType.FLOAT, DataType.BOOLEAN, DataType.ARRAY):
                raise ValueError(
                    f'[Qdrant Store] Unsupported array element type '
                    f'{v.element_type} for global metadata {k}')

    def _parse_index_kwargs(self):
        config = {}
        if not self._index_kwargs:
            return config
        entries = self._index_kwargs if isinstance(self._index_kwargs, list) else [self._index_kwargs]
        for item in entries:
            key = item.get('embed_key')
            if not key:
                raise ValueError(f'cannot find `embed_key` in `index_kwargs` of `{item}`')
            if key in config:
                raise ValueError(f'duplicate embed_key {key} in index_kwargs')
            distance = (item.get('distance') or '').upper()
            if distance:
                try:
                    qdrant_client.models.Distance[distance]
                except KeyError:
                    raise ValueError(f'[Qdrant Store] Unsupported distance: {distance}')
            config[key] = distance or None
        return config

    def _ingest_schema(self, collections):
        for collection_name in collections:
            try:
                if not self._client.collection_exists(collection_name):
                    continue
                params = self._client.get_collection(collection_name).config.params
                for name, spec in (params.vectors if isinstance(params.vectors, dict) else {}).items():
                    if not name.startswith(EMBED_PREFIX):
                        continue
                    key = name[len(EMBED_PREFIX):]
                    if key not in self._embed_datatypes:
                        self._embed_dims[key] = int(spec.size)
                        self._embed_datatypes[key] = DataType.FLOAT_VECTOR
                    if self._index_config.get(key) is None and spec.distance is not None:
                        self._index_config[key] = spec.distance.name
                for name in (params.sparse_vectors or {}):
                    if not name.startswith(EMBED_PREFIX):
                        continue
                    key = name[len(EMBED_PREFIX):]
                    if key not in self._embed_datatypes:
                        self._embed_datatypes[key] = DataType.SPARSE_FLOAT_VECTOR
            except Exception as e:
                LOG.warning(f'[Qdrant Store] Could not read embed dims from schema: {e}')

    def _distance_for(self, embed_key):
        return self._index_config.get(embed_key) or {
            DataType.FLOAT_VECTOR: 'COSINE',
            DataType.SPARSE_FLOAT_VECTOR: 'DOT',
        }[self._embed_datatypes[embed_key]]

    @override
    def upsert(self, collection_name, data):
        try:
            if not data:
                return True
            collection_exists = self._client.collection_exists(collection_name)
            required_keys = (self._required_keys_from_schema(collection_name) if collection_exists
                             else self._required_keys_from_data(data))
            if not required_keys:
                return True
            valid_data = self._drop_invalid(collection_name, data, required_keys)
            if not valid_data:
                return True
            if not collection_exists:
                self._create_collection(collection_name, required_keys)
            self._batch_upsert(collection_name, valid_data, required_keys)
            return True
        except Exception:
            LOG.error(f'[Qdrant Store - upsert] error: {traceback.format_exc()}')
            return False

    def _required_keys_from_schema(self, collection_name):
        params = self._client.get_collection(collection_name).config.params
        keys = set()
        for name in (params.vectors if isinstance(params.vectors, dict) else {}):
            if name.startswith(EMBED_PREFIX):
                keys.add(name[len(EMBED_PREFIX):])
        for name in (params.sparse_vectors or {}):
            if name.startswith(EMBED_PREFIX):
                keys.add(name[len(EMBED_PREFIX):])
        return keys

    def _required_keys_from_data(self, data):
        keys = set()
        for row in data:
            emb = row.get('embedding')
            if isinstance(emb, dict):
                keys.update(k for k, v in emb.items() if not is_empty_embedding_value(v))
        return keys

    def _drop_invalid(self, collection_name, data, required_keys):
        def _valid(d):
            emb = d.get('embedding')
            return (bool(emb) and isinstance(emb, dict)
                    and all(not is_empty_embedding_value(emb.get(k)) for k in required_keys))

        valid = [d for d in data if _valid(d)]
        if len(valid) != len(data):
            LOG.warning(
                f'[Qdrant Store - upsert] Dropping {len(data) - len(valid)} rows with'
                f' missing/empty embedding for collection {collection_name},'
                f' required embeddings: {sorted(required_keys)}.')
        return valid

    def _create_collection(self, collection_name, required_keys):
        self._resolve_missing_embed_specs(required_keys)
        vectors_config = {EMBED_PREFIX + k: qdrant_client.models.VectorParams(
            size=self._embed_dims[k],
            distance=qdrant_client.models.Distance[self._distance_for(k)])
            for k in required_keys if self._embed_datatypes[k] == DataType.FLOAT_VECTOR}
        sparse_vectors_config = {EMBED_PREFIX + k: qdrant_client.models.SparseVectorParams()
                                 for k in required_keys
                                 if self._embed_datatypes[k] == DataType.SPARSE_FLOAT_VECTOR}
        try:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config or None)
        except Exception:
            if not self._client.collection_exists(collection_name):
                raise

    def _batch_upsert(self, collection_name, data, required_keys):
        for i in range(0, len(data), 500):
            self._client.upsert(
                collection_name=collection_name,
                points=[self._to_point(d, required_keys) for d in data[i:i + 500]])

    def _uid_uuids(self, criteria):
        if not criteria or self._primary_key not in criteria:
            return None
        ids = criteria[self._primary_key]
        if isinstance(ids, str):
            ids = [ids]
        return [self._to_uuid(uid) for uid in ids]

    @override
    def delete(self, collection_name, criteria=None, **kwargs):
        try:
            if not self._client.collection_exists(collection_name):
                return True
            if not criteria:
                self._client.delete_collection(collection_name)
                return True
            selector = self._uid_uuids(criteria)
            if selector is None:
                selector = self._build_filter(criteria)
            if selector:
                self._client.delete(collection_name=collection_name, points_selector=selector)
            return True
        except Exception:
            LOG.error(f'[Qdrant Store - delete] error: {traceback.format_exc()}')
            return False

    @override
    def get(self, collection_name, criteria=None, **kwargs):
        try:
            if not self._client.collection_exists(collection_name):
                return []
            uuids = self._uid_uuids(criteria)
            if uuids is not None:
                records = self._client.retrieve(
                    collection_name,
                    ids=uuids,
                    with_payload=True,
                    with_vectors=True)
                return [self._from_point(r) for r in records]
            qf = self._build_filter(criteria)
            records, offset = [], None
            while True:
                batch, offset = self._client.scroll(
                    collection_name, scroll_filter=qf, limit=1000, offset=offset,
                    with_payload=True, with_vectors=True)
                records.extend(batch)
                if offset is None:
                    break
            return [self._from_point(r) for r in records]
        except Exception:
            LOG.error(f'[Qdrant Store - get] error: {traceback.format_exc()}')
            return []

    @override
    def collection_exists(self, collection_name):
        try:
            return self._client.collection_exists(collection_name)
        except Exception:
            LOG.warning(f'[Qdrant Store - collection_exists] error checking {collection_name}')
            return False

    @override
    def search(self, collection_name, query_embedding=None, topk=10, filters=None, embed_key=None, **kwargs):
        if not embed_key or embed_key not in self._embed_datatypes:
            raise ValueError(f'[Qdrant Store - search] Not supported or None `embed_key`: {embed_key}')
        if not self._client.collection_exists(collection_name):
            return []
        if topk <= 0:
            return []
        if query_embedding is None:
            raise ValueError('[Qdrant Store - search] query_embedding must be provided')

        if self._embed_datatypes[embed_key] == DataType.SPARSE_FLOAT_VECTOR:
            query_vector = self._to_sparse_vector(query_embedding)
        else:
            query_vector = list(query_embedding)

        res = self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=EMBED_PREFIX + embed_key,
            query_filter=self._build_filter(filters),
            limit=topk,
            with_payload=True)

        flip = self._distance_for(embed_key) in ('EUCLID', 'MANHATTAN')
        return [{'uid': (p.payload or {}).get(self._primary_key),
                 'score': -float(p.score) if flip else float(p.score)}
                for p in res.points
                if (p.payload or {}).get(self._primary_key)]

    def _to_sparse_vector(self, emb):
        return qdrant_client.models.SparseVector(
            indices=[int(i) for i in emb.keys()], values=[float(v) for v in emb.values()])

    def _to_point(self, d, required_keys):
        vector = {}
        for key in required_keys:
            emb = d.get('embedding', {}).get(key)
            if emb is not None:
                name = EMBED_PREFIX + key
                if self._embed_datatypes.get(key) == DataType.SPARSE_FLOAT_VECTOR:
                    vector[name] = self._to_sparse_vector(emb)
                else:
                    vector[name] = emb
        payload = {self._primary_key: d.get(self._primary_key, '')}
        for name, desc in self._global_metadata_desc.items():
            v = d.get('global_meta', {}).get(name, desc.default_value)
            if v is not None:
                payload[GLOBAL_META_KEY_PREFIX + name] = v
        return qdrant_client.models.PointStruct(
            id=self._to_uuid(d.get(self._primary_key, '')), vector=vector, payload=payload)

    def _from_point(self, record):
        res = {'uid': (record.payload or {}).get(self._primary_key, str(record.id)), 'embedding': {}}
        for name, vec in (record.vector or {}).items():
            if name.startswith(EMBED_PREFIX):
                key = name[len(EMBED_PREFIX):]
                if isinstance(vec, (list, tuple)):
                    res['embedding'][key] = list(vec)
                elif isinstance(vec, dict):
                    res['embedding'][key] = {str(i): float(v) for i, v in zip(vec['indices'], vec['values'])}
                else:
                    res['embedding'][key] = {str(i): float(v) for i, v in zip(vec.indices, vec.values)}
        return res

    def _to_uuid(self, uid):
        try:
            return str(uuid.UUID(str(uid)))
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, f'lazyllm:{uid}'))

    def _build_filter(self, filters):
        if not filters:
            return None
        must, should = [], []
        for name, candidates in filters.items():
            desc = self._global_metadata_desc.get(name)
            if not desc:
                raise ValueError(f'cannot find desc of field [{name}]')
            if isinstance(candidates, str):
                candidates = [candidates]
            elif not isinstance(candidates, (list, set, tuple)):
                candidates = [candidates]
            if not candidates:
                continue
            key = GLOBAL_META_KEY_PREFIX + name
            if desc.data_type == DataType.FLOAT:
                conds = [qdrant_client.models.FieldCondition(
                    key=key, range=qdrant_client.models.Range(gte=float(v), lte=float(v)))
                    for v in candidates]
                (should if len(conds) > 1 else must).extend(conds)
            elif desc.data_type == DataType.ARRAY or len(candidates) > 1:
                must.append(qdrant_client.models.FieldCondition(
                    key=key, match=qdrant_client.models.MatchAny(any=list(candidates))))
            else:
                must.append(qdrant_client.models.FieldCondition(
                    key=key, match=qdrant_client.models.MatchValue(value=candidates[0])))
        if should:
            return qdrant_client.models.Filter(must=must, should=should)
        return qdrant_client.models.Filter(must=must)
