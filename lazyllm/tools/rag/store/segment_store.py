import hashlib
import os
import re

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import lazyllm

from .hybrid import HybridStore, MapStore
from .store_base import BUILDIN_GLOBAL_META_DESC, LazyLLMStoreBase, StoreCapability


class SegmentStoreConflictError(RuntimeError):
    pass


class SegmentStoreUnsupportedError(RuntimeError):
    pass


class SegmentStore:
    '''Public facade for canonical text-segment persistence.

    The facade owns backend construction, lazy connection and collection name
    normalization.  Domain users exchange ``id/content/metadata`` records;
    existing RAG callers may access ``backend`` without changing their layout.
    '''

    _COLLECTION_NAME_PATTERN = re.compile(r'[^a-z0-9_]+')
    _COLLECTION_NAME_MAX_LEN = 255

    def __init__(self, store: Union[Dict[str, Any], LazyLLMStoreBase], *,
                 global_metadata_desc=None):
        self._global_metadata_desc = global_metadata_desc or BUILDIN_GLOBAL_META_DESC
        self.backend = self.create_backend(store)
        self._connected = False

    @classmethod
    def create_backend(cls, store):
        if not isinstance(store, dict):
            if store.capability == StoreCapability.VECTOR:
                segment_store = MapStore(uri=os.path.join(store.dir, 'segments.db') if store.dir else None)
                return HybridStore(segment_store=segment_store, vector_store=store)
            return store
        cfg = dict(store)
        if cfg.get('indices'):
            indices = cfg.pop('indices')
            index_config = indices.get('smart_embedding_index')
            if not index_config or not index_config.get('backend'):
                raise ValueError('smart_embedding_index backend is required')
            cfg = {'type': index_config['backend'], 'kwargs': index_config.get('kwargs', {})}
        if 'type' in cfg:
            store_cls = getattr(lazyllm.store, cfg['type'], None)
            if not store_cls:
                raise NotImplementedError(f'Not implemented store type: {cfg["type"]}')
            impl = store_cls(**cfg.get('kwargs', {}))
            if impl.capability == StoreCapability.VECTOR:
                path = Path(impl.dir) if impl.dir else None
                uri = str(path.with_name(f'lazyllm_{path.stem}_segments.db')) if path else None
                return HybridStore(segment_store=MapStore(uri=uri), vector_store=impl)
            return impl
        seg_cfg = cfg.get('segment_store') or {}
        vec_cfg = cfg.get('vector_store') or {}

        def make(value):
            return getattr(lazyllm.store, value['type'])(**value.get('kwargs', {})) if value else None
        segment, vector = make(seg_cfg), make(vec_cfg)
        if segment and vector:
            return HybridStore(segment_store=segment, vector_store=vector)
        if segment:
            return segment
        if vector:
            path = Path(vector.dir) if vector.dir else None
            uri = str(path.with_name(f'lazyllm_{path.stem}_segments.db')) if path else None
            return HybridStore(segment_store=MapStore(uri=uri),
                               vector_store=vector)
        raise ValueError('Provide either type or segment_store/vector_store config')

    @classmethod
    def normalize_collection_name(cls, raw_name: str) -> str:
        normalized = cls._COLLECTION_NAME_PATTERN.sub('_', raw_name.lower()).strip('_') or 'col'
        if normalized[0].isdigit():
            normalized = f'col_{normalized}'
        if normalized == raw_name and len(normalized) <= cls._COLLECTION_NAME_MAX_LEN:
            return normalized
        digest = hashlib.sha1(raw_name.encode()).hexdigest()[:12]
        prefix = normalized[:cls._COLLECTION_NAME_MAX_LEN - 13].rstrip('_') or 'col'
        return f'{prefix}_{digest}'

    def connect(self):
        if not self._connected:
            target = getattr(self.backend, 'segment_store', self.backend)
            target.connect(global_metadata_desc=self._global_metadata_desc)
            self._connected = True
        return self

    @property
    def _segment_backend(self):
        self.connect()
        return getattr(self.backend, 'segment_store', self.backend)

    @staticmethod
    def _to_raw(item: Dict[str, Any]) -> dict:
        metadata = dict(item.get('metadata') or {})
        user_id = metadata.get('user_id')
        return {
            'uid': item['id'], 'doc_id': item['id'], 'group': 'segment',
            'content': item.get('content', ''), 'meta': metadata,
            'global_meta': metadata, 'kb_id': user_id or '__default__',
            'number': int(metadata.get('hit_count', 0)), 'type': 1,
        }

    @staticmethod
    def _from_raw(item: dict) -> dict:
        metadata = dict(item.get('meta') or item.get('global_meta') or {})
        metadata.setdefault('user_id', item.get('kb_id'))
        metadata['hit_count'] = int(item.get('number', metadata.get('hit_count', 0)) or 0)
        result = {'id': item.get('uid'), 'content': item.get('content', ''), 'metadata': metadata}
        if 'score' in item:
            result['score'] = float(item['score'] or 0)
        return result

    @staticmethod
    def _filters(filters: Optional[dict]) -> dict:
        result = dict(filters or {})
        if 'id' in result:
            value = result.pop('id')
            result['uid'] = value if isinstance(value, (list, set, tuple)) else [value]
        if 'user_id' in result:
            result['kb_id'] = result.pop('user_id')
        return result

    def create_collection(self, name: str) -> bool:
        return bool(self._segment_backend.create_collection(self.normalize_collection_name(name)))

    def drop_collection(self, name: str) -> bool:
        return bool(self._segment_backend.drop_collection(self.normalize_collection_name(name)))

    def collection_exists(self, name: str) -> bool:
        return bool(self._segment_backend.collection_exists(self.normalize_collection_name(name)))

    def create(self, name: str, data: List[dict]) -> bool:
        try:
            return bool(self._segment_backend.create(
                self.normalize_collection_name(name), [self._to_raw(item) for item in data],
            ))
        except NotImplementedError as exc:
            raise SegmentStoreUnsupportedError(str(exc)) from exc
        except FileExistsError as exc:
            raise SegmentStoreConflictError(str(exc)) from exc

    def upsert(self, name: str, data: List[dict]) -> bool:
        return bool(self._segment_backend.upsert(
            self.normalize_collection_name(name), [self._to_raw(item) for item in data],
        ))

    def get(self, name: str, filters: dict, *, strict: bool = False) -> List[dict]:
        read_options = {'raise_on_error': True} if strict else {}
        return [self._from_raw(item) for item in self._segment_backend.get(
            self.normalize_collection_name(name), self._filters(filters), **read_options)]

    def search(self, name: str, query: str, *, topk: int = 10,
               filters: Optional[dict] = None, query_fields: Optional[List[str]] = None,
               match_mode: Optional[str] = None, strict: bool = False) -> List[dict]:
        search_options = {}
        if query_fields is not None:
            if not isinstance(query_fields, list) or not query_fields:
                raise ValueError('query_fields must be a non-empty list of field names')
            if any(not isinstance(field, str) or not field.strip() for field in query_fields):
                raise ValueError('query_fields must contain only non-empty field names')
            normalized_fields = [field.strip() for field in query_fields]
            search_options['query_fields'] = normalized_fields
        if match_mode is not None:
            if match_mode not in ('any', 'all'):
                raise ValueError("match_mode must be 'any', 'all', or None")
            search_options['match_mode'] = match_mode
        backend = self._segment_backend
        if search_options and not getattr(backend, 'supports_query_fields_match_mode', False):
            raise SegmentStoreUnsupportedError(
                f'{type(backend).__name__} does not support query_fields/match_mode'
            )
        supported_fields = getattr(backend, 'supported_query_fields', None)
        if query_fields is not None and supported_fields is not None:
            unsupported_fields = set(normalized_fields) - set(supported_fields)
            if unsupported_fields:
                raise SegmentStoreUnsupportedError(
                    f'{type(backend).__name__} does not support query fields: '
                    f'{sorted(unsupported_fields)!r}'
                )
        if strict:
            search_options['raise_on_error'] = True
        return [self._from_raw(item) for item in backend.search(
            self.normalize_collection_name(name), query=query, topk=topk,
            filters=self._filters(filters), **search_options)]

    def patch(self, name: str, filters: dict, *, set_fields=None, inc_fields=None) -> int:
        mapped_set = dict(set_fields or {})
        mapped_inc = dict(inc_fields or {})
        if 'hit_count' in mapped_set:
            mapped_set['number'] = mapped_set.pop('hit_count')
        if 'hit_count' in mapped_inc:
            mapped_inc['number'] = mapped_inc.pop('hit_count')
        try:
            return self._segment_backend.patch(self.normalize_collection_name(name), self._filters(filters),
                                               mapped_set, mapped_inc)
        except NotImplementedError as exc:
            raise SegmentStoreUnsupportedError(str(exc)) from exc

    def delete(self, name: str, filters: dict) -> bool:
        return bool(self._segment_backend.delete(self.normalize_collection_name(name), self._filters(filters)))
