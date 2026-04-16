'''Helpers for inspecting ``store_conf`` dicts consumed by ``Document``.'''
from typing import Dict, Optional
from urllib.parse import urlparse


_REMOTE_STORE_SCHEMES = frozenset({'http', 'https', 'tcp', 'grpc', 'unix'})


def is_local_map_store(store_conf: Optional[Dict]) -> bool:
    return isinstance(store_conf, dict) and store_conf.get('type') == 'map'


def is_persistent_store(store_conf: Optional[Dict]) -> bool:
    if not isinstance(store_conf, dict):
        return False
    if store_conf.get('type') is not None:
        return store_conf['type'] != 'map'
    for key in ('vector_store', 'segment_store'):
        sub = store_conf.get(key)
        if isinstance(sub, dict) and sub.get('type', 'map') != 'map':
            return True
    return False


def _endpoint_scheme(value: str) -> str:
    if not isinstance(value, str) or not value:
        return ''
    scheme = urlparse(value).scheme.lower()
    if scheme.startswith('chroma+'):
        scheme = scheme.split('+', 1)[1]
    return scheme


def _embedded_reason_for_single_cfg(cfg: Dict) -> Optional[str]:
    store_type = cfg.get('type') or cfg.get('backend')
    if not store_type or store_type == 'map':
        return None
    kwargs = cfg.get('kwargs') or {}
    for field in ('uri', 'url', 'endpoint'):
        value = kwargs.get(field)
        if isinstance(value, str) and value:
            if _endpoint_scheme(value) not in _REMOTE_STORE_SCHEMES:
                return f'{store_type}: {field}={value!r}'
            return None
    dir_value = kwargs.get('dir')
    if isinstance(dir_value, str) and dir_value:
        return f'{store_type}: dir={dir_value!r}'
    return None


def iter_embedded_store_endpoints(store_conf: Optional[Dict]):
    '''Yield a description for every embedded (filesystem-bound) backend in ``store_conf``.

    Used to reject service-mode RAG paired with single-process stores like
    milvus_lite or ChromaStore PersistentClient that would race subprocesses
    over the same on-disk state.
    '''
    if not isinstance(store_conf, dict):
        return
    pending = [store_conf]
    while pending:
        cfg = pending.pop()
        if not isinstance(cfg, dict):
            continue
        reason = _embedded_reason_for_single_cfg(cfg)
        if reason is not None:
            yield reason
        for key in ('vector_store', 'segment_store', 'metadata_store'):
            sub = cfg.get(key)
            if isinstance(sub, dict):
                pending.append(sub)
        indices = cfg.get('indices')
        if isinstance(indices, dict):
            for idx in indices.values():
                if isinstance(idx, dict):
                    pending.append(idx)
