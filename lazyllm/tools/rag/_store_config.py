'''Helpers for inspecting ``store_conf`` dicts consumed by ``Document``.

Extension points — single place to update when a new backend is introduced:
- ``NON_PERSISTENT_STORE_TYPES``: add a type string here when the backend is
  purely volatile (in-memory / mock). Everything else is treated as persistent
  by default (the safe direction: unknown types trigger DocServer auto-upgrade
  instead of silently bypassing it).
- ``SPLIT_STORE_KEYS``: add a sub-key here when ``store_conf`` grows a new
  split-form slot that can carry its own ``type``.
- ``_REMOTE_STORE_SCHEMES``: add a URL scheme string here when introducing a
  new remote transport.

New persistent remote stores (Qdrant, Weaviate, ...) require no change — they
are automatically classified as persistent and remote by the default path.
'''
from typing import Dict, FrozenSet, Optional, Tuple
from urllib.parse import urlparse


NON_PERSISTENT_STORE_TYPES: FrozenSet[str] = frozenset({'map'})
'''Volatile backends whose state does not survive process restart.'''

SPLIT_STORE_KEYS: Tuple[str, ...] = ('vector_store', 'segment_store')
'''Sub-keys carrying an independent ``type`` in split-form ``store_conf``.
``metadata_store`` is intentionally NOT here: its presence alone does not make
the config persistent (the vector backend is still map by default).'''

_NESTED_STORE_KEYS: Tuple[str, ...] = SPLIT_STORE_KEYS + ('metadata_store',)
'''All sub-keys walked when iterating nested store configs (includes
``metadata_store`` because embedded-store detection must recurse into it).'''

_REMOTE_STORE_SCHEMES = frozenset({'http', 'https', 'tcp', 'grpc', 'unix'})


def is_local_map_store(store_conf: Optional[Dict]) -> bool:
    return isinstance(store_conf, dict) and store_conf.get('type') == 'map'


def is_persistent_store(store_conf: Optional[Dict]) -> bool:
    if not isinstance(store_conf, dict):
        return False
    top_type = store_conf.get('type')
    if top_type is not None:
        return top_type not in NON_PERSISTENT_STORE_TYPES
    for key in SPLIT_STORE_KEYS:
        sub = store_conf.get(key)
        if isinstance(sub, dict) and sub.get('type', 'map') not in NON_PERSISTENT_STORE_TYPES:
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
    if not store_type or store_type in NON_PERSISTENT_STORE_TYPES:
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
        for key in _NESTED_STORE_KEYS:
            sub = cfg.get(key)
            if isinstance(sub, dict):
                pending.append(sub)
        indices = cfg.get('indices')
        if isinstance(indices, dict):
            for idx in indices.values():
                if isinstance(idx, dict):
                    pending.append(idx)
