# Copyright (c) 2026 LazyAGI. All rights reserved.
import re
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional
from lazyllm import globals
from lazyllm.tools.fs.base import LinkDocumentFSBase

_PROTOCOL_RE = re.compile(r'^([a-zA-Z][a-zA-Z0-9+\-.]*)(@[^:/]+)?:/(.*)$')
_FEISHU_BARE_URL_RE = re.compile(r'^https?://[^/]*(?:feishu\.cn|larksuite\.com)/', re.IGNORECASE)
_FEISHU_WIKI_PATH_PREFIXES = ('~link/', '~node/', '~docx/', '~doc/')
_NOTION_BARE_URL_RE = re.compile(r'^https?://[^/]*(?:notion\.so|notion\.site)/', re.IGNORECASE)
_NOTION_LINK_PATH_PREFIXES = ('~link/', '~page/', '~database/', '~block/')


def _lookup_fs_cls(protocol: str):
    from lazyllm.common import LazyLLMRegisterMetaClass
    registry = LazyLLMRegisterMetaClass.all_clses.get('fs', {})
    cls = registry.get(protocol + 'fs')
    if cls is None:
        available = sorted(k[:-2] for k in registry if k.endswith('fs'))
        raise ValueError(
            f'Unknown FS protocol: {protocol!r}. '
            f'Supported protocols: {", ".join(available)}.'
        )
    return cls


def _feishu_needs_wiki(space_id: Optional[str], real_path: str) -> bool:
    if space_id:
        return True
    norm = real_path.lstrip('/')
    if any(norm.startswith(p) for p in _FEISHU_WIKI_PATH_PREFIXES):
        return True
    return bool((globals.config.get('feishu_wiki_space_id') or '').strip())


class _FSRouter:

    def __init__(self) -> None:
        self._instances: Dict[tuple, Any] = {}
        self._lock = threading.Lock()

    def _parse(self, path: str):
        if _FEISHU_BARE_URL_RE.match(path):
            return 'feishu', 'dynamic', LinkDocumentFSBase.to_link_path(path)
        if _NOTION_BARE_URL_RE.match(path):
            return 'notion', 'dynamic', LinkDocumentFSBase.to_link_path(path)
        m = _PROTOCOL_RE.match(path)
        if not m:
            return 'file', None, path
        protocol = m.group(1).lower()
        at_id = m.group(2)
        rest = '/' + m.group(3)
        space_id = at_id[1:] if at_id else None
        if protocol == 'feishu' and space_id is None:
            norm = rest.lstrip('/')
            if any(norm.startswith(p) for p in _FEISHU_WIKI_PATH_PREFIXES):
                space_id = 'dynamic'
        if protocol == 'notion' and space_id is None:
            norm = rest.lstrip('/')
            if any(norm.startswith(p) for p in _NOTION_LINK_PATH_PREFIXES):
                space_id = 'dynamic'
        return protocol, space_id, rest

    def _get_or_create_fs(self, protocol: str, space_id: Optional[str], real_path: str = '') -> Any:
        effective_space = space_id
        if protocol == 'feishu' and effective_space is None and _feishu_needs_wiki(None, real_path):
            effective_space = 'dynamic'
        key = (protocol, effective_space)
        if key not in self._instances:
            with self._lock:
                if key not in self._instances:
                    cls = _lookup_fs_cls(protocol)
                    import inspect
                    init_params = inspect.signature(cls.__init__).parameters
                    kwargs: Dict[str, Any] = {'dynamic_auth': True}
                    if effective_space and 'space_id' in init_params:
                        kwargs['space_id'] = effective_space
                    self._instances[key] = cls(**kwargs)
        return self._instances[key]

    def _dispatch(self, method: str, path: str, *args, **kwargs) -> Any:
        protocol, space_id, real_path = self._parse(path)
        if protocol == 'file':
            import fsspec.implementations.local as _local_fs
            fs = _local_fs.LocalFileSystem()
            return getattr(fs, method)(real_path, *args, **kwargs)
        fs = self._get_or_create_fs(protocol, space_id, real_path)
        return getattr(fs, method)(real_path, *args, **kwargs)

    def open(self, path: str, mode: str = 'rb', **kwargs) -> Any:
        return self._dispatch('open', path, mode, **kwargs)

    def ls(self, path: str, detail: bool = True, **kwargs) -> Any:
        return self._dispatch('ls', path, detail=detail, **kwargs)

    def info(self, path: str, **kwargs) -> Any:
        return self._dispatch('info', path, **kwargs)

    def mkdir(self, path: str, create_parents: bool = True, **kwargs) -> None:
        self._dispatch('mkdir', path, create_parents=create_parents, **kwargs)

    def rm(self, path: str, recursive: bool = False) -> None:
        self._dispatch('rm', path, recursive=recursive)

    def exists(self, path: str, **kwargs) -> bool:
        return self._dispatch('exists', path, **kwargs)

    def read_bytes(self, path: str, **kwargs) -> bytes:
        return self._dispatch('read_bytes', path, **kwargs)

    def read_file(self, path: str) -> str:
        return self._dispatch('read_file', path)

    def write_file(self, path: str, data: bytes) -> None:
        self._dispatch('write_file', path, data)

    def copy(self, path1: str, path2: str, recursive: bool = False, **kwargs) -> None:
        self._dispatch('copy', path1, path2, recursive=recursive, **kwargs)

    def move(self, path1: str, path2: str, recursive: bool = False, **kwargs) -> None:
        self._dispatch('move', path1, path2, recursive=recursive, **kwargs)


FS = _FSRouter()


@contextmanager
def dynamic_fs_config(source_token_map: Dict[str, str]) -> Iterator[None]:
    old = globals.config['dynamic_fs_auth']
    globals.config['dynamic_fs_auth'] = {**(old or {}), **source_token_map}
    try:
        yield
    finally:
        globals.config['dynamic_fs_auth'] = old
