# Copyright (c) 2026 LazyAGI. All rights reserved.
import re
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional
from lazyllm import globals

_PROTOCOL_RE = re.compile(r'^([a-zA-Z][a-zA-Z0-9+\-.]*)(@[^:/]+)?:/(.*)$')


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


class _FSRouter:

    def __init__(self) -> None:
        self._instances: Dict[tuple, Any] = {}

    def _parse(self, path: str):
        m = _PROTOCOL_RE.match(path)
        if not m:
            return 'file', None, path
        protocol = m.group(1).lower()
        at_id = m.group(2)
        rest = '/' + m.group(3)
        return protocol, (at_id[1:] if at_id else None), rest

    def _get_or_create_fs(self, protocol: str, space_id: Optional[str]) -> Any:
        key = (protocol, space_id)
        if key not in self._instances:
            cls = _lookup_fs_cls(protocol)
            self._instances[key] = cls(space_id=space_id, dynamic_auth=True) if space_id else cls(dynamic_auth=True)
        return self._instances[key]

    def _dispatch(self, method: str, path: str, *args, **kwargs) -> Any:
        protocol, space_id, real_path = self._parse(path)
        if protocol == 'file':
            if method == 'open':
                mode = args[0] if args else kwargs.pop('mode', 'rb')
                return open(real_path, mode)
            raise ValueError(f'Local file system does not support {method!r} via FS router; use open() directly.')
        fs = self._get_or_create_fs(protocol, space_id)
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

    def read_bytes(self, path: str) -> bytes:
        return self._dispatch('read_bytes', path)

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
