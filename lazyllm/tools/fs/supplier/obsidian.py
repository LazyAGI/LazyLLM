# Copyright (c) 2026 LazyAGI. All rights reserved.
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import LazyLLMFSBase, CloudFSBufferedFile


class ObsidianFS(LazyLLMFSBase):

    def __init__(
        self,
        token: str = '',
        base_url: Optional[str] = None,
        asynchronous: bool = False,
        use_listings_cache: bool = False,
        skip_instance_cache: bool = False,
        loop: Optional[Any] = None,
    ):
        vault = (token or '').strip() or '.'
        self._vault_root = os.path.abspath(os.path.expanduser(vault))
        super().__init__(
            token=vault,
            base_url=base_url,
            asynchronous=asynchronous,
            use_listings_cache=use_listings_cache,
            skip_instance_cache=skip_instance_cache,
            loop=loop,
        )

    def _setup_auth(self) -> None:
        if not os.path.isdir(self._vault_root):
            raise FileNotFoundError(
                'Obsidian vault path is not a directory: %r' % (self._vault_root,)
            )

    def _abspath(self, path: str) -> str:
        parts = self._parse_path(path)
        vault_p = Path(self._vault_root).resolve()
        if not parts:
            return str(vault_p)
        full = vault_p.joinpath(*parts).resolve()
        try:
            full.relative_to(vault_p)
        except ValueError:
            raise PermissionError(
                'Path %r escapes Obsidian vault %r' % (path, self._vault_root)
            )
        return str(full)

    def _relpath(self, full_path: str) -> str:
        return os.path.relpath(full_path, self._vault_root).replace(os.sep, '/')

    def ls(self, path: str, detail: bool = True, **kwargs) -> List:
        full_dir = self._abspath(path)
        if not os.path.isdir(full_dir):
            raise FileNotFoundError(path)
        prefix = path.strip('/')
        results = []
        for name in sorted(os.listdir(full_dir)):
            if name.startswith('.'):
                continue
            child_full = os.path.join(full_dir, name)
            rel = prefix + '/' + name if prefix else name
            if detail:
                st = os.stat(child_full)
                ftype = 'directory' if os.path.isdir(child_full) else 'file'
                sz = 0 if os.path.isdir(child_full) else st.st_size
                results.append(
                    self._entry(name=rel, size=sz, ftype=ftype, mtime=st.st_mtime)
                )
            else:
                results.append(rel)
        return results

    def info(self, path: str, **kwargs) -> Dict[str, Any]:
        parts = self._parse_path(path)
        if not parts:
            return self._entry(name='/', ftype='directory')
        full = self._abspath(path)
        if not os.path.exists(full):
            raise FileNotFoundError(path)
        rel = self._relpath(full)
        if os.path.isdir(full):
            return self._entry(name=rel, ftype='directory')
        st = os.stat(full)
        return self._entry(name=rel, size=st.st_size, ftype='file', mtime=st.st_mtime)

    def _open(self, path: str, mode: str = 'rb',
              block_size: Optional[int] = None,
              autocommit: bool = True,
              cache_options: Optional[Dict] = None,
              **kwargs) -> CloudFSBufferedFile:
        return CloudFSBufferedFile(
            self, path, mode=mode,
            block_size=block_size or self.blocksize,
            autocommit=autocommit, cache_options=cache_options,
        )

    def mkdir(self, path: str, create_parents: bool = True, **kwargs) -> None:
        os.makedirs(self._abspath(path), exist_ok=create_parents)

    def rmdir(self, path: str) -> None:
        full = self._abspath(path)
        if not os.path.exists(full):
            raise FileNotFoundError(path)
        if not os.path.isdir(full):
            raise NotADirectoryError(path)
        os.rmdir(full)

    def rm_file(self, path: str) -> None:
        full = self._abspath(path)
        if os.path.isdir(full):
            raise IsADirectoryError(path)
        os.remove(full)

    def _download_range(self, path: str, start: int, end: int) -> bytes:
        full = self._abspath(path)
        with open(full, 'rb') as fh:
            fh.seek(start)
            return fh.read(end - start)

    def _upload_data(self, path: str, data: bytes) -> None:
        full = self._abspath(path)
        parent = os.path.dirname(full)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        with open(full, 'wb') as fh:
            fh.write(data)
