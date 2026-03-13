# Copyright (c) 2026 LazyAGI. All rights reserved.
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import requests

from lazyllm import thirdparty
from lazyllm.common.registry import LazyLLMRegisterMetaABCClass

AbstractFileSystem = thirdparty.fsspec.spec.AbstractFileSystem
AbstractBufferedFile = thirdparty.fsspec.spec.AbstractBufferedFile


class CloudFSBufferedFile(AbstractBufferedFile):

    def _fetch_range(self, start: int, end: int) -> bytes:
        return self.fs._download_range(self.path, start, end)

    def _initiate_upload(self) -> None:
        pass

    def _upload_chunk(self, final: bool = False) -> bool:
        if not final:
            return False
        self.buffer.seek(0)
        data = self.buffer.read()
        self.fs._upload_data(self.path, data)
        return True


_CloudFSMeta = type('_CloudFSMeta', (LazyLLMRegisterMetaABCClass, type(AbstractFileSystem)), {})


class LazyLLMFSBase(AbstractFileSystem, metaclass=_CloudFSMeta):

    protocol: str = 'cloudfs'

    def __init__(self, token: str, base_url: Optional[str] = None, **storage_options):
        AbstractFileSystem.__init__(self, **storage_options)
        self._token = token
        self._base_url = (base_url or '').rstrip('/')
        self._session = requests.Session()
        self._session.headers.update({'User-Agent': 'lazyllm-fs/1.0'})
        self._setup_auth()

    def close(self) -> None:
        self._session.close()

    @abstractmethod
    def _setup_auth(self) -> None:
        pass

    @abstractmethod
    def ls(self, path: str, detail: bool = True, **kwargs) -> List:
        pass

    @abstractmethod
    def info(self, path: str, **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _open(self, path: str, mode: str = 'rb',
              block_size: Optional[int] = None,
              autocommit: bool = True,
              cache_options: Optional[Dict] = None,
              **kwargs) -> CloudFSBufferedFile:
        pass

    def mkdir(self, path: str, create_parents: bool = True, **kwargs) -> None:
        pass

    def makedirs(self, path: str, exist_ok: bool = False) -> None:
        self.mkdir(path, create_parents=True)

    def rmdir(self, path: str) -> None:
        pass

    def rm_file(self, path: str) -> None:
        raise NotImplementedError(f'{self.__class__.__name__}.rm_file is not implemented')

    def rm(self, path: str, recursive: bool = False, maxdepth: Optional[int] = None) -> None:
        if self.isdir(path) and recursive:  # type: ignore[attr-defined]
            path_rstrip = path.rstrip(self.sep)
            for entry in self.ls(path, detail=True):
                child_name = entry['name']
                full_path = f'{path_rstrip}{self.sep}{child_name}' if path_rstrip else child_name
                self.rm(full_path, recursive=True)
            self.rmdir(path)
        else:
            self.rm_file(path)

    def put_file(self, lpath: str, rpath: str, **kwargs) -> None:
        with open(lpath, 'rb') as fh:
            data = fh.read()
        self._upload_data(rpath, data)

    def get_file(self, rpath: str, lpath: str, **kwargs) -> None:
        with self.open(rpath, 'rb') as fh:
            data = fh.read()
        with open(lpath, 'wb') as fh:
            fh.write(data)

    def _platform_supports_webhook(self) -> bool:
        return False

    def supports_webhook(self) -> bool:
        return self._platform_supports_webhook()

    def register_webhook(
        self, path: str, webhook_url: str,
        events: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, Any]:
        if not self.supports_webhook():
            return {'mode': 'none'}
        reg = getattr(self, '_register_webhook', None)
        if not callable(reg):
            return {'mode': 'none'}
        return reg(webhook_url, events or ['*'], path)

    def _download_range(self, path: str, start: int, end: int) -> bytes:
        raise NotImplementedError(f'{self.__class__.__name__}._download_range is not implemented')

    def _upload_data(self, path: str, data: bytes) -> None:
        raise NotImplementedError(f'{self.__class__.__name__}._upload_data is not implemented')

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        resp = self._session.request(method, url, **kwargs)
        resp.raise_for_status()
        return resp

    def _json_or_empty(self, resp: requests.Response) -> Any:
        if not resp.content:
            return {}
        return resp.json()

    def _get(self, url: str, **kwargs) -> Any:
        return self._json_or_empty(self._request('GET', url, **kwargs))

    def _post(self, url: str, **kwargs) -> Any:
        return self._json_or_empty(self._request('POST', url, **kwargs))

    def _put(self, url: str, **kwargs) -> Any:
        return self._json_or_empty(self._request('PUT', url, **kwargs))

    def _patch(self, url: str, **kwargs) -> Any:
        return self._json_or_empty(self._request('PATCH', url, **kwargs))

    def _delete(self, url: str, **kwargs) -> Any:
        resp = self._request('DELETE', url, **kwargs)
        return self._json_or_empty(resp)

    @staticmethod
    def _entry(name: str, size: int = 0, ftype: str = 'file',
               mtime: Optional[float] = None, **extra) -> Dict[str, Any]:
        d: Dict[str, Any] = {'name': name, 'size': size, 'type': ftype}
        if mtime is not None:
            d['mtime'] = mtime
        d.update(extra)
        return d

    def _parse_path(self, path: str) -> Tuple[str, ...]:
        stripped = self._strip_protocol(path)
        parts = [p for p in stripped.split('/') if p]
        return tuple(parts)
