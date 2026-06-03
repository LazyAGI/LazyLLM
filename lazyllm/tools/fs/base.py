# Copyright (c) 2026 LazyAGI. All rights reserved.
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import io

import requests

from lazyllm import thirdparty, globals
from lazyllm.common import (
    AuthStrategy, BearerTokenStrategy, Credential, CredentialMixin, KeyAuthError,
)
from lazyllm.common.registry import LazyLLMRegisterMetaABCClass

AbstractFileSystem = thirdparty.fsspec.spec.AbstractFileSystem
AbstractBufferedFile = thirdparty.fsspec.spec.AbstractBufferedFile


class CloudFSBufferedFile(AbstractBufferedFile):

    def _fetch_range(self, start: int, end: int) -> bytes:
        return self.fs._download_range(self.path, start, end)

    def _initiate_upload(self) -> None:
        self.buffer = io.BytesIO()

    def _upload_chunk(self, final: bool = False) -> bool:
        if not final:
            return False
        if 'a' in self.mode:
            raise NotImplementedError('Append mode not supported')
        self.buffer.seek(0)
        data = self.buffer.read()
        self.fs._upload_data(self.path, data)
        return True


_CloudFSMeta = type('_CloudFSMeta', (LazyLLMRegisterMetaABCClass, type(AbstractFileSystem)), {})


class LazyLLMFSBase(AbstractFileSystem, CredentialMixin, metaclass=_CloudFSMeta):

    __public_apis__ = ['ls', 'info', 'mkdir', 'rm',
                       'exists', 'read', 'read_file', 'write', 'move', 'copy']
    protocol: str = 'cloudfs'

    def __init__(self, token: Any, base_url: Optional[str] = None, asynchronous: bool = False,
                 use_listings_cache: bool = False, skip_instance_cache: bool = False, loop: Optional[Any] = None,
                 dynamic_auth: bool = False, auth_strategy: Optional[AuthStrategy] = None):
        AbstractFileSystem.__init__(
            self, asynchronous=asynchronous, use_listings_cache=use_listings_cache,
            skip_instance_cache=skip_instance_cache, loop=loop,
        )
        self._base_url = (base_url or '').rstrip('/')
        self._session = requests.Session()
        credential = self._make_credential(token, dynamic_auth)
        self.__init_credential__(credential, strategy=auth_strategy or BearerTokenStrategy())
        self._setup_auth()
        if credential.kind in ('app_credentials', 'oauth2'):
            self._bootstrap_token()

    @staticmethod
    def __lazyllm_after_registry_hook__(cls, group_name: str, name: str, isleaf: bool):
        if isleaf:
            if not name.lower().endswith('fs'):
                raise ValueError(f'Class name {name} must follow the schema of <SupplierType>FS, like <GoogleDriveFS>')
            cls.protocol = cls._fs_protocol_key = name[:-2].lower()

    def close(self) -> None:
        self._session.close()

    def _make_credential(self, token: Any, dynamic_auth: bool) -> Credential:
        # If a subclass implements _do_acquire_without_refresh, treat it as app_credentials.
        if not dynamic_auth and self._has_app_credentials_flow():
            return Credential(kind='app_credentials', secret_key=token)
        return self._default_credential(token, dynamic_auth)

    @classmethod
    def _has_app_credentials_flow(cls) -> bool:
        return cls._do_acquire_without_refresh is not CredentialMixin._do_acquire_without_refresh

    def _resolve_dynamic_token(self) -> str:
        mapping = globals.config['dynamic_fs_auth'] or {}
        return mapping.get(self._fs_protocol_key, '')

    def _missing_dynamic_token_error(self) -> str:
        return (f'dynamic_fs_auth["{self.protocol}"] is not set in globals.config; '
                f'use dynamic_fs_config() or set globals.config["dynamic_fs_auth"] '
                f'before calling FS methods')

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
    def _open(self, path: str, mode: str = 'rb', block_size: Optional[int] = None, autocommit: bool = True,
              cache_options: Optional[Dict] = None, **kwargs) -> CloudFSBufferedFile:
        pass

    def mkdir(self, path: str, create_parents: bool = True, **kwargs) -> None:
        pass

    def makedirs(self, path: str, exist_ok: bool = False) -> None:
        self.mkdir(path, create_parents=True)

    def rmdir(self, path: str) -> None:
        pass

    def rm_file(self, path: str) -> None:
        raise NotImplementedError(f'{self.__class__.__name__}.rm_file is not implemented')

    def rm(self, path: str, recursive: bool = False) -> None:
        if self.isdir(path) and recursive:  # type: ignore[attr-defined]
            for entry in self.ls(path, detail=True):
                self.rm(entry['name'], recursive=True)
            self.rmdir(path)
        else:
            self.rm_file(path)

    def put_file(self, lpath: str, rpath: str, **kwargs) -> None:
        with open(lpath, 'rb') as fh:
            data = fh.read()
        if kwargs.get('content_type') is None and lpath.lower().endswith('.md'):
            kwargs = {**kwargs, 'content_type': 'markdown'}
        self._upload_data(rpath, data, **kwargs)

    def get_file(self, rpath: str, lpath: str, **kwargs) -> None:
        with self.open(rpath, 'rb') as fh:
            data = fh.read()
        with open(lpath, 'wb') as fh:
            fh.write(data)

    def read_bytes(self, path: str) -> bytes:
        if not self.exists(path):
            raise FileNotFoundError(f'File {path} not found')
        with self.open(path, 'rb') as fh:
            return fh.read()

    def read(self, path: str) -> str:
        return self.read_bytes(path).decode('utf-8')

    def read_file(self, path: str) -> str:
        return self.read_bytes(path).decode('utf-8')

    def write_file(self, path: str, data: bytes) -> None:
        self._upload_data(path, data)

    def write(self, path: str, content: str) -> None:
        self._upload_data(path, content.encode('utf-8'))

    def copy(self, path1: str, path2: str, recursive: bool = False, **kwargs) -> None:
        raise NotImplementedError(f'{self.__class__.__name__}.copy is not implemented')

    def move(self, path1: str, path2: str, recursive: bool = False, **kwargs) -> None:
        raise NotImplementedError(f'{self.__class__.__name__}.move is not implemented')

    def _platform_supports_webhook(self) -> bool:
        return False

    def supports_webhook(self) -> bool:
        return self._platform_supports_webhook()

    def _register_webhook(self, webhook_url: str, events: List[str], path: str) -> Dict[str, Any]:
        return {'mode': 'none'}

    def register_webhook(self, path: str, webhook_url: str, events: Optional[List[str]] = None) -> Dict[str, Any]:
        if not self.supports_webhook():
            return {'mode': 'none'}
        return self._register_webhook(webhook_url, events or ['*'], path)

    def _download_range(self, path: str, start: int, end: int) -> bytes:
        raise NotImplementedError(f'{self.__class__.__name__}._download_range is not implemented')

    def _upload_data(self, path: str, data: bytes) -> None:
        raise NotImplementedError(f'{self.__class__.__name__}._upload_data is not implemented')

    def _http_execute(self, method: str, url: str, **kwargs) -> requests.Response:
        resp = self._session.request(method, url, **kwargs)
        if self._is_key_auth_error(resp):
            raise KeyAuthError(f'{resp.status_code} for {url}')
        if not resp.ok:
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            raise requests.HTTPError(
                f'{resp.status_code} {resp.reason} — {body}', response=resp,
            )
        return resp

    def _json_or_empty(self, resp: requests.Response) -> Any:
        return resp.json() if resp.content else {}

    def _get(self, url: str, **kwargs) -> Any:
        return self._json_or_empty(self._request('GET', url, **kwargs))

    def _post(self, url: str, **kwargs) -> Any:
        return self._json_or_empty(self._request('POST', url, **kwargs))

    def _put(self, url: str, **kwargs) -> Any:
        return self._json_or_empty(self._request('PUT', url, **kwargs))

    def _patch(self, url: str, **kwargs) -> Any:
        return self._json_or_empty(self._request('PATCH', url, **kwargs))

    def _delete(self, url: str, **kwargs) -> Any:
        return self._json_or_empty(self._request('DELETE', url, **kwargs))

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
        return tuple(p for p in stripped.split('/') if p)


globals.config.add('dynamic_fs_auth', dict, None, 'DYNAMIC_FS_AUTH',
                   description='Per-source dynamic FS auth: {source: token}.')
