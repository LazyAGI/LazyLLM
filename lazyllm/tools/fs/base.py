# Copyright (c) 2026 LazyAGI. All rights reserved.
import io
import threading
import time
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from fsspec.spec import AbstractFileSystem, AbstractBufferedFile

from lazyllm.common.registry import LazyLLMRegisterMetaABCClass


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


class CloudFSWatchMixin:

    def __init__(self) -> None:
        self._watchers: List[Dict[str, Any]] = []
        self._watch_lock = threading.Lock()
        self._watch_thread: Optional[threading.Thread] = None
        self._watch_running = False

    def _platform_supports_webhook(self) -> bool:
        return False

    def _register_webhook(self, webhook_url: str, events: List[str], path: str) -> Dict[str, Any]:
        raise NotImplementedError(f'{self.__class__.__name__} does not support webhooks')

    def watch(self, path: str, callback: Callable[[str, str, Dict], None],
              polling_interval: int = 30) -> str:
        watcher_id = f'watcher-{id(callback)}-{int(time.time())}'
        entry: Dict[str, Any] = {
            'id': watcher_id,
            'path': path,
            'callback': callback,
            'interval': polling_interval,
            'last_snapshot': self._snapshot(path),
            'type': 'polling',
        }
        with self._watch_lock:
            self._watchers.append(entry)
            if not self._watch_running:
                self._watch_running = True
                self._watch_thread = threading.Thread(
                    target=self._polling_loop, daemon=True)
                self._watch_thread.start()
        return watcher_id

    def register_webhook(self, path: str, webhook_url: str,
                         events: Optional[List[str]] = None,
                         callback: Optional[Callable] = None) -> Dict[str, Any]:
        events = events or ['*']
        if self._platform_supports_webhook():
            info = self._register_webhook(webhook_url, events, path)
            if callback:
                self.watch(path, callback)
            return info
        if callback:
            watcher_id = self.watch(path, callback)
            return {'mode': 'polling', 'watcher_id': watcher_id}
        return {'mode': 'none'}

    def unwatch(self, watcher_id: str) -> bool:
        with self._watch_lock:
            before = len(self._watchers)
            self._watchers = [w for w in self._watchers if w['id'] != watcher_id]
            removed = len(self._watchers) < before
            if not self._watchers:
                self._watch_running = False
        return removed

    def _snapshot(self, path: str) -> Dict[str, Any]:
        try:
            entries = self.ls(path, detail=True)  # type: ignore[attr-defined]
            return {e['name']: e.get('mtime') or e.get('last_modified') for e in entries}
        except Exception:
            return {}

    def _polling_loop(self) -> None:
        while self._watch_running:
            with self._watch_lock:
                watchers = list(self._watchers)
            for watcher in watchers:
                self._check_watcher(watcher)
            time.sleep(min(w['interval'] for w in watchers) if watchers else 30)

    def _check_watcher(self, watcher: Dict[str, Any]) -> None:
        path = watcher['path']
        callback = watcher['callback']
        old = watcher['last_snapshot']
        try:
            new = self._snapshot(path)
        except Exception:
            return
        old_keys = set(old.keys())
        new_keys = set(new.keys())
        for name in new_keys - old_keys:
            callback('created', name, {'path': name})
        for name in old_keys - new_keys:
            callback('deleted', name, {'path': name})
        for name in old_keys & new_keys:
            if old[name] != new[name]:
                callback('modified', name, {'path': name})
        watcher['last_snapshot'] = new


_CloudFSMeta = type('_CloudFSMeta', (LazyLLMRegisterMetaABCClass, type(AbstractFileSystem)), {})


class LazyLLMFSBase(CloudFSWatchMixin, AbstractFileSystem, metaclass=_CloudFSMeta):

    protocol: str = 'cloudfs'

    def __init__(self, token: str, base_url: Optional[str] = None, **storage_options):
        CloudFSWatchMixin.__init__(self)
        AbstractFileSystem.__init__(self, **storage_options)
        self._token = token
        self._base_url = (base_url or '').rstrip('/')
        self._session = requests.Session()
        self._session.headers.update({'User-Agent': 'lazyllm-fs/1.0'})
        self._setup_auth()

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
            for entry in self.ls(path, detail=True):
                self.rm(entry['name'], recursive=True)
            self.rmdir(path)
        else:
            self.rm_file(path)

    def put_file(self, lpath: str, rpath: str, **kwargs) -> None:
        with open(lpath, 'rb') as fh:
            data = fh.read()
        self._upload_data(rpath, data)

    def _download_range(self, path: str, start: int, end: int) -> bytes:
        raise NotImplementedError(f'{self.__class__.__name__}._download_range is not implemented')

    def _upload_data(self, path: str, data: bytes) -> None:
        raise NotImplementedError(f'{self.__class__.__name__}._upload_data is not implemented')

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        resp = self._session.request(method, url, **kwargs)
        resp.raise_for_status()
        return resp

    def _get(self, url: str, **kwargs) -> Any:
        return self._request('GET', url, **kwargs).json()

    def _post(self, url: str, **kwargs) -> Any:
        return self._request('POST', url, **kwargs).json()

    def _put(self, url: str, **kwargs) -> Any:
        return self._request('PUT', url, **kwargs).json()

    def _patch(self, url: str, **kwargs) -> Any:
        return self._request('PATCH', url, **kwargs).json()

    def _delete(self, url: str, **kwargs) -> Any:
        resp = self._request('DELETE', url, **kwargs)
        if resp.content:
            return resp.json()
        return {}

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
