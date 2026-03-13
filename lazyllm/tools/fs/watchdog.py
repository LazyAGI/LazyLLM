import threading
import time
from typing import Any, Callable, Dict, List, Optional

import lazyllm

from .base import LazyLLMFSBase


class CloudFsWatchdog:

    def __init__(self, fs: LazyLLMFSBase) -> None:
        self._fs = fs
        self._watchers: List[Dict[str, Any]] = []
        self._watch_lock = threading.Lock()
        self._watch_thread: Optional[threading.Thread] = None
        self._running = False

    # ---------- polling based watch ----------

    def watch(
        self,
        path: str,
        callback: Callable[[str, str, Dict[str, Any]], None],
        polling_interval: int = 30,
    ) -> str:
        watcher_id = f'watcher-{id(callback)}-{int(time.time())}'
        entry: Dict[str, Any] = {
            'id': watcher_id,
            'path': path,
            'callback': callback,
            'interval': polling_interval,
            'last_snapshot': self._snapshot(path),
        }
        with self._watch_lock:
            self._watchers.append(entry)
            if not self._running:
                self._running = True
                self._watch_thread = threading.Thread(
                    target=self._polling_loop, daemon=True
                )
                self._watch_thread.start()
        return watcher_id

    def unwatch(self, watcher_id: str) -> bool:
        with self._watch_lock:
            before = len(self._watchers)
            self._watchers = [w for w in self._watchers if w['id'] != watcher_id]
            removed = len(self._watchers) < before
            if not self._watchers:
                self._running = False
        return removed

    def stop(self) -> None:
        with self._watch_lock:
            self._running = False

    # ---------- sugar for event-specific handlers ----------

    def on(
        self,
        path: str,
        events: Optional[List[str]],
        handler: Callable[[str, str, Dict[str, Any]], None],
        polling_interval: int = 30,
    ) -> str:
        event_set = set(events or ['created', 'deleted', 'modified'])

        def _wrapper(event: str, name: str, info: Dict[str, Any]) -> None:
            if event in event_set:
                handler(event, name, info)

        return self.watch(path, _wrapper, polling_interval=polling_interval)

    def on_created(
        self,
        path: str,
        handler: Callable[[str, str, Dict[str, Any]], None],
        polling_interval: int = 30,
    ) -> str:
        return self.on(path, ['created'], handler, polling_interval=polling_interval)

    def on_deleted(
        self,
        path: str,
        handler: Callable[[str, str, Dict[str, Any]], None],
        polling_interval: int = 30,
    ) -> str:
        return self.on(path, ['deleted'], handler, polling_interval=polling_interval)

    def on_modified(
        self,
        path: str,
        handler: Callable[[str, str, Dict[str, Any]], None],
        polling_interval: int = 30,
    ) -> str:
        return self.on(path, ['modified'], handler, polling_interval=polling_interval)

    # ---------- webhook helpers ----------

    def supports_webhook(self) -> bool:
        return self._fs.supports_webhook()

    def register_webhook(
        self,
        path: str,
        webhook_url: str,
        events: Optional[List[str]] = None,
        callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        events = events or ['*']
        if self._fs.supports_webhook():
            info = self._fs.register_webhook(path=path, webhook_url=webhook_url, events=events)
            if callback:
                self.watch(path, callback)
            info.setdefault('mode', 'webhook')
            return info

        if callback:
            watcher_id = self.watch(path, callback)
            return {'mode': 'polling', 'watcher_id': watcher_id}

        return {'mode': 'none'}

    # ---------- internal helpers ----------

    def _snapshot(self, path: str) -> Dict[str, Any]:
        try:
            entries = self._fs.ls(path, detail=True)
            return {
                e['name']: e.get('mtime') or e.get('last_modified')
                for e in entries
            }
        except Exception as e:  # pragma: no cover - defensive
            lazyllm.LOG.debug(
                f"Failed to create snapshot for path '{path}': {e}"
            )
            return {}

    def _polling_loop(self) -> None:
        _min_interval = 1
        while True:
            with self._watch_lock:
                if not self._running:
                    break
                watchers = list(self._watchers)
            for watcher in watchers:
                self._check_watcher(watcher)
            interval = (
                max(_min_interval, min(w['interval'] for w in watchers))
                if watchers
                else 30
            )
            time.sleep(interval)

    def _check_watcher(self, watcher: Dict[str, Any]) -> None:
        path = watcher['path']
        callback = watcher['callback']
        old = watcher['last_snapshot']
        try:
            new = self._snapshot(path)
        except Exception:  # pragma: no cover - defensive
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
