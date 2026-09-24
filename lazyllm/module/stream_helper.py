import asyncio
import json
import re
import sys
import time
from concurrent.futures import CancelledError
from threading import Lock
from typing import Callable, Optional

import lazyllm

_ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*m')

_g_stream_thread_pool = lazyllm.ThreadPoolExecutor(
    max_workers=lazyllm.config['thread_pool_worker_num'], local_scope='inherit')


def _clean_chunk(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub('', text)


class StreamCallHelper:
    def __init__(self, impl: Callable, interval: float = 0.1, *, init_sid: Optional[bool] = True,
                 on_cancel: Optional[Callable[[], None]] = None):
        self._impl = impl
        self._sleep_interval = interval
        self.init_sid = init_sid
        self.future = None
        self._on_cancel = on_cancel
        self._cancel_notified = False
        self._cancel_lock = Lock()

    def _submit(self, *args, **kwargs):
        if self.init_sid:
            lazyllm.globals._init_sid()
            lazyllm.locals._init_sid()
        lazyllm.FileSystemQueue().clear()
        self._cancel_notified = False
        self.future = _g_stream_thread_pool.submit(self._impl, *args, **kwargs)
        return self.future

    def __call__(self, *args, **kwargs):
        future = self._submit(*args, **kwargs)
        try:
            yield from self._drain(future, time.sleep)
        finally:
            self.close()

    async def astream(self, *args, **kwargs):
        future = self._submit(*args, **kwargs)
        iterator = self._adrain(future)
        try:
            async for item in iterator:
                yield item
        finally:
            error = sys.exc_info()[1]
            await iterator.aclose()
            try:
                await self.aclose()
            except asyncio.CancelledError:
                if error is None:
                    raise

    def _request_cancel(self):
        future = self.future
        with self._cancel_lock:
            if future is None or future.done() or future.cancel() or self._cancel_notified:
                return
            self._cancel_notified = True
        if self._on_cancel is not None:
            try:
                self._on_cancel()
            except BaseException as error:  # noqa: B036 - A failing callback must not skip producer cleanup.
                lazyllm.LOG.warning(f'Stream cancellation callback failed: {error!r}')

    def close(self) -> None:
        self._request_cancel()
        if self.future is not None:
            try:
                self.future.exception()
            except CancelledError:
                pass

    async def aclose(self) -> None:
        self._request_cancel()
        future = self.future
        if future is None or future.done():
            return
        loop = asyncio.get_running_loop()
        completed = loop.create_future()
        future.add_done_callback(lambda _: loop.call_soon_threadsafe(completed.set_result, None))
        cancellation = None
        while not completed.done():
            try:
                await asyncio.shield(completed)
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
        if cancellation is not None:
            raise cancellation

    def _drain(self, future, sleep):
        q = lazyllm.FileSystemQueue()
        while not future.done():
            drained = False
            for item in self._drain_queue(q):
                drained = True
                yield item
            if not drained:
                sleep(self._sleep_interval)
        for item in self._drain_queue(q):
            yield item

    async def _adrain(self, future):
        q = lazyllm.FileSystemQueue()
        while not future.done():
            drained = False
            for item in self._drain_queue(q):
                drained = True
                yield item
            if not drained:
                await asyncio.sleep(self._sleep_interval)
        for item in self._drain_queue(q):
            yield item

    def _drain_queue(self, q):
        if values := q.dequeue():
            for raw in values:
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if 'delta' in payload:
                    cleaned = _clean_chunk(str(payload['delta']))
                    if not cleaned:
                        continue
                    payload['delta'] = cleaned
                yield payload
            return True
        return False
