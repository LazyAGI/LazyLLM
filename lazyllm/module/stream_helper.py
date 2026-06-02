import asyncio
import json
import re
import time
from typing import Callable, Optional

import lazyllm

_ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*m')

_g_stream_thread_pool = lazyllm.ThreadPoolExecutor(max_workers=lazyllm.config['thread_pool_worker_num'])


def _clean_chunk(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub('', text)


class StreamCallHelper:
    def __init__(self, impl: Callable, interval: float = 0.1, *, sid: Optional[str] = None):
        self._impl = impl
        self._sleep_interval = interval
        self._sid = sid
        self.future = None

    def _submit(self, *args, **kwargs):
        if self._sid:
            lazyllm.globals._init_sid(sid=self._sid)
            lazyllm.locals._init_sid(sid=self._sid)
        else:
            lazyllm.globals._init_sid()
        lazyllm.FileSystemQueue().clear()
        self.future = _g_stream_thread_pool.submit(self._impl, *args, **kwargs)
        return self.future

    def __call__(self, *args, **kwargs):
        future = self._submit(*args, **kwargs)
        yield from self._drain(future, time.sleep)

    async def astream(self, *args, **kwargs):
        future = self._submit(*args, **kwargs)
        async for item in self._adrain(future):
            yield item

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
