import asyncio
import concurrent.futures
import json
import re
import time
from contextvars import copy_context
from typing import Callable, Optional

import lazyllm
from lazyllm.common.globals import filter_session_for_propagation

_ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*m')

_g_stream_thread_pool = lazyllm.ThreadPoolExecutor(max_workers=lazyllm.config['thread_pool_worker_num'])
# init_sid=False must bypass lazyllm.ThreadPoolExecutor: its submit() re-inits sid
# before ctx.run and does not restore session bucket data (relay server pattern).
_g_context_stream_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=lazyllm.config['thread_pool_worker_num'],
)


def _clean_chunk(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub('', text)


class StreamCallHelper:
    def __init__(self, impl: Callable, interval: float = 0.1, *, init_sid: Optional[bool] = True):
        self._impl = impl
        self._sleep_interval = interval
        self.init_sid = init_sid
        self.future = None

    def _submit(self, *args, **kwargs):
        if self.init_sid:
            lazyllm.globals._init_sid()
            lazyllm.locals._init_sid()
        lazyllm.FileSystemQueue().clear()
        if self.init_sid:
            self.future = _g_stream_thread_pool.submit(self._impl, *args, **kwargs)
        else:
            # Snapshot sid + session buckets on the parent thread, then restore in the
            # worker before ctx.run — same pattern as relay server async_wrapper.
            sid = lazyllm.globals._sid
            session_data = filter_session_for_propagation(lazyllm.globals._data)
            local_data = dict(lazyllm.locals._data)
            ctx = copy_context()

            def _worker():
                lazyllm.globals._init_sid(sid)
                lazyllm.globals._update(session_data)
                lazyllm.locals._init_sid(sid)
                lazyllm.locals._update({
                    k: (v.copy() if hasattr(v, 'copy') else v)
                    for k, v in local_data.items()
                })
                return ctx.run(self._impl, *args, **kwargs)

            self.future = _g_context_stream_pool.submit(_worker)
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
