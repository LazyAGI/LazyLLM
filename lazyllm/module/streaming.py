import asyncio
import time
from typing import Callable

import lazyllm


_g_stream_thread_pool = lazyllm.ThreadPoolExecutor(max_workers=lazyllm.config['thread_pool_worker_num'])


class StreamCallHelper:
    def __init__(self, impl: Callable, interval: float = 0.1):
        self._impl = impl
        self._sleep_interval = interval
        adapter_factory = getattr(impl, '_lazyllm_stream_adapter', None)
        self._adapter = adapter_factory(interval) if callable(adapter_factory) else None

    def _submit(self, *args, **kwargs):
        lazyllm.globals._init_sid()
        if self._adapter:
            self._adapter.prepare()
        else:
            lazyllm.FileSystemQueue().clear()
        return _g_stream_thread_pool.submit(self._impl, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        future = self._submit(*args, **kwargs)
        if self._adapter:
            yield from self._adapter.drain(future, time.sleep)
        else:
            yield from self._drain_llm(future)

    async def astream(self, *args, **kwargs):
        future = self._submit(*args, **kwargs)
        if self._adapter:
            async for evt in self._adapter.adrain(future):
                yield evt
        else:
            async for chunk in self._adrain_llm(future):
                yield chunk

    def _drain_llm(self, future):
        str_total = ''
        while not future.done():
            if value := lazyllm.FileSystemQueue().dequeue():
                chunk = ''.join(value)
                str_total += chunk
                yield chunk
            else:
                time.sleep(self._sleep_interval)
        while value := lazyllm.FileSystemQueue().dequeue():
            chunk = ''.join(value)
            str_total += chunk
            yield chunk
        if (tail := self._finalize(future, str_total)) is not None:
            yield tail

    async def _adrain_llm(self, future):
        str_total = ''
        while not future.done():
            if value := lazyllm.FileSystemQueue().dequeue():
                chunk = ''.join(value)
                str_total += chunk
                yield chunk
            else:
                await asyncio.sleep(self._sleep_interval)
        while value := lazyllm.FileSystemQueue().dequeue():
            chunk = ''.join(value)
            str_total += chunk
            yield chunk
        if (tail := self._finalize(future, str_total)) is not None:
            yield tail

    @staticmethod
    def _finalize(future, str_total: str):
        result = future.result()
        lazyllm.FileSystemQueue().clear()
        if result is None:
            return None
        if isinstance(result, str):
            return None if str_total.endswith(result) else result
        return str(result)
