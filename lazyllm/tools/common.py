import lazyllm
from typing import Callable
import time

g_thread_pool = lazyllm.ThreadPoolExecutor(max_workers=lazyllm.config["thread_pool_worker_num"])

class StreamCallHelper:
    def __init__(self, impl: Callable, interval: float = 0.1):
        self._impl = impl
        self._sleep_interval = interval

    def __call__(self, *args, **kwargs):
        lazyllm.globals._init_sid()
        if lazyllm.FileSystemQueue().size() > 0:
            lazyllm.FileSystemQueue().clear()
        func_future = g_thread_pool.submit(self._impl, *args, **kwargs)
        need_continue = True
        while need_continue:
            if func_future.done:
                need_continue = False
            if value := lazyllm.FileSystemQueue().dequeue():
                yield ''.join(value)
            else:
                time.sleep(self._sleep_interval)
        result = func_future.result()
        yield result
        if lazyllm.FileSystemQueue().size() > 0:
            lazyllm.FileSystemQueue().clear()
