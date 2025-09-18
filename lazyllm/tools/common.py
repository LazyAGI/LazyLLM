import lazyllm
from typing import Callable
import time

g_thread_pool = lazyllm.ThreadPoolExecutor(max_workers=lazyllm.config['thread_pool_worker_num'])

class StreamCallHelper:
    """Helper class for streaming function calls, wrapping a blocking callable into a generator that yields results incrementally.

Args:
    impl (Callable): The function or callable to execute in streaming mode.
    interval (float): Time interval (in seconds) to poll the internal queue. Defaults to 0.1.
"""
    def __init__(self, impl: Callable, interval: float = 0.1):
        self._impl = impl
        self._sleep_interval = interval

    def __call__(self, *args, **kwargs):
        lazyllm.globals._init_sid()
        lazyllm.FileSystemQueue().clear()
        func_future = g_thread_pool.submit(self._impl, *args, **kwargs)
        need_continue = True
        str_total = ''
        while need_continue:
            if func_future.done():
                need_continue = False
            if value := lazyllm.FileSystemQueue().dequeue():
                str_streaming = ''.join(value)
                str_total += str_streaming
                yield str_streaming
            else:
                time.sleep(self._sleep_interval)
        result = func_future.result()
        if isinstance(result, str):
            if not str_total.endswith(result):
                yield result
        else:
            yield str(result)
        lazyllm.FileSystemQueue().clear()
