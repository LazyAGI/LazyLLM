import asyncio
import threading
from functools import wraps
from typing import Any, Callable


def run_async_in_new_loop(func_async: Callable, *args: Any, **kwargs: Any) -> Any:

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(func_async(*args, **kwargs))
    finally:
        loop.close()


def run_async_in_thread(func_async: Callable, *args: Any, **kwargs: Any) -> Any:

    result = None
    exception = None

    def run():
        nonlocal result, exception
        try:
            result = run_async_in_new_loop(func_async, *args, **kwargs)
        except Exception as e:
            exception = e

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    if exception:
        raise exception
    return result


def patch_sync(func_async: Callable) -> Callable:
    @wraps(func_async)
    def patched_sync(*args: Any, **kwargs: Any) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            return run_async_in_thread(func_async, *args, **kwargs)
        else:
            return run_async_in_new_loop(func_async, *args, **kwargs)

    return patched_sync
