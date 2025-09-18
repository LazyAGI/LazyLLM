import threading
from queue import Queue
import functools
from .globals import globals
from concurrent.futures import ThreadPoolExecutor as TPE

def _sid_setter(sid):
    globals._init_sid(sid)

class Thread(threading.Thread):
    """Enhanced thread class provided by LazyLLM, inheriting from Python's standard library `threading.Thread`. This class provides additional functionality including session ID management, pre-hook function support, and exception handling mechanisms.

Args:
    group: Thread group, default to ``None``
    target: Function to be executed in the thread, default to ``None``
    name: Thread name, default to ``None``
    args: Tuple of arguments to pass to the target function, default to ``()``
    kwargs: Dictionary of keyword arguments to pass to the target function, default to ``None``
    prehook: Function or list of functions to call before thread execution, default to ``None``
    daemon: Whether the thread is a daemon thread, default to ``None``


Examples:
    >>> import lazyllm
    >>> from lazyllm.common.threading import Thread
    >>> import time
    >>> def simple_task(name):
    ...     time.sleep(0.1)
    ...     return f"Hello from {name}"
    >>> thread = Thread(target=simple_task, args=("Worker",))
    >>> thread.start()
    >>> result = thread.get_result()
    >>> print(result)
    Hello from Worker
    >>> def setup_environment():
    ...     print("Setting up environment...")
    ...     return "environment_ready"
    >>> def validate_input(data):
    ...     print(f"Validating input: {data}")
    ...     if not isinstance(data, (int, float)):
    ...         raise ValueError("Input must be numeric")
    >>> def process_data(data):
    ...     print(f"Processing data: {data}")
    ...     time.sleep(0.1) 
    ...     return data * 2
    >>> thread = Thread(
    ...     target=process_data,
    ...     args=(42,),
    ...     prehook=[setup_environment, lambda: validate_input(42)]
    ... )
    >>> thread.start()
    Setting up environment...
    Validating input: 42
    Processing data: 42
    >>> result = thread.get_result()
    >>> print(f"Final result: {result}")
    Final result: 84
    """
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, *, prehook=None, daemon=None):
        self.q = Queue()
        if not isinstance(prehook, (tuple, list)): prehook = [prehook] if prehook else []
        prehook.insert(0, functools.partial(_sid_setter, sid=globals._sid))
        super().__init__(group, self.work, name, (prehook, target, args), kwargs, daemon=daemon)

    def work(self, prehook, target, args, **kw):
        """Core working method of the thread, responsible for executing pre-hook functions, target function, and handling exceptions and results.

Args:
    prehook: List of pre-hook functions to call before thread execution
    target: Target function to execute
    args: Arguments to pass to the target function
    **kw: Keyword arguments to pass to the target function

**Note**: This method is called internally by the `Thread` class, users typically don't need to call this method directly.
"""
        [p() for p in prehook]
        try:
            r = target(*args, **kw)
        except Exception as e:
            self.q.put(e)
        else:
            self.q.put(r)

    def get_result(self):
        """Method to retrieve the thread execution result. This method blocks until the thread execution is complete, then returns the execution result or re-raises the exception.

**Returns:**

- The result of thread execution. If the target function executes normally, returns its return value; if an exception occurs, re-raises that exception.

**Note**: This method should be used after calling `thread.start()` to retrieve the thread execution result.
"""
        r = self.q.get()
        if isinstance(r, Exception):
            raise r
        return r


class ThreadPoolExecutor(TPE):
    def submit(self, fn, /, *args, **kwargs):
        def impl(sid, *a, **kw):
            globals._init_sid(sid)
            return fn(*a, **kw)

        return super(__class__, self).submit(functools.partial(impl, globals._sid), *args, **kwargs)
