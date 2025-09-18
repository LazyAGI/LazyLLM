import multiprocessing
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor as PPE
import functools
import time
import atexit
from .utils import load_obj, dump_obj

@contextmanager
def _ctx(method='spawn'):
    m = multiprocessing.get_start_method()
    if m != method:
        multiprocessing.set_start_method(method, force=True)
    yield
    if m != method:
        multiprocessing.set_start_method(m, force=True)


class SpawnProcess(multiprocessing.Process):
    def start(self):
        """
Start the process using spawn method.

This method forces the use of spawn method when starting the process, which creates a brand new Python interpreter process. Spawn is safer than fork, especially in multi-threaded environments.

**Notes:**
- Uses spawn method to start new process, avoiding potential issues with fork
- Temporarily switches to spawn method and restores original method after execution
- Inherits all functionality from multiprocessing.Process.start()


Examples:
    
    ```python
    from lazyllm.common.multiprocessing import SpawnProcess
    
    def worker():
        print("Worker process running")
    
    # Create and start a process using spawn method
    process = SpawnProcess(target=worker)
    process.start()
    process.join()
    ```
    """
        with _ctx('spawn'):
            return super().start()


class ForkProcess(multiprocessing.Process):
    """
Enhanced process class provided by LazyLLM, inheriting from Python's standard library `multiprocessing.Process`. This class specifically uses the fork start method to create child processes and provides support for synchronous/asynchronous execution modes.

Args:
    group: Process group, default to ``None``
    target: Function to be executed in the process, default to ``None``
    name: Process name, default to ``None``
    args: Tuple of arguments to pass to the target function, default to ``()``
    kwargs: Dictionary of keyword arguments to pass to the target function, default to ``{}``
    daemon: Whether the process is a daemon process, default to ``None``
    sync: Whether to use synchronous mode, default to ``True``. In synchronous mode, the process automatically exits after executing the target function; in asynchronous mode, the process continues running until manually terminated.

**Note**: This class is primarily used for LazyLLM's internal process management, especially in long-running server processes.


Examples:
    
    >>> import lazyllm
    >>> from lazyllm.common import ForkProcess
    >>> import time
    >>> import os
    >>> def simple_task(task_id):
    ...     print(f"Process {os.getpid()} executing task {task_id}")
    ...     time.sleep(0.1)  
    ...     return f"Task {task_id} completed by process {os.getpid()}"
    >>> process = ForkProcess(target=simple_task, args=(1,), sync=True)
    >>> process.start()
    Process 12345 executing task 1
    """
    def __init__(self, group=None, target=None, name=None, args=(),
                 kwargs=None, *, daemon=None, sync=True):
        super().__init__(group, ForkProcess.work(target, sync), name, args, kwargs or {}, daemon=daemon)

    @staticmethod
    def work(f, sync):
        """
Core working method of ForkProcess, responsible for wrapping the target function and handling synchronous/asynchronous execution logic.

Args:
    f: Target function to execute
    sync: Whether to use synchronous mode. In synchronous mode, the process exits after executing the target function; in asynchronous mode, the process continues running.
"""
        def impl(*args, **kw):
            try:
                f(*args, **kw)
                if not sync:
                    while True: time.sleep(1)
            finally:
                atexit._run_exitfuncs()
        return impl

    def start(self):
        """
Start the ForkProcess. This method uses the fork start method to create a child process and begin executing the target function.

Features of this method:

- **Fork Start**: Uses fork method to create child processes, providing better performance on Unix/Linux systems
- **Context Management**: Automatically manages the context of process start methods, ensuring the correct start method is used
- **Parent Inheritance**: Inherits all functionality from `multiprocessing.Process.start()`

**Note**: This method actually creates a new process and begins execution, the process starts running immediately after calling.

"""
        with _ctx('fork'):
            return super().start()


def _worker(f):
    return load_obj(f)()

class ProcessPoolExecutor(PPE):
    def submit(self, fn, /, *args, **kwargs):
        """
Submit a task to the process pool for execution.

This method serializes a function and its arguments, then submits them to the process pool for execution. It returns a `Future` object to track the task's status or result.

Args:
    fn (Callable): The function to execute.
    *args: Positional arguments passed to the function.
    **kwargs: Keyword arguments passed to the function.

**Returns:**

- concurrent.futures.Future: A `Future` object representing the task's execution status.


Examples:
    
    >>> from lazyllm.common.multiprocessing import ProcessPoolExecutor
    >>> import time
    >>> 
    >>> def task(x):
    ...     time.sleep(1)
    ...     return x * 2
    ... 
    >>> with ProcessPoolExecutor(max_workers=2) as executor:
    ...     future = executor.submit(task, 5)
    ...     result = future.result()
    ...     print(result)
    10
    """
        f = dump_obj(functools.partial(fn, *args, **kwargs))
        return super(__class__, self).submit(_worker, f)
