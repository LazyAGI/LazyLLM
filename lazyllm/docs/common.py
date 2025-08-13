# flake8: noqa E501
from . import utils
import functools
import lazyllm

add_chinese_doc = functools.partial(utils.add_chinese_doc, module=lazyllm.common)
add_english_doc = functools.partial(utils.add_english_doc, module=lazyllm.common)
add_example = functools.partial(utils.add_example, module=lazyllm.common)

utils.add_doc.__doc__ = 'Add document for lazyllm functions'

add_chinese_doc('Register', '''\
LazyLLM提供的Component的注册机制，可以将任意函数注册成LazyLLM的Component。被注册的函数无需显式的import，即可通过注册器提供的分组机制，在任一位置被索引到。

''')

add_english_doc('Register', '''\
LazyLLM provides a registration mechanism for Components, allowing any function to be registered as a Component of LazyLLM. The registered functions can be indexed at any location through the grouping mechanism provided by the registrar, without the need for explicit import.

<span style="font-size: 18px;">&ensp;**`lazyllm.components.register(cls, *, rewrite_func)→ Decorator`**</span>

After the function is called, it returns a decorator which wraps the decorated function into a Component and registers it in a group named cls.

Args:
    cls (str) :The name of the group to which the function will be registered. The group must exist. Default groups include ``finetune`` and ``deploy``. Users can create new groups by calling the ``new_group`` function.
    rewrite_func (str) :The name of the function to be rewritten after registration. Default is ``apply``. When registering a bash command, you need to pass ``cmd`` as the argument.

**Examples:**\n
```python
>>> import lazyllm
>>> @lazyllm.component_register('mygroup')
... def myfunc(input):
...    return input
...
>>> lazyllm.mygroup.myfunc()(1)
1
```

<span style="font-size: 20px;">&ensp;**`register.cmd(cls)→ Decorator `**</span>

After the function is called, it returns a decorator that wraps the decorated function into a Component and registers it in a group named cls. The wrapped function needs to return an executable bash command.

Args:
    cls (str) :The name of the group to which the function will be registered. The group must exist. Default groups include ``finetune`` and ``deploy``. Users can create new groups by calling the ``new_group`` function.

**Examples:**\n
```python
>>> import lazyllm
>>> @lazyllm.component_register.cmd('mygroup')
... def mycmdfunc(input):
...     return f'echo {input}'
...
>>> lazyllm.mygroup.mycmdfunc()(1)
PID: 2024-06-01 00:00:00 lazyllm INFO: (lazyllm.launcher) Command: echo 1
PID: 2024-06-01 00:00:00 lazyllm INFO: (lazyllm.launcher) PID: 1
```
''')

add_english_doc('Register.new_group', '''\

Creates a new ComponentGroup. The newly created group will be automatically added to __builtin__ and can be accessed at any location without the need for import.

Args:
    group_name (str): The name of the group to be created.
''')

# add_example('Register', '''\
# >>> import lazyllm
# >>> @lazyllm.component_register('mygroup')
# ... def myfunc(input):
# ...     return input
# ''')

add_chinese_doc('compile_func', '''
将一段 python 函数字符串编译成一个可执行函数并返回。

Args:
    func_code (str): 包含 python 函数代码的字符串
    global_env (str): 在 python 函数中用到的包和全局变量
''')

add_english_doc('compile_func', '''
Compile a Python function string into an executable function and return it.

Args:
    func_code (str): A string containing Python function code
    global_env (str): Packages and global variables used in the Python function
''')

add_example('compile_func', '''
from lazyllm.common import compile_func
code_str = 'def Identity(v): return v'
identity = compile_func(code_str)
assert identity('hello') == 'hello'
''')

add_chinese_doc('ReadOnlyWrapper', '''
一个轻量级只读包装器，用于包裹任意对象并对外提供只读访问（实际并未完全禁止修改，但复制时不会携带原始对象）。包装器可以动态替换内部对象，并提供判断对象是否为空的辅助方法。
# ============= Threading
# Thread
add_chinese_doc('Thread', '''\
LazyLLM 提供的增强线程类，继承自 Python 标准库的 `threading.Thread`。此类提供了额外的功能，包括会话ID管理、预钩子函数支持和异常处理机制。

Args:
    group: 线程组，默认为 ``None``
    target: 要在线程中执行的函数，默认为 ``None``
    name: 线程名称，默认为 ``None``
    args: 传递给目标函数的参数元组，默认为 ``()``
    kwargs: 传递给目标函数的关键字参数字典，默认为 ``None``
    prehook: 在线程执行前要调用的函数或函数列表，默认为 ``None``
    daemon: 是否为守护线程，默认为 ``None``
''')

add_english_doc('Thread', '''\
Enhanced thread class provided by LazyLLM, inheriting from Python's standard library `threading.Thread`. This class provides additional functionality including session ID management, pre-hook function support, and exception handling mechanisms.

Args:
    group: Thread group, default to ``None``
    target: Function to be executed in the thread, default to ``None``
    name: Thread name, default to ``None``
    args: Tuple of arguments to pass to the target function, default to ``()``
    kwargs: Dictionary of keyword arguments to pass to the target function, default to ``None``
    prehook: Function or list of functions to call before thread execution, default to ``None``
    daemon: Whether the thread is a daemon thread, default to ``None``
''')

add_example('Thread', '''\
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
''')

# Thread.work
add_chinese_doc('Thread.work', '''\
线程的核心工作方法，负责执行预钩子函数、目标函数，并处理异常和结果。

Args:
    prehook: 预钩子函数列表，在线程执行前调用
    target: 要执行的目标函数
    args: 传递给目标函数的参数
    **kw: 传递给目标函数的关键字参数

**注意**: 此方法由 `Thread` 类内部调用，用户通常不需要直接调用此方法。
''')

add_english_doc('Thread.work', '''\
Core working method of the thread, responsible for executing pre-hook functions, target function, and handling exceptions and results.

Args:
    prehook: List of pre-hook functions to call before thread execution
    target: Target function to execute
    args: Arguments to pass to the target function
    **kw: Keyword arguments to pass to the target function

**Note**: This method is called internally by the `Thread` class, users typically don't need to call this method directly.
''')

# Thread.get_result
add_chinese_doc('Thread.get_result', '''\
获取线程执行结果的方法。此方法会阻塞直到线程执行完成，然后返回执行结果或重新抛出异常。

**Returns:**\n
- 线程执行的结果。如果目标函数正常执行，返回其返回值；如果发生异常，会重新抛出该异常。

**注意**: 此方法应该在调用 `thread.start()` 之后使用，用于获取线程的执行结果。
''')

add_english_doc('Thread.get_result', '''\
Method to retrieve the thread execution result. This method blocks until the thread execution is complete, then returns the execution result or re-raises the exception.

**Returns:**\n
- The result of thread execution. If the target function executes normally, returns its return value; if an exception occurs, re-raises that exception.

**Note**: This method should be used after calling `thread.start()` to retrieve the thread execution result.
''')
# ============= Bind/bind
add_chinese_doc('bind', '''\
Bind 类用于函数绑定与延迟调用，支持动态参数传入和上下文参数解析，实现灵活的函数组合与流水线式调用。

bind 函数能够将一个函数与固定的位置参数和关键字参数绑定，支持使用占位符（如 _0, _1）引用当前数据流中上游节点的输出，实现数据在流水线中的跳跃传递和灵活组合。

注意事项：
    - 绑定的参数可以是具体值，也可以是当前数据流中上游节点的输出占位符。
    - 参数绑定仅在当前数据流上下文内生效，不能跨数据流绑定或绑定外部变量。

Args:
    obj (Optional[Any]): 初始被包装的对象，默认为 None。
''')

add_english_doc('ReadOnlyWrapper', '''
A lightweight read-only wrapper that holds an arbitrary object and exposes its attributes. It supports swapping the internal object dynamically and provides utility for checking emptiness. Note: it does not enforce deep immutability, but deepcopy drops the wrapped object.
Args:
    obj (Optional[Any]): The initial wrapped object, defaults to None.
''')

add_chinese_doc('ReadOnlyWrapper.set', '''
替换当前包装的内部对象。

Args:
    obj (Any): 新的内部对象。
''')

add_english_doc('ReadOnlyWrapper.set', '''
Replace the currently wrapped internal object.

Args:
    obj (Any): New object to wrap.
''')

add_chinese_doc('ReadOnlyWrapper.isNone', '''
检查当前包装器是否未持有任何对象。

Args:
    None.

**Returns**\n
- bool: 如果内部对象为 None 返回 True，否则 False。
''')

add_english_doc('ReadOnlyWrapper.isNone', '''
Check whether the wrapper currently holds no object.

Args:
    None.

**Returns**\n
- bool: True if the internal object is None, otherwise False.
''')

add_chinese_doc('RedisQueue', '''
基于 Redis 实现的文件系统队列（继承自 FileSystemQueue），用于跨进程/节点的消息传递与队列管理。内部使用指定的 redis_url 初始化并管理底层存储，同时提供线程安全的初始化逻辑。

Args:
    klass (str): 队列的分类名称，用于区分不同队列实例，默认值为 '__default__'。
''')

add_english_doc('RedisQueue', '''
Redis-backed file system queue (inherits from FileSystemQueue) for cross-process/node message passing and queue management. It initializes its underlying storage using a configured Redis URL and employs thread-safe setup logic.

Args:
    klass (str): Classification name for the queue instance to distinguish different queues. Defaults to '__default__'.
    limit (int, optional): Maximum number of messages to retrieve at once. If None, retrieves all messages. Defaults to None.

**Returns:**\n
- list: List of retrieved messages.
""")

add_example('FileSystemQueue.dequeue', """\
>>> import lazyllm
>>> queue = lazyllm.FileSystemQueue(klass='dequeue_test')
>>> for i in range(5):
...     queue.enqueue(f"Message{i}")
>>> all_messages = queue.dequeue()
>>> all_messages
['Message0', 'Message1', 'Message2', 'Message3', 'Message4']
""")

add_chinese_doc('FileSystemQueue.peek', """\
查看队列头部的消息但不移除。

此方法允许查看队列中最早的消息，但不会将其从队列中移除。

**Returns:**\n
- 队列头部的消息，如果队列为空则返回None。
""")

add_english_doc('FileSystemQueue.peek', """\
View the message at the head of the queue without removing it.

This method allows viewing the earliest message in the queue without removing it from the queue.

**Returns:**\n
- The message at the head of the queue, or None if the queue is empty.
""")

add_example('FileSystemQueue.peek', """\
>>> import lazyllm
>>> queue = lazyllm.FileSystemQueue(klass='peek_test')
>>> queue.enqueue("First message")
>>> queue.enqueue("Second message")
>>> first_message = queue.peek()
>>> first_message
'First message'
>>> queue.peek()
'First message'
""")

add_chinese_doc('FileSystemQueue.size', """\
获取队列大小。

此方法返回当前队列中的消息数量。

**Returns:**\n
- int: 队列中的消息数量。
""")

add_english_doc('FileSystemQueue.size', """\
Get the queue size.

This method returns the current number of messages in the queue.

**Returns:**\n
- int: Number of messages in the queue.
""")

add_example('FileSystemQueue.size', """\
>>> import lazyllm
>>> queue = lazyllm.FileSystemQueue(klass='size_test')
>>> queue.size()
0
>>> queue.enqueue("Message1")
>>> queue.size()
1
>>> queue.enqueue("Message2")
>>> queue.size()
2
>>> queue.dequeue()
['Message1', 'Message2']
>>> queue.size()
0
""")

add_chinese_doc('FileSystemQueue.clear', """\
清空队列。

此方法移除队列中的所有消息，将队列重置为空状态。
""")

add_english_doc('FileSystemQueue.clear', """\
Clear the queue.

This method removes all messages from the queue, resetting it to an empty state.
""")

add_example('FileSystemQueue.clear', """\
>>> import lazyllm
>>> queue = lazyllm.FileSystemQueue(klass='clear_test')
>>> for i in range(10):
...     queue.enqueue(f"Message{i}")
>>> queue.size()
10
>>> queue.clear()
>>> queue.size()
0
>>> queue.peek() is None
True
""")

add_chinese_doc('multiprocessing.SpawnProcess.start', '''
使用spawn方式启动进程。

此方法在启动进程时强制使用spawn方式，这种方式会创建一个全新的Python解释器进程。spawn方式相比fork更安全，特别是在多线程环境下。

**说明:**
- 使用spawn方式启动新进程，避免了fork可能带来的问题
- 会临时切换启动方式为spawn，执行完后恢复原有启动方式
- 继承自multiprocessing.Process.start()的所有功能
''')

add_english_doc('multiprocessing.SpawnProcess.start', '''
Start the process using spawn method.

This method forces the use of spawn method when starting the process, which creates a brand new Python interpreter process. Spawn is safer than fork, especially in multi-threaded environments.

**Notes:**
- Uses spawn method to start new process, avoiding potential issues with fork
- Temporarily switches to spawn method and restores original method after execution
- Inherits all functionality from multiprocessing.Process.start()
''')

add_example('multiprocessing.SpawnProcess.start', '''
```python
from lazyllm.common.multiprocessing import SpawnProcess

def worker():
    print("Worker process running")

# Create and start a process using spawn method
process = SpawnProcess(target=worker)
process.start()
process.join()
```
''')

add_chinese_doc('LazyLLMCMD', '''\
命令行操作封装类，提供安全、灵活的命令行管理功能。

Args:
    cmd (Union[str, List[str], Callable]):命令行指令，支持三种形式：字符串命令,命令列表,可调用对象。
    return_value (Any):预设返回值。
    checkf(Any):命令验证函数。
    no_displays(Any):需要过滤的敏感参数名。
''')

add_english_doc('LazyLLMCMD', '''\
Command line operation wrapper class providing secure and flexible command management.

Args:
    cmd (Union[str, List[str], Callable]):Command input, supports three formats:String command,Command list,Callable object.
    return_value (Any):Preset return value.
    checkf(Any):Command validation function with signature.
    no_displays(Any):Sensitive parameter names to filter.

''')

add_example('LazyLLMCMD', '''\
>>> from lazyllm.common import LazyLLMCMD
>>> cmd = LazyLLMCMD("run --epochs=50 --batch-size=32")
>>> print(cmd.get_args("epochs"))
50
>>> print(cmd.get_args("batch-size")) 
32
>>> base = LazyLLMCMD("python train.py", checkf=lambda x: True)
>>> new = base.with_cmd("python predict.py")

''')

add_chinese_doc('LazyLLMCMD.with_cmd', '''\
创建新命令对象并继承当前配置。

参数:
    cmd: 新的命令内容（类型需与原始命令一致）

''')

add_english_doc('LazyLLMCMD.with_cmd', '''\
Create new command object inheriting current configuration.

Args:
    cmd: New command content (must be same type as original)

''')

add_chinese_doc('LazyLLMCMD.get_args', '''\
从命令字符串中提取指定参数的值。

参数:
    key: 要提取的参数名
''')

add_english_doc('LazyLLMCMD.get_args', '''\
Extracts specified argument value from command string.

Args:
    key: Argument name
''')

add_chinese_doc('queue.SQLiteQueue', '''\
基于 SQLite 的持久化文件系统队列。
该类扩展自 FileSystemQueue，使用 SQLite 数据库存储队列数据，通过 position 字段保证先进先出顺序，并支持并发安全的消息入队、出队、查看队头、队列大小查询和清空操作。
队列数据库默认存储在 ~/.lazyllm_filesystem_queue.db，通过文件锁机制确保多进程安全访问。
Args:
    klass (str): 队列分类名，用于逻辑隔离不同的队列，默认为 '__default__'。
''')

add_english_doc('queue.SQLiteQueue', '''\
Persistent file system queue backed by SQLite.
This class extends FileSystemQueue and stores queue data in an SQLite database. Messages are ordered by a position field to preserve FIFO behavior. The class supports concurrent-safe operations including enqueue, dequeue, peek, size checking, and clearing the queue.
The queue database is saved at ~/.lazyllm_filesystem_queue.db, with a file lock mechanism ensuring safe access in multi-process environments.
Args:
    klass (str): Name of the queue category used to logically separate queues. Default is '__default__'.
''')
