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

# ============= Bind/bind
add_chinese_doc('bind', '''\
Bind 类用于函数绑定与延迟调用，支持动态参数传入和上下文参数解析，实现灵活的函数组合与流水线式调用。

bind 函数能够将一个函数与固定的位置参数和关键字参数绑定，支持使用占位符（如 _0, _1）引用当前数据流中上游节点的输出，实现数据在流水线中的跳跃传递和灵活组合。

注意事项：
    - 绑定的参数可以是具体值，也可以是当前数据流中上游节点的输出占位符。
    - 参数绑定仅在当前数据流上下文内生效，不能跨数据流绑定或绑定外部变量。

Args:
    __bind_func (Callable 或 type): 要绑定的函数或函数类型，传入类型时会自动实例化。
    *args: 绑定时固定的位置参数，可以包含占位符。
    **kw: 绑定时固定的关键字参数，可以包含占位符。
''')

add_english_doc('bind', '''\
The Bind class provides function binding and deferred invocation capabilities, supporting dynamic argument passing and context-based argument resolution for flexible function composition and pipeline-style calls.

The bind function binds a callable with fixed positional and keyword arguments, supporting placeholders (e.g. _0, _1) to reference outputs of upstream nodes within the current pipeline, enabling flexible data jumps and function composition.

Notes:
    - Bound arguments can be concrete values or placeholders referring to upstream pipeline outputs.
    - Bindings are local to the current pipeline context and do not support cross-pipeline or external variable binding.

Args:
    __bind_func (Callable or type): The function or function type to bind. If a type is given, it will be instantiated automatically.
    *args: Fixed positional arguments to bind, supporting placeholders.
    **kw: Fixed keyword arguments to bind, supporting placeholders.
''')

add_example('bind', '''\
>>> from lazyllm import bind, _0, _1
>>> def f1(x):
...     return x ** 2
>>> def f21(input1, input2=0):
...     return input1 + input2 + 1
>>> def f22(input1, input2=0):
...     return input1 + input2 - 1
>>> def f3(in1='placeholder1', in2='placeholder2', in3='placeholder3'):
...     return f"get [input:{in1}], [f21:{in2}], [f22:{in3}]"

>>> from lazyllm import pipeline, parallel

>>> with pipeline() as ppl:
...     ppl.f1 = f1
...     with parallel() as ppl.subprl2:
...         ppl.subprl2.path1 = f21
...         ppl.subprl2.path2 = f22
...     ppl.f3 = bind(f3, ppl.input, _0, _1)
...
>>> print(ppl(2))
get [input:2], [f21:5], [f22:3]

>>> # Demonstrate operator '|' overloading for bind
>>> with pipeline() as ppl2:
...     ppl2.f1 = f1
...     with parallel().bind(ppl2.input, _0) as ppl2.subprl2:
...         ppl2.subprl2.path1 = f21
...         ppl2.subprl2.path2 = f22
...     ppl2.f3 = f3 | bind(ppl2.input, _0, _1)
...
>>> print(ppl2(2))
get [input:2], [f21:7], [f22:5]
''')

# ============= package
add_chinese_doc('package', '''\
package类用于封装流水线或并行模块的返回值，保证传递给下游模块时自动拆包，从而支持多个值的灵活传递。
''')

add_english_doc('package', '''\
The package class is used to encapsulate the return values of pipeline or parallel modules,
ensuring automatic unpacking when passing to the next module, thereby supporting flexible multi-value passing.
''')

add_example('package', '''\
>>> from lazyllm.common import package
>>> p = package(1, 2, 3)
>>> p
(1, 2, 3)
>>> p[1]
2
>>> p_slice = p[1:]
>>> isinstance(p_slice, package)
True
>>> p2 = package([4, 5])
>>> p + p2
(1, 2, 3, 4, 5)
''')

add_chinese_doc('FileSystemQueue', """\
基于文件系统的队列抽象基类。

FileSystemQueue是一个抽象基类，提供了基于文件系统的队列操作接口。它支持多种后端实现（如SQLite、Redis），用于在分布式环境中进行消息传递和数据流控制。

该类实现了单例模式，确保每个类名只有一个队列实例，并提供了线程安全的队列操作。

Args:
    klass (str, optional): 队列的类名标识符。默认为 ``'__default__'``。

**Returns:**\n
- FileSystemQueue: 队列实例（单例模式）
""")

add_english_doc('FileSystemQueue', """\
Abstract base class for file system-based queues.

FileSystemQueue is an abstract base class that provides a file system-based queue operation interface. It supports multiple backend implementations (such as SQLite, Redis) for message passing and data flow control in distributed environments.

This class implements the singleton pattern, ensuring only one queue instance per class name, and provides thread-safe queue operations.

Args:
    klass (str, optional): Class name identifier for the queue. Defaults to ``'__default__'``.

**Returns:**\n
- FileSystemQueue: Queue instance (singleton pattern)
""")

add_chinese_doc('FileSystemQueue.enqueue', """\
将消息加入队列。

此方法将指定的消息添加到队列的尾部，遵循先进先出（FIFO）的原则。

Args:
    message: 要加入队列的消息内容。
""")

add_english_doc('FileSystemQueue.enqueue', """\
Add a message to the queue.

This method adds the specified message to the tail of the queue, following the First-In-First-Out (FIFO) principle.

Args:
    message: The message content to be added to the queue.
""")

add_example('FileSystemQueue.enqueue', """\
>>> import lazyllm
>>> queue = lazyllm.FileSystemQueue(klass='enqueue_test')
>>> queue.enqueue(123)
>>> queue.peek()
'123'
""")

add_chinese_doc('FileSystemQueue.dequeue', """\
从队列中取出消息。

此方法从队列头部取出消息并移除它们，可以指定一次取出的消息数量。

Args:
    limit (int, optional): 一次取出的最大消息数量。如果为None，则取出所有消息。默认为None。

**Returns:**\n
- list: 取出的消息列表。
""")

add_english_doc('FileSystemQueue.dequeue', """\
Retrieve messages from the queue.

This method retrieves messages from the head of the queue and removes them, with the option to specify the number of messages to retrieve at once.

Args:
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

# ============= Multiprocessing
# ForkProcess
add_chinese_doc('ForkProcess', '''\
LazyLLM 提供的增强进程类，继承自 Python 标准库的 `multiprocessing.Process`。此类专门使用 fork 启动方法来创建子进程，并提供了同步/异步执行模式的支持。

Args:
    group: 进程组，默认为 ``None``
    target: 要在进程中执行的函数，默认为 ``None``
    name: 进程名称，默认为 ``None``
    args: 传递给目标函数的参数元组，默认为 ``()``
    kwargs: 传递给目标函数的关键字参数字典，默认为 ``{}``
    daemon: 是否为守护进程，默认为 ``None``
    sync: 是否为同步模式，默认为 ``True``。在同步模式下，进程执行完目标函数后会自动退出；在异步模式下，进程会持续运行直到被手动终止。

**注意**: 此类主要用于 LazyLLM 内部的进程管理，特别是在需要长期运行的服务器进程中。
''')

add_english_doc('ForkProcess', '''\
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
''')

add_example('ForkProcess', '''\
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
''')

# ForkProcess.work
add_chinese_doc('ForkProcess.work', '''\
ForkProcess 的核心工作方法，负责包装目标函数并处理同步/异步执行逻辑。

Args:
    f: 要执行的目标函数
    sync: 是否为同步模式。在同步模式下，执行完目标函数后进程会退出；在异步模式下，进程会持续运行。
''')

add_english_doc('ForkProcess.work', '''\
Core working method of ForkProcess, responsible for wrapping the target function and handling synchronous/asynchronous execution logic.

Args:
    f: Target function to execute
    sync: Whether to use synchronous mode. In synchronous mode, the process exits after executing the target function; in asynchronous mode, the process continues running.
''')

# ForkProcess.start
add_chinese_doc('ForkProcess.start', '''\
启动 ForkProcess 进程。此方法会使用 fork 启动方法来创建子进程，并开始执行目标函数。

此方法的特点：

- **Fork 启动**: 使用 fork 方法创建子进程，在 Unix/Linux 系统上提供更好的性能
- **上下文管理**: 自动管理进程启动方法的上下文，确保使用正确的启动方式
- **继承父类**: 继承自 `multiprocessing.Process.start()` 的所有功能

**注意**: 此方法会实际创建新的进程并开始执行，调用后进程会立即开始运行。

''')

add_english_doc('ForkProcess.start', '''\
Start the ForkProcess. This method uses the fork start method to create a child process and begin executing the target function.

Features of this method:

- **Fork Start**: Uses fork method to create child processes, providing better performance on Unix/Linux systems
- **Context Management**: Automatically manages the context of process start methods, ensuring the correct start method is used
- **Parent Inheritance**: Inherits all functionality from `multiprocessing.Process.start()`

**Note**: This method actually creates a new process and begins execution, the process starts running immediately after calling.

''')

# ============= Options
# Option
add_chinese_doc('Option', '''\
LazyLLM 提供的选项管理类，用于管理多个选项值并在它们之间进行迭代。此类主要用于参数网格搜索和超参数调优场景。

Args:
    *obj: 一个或多个选项值，可以是任意类型的对象。如果传入单个列表或元组，会自动展开。

此类的主要特性：

- **多选项管理**: 可以管理多个不同的选项值
- **迭代支持**: 支持标准的 Python 迭代协议，可以遍历所有选项
- **当前值访问**: 始终可以访问当前选中的选项值
- **深度复制**: 支持深度复制当前选中的选项值
- **多进程兼容**: 支持在多进程环境中使用

**注意**: 此类主要用于 LazyLLM 内部的参数搜索和试验管理，特别是在 TrialModule 中进行参数网格搜索时。

''')

add_english_doc('Option', '''\
Option management class provided by LazyLLM, used for managing multiple option values and iterating between them. This class is primarily used for parameter grid search and hyperparameter tuning scenarios.

Args:
    *obj: One or more option values, which can be objects of any type. If a single list or tuple is passed, it will be automatically expanded.

Key features of this class:

- **Multi-option Management**: Can manage multiple different option values
- **Iteration Support**: Supports standard Python iteration protocol, can iterate through all options
- **Current Value Access**: Always can access the currently selected option value
- **Deep Copy**: Supports deep copying of the currently selected option value
- **Multi-process Compatibility**: Supports usage in multi-process environments

**Note**: This class is primarily used for LazyLLM's internal parameter search and trial management, especially in TrialModule for parameter grid search.

''')

add_example('Option', '''\
>>> import lazyllm
>>> from lazyllm.common.option import Option
>>> learning_rates = Option(0.001, 0.01, 0.1)
>>> print(f"当前学习率: {learning_rates}")
当前学习率: <Option options="(0.001, 0.01, 0.1)" curr="0.001">
>>> print(f"所有选项: {list(learning_rates)}")
所有选项: [0.001, 0.01, 0.1]
''')
