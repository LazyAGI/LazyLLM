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

add_chinese_doc('registry.LazyDict', '''\
一个为懒惰的程序员设计的特殊字典类。支持多种便捷的访问和操作方式。

特性：
1. 使用点号代替['str']访问字典元素
2. 支持首字母小写来使语句更像函数调用
3. 当字典只有一个元素时支持直接调用
4. 支持动态默认键
5. 如果组名出现在名称中，允许省略组名

参数:
    name (str): 字典的名称，默认为空字符串。
    base: 基类引用，默认为None。
    *args: 位置参数，传递给dict父类。
    **kw: 关键字参数，传递给dict父类。
''')

add_english_doc('registry.LazyDict', '''\
A special dictionary class designed for lazy programmers. Supports various convenient access and operation methods.

Features:
1. Use dot notation instead of ['str'] to access dictionary elements
2. Support lowercase first character to make statements more like function calls
3. Support direct calls when dictionary has only one element
4. Support dynamic default keys
5. Allow omitting group name if it appears in the name

Args:
    name (str): Name of the dictionary, defaults to empty string.
    base: Base class reference, defaults to None.
    *args: Positional arguments passed to dict parent class.
    **kw: Keyword arguments passed to dict parent class.
''')

add_chinese_doc('registry.LazyDict.remove', '''\
从字典中移除指定的键值对。

参数:
    key (str): 要移除的键。支持与__getattr__相同的键匹配规则，包括首字母小写和组名省略等特性。

注意:
    如果找不到匹配的键，将抛出AttributeError异常。
''')

add_english_doc('registry.LazyDict.remove', '''\
Remove the specified key-value pair from the dictionary.

Args:
    key (str): The key to remove. Supports the same key matching rules as __getattr__, 
              including lowercase first character and group name omission features.

Note:
    Raises AttributeError if no matching key is found.
''')

add_chinese_doc('registry.LazyDict.set_default', '''\
设置字典的默认键。设置后可以通过.default属性访问该键对应的值。

参数:
    key (str): 要设置为默认的键名。

注意:
    - key必须是字符串类型
    - 设置后可以通过.default访问，或在字典只有一个元素时直接调用
''')

add_english_doc('registry.LazyDict.set_default', '''\
Set the default key for the dictionary. After setting, the value can be accessed through the .default property.

Args:
    key (str): The key name to set as default.

Note:
    - key must be a string type
    - After setting, can be accessed via .default, or called directly when dictionary has only one element
''')

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
    __bind_func (Callable 或 type): 要绑定的函数或函数类型，传入类型时会自动实例化。
    *args: 绑定时固定的位置参数，可以包含占位符。
    **kw: 绑定时固定的关键字参数，可以包含占位符。
''')

add_chinese_doc('common.CaseInsensitiveDict', '''\
大小写不敏感的字典类。

CaseInsensitiveDict 继承自 dict，提供大小写不敏感的键值存储和检索功能。所有的键都会被转换为小写形式存储，确保无论使用大写、小写或混合大小写的键名都能访问到相同的值。

特点：
    - 所有键在存储时自动转换为小写
    - 支持标准的字典操作（获取、设置、检查包含关系）
    - 保持字典的原有功能，只是键名处理方式不同

Args:
    *args: 传递给父类 dict 的位置参数
    **kwargs: 传递给父类 dict 的关键字参数
''')

add_english_doc('common.CaseInsensitiveDict', '''\
Case-insensitive dictionary class.

CaseInsensitiveDict inherits from dict and provides case-insensitive key-value storage and retrieval. All keys are converted to lowercase when stored, ensuring that values can be accessed regardless of whether the key name is uppercase, lowercase, or mixed case.

Features:
    - All keys are automatically converted to lowercase when stored
    - Supports standard dictionary operations (get, set, check containment)
    - Maintains all original dict functionality, only differs in key name handling

Args:
    *args: Positional arguments passed to the parent dict class
    **kwargs: Keyword arguments passed to the parent dict class
''')

add_example('common.CaseInsensitiveDict', '''\
>>> from lazyllm.common import CaseInsensitiveDict
>>> # 创建大小写不敏感的字典
>>> d = CaseInsensitiveDict({'Name': 'John', 'AGE': 25, 'City': 'New York'})
>>> 
>>> # 使用不同大小写访问相同的键
>>> print(d['name'])      # 使用小写
... 'John'
>>> print(d['NAME'])      # 使用大写
... 'John'
>>> print(d['Name'])      # 使用首字母大写
... 'John'
>>> 
>>> # 设置值时也会转换为小写
>>> d['EMAIL'] = 'john@example.com'
>>> print(d['email'])     # 使用小写访问
... 'john@example.com'
>>> 
>>> # 检查键是否存在（大小写不敏感）
>>> 'AGE' in d
True
>>> 'age' in d
True
>>> 'Age' in d
True
>>> 
>>> # 支持标准字典操作
>>> d['PHONE'] = '123-456-7890'
>>> print(d.get('phone'))
... '123-456-7890'
>>> print(len(d))
... 5
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


add_chinese_doc('common.ResultCollector', '''\
结果收集器，用于在流程或任务执行过程中按名称存储和访问结果。  
它通过调用自身（传入 name）返回一个可调用的 Impl 对象来收集指定名称的结果。  
适用于需要跨步骤共享中间结果的场景。
''')

add_english_doc('common.ResultCollector', '''\
A result collector used to store and access results by name during the execution of a flow or task.  
Calling the instance with a name returns a callable Impl object that collects results for that name.  
Useful for scenarios where intermediate results need to be shared across steps.
''')
add_chinese_doc('common.ResultCollector.Impl', '''\
ResultCollector 的内部实现类，负责为指定名称收集结果。  
不应直接实例化，需通过 ResultCollector(name) 获取。

Args:
    name (str): 结果名称。
    value (dict): 存储结果的字典引用。
''')

add_english_doc('common.ResultCollector.Impl', '''\
Internal implementation class of ResultCollector, responsible for collecting results for a given name.  
Should not be instantiated directly; obtain via ResultCollector(name).

Args:
    name (str): The result name.
    value (dict): A reference to the dictionary where results are stored.
''')


add_chinese_doc('common.ResultCollector.keys', '''\
获取所有已存储结果的名称。

**Returns**\n
- KeysView[str]: 结果名称集合。
''')

add_english_doc('common.ResultCollector.keys', '''\
Get all stored result names.

**Returns**\n
- KeysView[str]: A set-like object containing result names.
''')

add_chinese_doc('common.ResultCollector.items', '''\
获取所有已存储的 (名称, 值) 对。

**Returns**\n
- ItemsView[str, Any]: 结果的键值对集合。
''')

add_english_doc('common.ResultCollector.items', '''\
Get all stored (name, value) pairs.

**Returns**\n
- ItemsView[str, Any]: A set-like object containing name-value pairs of results.
''')

add_chinese_doc('common.EnvVarContextManager', '''\
环境变量上下文管理器，用于 在代码块执行期间临时设置环境变量，退出时自动恢复原始环境变量。

Args:
    env_vars_dict (dict): 需要临时设置的环境变量字典，值为 None 的变量将被忽略。
''')

add_english_doc('common.EnvVarContextManager', '''\
Environment variable context manager used to temporarily set environment variables during the execution of a code block, automatically restoring original environment variables upon exit.

Args:
    env_vars_dict (dict): Dictionary of environment variables to temporarily set; variables with None values are ignored.
''')

add_chinese_doc('ReadOnlyWrapper', '''\ 
一个轻量级只读包装器，用于包裹任意对象并对外提供只读访问（实际并未完全禁止修改，但复制时不会携带原始对象）。包装器可以动态替换内部对象，并提供判断对象是否为空的辅助方法。
Args:
    obj (Optional[Any]): 初始被包装的对象，默认为 None。
''')

add_english_doc('ReadOnlyWrapper', '''\
A lightweight read-only wrapper that holds an arbitrary object and exposes its attributes. It supports swapping the internal object dynamically and provides utility for checking emptiness. Note: it does not enforce deep immutability, but deepcopy drops the wrapped object.
Args:
    obj (Optional[Any]): The initial wrapped object, defaults to None.
''')

add_chinese_doc('ReadOnlyWrapper.set', '''\ 
替换当前包装的内部对象。

Args:
    obj (Any): 新的内部对象。
''')

add_english_doc('ReadOnlyWrapper.set', '''\
Replace the currently wrapped internal object.

Args:
    obj (Any): New object to wrap.
''')

add_chinese_doc('ReadOnlyWrapper.isNone', '''\ 
检查当前包装器是否未持有任何对象。

Args:
    None.

**Returns**\n
- bool: 如果内部对象为 None 返回 True，否则 False。
''')

add_english_doc('ReadOnlyWrapper.isNone', '''\
Check whether the wrapper currently holds no object.

Args:
    None.

**Returns**\n
- bool: True if the internal object is None, otherwise False.
''')

add_chinese_doc('queue.RedisQueue', '''\ 
基于 Redis 实现的文件系统队列（继承自 FileSystemQueue），用于跨进程/节点的消息传递与队列管理。内部使用指定的 redis_url 初始化并管理底层存储，同时提供线程安全的初始化逻辑。

Args:
    klass (str): 队列的分类名称，用于区分不同队列实例，默认值为 '__default__'。
''')

add_english_doc('queue.RedisQueue', '''\
Redis-backed file system queue (inherits from FileSystemQueue) for cross-process/node message passing and queue management. It initializes its underlying storage using a configured Redis URL and employs thread-safe setup logic.

Args:
    klass (str): Classification name for the queue instance to distinguish different queues. Defaults to '__default__'.
''')


add_chinese_doc('Identity', '''\
恒等模块，用于直接返回输入值。

该模块常用于模块拼接结构中占位，无实际处理逻辑。若输入为多个参数，将自动打包为一个整体结构输出。

Args:
    *args: 可选的位置参数，占位用。
    **kw: 可选的关键字参数，占位用。
''')

add_english_doc('Identity', '''\
Identity module that directly returns the input as output.

This module serves as a no-op placeholder in composition pipelines. If multiple inputs are provided, they are packed together before returning.

Args:
    *args: Optional positional arguments for placeholder compatibility.
    **kw: Optional keyword arguments for placeholder compatibility.
''')



add_chinese_doc('ProcessPoolExecutor.submit', '''\
将任务提交到进程池中执行。

此方法将一个函数及其参数序列化后提交到进程池中执行，返回一个 `Future` 对象，用于获取任务执行结果或状态。

Args:
    fn (Callable): 要执行的函数。
    *args: 传递给函数的位置参数。
    **kwargs: 传递给函数的关键字参数。

Returns:
    concurrent.futures.Future: 表示任务执行状态的 `Future` 对象。
''')

add_english_doc('ProcessPoolExecutor.submit', '''\
Submit a task to the process pool for execution.

This method serializes a function and its arguments, then submits them to the process pool for execution. It returns a `Future` object to track the task's status or result.

Args:
    fn (Callable): The function to execute.
    *args: Positional arguments passed to the function.
    **kwargs: Keyword arguments passed to the function.

Returns:
    concurrent.futures.Future: A `Future` object representing the task's execution status.
''')

add_example('ProcessPoolExecutor.submit', '''\
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
''')


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
