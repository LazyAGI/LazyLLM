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

add_chinese_doc('DynamicDescriptor', '''\
动态描述符类，实现了根据访问对象的不同动态绑定方法调用。  
通过封装函数，支持当描述符被访问时，根据是通过实例还是类访问，调用对应的方法。

Args:
    func (Callable): 作为描述符的函数。

''')

add_english_doc('DynamicDescriptor', '''\
A dynamic descriptor class that implements method binding dynamically based on the access object.  
By wrapping a function, it supports calling the method on the instance or the class depending on how the descriptor is accessed.

Args:
    func (Callable): The function to be used as a descriptor.

''')

add_chinese_doc('DynamicDescriptor.Impl', '''\
动态描述符的内部实现类，封装了函数及其访问上下文。  
根据访问者是实例还是类，调用对应的方法。

Args:
    func (Callable): 被封装的函数。
    instance (Optional[Any]): 调用该描述符的实例，若无则为 None。
    owner (type): 描述符所属的类。

**Returns:**\n
- 调用结果，与封装函数的返回值一致。
''')

add_english_doc('DynamicDescriptor.Impl', '''\
Internal implementation class of the dynamic descriptor, encapsulating the function and its context.  
Calls the function bound to either the instance or the class depending on the accessor.

Args:
    func (Callable): The wrapped function.
    instance (Optional[Any]): The instance accessing the descriptor, or None if accessed via class.
    owner (type): The class owning the descriptor.

**Returns:**\n
- The result of the function call, consistent with the wrapped function's return value.
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
