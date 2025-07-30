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

add_chinese_doc('SQLiteQueue', '''\
基于 SQLite 的持久化文件系统队列。

该类继承自 FileSystemQueue，并通过 SQLite 数据库存储队列内容，支持并发访问控制与消息顺序管理。每条队列消息按照 position 字段顺序排列，并提供入队、出队、查看、统计及清空操作。

数据库默认存储在 ~/.lazyllm_filesystem_queue.db，写操作使用文件锁防止并发冲突。

Args:
    klass (str): 队列类别名称，用于区分不同逻辑队列，默认为 '__default__'。
''')

add_english_doc('SQLiteQueue', '''\
Persistent file system queue backed by SQLite.

This class extends FileSystemQueue and stores queue entries in an SQLite database with ordered message positions. It supports concurrent-safe enqueue, dequeue, peek, count, and clear operations.

The queue database is saved at ~/.lazyllm_filesystem_queue.db, and file locking ensures safe concurrent access.

Args:
    klass (str): Name of the queue category, used to separate logical queues. Default is '__default__'.
''')

add_example('SQLiteQueue', ['''\
>>> from lazyllm.components import SQLiteQueue
>>> queue = SQLiteQueue(klass='demo')

>>> # Enqueue messages
>>> queue._enqueue('session1', 'Hello')
>>> queue._enqueue('session1', 'World')

>>> # Peek at the first message without removing
>>> print(queue._peek('session1'))
... 'Hello'

>>> # Dequeue messages
>>> messages = queue._dequeue('session1', limit=2)
>>> print(messages)
... ['Hello', 'World']

>>> # Check queue size (should be 0)
>>> print(queue._size('session1'))
... 0

>>> # Clear the queue (safe even if empty)
>>> queue._clear('session1')
'''])
