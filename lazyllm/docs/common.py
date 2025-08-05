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
''')
