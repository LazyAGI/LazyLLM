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
