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