# flake8: noqa E501
from . import utils
import functools
import lazyllm

# ============= Hook

add_chinese_doc = functools.partial(utils.add_chinese_doc, module=lazyllm.hook)
add_english_doc = functools.partial(utils.add_english_doc, module=lazyllm.hook)
add_example = functools.partial(utils.add_example, module=lazyllm.hook)

# LazyLLMHook
add_chinese_doc('LazyLLMHook', '''\
LazyLLM 提供的钩子系统抽象基类，用于在函数或方法执行前后插入自定义逻辑。

此类是一个抽象基类（ABC），定义了钩子系统的基本接口。通过继承此类并实现其抽象方法，可以创建自定义的钩子来监控、记录或修改函数执行过程。

Args:
    obj: 要监控的对象（通常是函数或方法）。此对象会被存储在钩子实例中，供其他方法使用。
                
**注意**: 此类是抽象基类，不能直接实例化。必须继承此类并实现所有抽象方法才能使用。
''')

add_english_doc('LazyLLMHook', '''\
Abstract base class for LazyLLM's hook system, used to insert custom logic before and after function or method execution.

This class is an abstract base class (ABC) that defines the basic interface for the hook system. By inheriting from this class and implementing its abstract methods, you can create custom hooks to monitor, log, or modify function execution processes.

Args:
    obj: The object to monitor (usually a function or method). This object will be stored in the hook instance for use by other methods.

**Note**: This class is an abstract base class and cannot be instantiated directly. You must inherit from this class and implement all abstract methods to use it.
''')

add_chinese_doc('LazyLLMHook.pre_hook', '''\
前置钩子方法，在函数执行前调用。

Args:
    *args: 传递给被监控函数的参数
    **kwargs: 传递给被监控函数的关键字参数
''')

add_english_doc('LazyLLMHook.pre_hook', '''\
Pre-hook method, called before function execution.

Args:
    *args: Arguments passed to the monitored function
    **kwargs: Keyword arguments passed to the monitored function
''')

add_chinese_doc('LazyLLMHook.post_hook', '''\
后置钩子方法，在函数执行后调用。

Args:
    output: 被监控函数的返回值

**Returns:**\n
- 处理后的输出值。通常返回原始输出，但也可以修改或包装输出。
''')

add_english_doc('LazyLLMHook.post_hook', '''\
Post-hook method, called after function execution.

Args:
    output: The return value of the monitored function

**Returns:**\n
- The processed output value. Usually returns the original output, but can also modify or wrap the output.
''')

add_chinese_doc('LazyLLMHook.report', '''\
生成钩子执行报告。

**Returns:**\n
- 钩子执行的相关信息或统计数据。
''')

add_english_doc('LazyLLMHook.report', '''\
Generate a report of hook execution.

**Returns:**\n
- Relevant information or statistics about hook execution.
''')
