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

add_chinese_doc('LazyLLMFuncHook', '''\
函数作为钩子的辅助函数，如果函数中存在 yield 语句，则 yield 之前的语句作为 pre_hook，yield 之后的语句作为 post_hook。

Args:
    func: 作为钩子的函数。
''')

add_english_doc('LazyLLMFuncHook', '''\
Helper class for hooking functions. if the function is a generator function, statements before yield
will be executed as pre_hook, and statements after yield will be executed as post_hook.

Args:
    func: The function to hook.
''')

add_chinese_doc('LazyLLMHook.pre_hook', '''\
前置钩子方法，在被监控函数执行前调用。

这是一个抽象方法，需要在子类中实现。

Args:
    *args: 传递给被监控函数的位置参数。
    **kwargs: 传递给被监控函数的关键字参数。
''')

add_english_doc('LazyLLMHook.pre_hook', '''\
Pre-hook method, called before the monitored function executes.

This is an abstract method and must be implemented in subclasses.

Args:
    *args: Positional arguments passed to the monitored function.
    **kwargs: Keyword arguments passed to the monitored function.
''')

add_chinese_doc('LazyLLMHook.post_hook', '''\
后置钩子方法，在被监控函数执行后调用。

这是一个抽象方法，需要在子类中实现。

Args:
    output: 被监控函数的返回值。
''')

add_english_doc('LazyLLMHook.post_hook', '''\
Post-hook method, called after the monitored function executes.

This is an abstract method and must be implemented in subclasses.

Args:
    output: The return value of the monitored function.
''')

add_chinese_doc('LazyLLMHook.on_error', '''\
异常处理钩子，在被监控函数执行抛出异常时调用。

默认实现为空操作，子类可以按需覆盖，用于记录错误状态、补充诊断信息或执行清理逻辑。

Args:
    exc: 被监控函数抛出的异常对象。
''')

add_english_doc('LazyLLMHook.on_error', '''\
Error-handling hook, called when the monitored function raises an exception.

The default implementation is a no-op. Subclasses can override it to record error status,
attach diagnostic information, or perform cleanup.

Args:
    exc: The exception raised by the monitored function.
''')

add_chinese_doc('LazyLLMHook.report', '''\
生成钩子的执行报告。

这是一个抽象方法，需要在子类中实现。
''')

add_english_doc('LazyLLMHook.report', '''\
Generate a report of the hook execution.

This is an abstract method and must be implemented in subclasses.
''')

add_chinese_doc('HookPhaseError', '''\
Hook 阶段错误，当一个 hook 阶段中有一个或多个 strict 模式的 hook 执行失败时抛出。

Args:
    phase (str): 发生错误的 hook 阶段名称（如 ``'post_hook'``、``'on_error'``、``'report'``）。
    errors: 包含 ``(hook_obj, exception)`` 元组的序列，记录所有失败的 hook 及其异常。
''')

add_english_doc('HookPhaseError', '''\
Raised when one or more strict-mode hooks fail during a hook phase.

Args:
    phase (str): The name of the hook phase where the error(s) occurred (e.g. ``'post_hook'``, ``'on_error'``, ``'report'``).
    errors: A sequence of ``(hook_obj, exception)`` tuples recording each failed hook and its exception.
''')

utils.add_chinese_doc('LazyTracingHook', '''\
为 flow 或 module 创建 tracing hook。

该 hook 会在执行生命周期中创建、更新并结束对应的 tracing span。

Args:
    obj: 要进行 tracing 的 flow 或 module 对象。
''', module=lazyllm.tracing)

utils.add_english_doc('LazyTracingHook', '''\
Create a tracing hook for a flow or module object.

This hook is responsible for creating, updating, and finishing the corresponding tracing span
during the execution lifecycle.

Args:
    obj: The flow or module object to be traced.
''', module=lazyllm.tracing)

utils.add_chinese_doc('LazyTracingHook.pre_hook', '''\
创建并激活当前 flow 或 module 对应的 tracing span。

该方法会在被包裹对象执行前调用，并根据当前调用参数初始化 span 上下文。

Args:
    *args: 传递给目标对象的位置参数。
    **kwargs: 传递给目标对象的关键字参数。
''', module=lazyllm.tracing)

utils.add_english_doc('LazyTracingHook.pre_hook', '''\
Create and activate the tracing span for the current flow or module.

This method is called before the wrapped object executes and initializes the span context
from the current call arguments.

Args:
    *args: Positional arguments passed to the target object.
    **kwargs: Keyword arguments passed to the target object.
''', module=lazyllm.tracing)

utils.add_chinese_doc('LazyTracingHook.post_hook', '''\
在 tracing span 上记录执行输出。

该方法会在被包裹对象成功执行后调用，把返回结果写入当前 span。

Args:
    output: 被包裹对象的返回值。
''', module=lazyllm.tracing)

utils.add_english_doc('LazyTracingHook.post_hook', '''\
Record the execution output on the active tracing span.

This method is called after the wrapped object completes successfully and writes the
returned result to the current span.

Args:
    output: The return value of the wrapped object.
''', module=lazyllm.tracing)

utils.add_chinese_doc('LazyTracingHook.on_error', '''\
在 tracing span 上记录异常状态。

当被包裹的 flow 或 module 执行失败时，该方法会把异常信息写入当前 span。

Args:
    exc: 执行过程中抛出的异常对象。
''', module=lazyllm.tracing)

utils.add_english_doc('LazyTracingHook.on_error', '''\
Record the error state on the active tracing span.

When the wrapped flow or module execution fails, this method writes the exception information
to the current span.

Args:
    exc: The exception raised during execution.
''', module=lazyllm.tracing)

utils.add_chinese_doc('LazyTracingHook.report', '''\
结束并上报当前 tracing span。

该方法会在 hook 生命周期结束时调用，用于关闭当前 span 并完成本次 tracing 记录。
''', module=lazyllm.tracing)

utils.add_english_doc('LazyTracingHook.report', '''\
Finish and report the current tracing span.

This method is called at the end of the hook lifecycle to close the current span and
complete the tracing record for this execution.
''', module=lazyllm.tracing)
