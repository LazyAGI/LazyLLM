# flake8: noqa: E501
from . import utils
import functools
import lazyllm

add_chinese_doc = functools.partial(utils.add_chinese_doc, module=lazyllm.flow)
add_english_doc = functools.partial(utils.add_english_doc, module=lazyllm.flow)
add_example = functools.partial(utils.add_example, module=lazyllm.flow)

add_chinese_doc('FlowBase', """\
用于构建流式结构的基类，可以容纳多个项目并组织成层次化结构。

该类允许将不同的对象（包括 ``FlowBase`` 实例或其他类型对象）组合在一起，
并为其分配可选的名称，从而支持按名称或索引访问。结构中的项目可以动态添加或遍历。

Args:
    *items: 要包含在流中的项目，可以是 ``FlowBase`` 的实例或其他对象。
    item_names (list of str, optional): 对应于每个项目的名称列表，会与 ``items`` 按顺序配对。
        如果未提供，所有项目的名称默认为 ``None``。
    auto_capture (bool, optional): 是否启用自动捕获。如果为 ``True``，在上下文管理器模式下，
        将自动捕获当前作用域中新定义的变量并加入流。默认为 ``False``。
""")

add_english_doc('FlowBase', """\
Base class for constructing flow-like structures that can hold multiple items and organize them hierarchically.

This class allows combining different objects (including ``FlowBase`` instances or other types)
into a structured flow, with optional names for each item, enabling both name-based and index-based access.
Items in the structure can be added or traversed dynamically.

Args:
    *items: Items to be included in the flow, which can be instances of ``FlowBase`` or other objects.
    item_names (list of str, optional): A list of names corresponding to the items, paired with ``items`` in order.
        If not provided, all items will be assigned ``None`` as their name.
    auto_capture (bool, optional): Whether to enable automatic variable capture. If ``True``, when used
        as a context manager, newly defined variables in the current scope will be automatically added to the flow.
        Defaults to ``False``.
""")

add_chinese_doc('FlowBase.id', """\
获取模块或流程的 ID。如果传入字符串则原样返回；如果传入已绑定的模块则返回其对应的 item_id；不传参时返回整个 flow 的唯一 id。

Args:
    module (Optional[Union[str, Any]]): 目标模块或字符串标识。

**Returns:**\n
- str: 对应的 ID 字符串。
""")

add_english_doc('FlowBase.id', """\
Get the identifier for a module or the flow itself. If a string is provided, it is returned as-is. If a bound module is provided, returns its associated item_id. If no argument is given, returns the unique id of the entire flow.

Args:
    module (Optional[Union[str, Any]]): Target module or string identifier.

**Returns:**\n
- str: Corresponding identifier string.
""")

add_chinese_doc('FlowBase.is_root', """\
一个属性，指示当前流项目是否是流结构的根。

**Returns:**\n
- bool: 如果当前项目没有父级（ ``_father`` 为None），则为True，否则为False。
""")

add_english_doc('FlowBase.is_root', """\
A property that indicates whether the current flow item is the root of the flow structure.

**Returns:**\n
- bool: True if the current item has no parent (`` _father`` is None), otherwise False.
""")

add_example('FlowBase.is_root', '''\
>>> import lazyllm
>>> p = lazyllm.pipeline()
>>> p.is_root
True
>>> p2 = lazyllm.pipeline(p)
>>> p.is_root
False
>>> p2.is_root
True
''')

add_chinese_doc('FlowBase.ancestor', """\
一个属性，返回当前流项目的最顶层祖先。

如果当前项目是根，则返回其自身。

**Returns:**\n
- FlowBase: 最顶层的祖先流项目。
""")

add_english_doc('FlowBase.ancestor', """\
A property that returns the topmost ancestor of the current flow item.

If the current item is the root, it returns itself.

**Returns:**\n
- FlowBase: The topmost ancestor flow item.
""")

add_example('FlowBase.ancestor', '''\
>>> import lazyllm
>>> p = lazyllm.pipeline()
>>> p2 = lazyllm.pipeline(p)
>>> p.ancestor is p2
True
''')

add_chinese_doc('FlowBase.for_each', """\
对流中每个通过过滤器的项目执行一个操作。

该方法递归地遍历流结构，将操作应用于通过过滤器的每个项目。

Args:
    filter (callable): 一个接受项目作为输入并返回bool的函数，如果该项目应该应用操作，则返回True。
    action (callable): 一个接受项目作为输入并对其执行某些操作的函数。

**Returns:**\n
- None
""")

add_english_doc('FlowBase.for_each', """\
Performs an action on each item in the flow that matches a given filter.

The method recursively traverses the flow structure, applying the action to each item that passes the filter.

Args:
    filter (callable): A function that takes an item as input and returns True if the item should have the action applied.
    action (callable): A function that takes an item as input and performs some operation on it.

**Returns:**\n
- None
""")

add_example('FlowBase.for_each', """\
>>> import lazyllm
>>> def test1(): print('1')
... 
>>> def test2(): print('2')
... 
>>> def test3(): print('3')
... 
>>> flow = lazyllm.pipeline(test1, lazyllm.pipeline(test2, test3))
>>> flow.for_each(lambda x: callable(x), lambda x: print(x))
<Function type=test1>
<Function type=test2>
<Function type=test3>
""")

add_chinese_doc('LazyLLMFlowsBase', """\
一个支持流程封装、钩子注册与调用逻辑的基础类。

`LazyLLMFlowsBase` 是 LazyLLM 中所有流程（Flow）的基类，用于组织一系列可调用模块的执行流程，并支持钩子（hook）机制、同步控制、后处理逻辑等功能。它的设计旨在统一封装执行调用、异常处理、后处理、流程表示等功能，适用于各种同步数据处理场景。

该类通常不直接使用，而是被诸如 `Pipeline`、`Parallel` 等具体流程类继承和使用。

```text
输入 --> [Flow模块1 -> Flow模块2 -> ... -> Flow模块N] --> 输出
                   ↑             ↓
               pre_hook       post_hook
```

Args:
    args: 可变长度参数列表。
    post_action: 在主流程结束后对输出进行进一步处理的可调用对象。默认为 ``None``。
    auto_capture: 如果为 True，在上下文管理器模式下将自动捕获当前作用域中新定义的变量加入流中。默认为 False。
    **kw: 命名组件的键值对。
""")

add_english_doc('LazyLLMFlowsBase', """\
A base class for flow structures with hook support and unified execution logic.

`LazyLLMFlowsBase` is the base class for all LazyLLM flow types. It organizes a sequence of callable modules into a flow and provides support for pre/post hooks, synchronization control, post-processing, and error-safe invocation. It is not intended for direct use but instead serves as a foundational class for concrete flow types like `Pipeline`, `Parallel`, etc.

```text
input --> [Flow module1 -> Flow module2 -> ... -> Flow moduleN] --> output
                   ↑             ↓
               pre_hook       post_hook
```

Args:
    args: A sequence of callables representing the flow modules.
    post_action: An optional callable applied to the output after main flow execution. Defaults to ``None``。
    auto_capture: If True, variables newly defined within the ``with`` block will be automatically added to the flow. Defaults to ``False``.
    **kw: Key-value pairs for named components.
""")

add_chinese_doc('LazyLLMFlowsBase.register_hook', '''\
注册一个 Hook 类型，用于在流程执行前后进行额外处理。

Args:
    hook_type (LazyLLMHook): 要注册的 Hook 类型或实例。
''')

add_english_doc('LazyLLMFlowsBase.register_hook', '''\
Register a hook type for additional processing before and after the flow execution.

Args:
    hook_type (LazyLLMHook): The hook type or instance to register.
''')

add_chinese_doc('LazyLLMFlowsBase.unregister_hook', '''\
注销已注册的 Hook。

Args:
    hook_type (LazyLLMHook): 要移除的 Hook 类型或实例。
''')

add_english_doc('LazyLLMFlowsBase.unregister_hook', '''\
Unregister a previously registered hook.

Args:
    hook_type (LazyLLMHook): The hook type or instance to remove.
''')
add_chinese_doc('LazyLLMFlowsBase.start', """\
启动流处理执行（已弃用）。

此方法已弃用，建议直接将流实例作为函数调用。执行流处理并返回结果。

Args:
    *args: 传递给流处理的可变位置参数。
    **kw: 传递给流处理的命名参数。

**Returns:**\n
- 流处理的结果。

**Note:**\n
- 此方法已标记为弃用，请使用流实例的直接调用方式代替。
""")

add_english_doc('LazyLLMFlowsBase.start', """\
Start flow processing execution (deprecated).

This method is deprecated, it is recommended to directly call the flow instance as a function. Executes the flow processing and returns the result.

Args:
    *args: Variable positional arguments passed to the flow processing.
    **kw: Named arguments passed to the flow processing.

**Returns:**\n
- The result of flow processing.

**Note:**\n
- This method is marked as deprecated, please use direct invocation of the flow instance instead.
""")
add_chinese_doc('LazyLLMFlowsBase.clear_hooks', '''\
清空所有已注册的 Hook。
''')

add_english_doc('LazyLLMFlowsBase.clear_hooks', '''\
Clear all registered hooks.
''')

add_chinese_doc('LazyLLMFlowsBase.set_sync', '''\
设置流程是否同步执行。

Args:
    sync (bool): 是否同步执行，默认为 True。

**Returns:**\n
- LazyLLMFlowsBase: 当前实例。
''')

add_english_doc('LazyLLMFlowsBase.set_sync', '''\
Set whether the flow executes synchronously.

Args:
    sync (bool): Whether to execute synchronously. Default is True.

**Returns:**\n
- LazyLLMFlowsBase: The current instance.
''')

add_chinese_doc('LazyLLMFlowsBase.wait', '''\
等待流程中所有异步任务完成。

**Returns:**\n
- LazyLLMFlowsBase: 当前实例。
''')

add_english_doc('LazyLLMFlowsBase.wait', '''\
Wait for all asynchronous tasks in the flow to complete.

**Returns:**\n
- LazyLLMFlowsBase: The current instance.
''')

add_chinese_doc('LazyLLMFlowsBase.invoke', '''\
调用指定对象（可为函数、模块或 bind 对象）并传入输入数据。  
支持对 bind 对象进行 root/pipeline 输出替换。

Args:
    it (Callable | bind): 要调用的对象。
    __input (Any): 输入数据。
    bind_args_source (Any, optional): 绑定参数来源。
    **kw: 其他关键字参数。
''')

add_english_doc('LazyLLMFlowsBase.invoke', '''\
Invoke a target (function, module, or bind object) with the given input.  
Supports root/pipeline output replacement for bind objects.

Args:
    it (Callable | bind): The target to invoke.
    __input (Any): Input data.
    bind_args_source (Any, optional): Source of bind arguments.
    **kw: Additional keyword arguments.
''')

add_chinese_doc('LazyLLMFlowsBase.bind', '''\
为当前流程绑定参数，生成一个 bind 对象。

Args:
    *args: 位置参数。
    **kw: 关键字参数。

**Returns:**\n
- bind: 绑定后的 bind 对象。
''')

add_english_doc('LazyLLMFlowsBase.bind', '''\
Bind arguments to the current flow, producing a bind object.

Args:
    *args: Positional arguments.
    **kw: Keyword arguments.

**Returns:**\n
- bind: The bound bind object.
''')

add_chinese_doc('Parallel', """\
用于管理LazyLLMFlows中的并行流的类。

这个类继承自LazyLLMFlowsBase，提供了一个并行或顺序运行操作的接口。它支持使用线程进行并发执行，并允许以字典形式返回结果。


可以这样可视化 ``Parallel`` 类：

```text
#       /> module11 -> ... -> module1N -> out1 \\\\
# input -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
#       \> module31 -> ... -> module3N -> out3 /
```        

可以这样可视化 ``Parallel.sequential`` 方法：

```text
# input -> module21 -> ... -> module2N -> out2 -> 
```

Args:
    args: 基类的可变长度参数列表。
    _scatter (bool, optional): 如果为 ``True``，输入将在项目之间分割。如果为 ``False``，相同的输入将传递给所有项目。默认为 ``False``。
    _concurrent (bool, optional): 如果为 ``True``，操作将使用线程并发执行。如果为 ``False``，操作将顺序执行。默认为 ``True``。
    multiprocessing (bool, optional): 如果为 ``True``，将使用多进程而不是多线程进行并行执行。这可以提供真正的并行性，但会增加进程间通信的开销。默认为 ``False``。
    auto_capture (bool, optional): 如果为 True，在上下文管理器模式下将自动捕获当前作用域中新定义的变量加入流中。默认为 ``False``。
    kwargs: 基类的任意关键字参数。

<span style="font-size: 20px;">&ensp;**`asdict property`**</span>

标记Parellel，使得Parallel每次调用时的返回值由package变为dict。当使用 ``asdict`` 时，请务必保证parallel的元素被取了名字，例如:  ``parallel(name=value)`` 。

<span style="font-size: 20px;">&ensp;**`astuple property`**</span>

标记Parellel，使得Parallel每次调用时的返回值由package变为tuple。

<span style="font-size: 20px;">&ensp;**`aslist property`**</span>

标记Parellel，使得Parallel每次调用时的返回值由package变为list。

<span style="font-size: 20px;">&ensp;**`sum property`**</span>

标记Parellel，使得Parallel每次调用时的返回值做一次累加。

<span style="font-size: 20px;">&ensp;**`join(self, string)`**</span>

标记Parellel，使得Parallel每次调用时的返回值通过 ``string`` 做一次join。
""")

add_english_doc('Parallel', """\
A class for managing parallel flows in LazyLLMFlows.

This class inherits from LazyLLMFlowsBase and provides an interface for running operations in parallel or sequentially. It supports concurrent execution using threads and allows for the return of results as a dictionary.


The ``Parallel`` class can be visualized as follows:

```text
#       /> module11 -> ... -> module1N -> out1 \\\\
# input -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
#       \> module31 -> ... -> module3N -> out3 /
```       

The ``Parallel.sequential`` method can be visualized as follows:

```text
# input -> module21 -> ... -> module2N -> out2 -> 
```

Args:
    args: Variable length argument list for the base class.
    _scatter (bool, optional): If ``True``, the input is split across the items. If ``False``, the same input is passed to all items. Defaults to ``False``.
    _concurrent (Union[bool, int], optional): If ``True``, operations will be executed concurrently using threading. If an integer, specifies the maximum number of concurrent executions. If ``False``, operations will be executed sequentially. Defaults to ``True``.
    multiprocessing (bool, optional): If ``True``, multiprocessing will be used instead of multithreading for parallel execution. This can provide true parallelism but adds overhead for inter-process communication. Defaults to ``False``.
    auto_capture (bool, optional): If True, variables newly defined within the ``with`` block will be automatically added to the flow. Defaults to ``False``.
    kwargs: Arbitrary keyword arguments for the base class.

`asdict property`

Tag ``Parallel`` so that the return value of each call to ``Parallel`` is changed from a tuple to a dict. When using ``asdict``, make sure that the elements of ``parallel`` are named, for example: ``parallel(name=value)``.

`astuple property`

Mark Parallel so that the return value of Parallel changes from package to tuple each time it is called.

`aslist property`

Mark Parallel so that the return value of Parallel changes from package to list each time it is called.

`sum property`

Mark Parallel so that the return value of Parallel is accumulated each time it is called.

`join(self, string)`

Mark Parallel so that the return value of Parallel is joined by ``string`` each time it is called.
""")

add_example(
    'Parallel',
    '''\
>>> import lazyllm
>>> test1 = lambda a: a + 1
>>> test2 = lambda a: a * 4
>>> test3 = lambda a: a / 2
>>> ppl = lazyllm.parallel(test1, test2, test3)
>>> assert ppl(1) == (2, 4, 0.5), "LAZYLLM_CHECK_FAILED"
>>> ppl = lazyllm.parallel(a=test1, b=test2, c=test3)
>>> ppl(1)
{2, 4, 0.5}
>>> ppl = lazyllm.parallel(a=test1, b=test2, c=test3).asdict
>>> assert  ppl(2) == {'a': 3, 'b': 8, 'c': 1.0}, "LAZYLLM_CHECK_FAILED"
>>> ppl = lazyllm.parallel(a=test1, b=test2, c=test3).astuple
>>> ppl(-1)
(0, -4, -0.5)
>>> ppl = lazyllm.parallel(a=test1, b=test2, c=test3).aslist
>>> ppl(0)
[1, 0, 0.0]
>>> ppl = lazyllm.parallel(a=test1, b=test2, c=test3).join('\\\\n')
>>> ppl(1)
'2\\\\n4\\\\n0.5'
''',
)

add_chinese_doc('Parallel.sequential', """\
创建一个顺序执行的Parallel实例。

这个类方法会将 ``_concurrent`` 设置为 ``False``，使得所有操作按顺序执行而不是并行执行。

可以这样可视化 ``Parallel.sequential`` 方法：

```text
# input -> module21 -> ... -> module2N -> out2 -> 
```

Args:
    args: 传递给 Parallel 构造函数的可变长度参数列表。
    kwargs: 传递给 Parallel 构造函数的关键字参数。

**Returns:**\n
- Parallel: 一个新的顺序执行的 Parallel 实例。
""")

add_english_doc('Parallel.sequential', """\
Creates a Parallel instance that executes sequentially.

This class method sets ``_concurrent`` to ``False``, causing all operations to be executed in sequence rather than in parallel.

The ``Parallel.sequential`` method can be visualized as follows:

```text
# input -> module21 -> ... -> module2N -> out2 -> 
```

Args:
    args: Variable length argument list passed to the Parallel constructor.
    kwargs: Keyword arguments passed to the Parallel constructor.
    _scatter (bool, optional): If ``True``, the input is split across the items. If ``False``, the same input is passed to all items. Defaults to ``False``.
    _concurrent (bool, optional): If ``True``, operations will be executed concurrently using threading. If ``False``, operations will be executed sequentially. Defaults to ``True``.
    multiprocessing (bool, optional): If ``True``, multiprocessing will be used instead of multithreading for parallel execution. This can provide true parallelism but adds overhead for inter-process communication. Defaults to ``False``.
    auto_capture (bool, optional): If True, variables newly defined within the ``with`` block will be automatically added to the flow. Defaults to ``False``.
    args: Variable length argument list for the base class.
    kwargs: Arbitrary keyword arguments for the base class.

**Returns:**\n
- Parallel: A new Parallel instance configured for sequential execution.
""")

add_chinese_doc('Parallel.join', """\
标记Parallel，使得每次调用时的返回值通过指定字符串连接。

Args:
    string (str): 用于连接结果的字符串。默认为空字符串。

**Returns:**\n
- Parallel: 返回当前 Parallel 实例，其结果将被字符串连接。

**示例:**\n
```python
>>> ppl = lazyllm.parallel(a=test1, b=test2, c=test3).join('\\n')
>>> ppl(1)
'2\\n4\\n0.5'
```
""")

add_english_doc('Parallel.join', """\
Marks the Parallel instance to join its results with the specified string on each call.

Args:
    string (str): The string to use for joining results. Defaults to an empty string.

**Returns:**\n
- Parallel: Returns the current Parallel instance configured to join results with the specified string.

**Example:**\n
```python
>>> ppl = lazyllm.parallel(a=test1, b=test2, c=test3).join('\\n')
>>> ppl(1)
'2\\n4\\n0.5'
```
""")


add_chinese_doc('Pipeline', """\
一个形成处理阶段管道的顺序执行模型。

 ``Pipeline``类是一个处理阶段的线性序列，其中一个阶段的输出成为下一个阶段的输入。它支持在最后一个阶段之后添加后续操作。它是 ``LazyLLMFlowsBase``的子类，提供了一个延迟执行模型，并允许以延迟方式包装和注册函数。

Args:
    args (list of callables or single callable): 管道的处理阶段。每个元素可以是一个可调用的函数或 ``LazyLLMFlowsBase.FuncWrap``的实例。如果提供了单个列表或元组，则将其解包为管道的阶段。
    post_action (callable, optional): 在管道的最后一个阶段之后执行的可选操作。默认为None。
    auto_capture (bool, optional): 如果为 True，在上下文管理器模式下将自动捕获当前作用域中新定义的变量加入流中。默认为 ``False``。
    kwargs (dict of callables): 管道的命名处理阶段。每个键值对表示一个命名阶段，其中键是名称，值是可调用的阶段。

**Returns:**\n
- 管道的最后一个阶段的输出。
""")

add_english_doc('Pipeline', """\
A sequential execution model that forms a pipeline of processing stages.

The ``Pipeline`` class is a linear sequence of processing stages, where the output of one stage becomes the input to the next. It supports the addition of post-actions that can be performed after the last stage. It is a subclass of ``LazyLLMFlowsBase`` which provides a lazy execution model and allows for functions to be wrapped and registered in a lazy manner.

Args:
    args (list of callables or single callable): The processing stages of the pipeline. Each element can be a callable function or an instance of ``LazyLLMFlowsBase.FuncWrap``. If a single list or tuple is provided, it is unpacked as the stages of the pipeline.
    post_action (callable, optional): An optional action to perform after the last stage of the pipeline. Defaults to None.
    auto_capture (bool, optional): If True, variables newly defined within the ``with`` block will be automatically added to the flow. Defaults to ``False``.
    kwargs (dict of callables): Named processing stages of the pipeline. Each key-value pair represents a named stage, where the key is the name and the value is the callable stage.

**Returns:**\n
- The output of the last stage of the pipeline.
""")

add_example('Pipeline', """\
>>> import lazyllm
>>> ppl = lazyllm.pipeline(
...     stage1=lambda x: x+1,
...     stage2=lambda x: f'get {x}'
... )
>>> ppl(1)
'get 2'
>>> ppl.stage2
<Function type=lambda>
""")

add_chinese_doc('Pipeline.output', '''\
获取流水线中指定模块的输出结果。

Args:
    module: 要获取输出的模块。可以是模块对象或模块名称。
    unpack (bool): 是否解包输出结果。默认为False。

**Returns:**\n
- bind.Args: 一个绑定参数对象，用于在流水线中传递数据。
''')

add_english_doc('Pipeline.output', '''\
Get the output result of a specified module in the pipeline.

Args:
    module: The module to get output from. Can be a module object or module name.
    unpack (bool): Whether to unpack the output result. Defaults to False.

**Returns:**\n
- bind.Args: A bound argument object for data passing in the pipeline.
''')

add_chinese_doc('save_pipeline_result', """\
一个上下文管理器，用于临时设置是否保存流水线中的中间执行结果。

在进入上下文时，会将 `Pipeline.g_save_flow_result` 设置为指定值；退出上下文后会恢复为原来的状态。适用于调试或需要记录中间输出的场景。

Args:
    flag (bool): 是否启用结果保存功能，默认为 True。

**Returns:**\n
- ContextManager: 上下文管理器。
""")

add_english_doc('save_pipeline_result', """\
A context manager that temporarily sets whether to save intermediate results during pipeline execution.

When entering the context, `Pipeline.g_save_flow_result` is set to the given value. After exiting, it restores the previous value. Useful for debugging or recording intermediate outputs.

Args:
    flag (bool): Whether to enable result saving. Defaults to True.

**Returns:**\n
- ContextManager: A context manager.
""")

add_example('save_pipeline_result', '''\
>>> import lazyllm
>>> pipe = lazyllm.pipeline(lambda x: x + 1, lambda x: x * 2)
>>> with lazyllm.save_pipeline_result(True):
...     result = pipe(1)
>>> result
4
''')

add_chinese_doc('Loop', '''\
初始化一个循环流结构，该结构将一系列函数重复应用于输入，直到满足停止条件或达到指定的迭代次数。

Loop结构允许定义一个简单的控制流，其中一系列步骤在循环中应用，可以使用可选的停止条件来根据步骤的输出提前退出循环。

Args:
    item (callable or list of callables): 将在循环中应用的函数或可调用对象。
    stop_condition (callable, optional): 一个函数，它接受循环中最后一个项目的输出作为输入并返回一个布尔值。如果返回 ``True``，循环将停止。如果为 ``None``，循环将继续直到达到 ``count``。默认为 ``None``。
    count (int, optional): 运行循环的最大迭代次数。默认为 ``sys.maxsize``。
    post_action (callable, optional): 循环结束后调用的函数。默认为 ``None``。
    auto_capture (bool, optional): 如果为 True，在上下文管理器模式下将自动捕获当前作用域中新定义的变量加入流中。默认为 ``False``。
    judge_on_full_input (bool): 如果设置为 ``True`` ，则通过 ``stop_condition`` 的输入进行条件判断；否则会将输入拆成判定条件和真实的输入两部分，仅对判定条件进行判断。

Raises:
    AssertionError: 如果提供的 ``stop_condition`` 既不是 ``callable`` 也不是 ``None``。
''')

add_english_doc('Loop', '''\
Initializes a Loop flow structure which repeatedly applies a sequence of functions to an input until a stop condition is met or a specified count of iterations is reached.

The Loop structure allows for the definition of a simple control flow where a series of steps are applied in a loop, with an optional stop condition that can be used to exit the loop early based on the output of the steps.

Args:
    *item (callable or list of callables): The function(s) or callable object(s) that will be applied in the loop.
    stop_condition (callable, optional): A function that takes the output of the last item in the loop as input and returns a boolean. If it returns ``True``, the loop will stop. If ``None``, the loop will continue until ``count`` is reached. Defaults to ``None``.
    count (int, optional): The maximum number of iterations to run the loop for. Defaults to ``sys.maxsize``.
    post_action (callable, optional): A function to be called with the final output after the loop ends. Defaults to ``None``.
    auto_capture (bool, optional): If True, variables newly defined within the ``with`` block will be automatically added to the flow. Defaults to ``False``.
    judge_on_full_input (bool): If set to ``True``, the conditional judgment will be performed through the input of ``stop_condition``; otherwise, the input will be split into two parts: the judgment condition and the actual input, and only the judgment condition will be judged.

Raises:
    AssertionError: If the provided ``stop_condition`` is neither callable nor ``None``.
''')

add_example('Loop', '''\
>>> import lazyllm
>>> loop = lazyllm.loop(lambda x: x * 2, stop_condition=lambda x: x > 10, judge_on_full_input=True)
>>> loop(1)
16
>>> loop(3)
12
>>>
>>> with lazyllm.loop(stop_condition=lambda x: x > 10, judge_on_full_input=True) as lp:
...    lp.f1 = lambda x: x + 1
...    lp.f2 = lambda x: x * 2
...
>>> lp(0)
14
''')

add_chinese_doc('IFS', '''\
在LazyLLMFlows框架中实现If-Else功能。

IFS（If-Else Flow Structure）类设计用于根据给定条件的评估有条件地执行两个提供的路径之一（真路径或假路径）。执行选定路径后，可以应用可选的后续操作，并且如果指定，输入可以与输出一起返回。

Args:
    cond (callable): 一个接受输入并返回布尔值的可调用对象。它决定执行哪个路径。如果 ``cond(input)`` 评估为True，则执行 ``tpath`` ；否则，执行 ``fpath`` 。
    tpath (callable): 如果条件为True，则执行的路径。
    fpath (callable): 如果条件为False，则执行的路径。
    post_action (callable, optional): 执行选定路径后执行的可选可调用对象。可以用于进行清理或进一步处理。默认为None。

**Returns:**\n
- 执行路径的输出。
''')

add_english_doc('IFS', '''\
Implements an If-Else functionality within the LazyLLMFlows framework.

The IFS (If-Else Flow Structure) class is designed to conditionally execute one of two provided
paths (true path or false path) based on the evaluation of a given condition. After the execution
of the selected path, an optional post-action can be applied, and the input can be returned alongside
the output if specified.

Args:
    cond (callable): A callable that takes the input and returns a boolean. It determines which path
                        to execute. If ``cond(input)`` evaluates to True, ``tpath`` is executed; otherwise,
                        ``fpath`` is executed.
    tpath (callable): The path to be executed if the condition is True.
    fpath (callable): The path to be executed if the condition is False.
    post_action (callable, optional): An optional callable that is executed after the selected path.
                                        It can be used to perform cleanup or further processing. Defaults to None.

**Returns:**\n
- The output of the executed path.
''')

add_example('IFS', '''\
>>> import lazyllm
>>> cond = lambda x: x > 0
>>> tpath = lambda x: x * 2
>>> fpath = lambda x: -x
>>> ifs_flow = lazyllm.ifs(cond, tpath, fpath)
>>> ifs_flow(10)
20
>>> ifs_flow(-5)
5
''')

add_chinese_doc('Switch', """\
一个根据条件选择并执行流的控制流机制。

 ``Switch``类提供了一种根据表达式的值或条件的真实性选择不同流的方法。它类似于其他编程语言中找到的switch-case语句。

```text
# switch(exp):
#     case cond1: input -> module11 -> ... -> module1N -> out; break
#     case cond2: input -> module21 -> ... -> module2N -> out; break
#     case cond3: input -> module31 -> ... -> module3N -> out; break
```   

Args:
    args: 可变长度参数列表，交替提供条件和对应的流或函数。条件可以是返回布尔值的可调用对象或与输入表达式进行比较的值。
    conversion (callable, optional): 在进行条件匹配之前，对判定表达式 ``exp`` 进行转换或预处理的函数。默认为 ``None``。
    post_action (callable, optional): 在执行选定流后要调用的函数。默认为 ``None``。
    judge_on_full_input(bool): 如果设置为 ``True`` ， 则通过 ``switch`` 的输入进行条件判断，否则会将输入拆成判定条件和真实的输入两部分，仅对判定条件进行判断。

Raises:
    TypeError: 如果提供的参数数量为奇数，或者如果第一个参数不是字典且条件没有成对提供。
""")

add_english_doc('Switch', """\
A control flow mechanism that selects and executes a flow based on a condition.

The ``Switch`` class provides a way to choose between different flows depending on the value of an expression or the truthiness of conditions. It is similar to a switch-case statement found in other programming languages.

```text
# switch(exp):
#     case cond1: input -> module11 -> ... -> module1N -> out; break
#     case cond2: input -> module21 -> ... -> module2N -> out; break
#     case cond3: input -> module31 -> ... -> module3N -> out; break
``` 

Args:
    args: A variable length argument list, alternating between conditions and corresponding flows or functions. Conditions are either callables returning a boolean or values to be compared with the input expression.
    conversion (callable, optional): A function used to transform or preprocess the evaluation expression ``exp`` before performing condition matching. Defaults to ``None``.
    post_action (callable, optional): A function to be called on the output after the selected flow is executed. Defaults to ``None``.
    judge_on_full_input(bool): If set to ``True``, the conditional judgment will be performed through the input of ``switch``, otherwise the input will be split into two parts: the judgment condition and the actual input, and only the judgment condition will be judged.

Raises:
    TypeError: If an odd number of arguments are provided, or if the first argument is not a dictionary and the conditions are not provided in pairs.
""")

add_example('Switch', """\
>>> import lazyllm
>>> def is_positive(x): return x > 0
...
>>> def is_negative(x): return x < 0
...
>>> switch = lazyllm.switch(is_positive, lambda x: 2 * x, is_negative, lambda x : -x, 'default', lambda x : '000', judge_on_full_input=True)
>>>
>>> switch(1)
2
>>> switch(0)
'000'
>>> switch(-4)
4
>>>
>>> def is_1(x): return True if x == 1 else False
...
>>> def is_2(x): return True if x == 2 else False
...
>>> def is_3(x): return True if x == 3 else False
...
>>> def t1(x): return 2 * x
...
>>> def t2(x): return 3 * x
...
>>> def t3(x): return x
...
>>> with lazyllm.switch(judge_on_full_input=True) as sw:
...     sw.case[is_1::t1]
...     sw.case(is_2, t2)
...     sw.case[is_3, t3]
...
>>> sw(1)
2
>>> sw(2)
6
>>> sw(3)
3
""")

add_chinese_doc('Diverter', """\
一个流分流器，将输入通过不同的模块以并行方式路由。

Diverter类是一种专门的并行处理形式，其中多个输入分别通过一系列模块并行处理。然后将输出聚合并作为元组返回。

当您拥有可以并行执行的不同数据处理管道，并希望在单个流构造中管理它们时，此类非常有用。

```text
#                 /> in1 -> module11 -> ... -> module1N -> out1 \\\\
# (in1, in2, in3) -> in2 -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
#                 \> in3 -> module31 -> ... -> module3N -> out3 /
```                    

Args:
    args: 可变长度参数列表，代表并行执行的模块。
    _concurrent (bool, optional): 控制模块是否应并行执行的标志。默认为 ``True``。可用 ``Diverter.sequential`` 代替 ``Diverter`` 来设置此变量。
    auto_capture (bool, optional): 如果为 True，在上下文管理器模式下将自动捕获当前作用域中新定义的变量加入流中。默认为 ``False``。
    kwargs: 代表额外模块的任意关键字参数，其中键是模块的名称。

.. property:: 
    asdict

    和 ``parallel.asdict`` 一样
""")

add_english_doc('Diverter', """\
A flow diverter that routes inputs through different modules in parallel.

The Diverter class is a specialized form of parallel processing where multiple inputs are each processed by a separate sequence of modules in parallel. The outputs are then aggregated and returned as a tuple.

This class is useful when you have distinct data processing pipelines that can be executed concurrently, and you want to manage them within a single flow construct.

```text
#                 /> in1 -> module11 -> ... -> module1N -> out1 \\\\
# (in1, in2, in3) -> in2 -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
#                 \> in3 -> module31 -> ... -> module3N -> out3 /
```                    

Args:
    args : Variable length argument list representing the modules to be executed in parallel.
    _concurrent (bool, optional): A flag to control whether the modules should be run concurrently. Defaults to ``True``. You can use ``Diverter.sequential`` instead of ``Diverter`` to set this variable.
    auto_capture (bool, optional): If True, variables newly defined within the ``with`` block will be automatically added to the flow. Defaults to ``False``.
    kwargs : Arbitrary keyword arguments representing additional modules, where the key is the name of the module.
""")

add_example('Diverter', """\
>>> import lazyllm
>>> div = lazyllm.diverter(lambda x: x+1, lambda x: x*2, lambda x: -x)
>>> div(1, 2, 3)
(2, 4, -3)
>>> div = lazyllm.diverter(a=lambda x: x+1, b=lambda x: x*2, c=lambda x: -x).asdict
>>> div(1, 2, 3)
{'a': 2, 'b': 4, 'c': -3}
>>> div(dict(c=3, b=2, a=1))
{'a': 2, 'b': 4, 'c': -3}
""")

add_chinese_doc('Warp', """\
一个流形变器，将单个模块并行应用于多个输入。

Warp类设计用于将同一个处理模块应用于一组输入。它有效地将单个模块“形变”到输入上，使每个输入都并行处理。输出被收集并作为元组返回。需要注意的是，这个类不能用于异步任务，如训练和部署。

```text
#                 /> in1 \                            /> out1 \\
# (in1, in2, in3) -> in2 -> module1 -> ... -> moduleN -> out2 -> (out1, out2, out3)
#                 \> in3 /                            \> out3 /
```

Args:
    args: 可变长度参数列表，代表要应用于所有输入的单个模块。
    _scatter (bool): 是否以分片方式拆分输入，默认 False。
    _concurrent (bool | int): 是否启用并发执行，可设定最大并发数。默认启用并发。
    auto_capture (bool, optional): 如果为 True，在上下文管理器模式下将自动捕获当前作用域中新定义的变量加入流中。默认为 ``False``。
    kwargs: 未来扩展的任意关键字参数。

注意:
    - 只允许一个函数在warp中。
    - Warp流不应用于异步任务，如训练和部署。
""")

add_english_doc('Warp', """\
A flow warp that applies a single module to multiple inputs in parallel.

The Warp class is designed to apply the same processing module to a set of inputs. It effectively 'warps' the single module around the inputs so that each input is processed in parallel. The outputs are collected and returned as a tuple. It is important to note that this class cannot be used for asynchronous tasks, such as training and deployment.

```text
#                 /> in1 \                            /> out1 \\\\
# (in1, in2, in3) -> in2 -> module1 -> ... -> moduleN -> out2 -> (out1, out2, out3)
#                 \> in3 /                            \> out3 /
``` 

Args:
    args: Variable length argument list representing the single module to be applied to all inputs.
    _scatter (bool): Whether to scatter inputs into parts before processing. Defaults to False.
    _concurrent (bool | int): Whether to execute in parallel. Can be a boolean or a max concurrency limit. Defaults to True.
    auto_capture (bool): If True, variables newly defined within the ``with`` block will be automatically added to the flow. Defaults to ``False``.
    kwargs: Arbitrary keyword arguments for future extensions.

Note:
    - Only one function is allowed in warp.
    - The Warp flow should not be used for asynchronous tasks such as training and deployment.
""")

add_example('Warp', """\
>>> import lazyllm
>>> warp = lazyllm.warp(lambda x: x * 2)
>>> warp(1, 2, 3, 4)
(2, 4, 6, 8)
>>> warp = lazyllm.warp(lazyllm.pipeline(lambda x: x * 2, lambda x: f'get {x}'))
>>> warp(1, 2, 3, 4)
('get 2', 'get 4', 'get 6', 'get 8')

>>> from lazyllm import package
>>> warp1 = lazyllm.warp(lambda x, y: x * 2 + y)
>>> print(warp1([package(1,2), package(10, 20)]))
(4, 40)
""")

add_chinese_doc('Graph', """\
一个基于有向无环图（DAG）的复杂流控制结构。

Graph类允许您创建复杂的处理图，其中节点表示处理函数，边表示数据流。它支持拓扑排序来确保正确的执行顺序，并可以处理多输入和多输出的复杂依赖关系。

Graph类特别适用于需要复杂数据流和依赖管理的场景，如机器学习管道、数据处理工作流等。

Args:
    post_action (callable, optional): 在图执行完成后要调用的函数。默认为 ``None``。
    auto_capture (bool, optional): 是否自动捕获上下文中的变量。默认为 ``False``。
    kwargs: 代表命名节点和对应函数的任意关键字参数。

**Returns:**\n
- 图的最终输出结果。
""")

add_english_doc('Graph', """\
A complex flow control structure based on Directed Acyclic Graph (DAG).

The Graph class allows you to create complex processing graphs where nodes represent processing functions and edges represent data flow. It supports topological sorting to ensure correct execution order and can handle complex dependencies with multiple inputs and outputs.

The Graph class is particularly suitable for scenarios requiring complex data flow and dependency management, such as machine learning pipelines, data processing workflows, etc.

Args:
    post_action (callable, optional): A function to be called after the graph execution is complete. Defaults to ``None``.
    auto_capture (bool, optional): Whether to automatically capture variables from context. Defaults to ``False``.
    kwargs: Arbitrary keyword arguments representing named nodes and corresponding functions.

**Returns:**\n
- The final output result of the graph.
""")

add_chinese_doc('Graph.Node', """\
表示图中单个节点的类。

Node类封装了图中节点的所有信息，包括处理函数、名称、输入输出连接等。

Args:
    func (callable): 节点要执行的函数。
    name (str): 节点的名称。
    arg_names (list, optional): 函数参数的名称列表。默认为 ``None``。
    inputs (dict): 输入连接的字典，键为源节点名，值为格式化函数。
    outputs (list): 输出连接的节点列表。

**Returns:**\n
- Node: 新创建的节点对象。
""")

add_english_doc('Graph.Node', """\
A class representing a single node in the graph.

The Node class encapsulates all information about a node in the graph, including the processing function, name, input/output connections, etc.

Args:
    func (callable): The function to be executed by the node.
    name (str): The name of the node.
    arg_names (list, optional): List of function parameter names. Defaults to ``None``.
    inputs (dict): Dictionary of input connections, with source node names as keys and formatter functions as values.
    outputs (list): List of output connected nodes.

**Returns:**\n
- Node: The newly created node object.
""")

add_example('Graph.Node', """\
>>> import lazyllm
>>> node = lazyllm.graph.Node(lambda x: x * 2, "multiply_node", ["input"])
>>> node.name
'multiply_node'
>>> node.func(5)
10
""")

add_chinese_doc('Graph.set_node_arg_name', """\
设置节点的参数名称。

此方法用于为图中的节点设置函数参数的名称，这对于多参数函数的正确调用很重要。

Args:
    arg_names (list): 参数名称的列表，与节点创建时的顺序对应。
""")

add_english_doc('Graph.set_node_arg_name', """\
Set the argument names for nodes.

This method is used to set the names of function arguments for nodes in the graph, which is important for correct invocation of multi-parameter functions.

Args:
    arg_names (list): List of argument names, corresponding to the order when nodes were created.
""")

add_example('Graph.set_node_arg_name', """\
>>> import lazyllm
>>> with lazyllm.graph() as g:
...     g.add = lambda a, b: a + b
...     g.multiply = lambda x, y: x * y
>>> g.set_node_arg_name([['x', 'y'], ['a', 'b']])
>>> g._nodes['add'].arg_names
['x', 'y']
>>> g._nodes['multiply'].arg_names
['a', 'b']
""")

add_chinese_doc('Graph.start_node', """\
获取图的起始节点。

**Returns:**\n
- Node: 图的起始节点（__start__）对象。
""")

add_english_doc('Graph.start_node', """\
Get the start node of the graph.

**Returns:**\n
- Node: The start node (__start__) object of the graph.
""")

add_example('Graph.start_node', """\
>>> import lazyllm
>>> with lazyllm.graph() as g:
...     g.process = lambda x: x * 2
>>> start = g.start_node
>>> start.name
'__start__'
""")

add_chinese_doc('Graph.end_node', """\
获取图的结束节点。

**Returns:**\n
- Node: 图的结束节点（__end__）对象。
""")

add_english_doc('Graph.end_node', """\
Get the end node of the graph.

**Returns:**\n
- Node: The end node (__end__) object of the graph.
""")

add_example('Graph.end_node', """\
>>> import lazyllm
>>> with lazyllm.graph() as g:
...     g.process = lambda x: x * 2
>>> end = g.end_node
>>> end.name
'__end__'
""")

add_chinese_doc('Graph.add_edge', """\
在图中添加一条边，定义节点之间的数据流。

此方法用于定义图中节点之间的连接关系，指定数据如何从一个节点流向另一个节点。

Args:
    from_node (str or Node): 源节点的名称或Node对象。
    to_node (str or Node): 目标节点的名称或Node对象。
    formatter (callable, optional): 可选的格式化函数，用于在传递数据时进行转换。默认为 ``None``。
""")

add_english_doc('Graph.add_edge', """\
Add an edge to the graph, defining data flow between nodes.

This method is used to define connection relationships between nodes in the graph, specifying how data flows from one node to another.

Args:
    from_node (str or Node): The name or Node object of the source node.
    to_node (str or Node): The name or Node object of the target node.
    formatter (callable, optional): Optional formatting function for data transformation during transfer. Defaults to ``None``.
""")

add_example('Graph.add_edge', """\
>>> import lazyllm
>>> with lazyllm.graph() as g:
...     g.node1 = lambda x: x * 2
...     g.node2 = lambda x: x + 1
...     g.node3 = lambda x, y: x + y
>>> g.add_edge('__start__', 'node1')
>>> g.add_edge('node1', 'node2')
>>> g.add_edge('node3', '__end__')
>>> g._nodes['node1'].outputs
[<Flow type=Node name=node2>]
>>> def double_input(data):
...     return data * 2
>>> g.add_edge('node1', 'node3', formatter=double_input)
>>> g._nodes['node3'].inputs
{'node1': <function double_input at ...>}
""")

add_chinese_doc('Graph.add_const_edge', """\
添加一个常量边，将固定值传递给指定节点。

此方法用于将常量值作为输入传递给图中的节点，无需从其他节点获取数据。

Args:
    constant: 要传递的常量值。
    to_node (str or Node): 目标节点的名称或Node对象。
""")

add_english_doc('Graph.add_const_edge', """\
Add a constant edge that passes a fixed value to a specified node.

This method is used to pass constant values as input to nodes in the graph without needing to get data from other nodes.

Args:
    constant: The constant value to pass.
    to_node (str or Node): The name or Node object of the target node.
""")

add_example('Graph.add_const_edge', """\
>>> import lazyllm
>>> with lazyllm.graph() as g:
...     g.add = lambda x, y: x + y
>>> g.add_const_edge(10, 'add')
>>> g._constants
[10]
""")

add_chinese_doc('Graph.topological_sort', """\
执行拓扑排序，返回正确的节点执行顺序。

此方法使用Kahn算法对有向无环图进行拓扑排序，确保所有依赖关系都得到满足。

**Returns:**\n
- List[Node]: 按拓扑顺序排列的节点列表。

Raises:
- ValueError: 如果图中存在循环依赖。
""")

add_english_doc('Graph.topological_sort', """\
Perform topological sorting to return the correct node execution order.

This method uses Kahn's algorithm to perform topological sorting on the directed acyclic graph, ensuring all dependencies are satisfied.

**Returns:**\n
- List[Node]: List of nodes arranged in topological order.

Raises:
- ValueError: If there are circular dependencies in the graph.
""")

add_example('Graph.topological_sort', """\
>>> import lazyllm
>>> with lazyllm.graph() as g:
...     g.node1 = lambda x: x * 2
...     g.node2 = lambda x: x + 1
...     g.node3 = lambda x, y: x + y
>>> g.add_edge('__start__', 'node1')
>>> g.add_edge('node1', 'node2')
>>> g.add_edge('node1', 'node3')
>>> g.add_edge('node2', 'node3')
>>> g.add_edge('node3', '__end__')
>>> sorted_nodes = g.topological_sort()
>>> [node.name for node in sorted_nodes]
['__start__', 'node1', 'node2', 'node3', '__end__']
>>> g.add_edge('node3', 'node1')
>>> try:
...     g.topological_sort()
... except ValueError as e:
...     print("检测到循环依赖")
检测到循环依赖
""")

add_chinese_doc('Graph.compute_node', """\
计算单个节点的输出结果。

此方法是图的内部方法，用于执行单个节点的计算，包括获取输入数据、应用格式化函数、调用节点函数等。

Args:
    sid: 会话ID。
    node (Node): 要计算的节点。
    intermediate_results (dict): 中间结果存储。
    futures (dict): 异步任务字典。

**Returns:**\n
- 节点的计算结果。
""")

add_english_doc('Graph.compute_node', """\
Compute the output result of a single node.

This is an internal method of the graph, used to execute the computation of a single node, including getting input data, applying formatter functions, calling node functions, etc.

Args:
    sid: Session ID.
    node (Node): The node to compute.
    intermediate_results (dict): Intermediate result storage.
    futures (dict): Async task dictionary.

**Returns:**\n
- The computation result of the node.
""")

add_example('Graph.compute_node', """\
>>> import lazyllm
>>> with lazyllm.graph() as g:
...     g.add = lambda x, y: x + y
...     g.multiply = lambda x: x * 2
>>> g.add_edge('__start__', 'add')
>>> g.add_const_edge(5, 'add')
>>> g.add_edge('add', 'multiply')
>>> g.add_edge('multiply', '__end__')
>>> result = g(3)  # x=3, y=5 (常量)
>>> result
16
""")
