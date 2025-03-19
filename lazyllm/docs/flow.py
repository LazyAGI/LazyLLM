# flake8: noqa: E501
from . import utils
import functools
import lazyllm

add_chinese_doc = functools.partial(utils.add_chinese_doc, module=lazyllm.flow)
add_english_doc = functools.partial(utils.add_english_doc, module=lazyllm.flow)
add_example = functools.partial(utils.add_example, module=lazyllm.flow)

add_chinese_doc('FlowBase', """\
一个用于创建可以包含各种项目的流式结构的基类。
这个类提供了一种组织项目的方式，这些项目可以是 ``FlowBase`` 的实例或其他类型，组织成一个层次结构。每个项目都可以有一个名称，结构可以动态地遍历或修改。

Args:
    items (iterable): 要包含在流中的项目的可迭代对象。这些可以是 ``FlowBase`` 的实例或其他对象。
    item_names (list of str, optional): 对应于项目的名称列表。这允许通过名称访问项目。如果未提供，则只能通过索引访问项目。

""")

add_english_doc('FlowBase', """\
A base class for creating flow-like structures that can contain various items.

This class provides a way to organize items, which can be instances of ``FlowBase`` or other types, into a hierarchical structure. Each item can have a name and the structure can be traversed or modified dynamically.

Args:
    items (iterable): An iterable of items to be included in the flow. These can be instances of ``FlowBase`` or other objects.
    item_names (list of str, optional): A list of names corresponding to the items. This allows items to be accessed by name. If not provided, items can only be accessed by index.

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
    _scatter (bool, optional): 如果为 ``True``，输入将在项目之间分割。如果为 ``False``，相同的输入将传递给所有项目。默认为 ``False``。
    _concurrent (bool, optional): 如果为 ``True``，操作将使用线程并发执行。如果为 ``False``，操作将顺序执行。默认为 ``True``。
    args: 基类的可变长度参数列表。
    kwargs: 基类的任意关键字参数。

<span style="font-size: 20px;">&ensp;**`asdict property`**</span>

标记Parellel，使得Parallel每次调用时的返回值由package变为dict。当使用 ``asdict`` 时，请务必保证parallel的元素被取了名字，例如:  ``parallel(name=value)`` 。

<span style="font-size: 20px;">&ensp;**`tuple property`**</span>

标记Parellel，使得Parallel每次调用时的返回值由package变为tuple。

<span style="font-size: 20px;">&ensp;**`list property`**</span>

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
    _scatter (bool, optional): If ``True``, the input is split across the items. If ``False``, the same input is passed to all items. Defaults to ``False``.
    _concurrent (bool, optional): If ``True``, operations will be executed concurrently using threading. If ``False``, operations will be executed sequentially. Defaults to ``True``.
    args: Variable length argument list for the base class.
    kwargs: Arbitrary keyword arguments for the base class.

`asdict property`

Tag ``Parallel`` so that the return value of each call to ``Parallel`` is changed from a tuple to a dict. When using ``asdict``, make sure that the elements of ``parallel`` are named, for example: ``parallel(name=value)``.

`tuple property`

Mark Parallel so that the return value of Parallel changes from package to tuple each time it is called.

`list property`

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

add_chinese_doc('Pipeline', """\
一个形成处理阶段管道的顺序执行模型。

 ``Pipeline``类是一个处理阶段的线性序列，其中一个阶段的输出成为下一个阶段的输入。它支持在最后一个阶段之后添加后续操作。它是 ``LazyLLMFlowsBase``的子类，提供了一个延迟执行模型，并允许以延迟方式包装和注册函数。

Args:
    args (list of callables or single callable): 管道的处理阶段。每个元素可以是一个可调用的函数或 ``LazyLLMFlowsBase.FuncWrap``的实例。如果提供了单个列表或元组，则将其解包为管道的阶段。
    post_action (callable, optional): 在管道的最后一个阶段之后执行的可选操作。默认为None。
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

add_chinese_doc('Loop', '''\
初始化一个循环流结构，该结构将一系列函数重复应用于输入，直到满足停止条件或达到指定的迭代次数。

Loop结构允许定义一个简单的控制流，其中一系列步骤在循环中应用，可以使用可选的停止条件来根据步骤的输出退出循环。

Args:
    item (callable or list of callables): 将在循环中应用的函数或可调用对象。
    stop_condition (callable, optional): 一个函数，它接受循环中最后一个项目的输出作为输入并返回一个布尔值。如果返回 ``True``，循环将停止。如果为 ``None``，循环将继续直到达到 ``count``。默认为 ``None``。
    count (int, optional): 运行循环的最大迭代次数。如果为 ``None``，循环将无限期地继续或直到 ``stop_condition`` 返回 ``True``。默认为 ``None``。
    post_action (callable, optional): 循环结束后调用的函数。默认为 ``None``。
    judge_on_full_input(bool): 如果设置为 ``True`` ， 则通过 ``stop_condition`` 的输入进行条件判断，否则会将输入拆成判定条件和真实的输入两部分，仅对判定条件进行判断。

抛出:
    AssertionError: 如果同时提供了 ``stop_condition`` 和 ``count``，或者当提供的 ``count``不是一个整数。
''')

add_english_doc('Loop', '''\
Initializes a Loop flow structure which repeatedly applies a sequence of functions to an input until a stop condition is met or a specified count of iterations is reached.

The Loop structure allows for the definition of a simple control flow where a series of steps are applied in a loop, with an optional stop condition that can be used to exit the loop based on the output of the steps.

Args:
    *item (callable or list of callables): The function(s) or callable object(s) that will be applied in the loop.
    stop_condition (callable, optional): A function that takes the output of the last item in the loop as input and returns a boolean. If it returns ``True``, the loop will stop. If ``None``, the loop will continue until ``count`` is reached. Defaults to ``None``.
    count (int, optional): The maximum number of iterations to run the loop for. If ``None``, the loop will continue indefinitely or until ``stop_condition`` returns ``True``. Defaults to ``None``.
    post_action (callable, optional): A function to be called with the final output after the loop ends. Defaults to ``None``.
    judge_on_full_input(bool): If set to ``True``, the conditional judgment will be performed through the input of ``stop_condition``, otherwise the input will be split into two parts: the judgment condition and the actual input, and only the judgment condition will be judged.

Raises:
    AssertionError: If both ``stop_condition`` and ``count`` are provided or if ``count`` is not an integer when provided.
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
    post_action (callable, optional): 在执行选定流后要调用的函数。默认为 ``None``。
    judge_on_full_input(bool): 如果设置为 ``True`` ， 则通过 ``switch`` 的输入进行条件判断，否则会将输入拆成判定条件和真实的输入两部分，仅对判定条件进行判断。
    kwargs: 代表命名条件和对应流或函数的任意关键字参数。

抛出:
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
    post_action (callable, optional): A function to be called on the output after the selected flow is executed. Defaults to ``None``.
    judge_on_full_input(bool): If set to ``True``, the conditional judgment will be performed through the input of ``switch``, otherwise the input will be split into two parts: the judgment condition and the actual input, and only the judgment condition will be judged.
    kwargs: Arbitrary keyword arguments representing named conditions and corresponding flows or functions.

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
""")
