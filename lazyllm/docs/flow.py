# flake8: noqa: E501
from . import utils
import functools
import lazyllm


flow_ch = functools.partial(utils.add_chinese_doc, module=lazyllm.flow)
flow_en = functools.partial(utils.add_english_doc, module=lazyllm.flow)
flow_eg = functools.partial(utils.add_example, module=lazyllm.flow)

flow_ch('FlowBase', r"""
一个用于创建可以包含各种项目的流式结构的基类。

这个类提供了一种组织项目的方式，这些项目可以是`FlowBase`的实例或其他类型，组织成一个层次结构。每个项目都可以有一个名称，结构可以动态地遍历或修改。

参数:
    items (iterable): 要包含在流中的项目的可迭代对象。这些可以是`FlowBase`的实例或其他对象。
    item_names (list of str, optional): 对应于项目的名称列表。这允许通过名称访问项目。如果未提供，则只能通过索引访问项目。

""")

flow_en('FlowBase', r"""
A base class for creating flow-like structures that can contain various items.

This class provides a way to organize items, which can be instances of `FlowBase` or other types, into a hierarchical structure. Each item can have a name and the structure can be traversed or modified dynamically.

Arguments:
    items (iterable): An iterable of items to be included in the flow. These can be instances of `FlowBase` or other objects.
    item_names (list of str, optional): A list of names corresponding to the items. This allows items to be accessed by name. If not provided, items can only be accessed by index.

""")

flow_eg('FlowBase', r"""
>>> flow = FlowBase('item1', 'item2', item_names=['name1', 'name2'])
>>> flow.is_root
True
>>> flow.ancestor
<FlowBase object at ...>
>>> flow.for_each(lambda x: isinstance(x, str), print)
item1
item2
""")

flow_en('FlowBase.is_root', r"""
A property that indicates whether the current flow item is the root of the flow structure.

Returns:
    bool: True if the current item has no parent (`_father` is None), otherwise False.
""")
flow_en('FlowBase.ancestor', r"""
A property that returns the topmost ancestor of the current flow item.

If the current item is the root, it returns itself.

Returns:
    FlowBase: The topmost ancestor flow item.
""")


flow_ch('FlowBase.for_each', r"""
对流中每个匹配给定过滤器的项目执行一个操作。

该方法递归地遍历流结构，将操作应用于通过过滤器的每个项目。

参数:
    filter (callable): 一个接受项目作为输入并返回True的函数，如果该项目应该应用操作。
    action (callable): 一个接受项目作为输入并对其执行某些操作的函数。

返回:
    None

""")

flow_en('FlowBase.for_each', r"""
Performs an action on each item in the flow that matches a given filter.

The method recursively traverses the flow structure, applying the action to each item that passes the filter.

Arguments:
    filter (callable): A function that takes an item as input and returns True if the item should have the action applied.
    action (callable): A function that takes an item as input and performs some operation on it.

Returns:
    None

""")
flow_eg('FlowBase.for_each', r"""
>>> flow = FlowBase('item1', FlowBase('item2'), item_names=['name1', 'name2'])
>>> flow.for_each(lambda x: isinstance(x, FlowBase), lambda x: setattr(x, '_flow_name', 'NamedFlow'))
>>> flow.items[1]._flow_name
'NamedFlow'
""")


flow_ch('Parallel', r"""
用于管理LazyLLMFlows中的并行流的类。

这个类继承自LazyLLMFlowsBase，提供了一个并行或顺序运行操作的接口。它支持使用线程进行并发执行，并允许以字典形式返回结果。


可以这样可视化`Parallel`类：

.. code-block:: text

        //> module11 -> ... -> module1N -> out1 \\
    input -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
        \\> module31 -> ... -> module3N -> out3 //
        

可以这样可视化`Parallel.sequential`方法：

.. code-block:: text

     input -> module21 -> ... -> module2N -> out2 -> 

参数:
    _scatter (bool, optional): 如果为`True`，输入将在项目之间分割。如果为`False`，相同的输入将传递给所有项目。默认为`False`。
    _concurrent (bool, optional): 如果为`True`，操作将使用线程并发执行。如果为`False`，操作将顺序执行。默认为`True`。
    *args: 基类的可变长度参数列表。
    **kw: 基类的任意关键字参数。
            """)

flow_en('Parallel', r"""
A class for managing parallel flows in LazyLLMFlows.

This class inherits from LazyLLMFlowsBase and provides an interface for running operations in parallel or sequentially. It supports concurrent execution using threads and allows for the return of results as a dictionary.


The `Parallel` class can be visualized as follows:

.. code-block:: text

        //> module11 -> ... -> module1N -> out1 \\
    input -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
        \\> module31 -> ... -> module3N -> out3 //
        

The `Parallel.sequential` method can be visualized as follows:

.. code-block:: text

     input -> module21 -> ... -> module2N -> out2 -> 

Arguments:
    _scatter (bool, optional): If `True`, the input is split across the items. If `False`, the same input is passed to all items. Defaults to `False`.
    _concurrent (bool, optional): If `True`, operations will be executed concurrently using threading. If `False`, operations will be executed sequentially. Defaults to `True`.
    *args: Variable length argument list for the base class.
    **kw: Arbitrary keyword arguments for the base class.
            """)

flow_eg('Parallel', r"""
>>> parallel_flow = Parallel()
>>> results = parallel_flow._run(input_data)
        """)



flow_ch('Pipeline', r"""
一个形成处理阶段管道的顺序执行模型。

`Pipeline`类是一个处理阶段的线性序列，其中一个阶段的输出成为下一个阶段的输入。它支持添加在最后一个阶段之后执行的后续操作。它是`LazyLLMFlowsBase`的子类，提供了一个延迟执行模型，并允许以延迟方式包装和注册函数。

参数:
    *args (list of callables or single callable): 管道的处理阶段。每个元素可以是一个可调用的函数或`LazyLLMFlowsBase.FuncWrap`的实例。如果提供了单个列表或元组，则将其解包为管道的阶段。
    post_action (callable, optional): 在管道的最后一个阶段之后执行的可选操作。默认为None。
    return_input (bool, optional): 如果设置为`True`，原始输入将与输出一起返回。默认为`False`。
    **kw (dict of callables): 管道的命名处理阶段。每个键值对表示一个命名阶段，其中键是名称，值是可调用的阶段。

返回:
    管道的最后一个阶段的输出，如果`return_input`为`True`，则可选地与原始输入一起返回。

方法:
    barrier(args):
            在继续之前等待所有线程到达同步点。

            这个方法与名为_barr.impl的threading.local实例相关联，它是一个threading.Barrier对象。
            在假设的Pipeline类的并行扩展中并行执行多个线程时，每个线程在开始执行之前会调用这个方法。
            这确保所有线程在同一时间达到执行的同一点，从而同步线程的并发执行
            并防止数据竞争和其他并发问题。

            返回:
                未修改的输入args, 允许线程函数继续执行而不影响其执行的同时确保同步。



""")


flow_en('Pipeline', r"""
A sequential execution model that forms a pipeline of processing stages.

The `Pipeline` class is a linear sequence of processing stages, where the output of one stage becomes the input to the next. It supports the addition of post-actions that can be performed after the last stage. It is a subclass of `LazyLLMFlowsBase` which provides a lazy execution model and allows for functions to be wrapped and registered in a lazy manner.

Arguments:
    *args (list of callables or single callable): The processing stages of the pipeline. Each element can be a callable function or an instance of `LazyLLMFlowsBase.FuncWrap`. If a single list or tuple is provided, it is unpacked as the stages of the pipeline.
    post_action (callable, optional): An optional action to perform after the last stage of the pipeline. Defaults to None.
    return_input (bool, optional): If set to `True`, the original input along with the output will be returned. Defaults to `False`.
    **kw (dict of callables): Named processing stages of the pipeline. Each key-value pair represents a named stage, where the key is the name and the value is the callable stage.

Returns:
    The output of the last stage of the pipeline, optionally along with the original input if `return_input` is `True`.

Methods:
    barrier(args):
            Waits for all threads to reach a synchronization point before proceeding.

            This method is associated with a threading.local instance named _barr.impl, which is a threading.Barrier object.
            When executing multiple threads concurrently in a hypothetical parallel extension of the Pipeline class, this method would be invoked by each thread before it begins execution.
            This ensures that all threads reach the same point of execution at the same time, thereby synchronizing the concurrent execution of threads
            and preventing data races and other concurrency issues.

            Returns:
                The unmodified input args, allowing thread functions to proceed without affecting their execution while ensuring synchronization.



""")

flow_eg('Pipeline', r"""
>>> pipeline = Pipeline(
...     preprocess_stage,
...     compute_stage,
...     postprocess_stage,
...     post_action=final_stage
... )
>>> result = pipeline(input_data)
""")



flow_ch('Loop',r'''
初始化一个循环流结构，该结构将一系列函数重复应用于输入，直到满足停止条件或达到指定的迭代次数。

Loop结构允许定义一个简单的控制流，其中一系列步骤在循环中应用，可以使用可选的停止条件来根据步骤的输出退出循环。

参数:
    *item (callable or list of callables): 将在循环中应用的函数或可调用对象。
    stop_condition (callable, optional): 一个函数，它接受循环中最后一个项目的输出作为输入并返回一个布尔值。如果返回`True`，循环将停止。如果为`None`，循环将继续直到达到`count`。默认为`None`。
    count (int, optional): 运行循环的最大迭代次数。如果为`None`，循环将无限期地继续或直到`stop_condition`返回`True`。默认为`None`。
    post_action (callable, optional): 循环结束后调用的函数。默认为`None`。
    return_input (bool, optional): 如果为`True`，最终输出将包括初始输入和最后一次迭代的输出。默认为`False`。

抛出:
    AssertionError: 如果同时提供了`stop_condition`和`count`，或者当提供`count`时它不是一个整数。
            ''')


flow_en('Loop',r'''
Initializes a Loop flow structure which repeatedly applies a sequence of functions to an input until a stop condition is met or a specified count of iterations is reached.

The Loop structure allows for the definition of a simple control flow where a series of steps are applied in a loop, with an optional stop condition that can be used to exit the loop based on the output of the steps.

Arguments:
    *item (callable or list of callables): The function(s) or callable object(s) that will be applied in the loop.
    stop_condition (callable, optional): A function that takes the output of the last item in the loop as input and returns a boolean. If it returns `True`, the loop will stop. If `None`, the loop will continue until `count` is reached. Defaults to `None`.
    count (int, optional): The maximum number of iterations to run the loop for. If `None`, the loop will continue indefinitely or until `stop_condition` returns `True`. Defaults to `None`.
    post_action (callable, optional): A function to be called with the final output after the loop ends. Defaults to `None`.
    return_input (bool, optional): If `True`, the final output will include both the initial input and the output of the last iteration. Defaults to `False`.

Raises:
    AssertionError: If both `stop_condition` and `count` are provided or if `count` is not an integer when provided.
            ''')
flow_eg('Loop',r'''
    >>> loop = Loop(my_step_function, stop_condition=lambda x: x > 10)
    >>> final_output = loop(initial_input)
            ''')




flow_ch('FlowBase.is_root', r"""
一个属性，指示当前流项目是否是流结构的根。

返回:
    bool: 如果当前项目没有父级（`_father`为None），则为True，否则为False。
""")

flow_ch('FlowBase.ancestor', r"""
一个属性，返回当前流项目的最顶层祖先。

如果当前项目是根，则返回其自身。

返回:
    FlowBase: 最顶层的祖先流项目。
""")


flow_ch('IFS',r'''
在LazyLLMFlows框架中实现If-Else功能。

IFS（If-Else Flow Structure）类设计用于根据给定条件的评估有条件地执行两个提供的路径之一（真路径或假路径）。执行选定路径后，可以应用可选的后续操作，并且如果指定，输入可以与输出一起返回。

参数:
    cond (callable): 一个接受输入并返回布尔值的可调用对象。它决定执行哪个路径。如果`cond(input)`评估为True，则执行`tpath`；否则，执行`fpath`。
    tpath (callable): 如果条件为True，则执行的路径。
    fpath (callable): 如果条件为False，则执行的路径。
    post_action (callable, optional): 执行选定路径后执行的可选可调用对象。可以用于进行清理或进一步处理。默认为None。
    return_input (bool, optional): 如果设置为True，原始输入也将与执行路径的输出一起返回。默认为False。

返回:
    执行路径的输出，如果`return_input`为True，则可选地与原始输入一起。
        ''')

flow_ch('Switch', r"""
一个根据条件选择并执行流的控制流机制。

`Switch`类提供了一种根据表达式的值或条件的真实性选择不同流的方法。它类似于其他编程语言中找到的switch-case语句。
.. code-block:: text

    switch(exp):
        case cond1: input -> module11 -> ... -> module1N -> out; break
        case cond2: input -> module21 -> ... -> module2N -> out; break
        case cond3: input -> module31 -> ... -> module3N -> out; break
     
参数:
    *args: 可变长度参数列表，交替提供条件和对应的流或函数。条件可以是返回布尔值的可调用对象或与输入表达式进行比较的值。
    post_action (callable, optional): 在执行选定流后要调用的函数。默认为`None`。
    return_input (bool, optional): 如果设置为`True`，原始输入将与输出一起返回。默认为`False`。
    **kw: 代表命名条件和对应流或函数的任意关键字参数。

抛出:
    TypeError: 如果提供的参数数量为奇数，或者如果第一个参数不是字典且条件没有成对提供。
""")

flow_ch('Diverter', r"""
一个流分流器，将输入通过不同的模块以并行方式路由。

Diverter类是一种专门的并行处理形式，其中多个输入分别通过一系列模块并行处理。然后将输出聚合并作为元组返回。

当您拥有可以并行执行的不同数据处理管道，并希望在单个流构造中管理它们时，此类非常有用。

.. code-block:: text

                    /> in1 -> module11 -> ... -> module1N -> out1 \
    (in1, in2, in3) -> in2 -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
                    \> in3 -> module31 -> ... -> module3N -> out3 /
                    
参数:
    *args: 可变长度参数列表，代表并行执行的模块。
    _concurrent (bool, optional): 控制模块是否应并行或顺序运行的标志。默认为`True`。
    **kw: 代表额外模块的任意关键字参数，其中键是模块的名称。

""")

flow_ch('Warp', r"""
一个流形变器，将单个模块并行应用于多个输入。

Warp类设计用于将同一个处理模块应用于一组输入。它有效地将单个模块“形变”到输入上，使每个输入都并行处理。输出被收集并作为元组返回。需要注意的是，这个类不能用于异步任务，如训练和部署。

.. code-block:: text

                    /> in1 \                            /> out1 \
    (in1, in2, in3) -> in2 -> module1 -> ... -> moduleN -> out2 -> (out1, out2, out3)
                    \> in3 /                            \> out3 /
                
参数:
    *args: 可变长度参数列表，代表要应用于所有输入的单个模块。
    **kw: 未来扩展的任意关键字参数。

注意:
    - 只允许一个函数在warp中。
    - Warp流不应用于异步任务，如训练和部署。
""")



flow_en('IFS',r'''
Implements an If-Else functionality within the LazyLLMFlows framework.

The IFS (If-Else Flow Structure) class is designed to conditionally execute one of two provided
paths (true path or false path) based on the evaluation of a given condition. After the execution
of the selected path, an optional post-action can be applied, and the input can be returned alongside
the output if specified.

Arguments:
    cond (callable): A callable that takes the input and returns a boolean. It determines which path
                        to execute. If `cond(input)` evaluates to True, `tpath` is executed; otherwise,
                        `fpath` is executed.
    tpath (callable): The path to be executed if the condition is True.
    fpath (callable): The path to be executed if the condition is False.
    post_action (callable, optional): An optional callable that is executed after the selected path.
                                        It can be used to perform cleanup or further processing. Defaults to None.
    return_input (bool, optional): If set to True, the original input is also returned alongside the output
                                    of the executed path. Defaults to False.

Returns:
    The output of the executed path, optionally paired with the original input if `return_input` is True.
        ''')


flow_eg('IFS',r'''
>>> cond = lambda x: x > 0
>>> tpath = lambda x: x * 2
>>> fpath = lambda x: -x
>>> ifs_flow = IFS(cond, tpath, fpath)
>>> ifs_flow(10)
20
>>> ifs_flow(-5)
5
''')

flow_en('Switch', r"""
A control flow mechanism that selects and executes a flow based on a condition.

The `Switch` class provides a way to choose between different flows depending on the value of an expression or the truthiness of conditions. It is similar to a switch-case statement found in other programming languages.
.. code-block:: text

    switch(exp):
        case cond1: input -> module11 -> ... -> module1N -> out; break
        case cond2: input -> module21 -> ... -> module2N -> out; break
        case cond3: input -> module31 -> ... -> module3N -> out; break
     
Arguments:
    *args: A variable length argument list, alternating between conditions and corresponding flows or functions. Conditions are either callables returning a boolean or values to be compared with the input expression.
    post_action (callable, optional): A function to be called on the output after the selected flow is executed. Defaults to `None`.
    return_input (bool, optional): If set to `True`, the original input is returned along with the output. Defaults to `False`.
    **kw: Arbitrary keyword arguments representing named conditions and corresponding flows or functions.

Raises:
    TypeError: If an odd number of arguments are provided, or if the first argument is not a dictionary and the conditions are not provided in pairs.

""")
flow_eg('Switch', r"""
>>> def is_positive(x): return x > 0
>>> def is_negative(x): return x < 0
>>> switch = Switch(is_positive, flow_positive, is_negative, flow_negative, 'default', flow_default)
>>> result = switch(input_value)  # Executes the flow corresponding to the first true condition or 'default' if none match.

""")

flow_en('Diverter', r"""
A flow diverter that routes inputs through different modules in parallel.

The Diverter class is a specialized form of parallel processing where multiple inputs are each processed by a separate sequence of modules in parallel. The outputs are then aggregated and returned as a tuple.

This class is useful when you have distinct data processing pipelines that can be executed concurrently, and you want to manage them within a single flow construct.

.. code-block:: text

                    /> in1 -> module11 -> ... -> module1N -> out1 \
    (in1, in2, in3) -> in2 -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
                    \> in3 -> module31 -> ... -> module3N -> out3 /
                    
Arguments:
    *args: Variable length argument list representing the modules to be executed in parallel.
    _concurrent (bool, optional): A flag to control whether the modules should be run concurrently or sequentially. Defaults to `True`.
    **kw: Arbitrary keyword arguments representing additional modules, where the key is the name of the module.

""")

flow_eg('Diverter', r"""
>>> diverter_flow = Diverter(module1, module2, module3, _concurrent=False)
>>> result = diverter_flow(input1, input2, input3)
>>> print(result)
(output1, output2, output3)
""")

flow_en('Warp', r"""
A flow warp that applies a single module to multiple inputs in parallel.

The Warp class is designed to apply the same processing module to a set of inputs. It effectively 'warps' the single module around the inputs so that each input is processed in parallel. The outputs are collected and returned as a tuple. It is important to note that this class cannot be used for asynchronous tasks, such as training and deployment.

.. code-block:: text

                    /> in1 \                            /> out1 \
    (in1, in2, in3) -> in2 -> module1 -> ... -> moduleN -> out2 -> (out1, out2, out3)
                    \> in3 /                            \> out3 /
                
Arguments:
    *args: Variable length argument list representing the single module to be applied to all inputs.
    **kw: Arbitrary keyword arguments for future extensions.

Note:
    - Only one function is allowed in warp.
    - The Warp flow should not be used for asynchronous tasks such as training and deployment.
""")

flow_eg('Warp', r"""
>>> warp_flow = Warp(single_module)
>>> result = warp_flow(input1, input2, input3)
>>> print(result)
(output1, output2, output3)
""")

