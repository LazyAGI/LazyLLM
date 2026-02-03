# flake8: noqa E501
from . import utils
import functools
import lazyllm

add_chinese_doc = functools.partial(utils.add_chinese_doc, module=lazyllm.module)
add_english_doc = functools.partial(utils.add_english_doc, module=lazyllm.module)
add_example = functools.partial(utils.add_example, module=lazyllm.module)

add_chinese_doc('ModuleBase', '''\
ModuleBase 是 LazyLLM 的核心基类，定义了所有模块的统一接口和基础能力。  
它抽象了模块的训练、部署、推理和评测逻辑，并提供了子模块管理、钩子注册、参数传递和递归更新等机制。  
用户自定义的模块需要继承 ModuleBase，并实现 ``forward`` 方法来定义具体的推理逻辑。  

功能特性:
    - 统一管理子模块 (submodules)，自动追踪被持有的 ModuleBase 实例。
    - 支持 Option 类型的超参数设置，方便网格搜索与自动调参。
    - 提供钩子 (hook) 机制，可在调用前后执行自定义逻辑。
    - 封装训练 (train)、服务部署 (server)、评测 (eval) 的更新流程。
    - 支持 evalset 的加载与自动并行推理评测。

Args:
    return_trace (bool): 是否将推理结果写入 trace 队列，用于调试和追踪。默认为 ``False``。

使用场景:
    1. 当你需要组合训练、部署、推理和评测中的部分或全部能力时，例如一个 Embedding 模型需要同时训练与推理。
    2. 当你希望通过根模块调用 ``start``、``update``、``eval`` 等方法，递归管理其持有的子模块。
    3. 当你希望用户参数从外层模块自动传递到内部实现（参考 WebModule）。
    4. 当你希望自定义模块支持参数网格搜索（参考 TrialModule）。
''')

add_english_doc('ModuleBase', '''\
ModuleBase is the core base class in LazyLLM, defining the common interface and fundamental capabilities for all modules.  
It abstracts training, deployment, inference, and evaluation logic, while also providing mechanisms for submodule management, hook registration, parameter passing, and recursive updates.  
Custom modules should inherit from ModuleBase and implement the ``forward`` method to define specific inference logic.  

Key Features:
    - Unified management of submodules, automatically tracking held ModuleBase instances.
    - Support for Option type hyperparameters, enabling grid search and automated tuning.
    - Hook system that allows executing custom logic before and after calls.
    - Encapsulated update pipeline covering training, server deployment, and evaluation.
    - Built-in evalset loading and parallel inference evaluation.

Args:
    return_trace (bool): Whether to write inference results into the trace queue for debugging and tracking. Default is ``False``.

Use Cases:
    1. When combining some or all of training, deployment, inference, and evaluation capabilities, e.g., an embedding model requiring both training and inference.
    2. When you want to recursively manage submodules through root-level methods such as ``start``, ``update``, and ``eval``.
    3. When you want user parameters to be automatically propagated from outer modules to inner implementations (see WebModule).
    4. When you want the module to support parameter grid search (see TrialModule).
''')

add_example('ModuleBase', '''\
>>> import lazyllm
>>> class Module(lazyllm.module.ModuleBase):
...     pass
... 
>>> class Module2(lazyllm.module.ModuleBase):
...     def __init__(self):
...         super(__class__, self).__init__()
...         self.m = Module()
... 
>>> m = Module2()
>>> m.submodules
[<Module type=Module>]
>>> m.m3 = Module()
>>> m.submodules
[<Module type=Module>, <Module type=Module>]
''')

add_chinese_doc('ModuleBase.stream_output', '''\
上下文管理器，用于在推理或执行过程中进行流式输出。  
当提供字典类型的 ``stream_output`` 时，可指定输出前缀和后缀，以及对应颜色。

Args:
    stream_output (Optional[Union[bool, Dict]]): 流式输出配置。

        - 如果为布尔值 True，则开启默认流式输出。
        - 如果为字典，可包含以下键：

            - 'prefix' (str): 输出前缀文本。
            - 'prefix_color' (str, optional): 前缀颜色。
            - 'suffix' (str): 输出后缀文本。
            - 'suffix_color' (str, optional): 后缀颜色。
''')

add_english_doc('ModuleBase.stream_output', '''\
Context manager for streaming output during inference or execution.  
When a dictionary is provided to ``stream_output``, a prefix and suffix can be specified along with optional colors.

Args:
    stream_output (Optional[Union[bool, Dict]]): Configuration for streaming output.

        - If True, enables default streaming output.
        - If a dictionary, may include:

            - 'prefix' (str): Text to output at the beginning.
            - 'prefix_color' (str, optional): Color of the prefix.
            - 'suffix' (str): Text to output at the end.
            - 'suffix_color' (str, optional): Color of the suffix.
''')

add_chinese_doc('ModuleBase.used_by', '''\
设置当前模块被哪个模块使用，用于标记模块的调用关系。  
可链式调用，返回模块自身。

Args:
    module_id (str): 调用该模块的上层模块的唯一 ID。

**Returns:**\n
- ModuleBase: 返回模块自身，用于链式调用。
''')

add_english_doc('ModuleBase.used_by', '''\
Mark which module is using the current module, indicating the calling relationship.  
Supports chaining by returning the module itself.

Args:
    module_id (str): Unique ID of the parent module that uses this module.

**Returns:**\n
- ModuleBase: Returns the module itself for method chaining.
''')

add_chinese_doc('ModuleBase.forward', '''\
前向计算接口，需要子类实现。  
该方法定义了模块接收输入并返回输出的逻辑，是模块作为仿函数的核心函数。

Args:
    *args: 可变位置参数，子类可根据实际需求定义输入。
    **kw: 可变关键字参数，子类可根据实际需求定义输入。
''')

add_english_doc('ModuleBase.forward', '''\
Forward computation interface that must be implemented by subclasses.  
This method defines the logic for receiving inputs and returning outputs, and is the core function of the module as a functor.

Args:
    *args: Variable positional arguments, subclass can define the input as needed.
    **kw: Variable keyword arguments, subclass can define the input as needed.
''')

add_chinese_doc('ModuleBase.register_hook', '''\
注册一个钩子（Hook），在模块调用时执行特定逻辑。  
钩子需要继承自 ``LazyLLMHook``，可用于在模块前向计算前后添加自定义操作，例如日志记录或统计。

Args:
    hook_type (LazyLLMHook): 待注册的钩子对象。
''')

add_english_doc('ModuleBase.register_hook', '''\
Register a hook to execute specific logic during module invocation.  
The hook must inherit from ``LazyLLMHook`` and can be used to add custom operations before or after the module's forward computation, such as logging or metrics collection.

Args:
    hook_type (LazyLLMHook): Hook object to register.
''')

add_chinese_doc('ModuleBase.unregister_hook', '''\
注销已注册的钩子。  
如果钩子存在于模块中，将其移除，使其不再在模块调用时执行。

Args:
    hook_type (LazyLLMHook): 待注销的钩子对象。
''')

add_english_doc('ModuleBase.unregister_hook', '''\
Unregister a previously registered hook.  
If the hook exists in the module, it will be removed and no longer executed during module invocation.

Args:
    hook_type (LazyLLMHook): Hook object to unregister.
''')

add_chinese_doc('ModuleBase.clear_hooks', '''\
清空模块中所有已注册的钩子。  
调用后模块将不再执行任何钩子逻辑。
''')

add_english_doc('ModuleBase.clear_hooks', '''\
Clear all hooks registered in the module.  
After calling this, the module will no longer execute any hook logic.
''')

add_chinese_doc('ModuleBase.start', '''\
启动模块及所有子模块的部署服务。该方法会确保模块和子模块的 server 功能被执行，适合用于初始化或重新启动服务。

**Returns:**\n
- ModuleBase: 返回自身实例，以支持链式调用
''')

add_english_doc('ModuleBase.start', '''\
Start the deployment services of the module and all its submodules. This ensures that the server functionality of the module and its submodules is executed, suitable for initialization or restarting services.

**Returns:**\n
- ModuleBase: Returns itself to support method chaining
''')

add_example('ModuleBase.start', '''\
>>> import lazyllm
>>> m = lazyllm.TrainableModule().deploy_method(lazyllm.deploy.dummy).prompt(None)
>>> m.start()
<Module type=Trainable mode=None basemodel= target= stream=False return_trace=False>
>>> m(1)
"reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}"
''')

add_chinese_doc('ModuleBase.restart', '''\
重启模块及其子模块的部署服务。内部会调用 ``start`` 方法，实现服务的重新启动。

**Returns:**\n
- ModuleBase: 返回自身实例，以支持链式调用
''')

add_english_doc('ModuleBase.restart', '''\
Restart the deployment services of the module and its submodules. Internally calls the ``start`` method to reinitialize the services.

**Returns:**\n
- ModuleBase: Returns itself to support method chaining
''')

add_example('ModuleBase.restart', '''\
>>> import lazyllm
>>> m = lazyllm.TrainableModule().deploy_method(lazyllm.deploy.dummy).prompt(None)
>>> m.restart()
<Module type=Trainable mode=None basemodel= target= stream=False return_trace=False>
>>> m(1)
"reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}"
''')

add_chinese_doc('ModuleBase.wait', '''\
等待模块或其子模块的执行完成。此方法在当前实现中为空，可由子类根据具体部署逻辑进行实现。
''')

add_english_doc('ModuleBase.wait', '''\
Wait for the module or its submodules to finish execution. Currently, this method is a no-op and can be implemented by subclasses according to specific deployment logic.
''')

add_chinese_doc('ModuleBase.stop', '''\
停止模块及其所有子模块的运行。该方法会递归调用子模块的 ``stop`` 方法，适用于释放资源或关闭服务。
''')

add_english_doc('ModuleBase.stop', '''\
Stop the module and all its submodules. This method recursively calls the ``stop`` method of each submodule, suitable for releasing resources or shutting down services.
''')

add_chinese_doc('ModuleBase.for_each', '''\
对模块的所有子模块执行指定操作。递归遍历所有子模块，如果子模块满足 ``filter`` 条件，则执行 ``action``。

Args:
    filter (Callable): 接受子模块作为输入并返回布尔值的函数，用于判断是否执行操作。
    action (Callable): 对满足条件的子模块执行的操作函数。
''')

add_english_doc('ModuleBase.for_each', '''\
Execute a specified action on all submodules of the module. Recursively traverses all submodules, and if a submodule satisfies the ``filter`` condition, executes the ``action``.

Args:
    filter (Callable): A function that takes a submodule as input and returns a boolean, used to determine whether to perform the action.
    action (Callable): A function to perform on submodules that meet the condition.
''')

add_chinese_doc('ModuleBase.update', '''\
更新模块（及所有的子模块）。当模块重写了 ``_get_train_tasks`` 方法后，模块会被更新。更新完后会自动进入部署和推理的流程。

Args:
    recursive (bool): 是否递归更新所有的子模块，默认为True
''')

add_english_doc('ModuleBase.update', '''\
Update the module (and all its submodules). The module will be updated when the ``_get_train_tasks`` method is overridden.

Args:
    recursive (bool): Whether to recursively update all submodules, default is True.
''')

add_example('ModuleBase.update', '''\
>>> import lazyllm
>>> m = lazyllm.module.TrainableModule().finetune_method(lazyllm.finetune.dummy).trainset("").deploy_method(lazyllm.deploy.dummy).mode('finetune').prompt(None)
>>> m.evalset([1, 2, 3])
>>> m.update()
INFO: (lazyllm.launcher) PID: dummy finetune!, and init-args is {}
>>> print(m.eval_result)
["reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 2, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 3, and parameters is {'do_sample': False, 'temperature': 0.1}"]
''')
add_chinese_doc('ModuleBase.use_cache', """\
启用或禁用模块的缓存功能。

此方法用于控制模块是否使用缓存来存储和检索执行结果，以提高性能并避免重复计算。

Args:
    flag (bool or str, optional): 缓存控制标志。如果为True，启用缓存；如果为False，禁用缓存；
                                 如果为字符串，使用特定的缓存标识符。默认为True。

**Returns:**\n
- 返回模块实例本身，支持方法链式调用。

""")

add_english_doc('ModuleBase.use_cache', """\
Enable or disable the caching functionality for the module.

This method controls whether the module uses caching to store and retrieve execution results, 
improving performance and avoiding redundant computations.

Args:
    flag (bool or str, optional): Cache control flag. If True, enables caching; if False, disables caching;
                                 if a string, uses a specific cache identifier. Defaults to True.

**Returns:**\n
- Returns the module instance itself, supporting method chaining.

""")
add_chinese_doc('ModuleBase.update_server', '''\
更新模块及其子模块的部署（server）部分。当模块或子模块实现了部署功能时，会进行相应的服务启动。  

Args:
    recursive (bool): 是否递归更新所有子模块的部署任务，默认为 True。
''')

add_english_doc('ModuleBase.update_server', '''\
Update the deployment (server) part of the module and its submodules. When a module or submodule implements deployment functionality, the corresponding services will be started.

Args:
    recursive (bool): Whether to recursively update deployment tasks of all submodules, default is True.
''')

add_chinese_doc('ModuleBase.evalset', '''\
为模块设置评测集（evaluation set）。  
模块在调用 ``update`` 或 ``eval`` 时会使用评测集进行推理，并将评测结果存储在 ``eval_result`` 变量中。  

Args:
    evalset (Union[list, str]): 评测数据列表，或者评测数据文件路径。
    load_f (Optional[Callable]): 当 ``evalset`` 为文件路径时，用于加载文件并返回列表的函数，默认为 None。
    collect_f (Callable): 对评测结果进行后处理的函数，默认为 ``lambda x: x``。
''')

add_english_doc('ModuleBase.evalset', '''\
Set the evaluation set for the module.  
During ``update`` or ``eval``, the module will perform inference on the evaluation set, and the results will be stored in the ``eval_result`` variable.  

Args:
    evalset (Union[list, str]): Evaluation data list or path to an evaluation data file.
    load_f (Optional[Callable]): Function to load and parse the evaluation file into a list if ``evalset`` is a file path, default is None.
    collect_f (Callable): Function to post-process evaluation results, default is ``lambda x: x``.
''')

add_example('ModuleBase.evalset', '''\
>>> import lazyllm
>>> m = lazyllm.module.TrainableModule().deploy_method(lazyllm.deploy.dummy).finetune_method(lazyllm.finetune.dummy).trainset("").mode("finetune").prompt(None)
>>> m.evalset([1, 2, 3])
>>> m.update()
INFO: (lazyllm.launcher) PID: dummy finetune!, and init-args is {}
>>> print(m.eval_result)
["reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 2, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 3, and parameters is {'do_sample': False, 'temperature': 0.1}"]
''')

add_chinese_doc('ModuleBase.eval', '''\
对模块（及所有的子模块）进行评测。当模块通过 ``evalset`` 设置了评测集之后，本函数生效。

Args:
    recursive (bool): 是否递归评测所有的子模块，默认为True
''')

add_english_doc('ModuleBase.eval', '''\
Evaluate the module (and all its submodules). This function takes effect after the module has been set with an evaluation set using 'evalset'.

Args:
    recursive (bool): Whether to recursively evaluate all submodules. Defaults to True.
''')

add_example('ModuleBase.eval', '''\
>>> import lazyllm
>>> class MyModule(lazyllm.module.ModuleBase):
...     def forward(self, input):
...         return f'reply for input'
... 
>>> m = MyModule()
>>> m.evalset([1, 2, 3])
>>> m.eval().eval_result
['reply for input', 'reply for input', 'reply for input']
''')

add_chinese_doc('ModuleBase._get_train_tasks', '''\
定义训练任务，该函数返回训练的pipeline，重写了此函数的子类可以在update阶段被训练/微调。
''')

add_english_doc('ModuleBase._get_train_tasks', '''\
Define a training task. This function returns a training pipeline. Subclasses that override this function can be trained or fine-tuned during the update phase.
''')

add_example('ModuleBase._get_train_tasks', '''\
>>> import lazyllm
>>> class MyModule(lazyllm.module.ModuleBase):
...     def _get_train_tasks(self):
...         return lazyllm.pipeline(lambda : 1, lambda x: print(x))
... 
>>> MyModule().update()
1
''')

add_chinese_doc('ModuleBase._get_deploy_tasks', '''\
定义部署任务，该函数返回训练的pipeline，重写了此函数的子类可以在update/start阶段被部署。
''')

add_english_doc('ModuleBase._get_deploy_tasks', '''\
Define a deployment task. This function returns a deployment pipeline. Subclasses that override this function can be deployed during the update/start phase.
''')

add_example('ModuleBase._get_deploy_tasks', '''\
>>> import lazyllm
>>> class MyModule(lazyllm.module.ModuleBase):
...     def _get_deploy_tasks(self):
...         return lazyllm.pipeline(lambda : 1, lambda x: print(x))
... 
>>> MyModule().start()
1
''')

add_chinese_doc('ActionModule', '''\
用于将函数、模块、flow、Module等可调用的对象包装一个Module。被包装的Module（包括flow中的Module）都会变成该Module的submodule。

Args:
    action (Callable|list[Callable]): 被包装的对象，是一个或一组可执行的对象。
    return_trace (bool): 是否开启 trace 模式，用于记录调用栈，默认为 ``False``。
''')

add_english_doc('ActionModule', '''\
Used to wrap a Module around functions, modules, flows, Module, and other callable objects. The wrapped Module (including the Module within the flow) will become a submodule of this Module.

Args:
    action (Callable|list[Callable]): The object to be wrapped, which is one or a set of callable objects.
    return_trace (bool): Whether to enable trace mode to record the execution stack. Defaults to ``False``.

**Examples:**\n
```python
>>> import lazyllm
>>> def myfunc(input): return input + 1
... 
>>> class MyModule1(lazyllm.module.ModuleBase):
...     def forward(self, input): return input * 2
... 
>>> class MyModule2(lazyllm.module.ModuleBase):
...     def _get_deploy_tasks(self): return lazyllm.pipeline(lambda : print('MyModule2 deployed!'))
...     def forward(self, input): return input * 4
... 
>>> class MyModule3(lazyllm.module.ModuleBase):
...     def _get_deploy_tasks(self): return lazyllm.pipeline(lambda : print('MyModule3 deployed!'))
...     def forward(self, input): return f'get {input}'
... 
>>> m = lazyllm.ActionModule(myfunc, lazyllm.pipeline(MyModule1(), MyModule2), MyModule3())
>>> print(m(1))
get 16
>>> 
>>> m.evalset([1, 2, 3])
>>> m.update()
MyModule2 deployed!
MyModule3 deployed!
>>> print(m.eval_result)
['get 16', 'get 24', 'get 32']
```


<span style="font-size: 20px;">**`evalset(evalset, load_f=None, collect_f=<function ModuleBase.<lambda>>)`**</span>

Set the evaluation set for the Module. Modules that have been set with an evaluation set will be evaluated during ``update`` or ``eval``, and the evaluation results will be stored in the eval_result variable. 


<span style="font-size: 18px;">&ensp;**`evalset(evalset, collect_f=lambda x: ...)→ None `**</span>


Args:
    evalset (list) :Evaluation set
    collect_f (Callable) :Post-processing method for evaluation results, no post-processing by default.\n


<span style="font-size: 18px;">&ensp;**`evalset(evalset, load_f=None, collect_f=lambda x: ...)→ None`**</span>


Args:
    evalset (str) :Path to the evaluation set
    load_f (Callable) :Method for loading the evaluation set, including parsing file formats and converting to a list
    collect_f (Callable) :Post-processing method for evaluation results, no post-processing by default.

**Examples:**\n
```python
>>> import lazyllm
>>> m = lazyllm.module.TrainableModule().deploy_method(deploy.dummy)
>>> m.evalset([1, 2, 3])
>>> m.update()
INFO: (lazyllm.launcher) PID: dummy finetune!, and init-args is {}
>>> m.eval_result
["reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 2, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 3, and parameters is {'do_sample': False, 'temperature': 0.1}"]
```


''')

add_chinese_doc('ActionModule.forward', '''\
执行被包装的 action，对输入参数进行前向计算。等效于调用该模块本身。

Args:
    args (list of callables or single callable): 传递给被包装 action 的位置参数。
    kwargs (dict of callables): 传递给被包装 action 的关键字参数。

**Returns:**\n
- 任意类型：被包装 action 的执行结果。
''')

add_english_doc('ActionModule.forward', '''\
Executes the wrapped action with the provided input arguments. Equivalent to directly calling the module.

Args:
    args (list of callables or single callable): Positional arguments to be passed to the wrapped action.
    kwargs (dict of callables): Keyword arguments to be passed to the wrapped action.

**Returns:**\n
- Any: The result of executing the wrapped action.
''')

add_chinese_doc('ActionModule.submodules', '''\
返回被包装 action 中所有属于 ModuleBase 类型的子模块。该属性会自动展开 Pipeline 中嵌套的模块。

**Returns:**\n
- list[ModuleBase]: 子模块列表
''')

add_english_doc('ActionModule.submodules', '''\
Returns all submodules of type ModuleBase contained in the wrapped action. This automatically traverses any nested modules inside a Pipeline.

**Returns:**\n
- list[ModuleBase]: List of submodules
''')

# add_example('ActionModule', '''\
# >>> import lazyllm
# >>> def myfunc(input): return input + 1
# ...
# >>> class MyModule1(lazyllm.module.ModuleBase):
# ...     def forward(self, input): return input * 2
# ...
# >>> class MyModule2(lazyllm.module.ModuleBase):
# ...     def _get_deploy_tasks(self): return lazyllm.pipeline(lambda : print('MyModule2 deployed!'))
# ...     def forward(self, input): return input * 4
# ...
# >>> class MyModule3(lazyllm.module.ModuleBase):
# ...     def _get_deploy_tasks(self): return lazyllm.pipeline(lambda : print('MyModule3 deployed!'))
# ...     def forward(self, input): return f'get {input}'
# ...
# >>> m = lazyllm.ActionModule(myfunc, lazyllm.pipeline(MyModule1(), MyModule2), MyModule3())
# >>> print(m(1))
# get 16
# >>>
# >>> m.evalset([1, 2, 3])
# >>> m.update()
# MyModule2 deployed!
# MyModule3 deployed!
# >>> print(m.eval_result)
# ['get 16', 'get 24', 'get 32']
# ''')

add_chinese_doc('servermodule.LLMBase', '''\
大语言模型模块的基类，继承自 ModuleBase。  
负责管理流式输出、Prompt 和格式化器的初始化与切换，处理输入中的文件信息，支持实例共享。

Args:
    stream (bool 或 dict): 是否启用流式输出或流式配置，默认为 False。
    return_trace (bool): 是否返回执行过程的 trace，默认为 False。
    init_prompt (bool): 是否在初始化时自动创建默认 Prompt，默认为 True。
''')

add_english_doc('servermodule.LLMBase', '''\
Base class for large language model modules, inheriting from ModuleBase.  
Manages initialization and switching of streaming output, prompts, and formatters; processes file information in inputs; supports instance sharing.

Args:
    stream (bool or dict): Whether to enable streaming output or streaming configuration, default is False.
    return_trace (bool): Whether to return execution trace, default is False.
    init_prompt (bool): Whether to automatically create a default prompt at initialization, default is True.
''')

add_chinese_doc('servermodule.LLMBase.prompt', '''\
设置或切换 Prompt。支持 None、PrompterBase 子类或字符串/字典类型创建 ChatPrompter。

Args:
    prompt (str/dict/PrompterBase/None): 要设置的 Prompt。
    history (list): 对话历史，仅当 prompt 为字符串或字典时有效。

**Returns:**\n
- self: 便于链式调用。
''')

add_english_doc('servermodule.LLMBase.prompt', '''\
Set or switch the prompt. Supports None, PrompterBase subclass, or string/dict to create ChatPrompter.

Args:
    prompt (str/dict/PrompterBase/None): The prompt to set.
    history (list): Conversation history, only valid when prompt is str or dict.

**Returns:**\n
- self: For chaining calls.
''')

add_chinese_doc('servermodule.LLMBase.formatter', '''\
设置或切换输出格式化器。支持 None、FormatterBase 子类或可调用对象。

Args:
    format (FormatterBase/Callable/None): 格式化器对象或函数，默认为 None。

**Returns:**\n
- self: 便于链式调用。
''')

add_english_doc('servermodule.LLMBase.formatter', '''\
Set or switch the output formatter. Supports None, FormatterBase subclass or callable.

Args:
    format (FormatterBase/Callable/None): Formatter object or function, default is None.

**Returns:**\n
- self: For chaining calls.
''')

add_chinese_doc('servermodule.LLMBase.share', '''\
创建当前实例的浅拷贝，并可重新设置 prompt、formatter、stream 等属性。  
适用于多会话或多 Agent 共享基础配置但个性化部分参数的场景。

Args:
    prompt (str/dict/PrompterBase/None): 新的 Prompt，可选。
    format (FormatterBase/None): 新的格式化器，可选。
    stream (bool/dict/None): 新的流式设置，可选。
    history (list/None): 新的对话历史，仅在设置 Prompt 时有效。

**Returns:**\n
- LLMBase: 新的共享实例。
''')

add_english_doc('servermodule.LLMBase.share', '''\
Creates a shallow copy of the current instance, with optional resetting of prompt, formatter, and stream attributes.  
Useful for scenarios where multiple sessions or agents share a base configuration but customize certain parameters.

Args:
    prompt (str/dict/PrompterBase/None): New prompt, optional.
    format (FormatterBase/None): New formatter, optional.
    stream (bool/dict/None): New streaming settings, optional.
    history (list/None): New conversation history, effective only when setting prompt.

**Returns:**\n
- LLMBase: The new shared instance.
''')

add_chinese_doc('TrainableModule', '''\
可训练模块，所有模型（包括LLM、Embedding等）都通过TrainableModule提供服务

<span style="font-size: 20px;">**`TrainableModule(base_model='', target_path='', *, stream=False, return_trace=False)`**</span>


Args:
    base_model (str): 基础模型的名称或路径。
    target_path (str): 保存微调任务的路径。
    stream (bool): 输出流式结果。     
    return_trace (bool): 在trace中记录结果。
    trust_remote_code (bool): 是否信任远程代码。
    type (str/LLMType): 模型类型。
    source (str): 模型来源，如果未设置，将从环境变量LAZYLLM_MODEL_SOURCE读取。

<span style="font-size: 20px;">**`TrainableModule.trainset(v):`**</span>

设置 TrainableModule 的训练集

Args:
    v (str): 训练/微调数据集的路径

**示例:**\n
```python
>>> import lazyllm
>>> m = lazyllm.module.TrainableModule().finetune_method(finetune.dummy).trainset('/file/to/path').deploy_method(None).mode('finetune')
>>> m.update()
INFO: (lazyllm.launcher) PID: dummy finetune!, and init-args is {}
```

<span style="font-size: 20px;">**`TrainableModule.train_method(v, **kw):`**</span>

设置 TrainableModule 的训练方法（暂不支持继续预训练，预计下一版本支持）

Args:
    v (LazyLLMTrainBase): 训练方法，可选值包括 ``train.auto`` 等
    kw (**dict): 训练方法所需的参数，对应 v 的参数

<span style="font-size: 20px;">**`TrainableModule.finetune_method(v, **kw):`**</span>

设置 TrainableModule 的微调方法及其参数

Args:
    v (LazyLLMFinetuneBase): 微调方法，可选值包括 ``finetune.auto`` / ``finetune.alpacalora`` / ``finetune.collie`` 等
    kw (**dict): 微调方法所需的参数，对应 v 的参数

**示例:**\n            
```python
>>> import lazyllm
>>> m = lazyllm.module.TrainableModule().finetune_method(finetune.dummy).deploy_method(None).mode('finetune')
>>> m.update()
INFO: (lazyllm.launcher) PID: dummy finetune!, and init-args is {}    
```

<span style="font-size: 20px;">**`TrainableModule.deploy_method(v, **kw):`**</span>

设置 TrainableModule 的部署方法及其参数

Args:
    v (LazyLLMDeployBase): 部署方法，可选值包括 ``deploy.auto`` / ``deploy.lightllm`` / ``deploy.vllm`` 等
    kw (**dict): 部署方法所需的参数，对应 v 的参数

**示例:**\n
```python
>>> import lazyllm
>>> m = lazyllm.module.TrainableModule().deploy_method(deploy.dummy).mode('finetune')
>>> m.evalset([1, 2, 3])
>>> m.update()
INFO: (lazyllm.launcher) PID: dummy finetune!, and init-args is {}
>>> m.eval_result
["reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 2, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 3, and parameters is {'do_sample': False, 'temperature': 0.1}"]                                   
```

<span style="font-size: 20px;">**`TrainableModule.mode(v):`**</span>

设置 TrainableModule 在 update 时执行训练还是微调

Args:
    v (str): 设置在 update 时执行训练还是微调，可选值为 'finetune' 和 'train'，默认为 'finetune'

**示例:**\n
```python
>>> import lazyllm
>>> m = lazyllm.module.TrainableModule().finetune_method(finetune.dummy).deploy_method(None).mode('finetune')
>>> m.update()
INFO: (lazyllm.launcher) PID: dummy finetune!, and init-args is {}            
```
<span style="font-size: 20px;">**`eval(*, recursive=True)`**</span>
评估该模块（及其所有子模块）。此功能需在模块通过evalset设置评估集后生效。

Args:
    recursive (bool) :是否递归评估所有子模块，默认为True。

<span style="font-size: 20px;">**`evalset(evalset, load_f=None, collect_f=<function ModuleBase.<lambda>>)`**</span>

为模块设置评估集。已设置评估集的模块将在执行``update``或``eval``时进行评估，评估结果将存储在eval_result变量中。

<span style="font-size: 18px;">&ensp;**`evalset(evalset, collect_f=lambda x: ...)→ None `**</span>


Args:
    evalset (list) :评估数据集
    collect_f (Callable) :评估结果的后处理方法，默认不进行后处理。\n


<span style="font-size: 18px;">&ensp;**`evalset(evalset, load_f=None, collect_f=lambda x: ...)→ None`**</span>


Args:
    evalset (str) :评估集路径
    load_f (Callable) :评估集加载方法，包括文件格式解析和列表转换
    collect_f (Callable) :评估结果后处理方法，默认不进行后处理

**示例:**\n
```python
>>> import lazyllm
>>> m = lazyllm.module.TrainableModule().deploy_method(deploy.dummy)
>>> m.evalset([1, 2, 3])
>>> m.update()
INFO: (lazyllm.launcher) PID: dummy finetune!, and init-args is {}
>>> m.eval_result
["reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 2, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 3, and parameters is {'do_sample': False, 'temperature': 0.1}"]                
```     

<span style="font-size: 20px;">**`restart() `**</span>

重启模块及其所有子模块

**示例:**\n
```python
>>> import lazyllm
>>> m = lazyllm.module.TrainableModule().deploy_method(deploy.dummy)
>>> m.restart()
>>> m(1)
"reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}"
```

<span style="font-size: 20px;">start() </span>

部署模块及其所有子模块

**示例:**\n
```python
>>> import lazyllm
>>> m = lazyllm.module.TrainableModule().deploy_method(deploy.dummy)
>>> m.start()
>>> m(1)
"reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}"     
```                

''')

add_english_doc('TrainableModule', '''\
Trainable module, all models (including LLM, Embedding, etc.) are served through TrainableModule

<span style="font-size: 20px;">**`TrainableModule(base_model='', target_path='', *, stream=False, return_trace=False)`**</span>


Args:
    base_model (str): Name or path of the base model. 
    target_path (str): Path to save the fine-tuning task. 
    stream (bool): Whether to output stream. 
    return_trace (bool): Record the results in trace.
    trust_remote_code (bool): Whether to trust remote code.
    type (str/LLMType): Model type.
    source (str): Model source. If not set, it will read the value from the environment variable LAZYLLM_MODEL_SOURCE.


<span style="font-size: 20px;">**`TrainableModule.trainset(v):`**</span>

Set the training set for TrainableModule


Args:
    v (str): Path to the training/fine-tuning dataset.

**Examples:**\n
```python
>>> import lazyllm
>>> m = lazyllm.module.TrainableModule().finetune_method(finetune.dummy).trainset('/file/to/path').deploy_method(None).mode('finetune')
>>> m.update()
INFO: (lazyllm.launcher) PID: dummy finetune!, and init-args is {}
```

<span style="font-size: 20px;">**`TrainableModule.train_method(v, **kw):`**</span>

Set the training method for TrainableModule. Continued pre-training is not supported yet, expected to be available in the next version.

Args:
    v (LazyLLMTrainBase): Training method, options include ``train.auto`` etc.
    kw (**dict): Parameters required by the training method, corresponding to v.

<span style="font-size: 20px;">**`TrainableModule.finetune_method(v, **kw):`**</span>

Set the fine-tuning method and its parameters for TrainableModule.

Args:
    v (LazyLLMFinetuneBase): Fine-tuning method, options include ``finetune.auto`` / ``finetune.alpacalora`` / ``finetune.collie`` etc.
    kw (**dict): Parameters required by the fine-tuning method, corresponding to v.

**Examples:**\n            
```python
>>> import lazyllm
>>> m = lazyllm.module.TrainableModule().finetune_method(finetune.dummy).deploy_method(None).mode('finetune')
>>> m.update()
INFO: (lazyllm.launcher) PID: dummy finetune!, and init-args is {}                
```

<span style="font-size: 20px;">**`TrainableModule.deploy_method(v, **kw):`**</span>

Set the deployment method and its parameters for TrainableModule.

Args:
    v (LazyLLMDeployBase): Deployment method, options include ``deploy.auto`` / ``deploy.lightllm`` / ``deploy.vllm`` etc.
    kw (**dict): Parameters required by the deployment method, corresponding to v.

**Examples:**\n
```python
>>> import lazyllm
>>> m = lazyllm.module.TrainableModule().deploy_method(deploy.dummy).mode('finetune')
>>> m.evalset([1, 2, 3])
>>> m.update()
INFO: (lazyllm.launcher) PID: dummy finetune!, and init-args is {}
>>> m.eval_result
["reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 2, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 3, and parameters is {'do_sample': False, 'temperature': 0.1}"]
```                


<span style="font-size: 20px;">**`TrainableModule.mode(v):`**</span>

Set whether to execute training or fine-tuning during update for TrainableModule.

Args:
    v (str): Sets whether to execute training or fine-tuning during update, options are 'finetune' and 'train', default is 'finetune'.

**Examples:**\n
```python
>>> import lazyllm
>>> m = lazyllm.module.TrainableModule().finetune_method(finetune.dummy).deploy_method(None).mode('finetune')
>>> m.update()
INFO: (lazyllm.launcher) PID: dummy finetune!, and init-args is {}
```    

<span style="font-size: 20px;">**`eval(*, recursive=True)`**</span>
Evaluate the module (and all its submodules). This function takes effect after the module has set an evaluation set through evalset.

Args:
    recursive (bool) :Whether to recursively evaluate all submodules, default is True.                         

<span style="font-size: 20px;">**`evalset(evalset, load_f=None, collect_f=<function ModuleBase.<lambda>>)`**</span>

Set the evaluation set for the Module. Modules that have been set with an evaluation set will be evaluated during ``update`` or ``eval``, and the evaluation results will be stored in the eval_result variable. 


<span style="font-size: 18px;">&ensp;**`evalset(evalset, collect_f=lambda x: ...)→ None `**</span>


Args:
    evalset (list) :Evaluation set
    collect_f (Callable) :Post-processing method for evaluation results, no post-processing by default.\n


<span style="font-size: 18px;">&ensp;**`evalset(evalset, load_f=None, collect_f=lambda x: ...)→ None`**</span>


Args:
    evalset (str) :Path to the evaluation set
    load_f (Callable) :Method for loading the evaluation set, including parsing file formats and converting to a list
    collect_f (Callable) :Post-processing method for evaluation results, no post-processing by default.

**Examples:**\n
```python
>>> import lazyllm
>>> m = lazyllm.module.TrainableModule().deploy_method(deploy.dummy)
>>> m.evalset([1, 2, 3])
>>> m.update()
INFO: (lazyllm.launcher) PID: dummy finetune!, and init-args is {}
>>> m.eval_result
["reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 2, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 3, and parameters is {'do_sample': False, 'temperature': 0.1}"]
```

<span style="font-size: 20px;">**`restart() `**</span>

Restart the module and all its submodules.

**Examples:**\n
```python
>>> import lazyllm
>>> m = lazyllm.module.TrainableModule().deploy_method(deploy.dummy)
>>> m.restart()
>>> m(1)
"reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}"
```

<span style="font-size: 20px;">**`start() `**</span> 

Deploy the module and all its submodules.

**Examples:**\n
```python
import lazyllm
m = lazyllm.module.TrainableModule().deploy_method(deploy.dummy)
m.start()
m(1)
"reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}"
```                                  
''')

add_chinese_doc('TrainableModule.wait', '''\
等待模型部署任务完成，该方法会阻塞当前线程直到部署完成。
''')

add_english_doc('TrainableModule.wait', '''\
Wait for the model deployment task to complete. This method blocks the current thread until the deployment is finished.
''')

add_example('TrainableModule.wait', '''\
>>> import lazyllm
>>> class Mywait(lazyllm.module.llms.TrainableModule):
...    def forward(self):
...        self.wait()
''')

add_chinese_doc('TrainableModule.stop', '''\
暂停模型特定任务。

Args:
    task_name(str): 需要暂停的任务名, 默认为None(默认暂停deploy任务)
''')

add_english_doc('TrainableModule.stop', '''\
Pause a specific task of the model.

Args:
    task_name (str): The name of the task to pause. Defaults to None (pauses the 'deploy' task by default).
''')

add_example('TrainableModule.stop', '''\
>>> import lazyllm
>>> class Mystop(lazyllm.module.llms.TrainableModule):
...    def forward(self, task):
...        self.stop(task)
''')

add_chinese_doc('TrainableModule.prompt', '''\
处理输入的prompt生成符合模型需求的格式。

Args:
    prompt(str): 输入的prompt, 默认为空。
    history(**List): 对话历史记忆。
''')

add_english_doc('TrainableModule.prompt', '''\
Processes the input prompt and generates a format compatible with the model.

Args:
    prompt (str): The input prompt. Defaults to an empty string.
    history (List): Conversation history.
''')

add_example('TrainableModule.prompt', '''\
>>> import lazyllm
>>> class Myprompt(lazyllm.module.llms.TrainableModule):
...    def forward(self, prompt, history):
...        self.prompt(prompt,history)
''')

add_chinese_doc('TrainableModule.forward', '''\
自动构建符合模型要求的输入数据结构，适配多模态场景。
''')

add_english_doc('TrainableModule.forward', '''\
Supports handling various input formats, automatically builds the input structure required by the model, and adapts to multimodal scenarios.
''')

add_example('TrainableModule.forward', '''\
>>> import lazyllm
>>> from lazyllm.module import TrainableModule
>>> class MyModule(TrainableModule):
...     def forward(self, __input, **kw):
...         return f"processed: {__input}"
...
>>> MyModule()("Hello")
'processed: Hello'
''')

add_english_doc("TrainableModule.get_all_models", '''\
get_all_models() -> List[str]

Returns a list of all fine-tuned model paths under the current target path.

**Returns:**\n
- List[str]: A list of fine-tuned model identifiers or directories.
''')

add_chinese_doc("TrainableModule.get_all_models", '''\
get_all_models() -> List[str]

返回当前目标路径下所有微调模型的路径列表。

**Returns:**\n
- List[str]：所有微调模型的名称或路径列表。
''')

add_english_doc("TrainableModule.status", '''\
status(task_name: Optional[str] = None) -> str

Returns the current status of a specific task in the module.

Args:
    task_name (Optional[str]): Name of the task (e.g., 'deploy'). Defaults to 'deploy' if not provided.

**Returns:**\n
- str: Status string such as 'running', 'finished', or 'stopped'.
''')

add_chinese_doc("TrainableModule.status", '''\
status(task_name: Optional[str] = None) -> str

返回模块中指定任务的当前状态。

Args：
    task_name (Optional[str])：任务名称（如 'deploy'），默认返回 'deploy' 任务的状态。

**Returns:**\n
- str：状态字符串，例如 'running'、'finished' 或 'stopped'。
''')

add_english_doc("TrainableModule.set_specific_finetuned_model", '''\
set_specific_finetuned_model(model_path: str) -> None

Sets the model to be used from a specific fine-tuned model path.

Args:
    model_path (str): The path to the fine-tuned model to use.
''')

add_chinese_doc("TrainableModule.set_specific_finetuned_model", '''\
set_specific_finetuned_model(model_path: str) -> None

设置要使用的特定微调模型路径。

Args:
    model_path (str)：要使用的微调模型的路径。
''')

add_english_doc("TrainableModule.set_default_parameters", '''\
set_default_parameters(*, optional_keys: List[str] = [], **kw) -> None

Sets the default parameters to be used during inference or evaluation.

Args:
    optional_keys (List[str]): A list of optional keys to allow additional parameters without error.
    **kw: Key-value pairs for default parameters such as temperature, top_k, etc.
''')

add_chinese_doc("TrainableModule.set_default_parameters", '''\
set_default_parameters(*, optional_keys: List[str] = [], **kw) -> None

设置用于推理或评估的默认参数。

Args:
    optional_keys (List[str])：允许传入额外参数的可选键列表。
    **kw：用于设置默认参数的键值对，如 temperature、top_k 等。
''')

add_chinese_doc('TrainableModule.log_path', """\
获取任务日志路径。

根据任务名称获取对应的日志文件路径，支持默认部署任务和手动指定任务。

Args:
    task_name (Optional[str]): 任务名称，默认为None（获取默认部署任务日志）

Returns:
    str: 日志文件路径
""")

add_english_doc('TrainableModule.log_path', """\
Get task log path.

Get corresponding log file path based on task name, supports default deployment tasks and manually specified tasks.

Args:
    task_name (Optional[str]): Task name, defaults to None (get default deployment task log)

Returns:
    str: Log file path
""")

add_chinese_doc('TrainableModule.forward_openai', """\
使用OpenAI兼容接口进行前向推理。

通过OpenAI标准API格式调用部署的模型服务，支持聊天历史、文件处理、工具调用和流式输出。

Args:
    __input (Union[Tuple[Union[str, Dict], str], str, Dict]): 输入数据，可以是文本、字典或打包数据
    llm_chat_history: 聊天历史记录
    lazyllm_files: 文件数据
    tools: 工具调用配置
    stream_output (bool): 是否流式输出
    **kw: 其他关键字参数

Returns:
    模型推理结果
""")

add_english_doc('TrainableModule.forward_openai', """\
Perform forward inference using OpenAI compatible interface.

Call deployed model service through OpenAI standard API format, supports chat history, file processing, tool calling and streaming output.

Args:
    __input (Union[Tuple[Union[str, Dict], str], str, Dict]): Input data, can be text, dictionary or packaged data
    llm_chat_history: Chat history records
    lazyllm_files: File data
    tools: Tool calling configuration
    stream_output (bool): Whether to stream output
    **kw: Other keyword arguments

Returns:
    Model inference result
""")

add_chinese_doc('TrainableModule.forward_standard', """\
使用标准接口进行前向推理。

通过自定义标准API格式调用部署的模型服务，支持模板消息、文件编码和流式输出。

Args:
    __input (Union[Tuple[Union[str, Dict], str], str, Dict]): 输入数据，可以是文本、字典或打包数据
    llm_chat_history: 聊天历史记录
    lazyllm_files: 文件数据
    tools: 工具调用配置
    stream_output (bool): 是否流式输出
    **kw: 其他关键字参数

Returns:
    模型推理结果
""")

add_english_doc('TrainableModule.forward_standard', """\
Perform forward inference using standard interface.

Call deployed model service through custom standard API format, supports template messages, file encoding and streaming output.

Args:
    __input (Union[Tuple[Union[str, Dict], str], str, Dict]): Input data, can be text, dictionary or packaged data
    llm_chat_history: Chat history records
    lazyllm_files: File data
    tools: Tool calling configuration
    stream_output (bool): Whether to stream output
    **kw: Other keyword arguments

Returns:
    Model inference result
""")
# add_example('TrainableModule', '''\
# >>> import lazyllm
# >>> m = lazyllm.module.TrainableModule().finetune_method(finetune.dummy).trainset('/file/to/path').deploy_method(None).mode('finetune')
# >>> m.update()
# INFO: (lazyllm.launcher) PID: dummy finetune!, and init-args is {}
# >>> import lazyllm
# >>> m = lazyllm.module.TrainableModule().finetune_method(finetune.dummy).deploy_method(None).mode('finetune')
# >>> m.update()
# INFO: (lazyllm.launcher) PID: dummy finetune!, and init-args is {}
# >>> import lazyllm
# >>> m = lazyllm.module.TrainableModule().deploy_method(deploy.dummy).mode('finetune')
# >>> m.evalset([1, 2, 3])
# >>> m.update()
# INFO: (lazyllm.launcher) PID: dummy finetune!, and init-args is {}
# >>> m.eval_result
# ["reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 2, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 3, and parameters is {'do_sample': False, 'temperature': 0.1}"]
# >>> import lazyllm
# >>> m = lazyllm.module.TrainableModule().finetune_method(finetune.dummy).deploy_method(None).mode('finetune')
# >>> m.update()
# INFO: (lazyllm.launcher) PID: dummy finetune!, and init-args is {}
# ''')

add_chinese_doc('UrlModule', '''\
可以将ServerModule部署得到的Url包装成一个Module，调用 ``__call__`` 时会访问该服务。

Args:
    url (str): 要包装的服务的Url，默认为空字符串
    stream (bool|Dict[str, str]): 是否流式请求和输出，默认为非流式
    return_trace (bool): 是否将结果记录在trace中，默认为False
    init_prompt (bool): 是否初始化prompt，默认为True
''')

add_english_doc('UrlModule', '''\
The URL obtained from deploying the ServerModule can be wrapped into a Module. When calling ``__call__`` , it will access the service.

Args:
    url (str): The URL of the service to be wrapped, defaults to empty string.
    stream (bool|Dict[str, str]): Whether to request and output in streaming mode, default is non-streaming.
    return_trace (bool): Whether to record the results in trace, default is False.
    init_prompt (bool): Whether to initialize prompt, defaults to True.
''')

add_example('UrlModule', '''\
>>> import lazyllm
>>> def demo(input): return input * 2
... 
>>> s = lazyllm.ServerModule(demo, launcher=lazyllm.launchers.empty(sync=False))
>>> s.start()
INFO:     Uvicorn running on http://0.0.0.0:35485
>>> u = lazyllm.UrlModule(url=s._url)
>>> print(u(1))
2
''')

add_chinese_doc('UrlModule.forward', '''\
定义了每次执行的计算步骤，ModuleBase的所有的子类都需要重写这个函数。
''')

add_english_doc('UrlModule.forward', '''\
Defines the computation steps to be executed each time. All subclasses of ModuleBase need to override this function.
''')

add_example('UrlModule.forward', '''\
>>> import lazyllm
>>> class MyModule(lazyllm.module.ModuleBase):
...    def forward(self, input):
...        return input + 1
...
>>> MyModule()(1)
2
''')

add_chinese_doc('ServerModule', '''\
ServerModule 类，继承自 UrlModule，封装了将任意可调用对象部署为 API 服务的能力。  
通过 FastAPI 实现，可以启动一个主服务和多个卫星服务，并支持流式调用、预处理和后处理逻辑。  
既可以传入本地可调用对象启动服务，也可以通过 URL 直接连接远程服务。

Args:
    m (Optional[Union[str, ModuleBase]]): 被包装成服务的模块或其名称。若为字符串则表示 URL，此时 `url` 必须为 None；若为 ModuleBase 则包装为服务。
    pre (Optional[Callable]): 前处理函数，在服务进程执行，默认为 ``None``。
    post (Optional[Callable]): 后处理函数，在服务进程执行，默认为 ``None``。
    stream (Union[bool, Dict]): 是否开启流式输出。可以是布尔值，或包含流式配置的字典，默认为 ``False``。
    return_trace (Optional[bool]): 是否返回调试追踪信息。默认为 ``False``。
    port (Optional[int]): 指定服务部署的端口。默认为 ``None``，将自动分配端口。
    pythonpath (Optional[str]): 传递给子进程的 PYTHONPATH 环境变量，默认为 ``None``。
    launcher (Optional[LazyLLMLaunchersBase]): 启动服务所使用的 Launcher，默认使用异步远程部署。
    url (Optional[str]): 已部署服务的 URL 地址。若提供，则 `m` 必须为 None。
''')

add_english_doc('ServerModule', '''\
The ServerModule class inherits from UrlModule and provides functionality to deploy any callable object as an API service.  
Built on FastAPI, it supports launching a main service with multiple satellite services, as well as preprocessing, postprocessing, and streaming capabilities.  
A local callable can be deployed as a service, or an existing service can be accessed directly via a URL.

Args:
    m (Optional[Union[str, ModuleBase]]): The module or its name to be wrapped as a service.  
        If a string is provided, it is treated as a URL and `url` must be None.  
        If a ModuleBase is provided, it will be wrapped as a service.
    pre (Optional[Callable]): Preprocessing function executed in the service process. Default is ``None``.
    post (Optional[Callable]): Postprocessing function executed in the service process. Default is ``None``.
    stream (Union[bool, Dict]): Whether to enable streaming output. Can be a boolean or a dictionary with streaming configuration. Default is ``False``.
    return_trace (Optional[bool]): Whether to return debug trace information. Default is ``False``.
    port (Optional[int]): Port to deploy the service. If ``None``, a random port will be assigned.
    pythonpath (Optional[str]): PYTHONPATH environment variable passed to the subprocess. Defaults to ``None``.
    launcher (Optional[LazyLLMLaunchersBase]): The launcher used to deploy the service. Defaults to asynchronous remote deployment.
    url (Optional[str]): URL of an already deployed service. If provided, `m` must be None.
''')

add_example('ServerModule', '''\
>>> import lazyllm
>>> def demo(input): return input * 2
...
>>> s = lazyllm.ServerModule(demo, launcher=launchers.empty(sync=False))
>>> s.start()
INFO:     Uvicorn running on http://0.0.0.0:35485
>>> print(s(1))
2

>>> class MyServe(object):
...     def __call__(self, input):
...         return 2 * input
...
...     @lazyllm.FastapiApp.post
...     def server1(self, input):
...         return f'reply for {input}'
...
...     @lazyllm.FastapiApp.get
...     def server2(self):
...        return f'get method'
...
>>> m = lazyllm.ServerModule(MyServe(), launcher=launchers.empty(sync=False))
>>> m.start()
INFO:     Uvicorn running on http://0.0.0.0:32028
>>> print(m(1))
2
''')

add_chinese_doc('ServerModule.wait', '''\
等待当前模块服务的启动或执行过程完成。  
通常用于阻塞主线程，直到服务正常结束或中断。  
''')

add_english_doc('ServerModule.wait', '''\
Wait for the current module service to finish starting or executing.  
Typically used to block the main thread until the service finishes or is interrupted.  
''')

add_chinese_doc('ServerModule.stop', '''\
停止当前模块服务以及其相关子进程。  
调用后，模块将不再响应请求。  
''')

add_english_doc('ServerModule.stop', '''\
Stop the current module service and its related subprocesses.  
After this call, the module will no longer respond to requests.  
''')

# add_example('ServerModule', '''\
# >>> import lazyllm
# >>> def demo(input): return input * 2
# ...
# >>> s = lazyllm.ServerModule(demo, launcher=launchers.empty(sync=False))
# >>> s.start()
# INFO:     Uvicorn running on http://0.0.0.0:35485
# >>> print(s(1))
# 2

# >>> class MyServe(object):
# ...     def __call__(self, input):
# ...         return 2 * input
# ...
# ...     @lazyllm.FastapiApp.post
# ...     def server1(self, input):
# ...         return f'reply for {input}'
# ...
# ...     @lazyllm.FastapiApp.get
# ...     def server2(self):
# ...        return f'get method'
# ...
# >>> m = lazyllm.ServerModule(MyServe(), launcher=launchers.empty(sync=False))
# >>> m.start()
# >>> print(m(1))
# INFO:     Uvicorn running on http://0.0.0.0:32028
# >>> print(m(1))
# 2
# ''')

add_chinese_doc('TrialModule', '''\
参数网格搜索模块，会遍历其所有的submodule，收集所有的可被搜索的参数，遍历这些参数进行微调、部署和评测

Args:
    m (Callable): 被网格搜索参数的子模块，微调、部署和评测都会基于这个模块进行
''')

add_english_doc('TrialModule', '''\
Parameter grid search module will traverse all its submodules, collect all searchable parameters, and iterate over these parameters for fine-tuning, deployment, and evaluation.

Args:
    m (Callable): The submodule whose parameters will be grid-searched. Fine-tuning, deployment, and evaluation will be based on this module.
''')

add_example('TrialModule', '''\
>>> import lazyllm
>>> from lazyllm import finetune, deploy
>>> m = lazyllm.TrainableModule('b1', 't').finetune_method(finetune.dummy, **dict(a=lazyllm.Option(['f1', 'f2'])))
>>> m.deploy_method(deploy.dummy).mode('finetune').prompt(None)
>>> s = lazyllm.ServerModule(m, post=lambda x, ori: f'post2({x})')
>>> s.evalset([1, 2, 3])
>>> t = lazyllm.TrialModule(s)
>>> t.update()
>>>
dummy finetune!, and init-args is {a: f1}
dummy finetune!, and init-args is {a: f2}
[["post2(reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1})", "post2(reply for 2, and parameters is {'do_sample': False, 'temperature': 0.1})", "post2(reply for 3, and parameters is {'do_sample': False, 'temperature': 0.1})"], ["post2(reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1})", "post2(reply for 2, and parameters is {'do_sample': False, 'temperature': 0.1})", "post2(reply for 3, and parameters is {'do_sample': False, 'temperature': 0.1})"]]
''')

add_chinese_doc('TrialModule.work', '''\
静态方法，用于在子进程中复制模块、执行更新操作，并将评测结果放入队列中。

Args:
    m (Callable): 要执行更新操作的模块。
    q (multiprocessing.Queue): 用于存放评测结果的队列。
''')

add_english_doc('TrialModule.work', '''\
Static method to deepcopy the module, perform update in a subprocess, and put the evaluation result into a queue.

Args:
    m (Callable): The module to perform update on.
    q (multiprocessing.Queue): Queue to store evaluation results.
''')

add_chinese_doc('TrialModule.update', '''\
遍历模块的所有配置选项，使用多进程并行执行模块更新，并收集每个配置的评测结果。
''')

add_english_doc('TrialModule.update', '''\
Iterates through all configuration options of the module, updates the module in parallel using multiprocessing, and collects the evaluation results for each configuration.
''')

add_chinese_doc('AutoModel', '''\
用于快速创建在线推理模块 OnlineModule 或本地 TrainableModule 的工厂类。它会优先采用用户传入的参数，若开启 ``config`` 则会根据 ``auto_model_config_map`` 中的配置进行覆盖，然后自动判断应当构建在线模块还是本地模块：\n
- 当判定为在线模块时，参数会透传给 OnlineModule（自动匹配 OnlineChatModule / OnlineEmbeddingModule / OnlineMultiModalModule）。\n
- 当判定为本地模块时，则以 ``model`` 与用户参数初始化 TrainableModule，并读取 config map 里的配置参数。

Args:
    model (str): 指定模型名称。例如 ``Qwen3-32B``。必填。
    config_id (Optional[str]): 指定配置文件里的id。默认为空。
    source (Optional[str]): 使用的服务提供方。为在线模块（``OnlineModule``）指定 ``qwen`` / ``glm`` / ``openai`` 等；若设为 ``local`` 则强制创建本地 TrainableModule。
    type (Optional[str]): 模型类型。若未指定会尝试从 kwargs 中获取或由在线模块自动推断。
    config (Union[str, bool]): 是否启用 ``auto_model_config_map`` 的覆盖逻辑，或者用户指定的 config 文件路径。默认为 True。
    **kwargs: 兼容 `model` 的同义字段 `base_model` 和 `embed_model_name`，不接收其他用户传入的字段。
''')

add_english_doc('AutoModel', '''\
A factory for quickly creating either an online ``OnlineModule`` or a local ``TrainableModule``. It prioritizes user-provided arguments; when ``config`` is enabled, settings in ``auto_model_config_map`` can override them, and it automatically decides which module to build: \n
- For online mode, arguments are passed through to ``OnlineModule`` (automatically matching OnlineChatModule / OnlineEmbeddingModule / OnlineMultiModalModule).\n
- For local mode, it initializes ``TrainableModule`` with ``model`` and user parameters, then reads the config map for configuration values.

Args:
    model (str): Name of the model, e.g., ``Qwen3-32B``. Required.
    config_id (Optional[str]): ID from the config file. Defaults to empty.
    source (Optional[str]): Provider for online modules (``qwen`` / ``glm`` / ``openai``). Set to ``local`` to force a local TrainableModule.
    type (Optional[str]): Model type. If omitted, it will try to fetch from kwargs or be inferred by the online module.
    config (Union[str, bool]): Whether to enable overrides from ``auto_model_config_map``, or a user-specified config file path. Defaults to True.
    **kwargs: Accepts `base_model` and `embed_model_name` as synonyms for `model`; does not accept other user-provided fields.
''')

add_chinese_doc('OnlineModule', '''\
在线模型基类，用来管理创建目前市面上公开的在线模型推理服务，包括LLM模块、Embedding模块以及多模态模块。
根据用户指定的在线模型类型和模型名自动创建对应的模块实例，目前支持的实例类型包括OnlineChatModule, OnlineEmbeddingModule和OnlineMultiModalModule。
                
Args:
    type (Optional[str]): 指定在线模型服务的类型，如果不指定则默认为 ``llm``。目前支持 ``llm`` / ``vlm`` / ``embed`` / ``cross_modal_embed`` / ``rerank`` / ``stt`` / ``tts`` / ``sd`` 这几类。
    model (Optional[str]): 指定要加载的模型名称，例如 ``internlm2-chat-7b``，可为空。为空时默认加载 ``internlm2-chat-7b``。
    source (Optional[str]): 指定要创建的模块类型，可选为 ``openai`` /  ``sensenova`` /  ``glm`` /  ``kimi`` /  ``qwen`` / ``doubao`` 等。
    url (Optional[str]): 指定要访问的平台的基础链接，默认是官方链接
    **kwargs: 其他传递给基类的参数。
''')

add_english_doc('OnlineModule', '''\
Base class for online models that orchestrates creation of publicly available online inference services, covering LLM, embedding, and multimodal modules.  
Automatically creates the proper module instance according to the requested model type and model name. Supported module classes currently include OnlineChatModule, OnlineEmbeddingModule, and OnlineMultiModalModule.
                
Args:
    type (Optional[str]): Category of the online service. Defaults to ``llm``. Supported options include ``llm`` / ``vlm`` / ``embed`` / ``cross_modal_embed`` / ``rerank`` / ``stt`` / ``tts`` / ``sd``.
    model (Optional[str]): Model to load, e.g., ``internlm2-chat-7b``. Defaults to ``internlm2-chat-7b`` when omitted.
    source (Optional[str]): Provider of the module to instantiate, e.g., ``openai`` / ``sensenova`` / ``glm`` / ``kimi`` / ``qwen`` / ``doubao``.
    url (Optional[str]): Base URL of the target platform, defaulting to the official endpoint.
    **kwargs: Additional keyword arguments passed to the base class.
''')

add_example('OnlineModule', '''\
>>> import lazyllm
>>> chat = lazyllm.OnlineModule(model="qwen-plus", source="qwen")
>>> isinstance(chat, lazyllm.OnlineChatModule)
True
>>> print(chat("Say hi in one sentence."))
Hi there! Happy to help.
>>> embed = lazyllm.OnlineModule(type="embed", source="qwen", model="text-embedding-v1")
>>> isinstance(embed, lazyllm.OnlineEmbeddingModule)
True
>>> vec = embed("LazyLLM routes models automatically.")
>>> len(vec)
1536
>>> tts = lazyllm.OnlineModule(type="tts", source="qwen", model="qwen-tts")
>>> isinstance(tts, lazyllm.OnlineMultiModalModule)
True
>>> audio_bytes = tts("Convert this line to speech.")
>>> len(audio_bytes) > 0
True
''')

add_chinese_doc('OnlineChatModule', '''\
用来管理创建目前市面上公开的大模型平台访问模块，目前支持openai、sensenova、glm、kimi、qwen、doubao、ppio、deekseek(由于该平台暂时不让充值了，暂时不支持访问)。平台的api key获取方法参见 [开始入门](/#platform)

Args:
    model (str): 指定要访问的模型 (注意使用豆包时需用 Model ID 或 Endpoint ID，获取方式详见 [获取推理接入点](https://www.volcengine.com/docs/82379/1099522)。使用模型前，要先在豆包平台开通对应服务。)，默认为 ``gpt-3.5-turbo(openai)`` / ``SenseChat-5(sensenova)`` / ``glm-4(glm)`` / ``moonshot-v1-8k(kimi)`` / ``qwen-plus(qwen)`` / ``mistral-7b-instruct-v0.2(doubao)`` / ``deepseek/deepseek-v3.2(ppio)`` 
    source (str): 指定要创建的模块类型，可选为 ``openai`` /  ``sensenova`` /  ``glm`` /  ``kimi`` /  ``qwen`` / ``doubao`` / ``ppio`` / ``deepseek(暂时不支持访问)``
    base_url (str): 指定要访问的平台的基础链接，默认是官方链接
    system_prompt (str): 指定请求的system prompt，默认是官方给的system prompt
    stream (bool): 是否流式请求和输出，默认为流式
    return_trace (bool): 是否将结果记录在trace中，默认为False
''')

add_english_doc('OnlineChatModule', '''\
Used to manage and create access modules for large model platforms currently available on the market. Currently, it supports openai, sensenova, glm, kimi, qwen, doubao, ppio and deepseek (since the platform does not allow recharges for the time being, access is not supported for the time being). For how to obtain the platform's API key, please visit [Getting Started](/#platform)

Args:
    model (str): Specify the model to access (Note that you need to use Model ID or Endpoint ID when using Doubao. For details on how to obtain it, see [Getting the Inference Access Point](https://www.volcengine.com/docs/82379/1099522). Before using the model, you must first activate the corresponding service on the Doubao platform.), default is ``gpt-3.5-turbo(openai)`` / ``SenseChat-5(sensenova)`` / ``glm-4(glm)`` / ``moonshot-v1-8k(kimi)`` / ``qwen-plus(qwen)`` / ``mistral-7b-instruct-v0.2(doubao)`` / ``deepseek/deepseek-v3.2(ppio)`` .
    source (str): Specify the type of module to create. Options include  ``openai`` /  ``sensenova`` /  ``glm`` /  ``kimi`` /  ``qwen`` / ``doubao`` / ``ppio`` / ``deepseek (not yet supported)`` .
    base_url (str): Specify the base link of the platform to be accessed. The default is the official link.
    system_prompt (str): Specify the requested system prompt. The default is the official system prompt.
    stream (bool): Whether to request and output in streaming mode, default is streaming.
    return_trace (bool): Whether to record the results in trace, default is False.      
''')

add_example('OnlineChatModule', '''\
>>> import lazyllm
>>> from functools import partial
>>> m = lazyllm.OnlineChatModule(source="sensenova", stream=True)
>>> query = "Hello!"
>>> with lazyllm.ThreadPoolExecutor(1) as executor:
...     future = executor.submit(partial(m, llm_chat_history=[]), query)
...     while True:
...         if value := lazyllm.FileSystemQueue().dequeue():
...             print(f"output: {''.join(value)}")
...         elif future.done():
...             break
...     print(f"ret: {future.result()}")
...
output: Hello
output: ! How can I assist you today?
ret: Hello! How can I assist you today?
>>> from lazyllm.components.formatter import encode_query_with_filepaths
>>> vlm = lazyllm.OnlineChatModule(source="sensenova", model="SenseChat-Vision")
>>> query = "what is it?"
>>> inputs = encode_query_with_filepaths(query, ["/path/to/your/image"])
>>> print(vlm(inputs))
''')

add_chinese_doc('llms.onlinemodule.supplier.doubao.DoubaoChat', '''\
豆包（Doubao）在线聊天模块，继承自 OnlineChatModuleBase。  
封装了对字节跳动 Doubao API 的调用，用于进行多轮问答交互。默认使用模型 `doubao-1-5-pro-32k-250115`，支持流式输出和调用链追踪。

Args:
    model (str): 使用的模型名称，默认为 `doubao-1-5-pro-32k-250115`。
    base_url (str): API 基础 URL，默认为 "https://ark.cn-beijing.volces.com/api/v3/"。
    api_key (Optional[str]): Doubao API Key，若未提供，则从 lazyllm.config['doubao_api_key'] 读取。
    stream (bool): 是否启用流式输出，默认为 True。
    return_trace (bool): 是否返回调用链追踪信息，默认为 False。
    **kwargs: 其他传递给基类 OnlineChatModuleBase 的参数。
''')

add_english_doc('llms.onlinemodule.supplier.doubao.DoubaoChat', '''\
Doubao online chat module, inheriting from OnlineChatModuleBase.  
Encapsulates the Doubao API (ByteDance) for multi-turn Q&A interactions. Defaults to model `doubao-1-5-pro-32k-250115`, supporting streaming and optional trace return.

Args:
    model (str): The model name to use. Defaults to `doubao-1-5-pro-32k-250115`.
    base_url (str): Base URL of the API, default is "https://ark.cn-beijing.volces.com/api/v3/".
    api_key (Optional[str]): Doubao API key. If not provided, it is read from `lazyllm.config['doubao_api_key']`.
    stream (bool): Whether to enable streaming output. Defaults to True.
    return_trace (bool): Whether to return trace information. Defaults to False.
    **kwargs: Additional arguments passed to the base class OnlineChatModuleBase.
''')

add_chinese_doc('llms.onlinemodule.supplier.ppio.PPIOChat', '''\
PPIO（派欧云）在线聊天模块，继承自 OnlineChatModuleBase。  
封装了对 PPIO (Paiou Cloud) API 的调用，用于进行多轮问答交互。默认使用模型 `deepseek/deepseek-v3.2`，支持流式输出和调用链追踪。PPIO 提供 OpenAI 兼容的 API 接口。

Args:
    model (str): 使用的模型名称，默认为 `deepseek/deepseek-v3.2`。
    base_url (str): API 基础 URL，默认为 "https://api.ppinfra.com/openai"。
    api_key (Optional[str]): PPIO API Key，若未提供，则从 lazyllm.config['ppio_api_key'] 读取。
    stream (bool): 是否启用流式输出，默认为 True。
    return_trace (bool): 是否返回调用链追踪信息，默认为 False。
    **kwargs: 其他传递给基类 OnlineChatModuleBase 的参数。
''')

add_english_doc('llms.onlinemodule.supplier.ppio.PPIOChat', '''\
PPIO (Paiou Cloud) online chat module, inheriting from OnlineChatModuleBase.  
Encapsulates the PPIO API for multi-turn Q&A interactions. Defaults to model `deepseek/deepseek-v3.2`, supporting streaming and optional trace return. PPIO provides OpenAI-compatible API interface.

Args:
    model (str): The model name to use. Defaults to `deepseek/deepseek-v3.2`.
    base_url (str): Base URL of the API, default is "https://api.ppinfra.com/openai".
    api_key (Optional[str]): PPIO API key. If not provided, it is read from `lazyllm.config['ppio_api_key']`.
    stream (bool): Whether to enable streaming output. Defaults to True.
    return_trace (bool): Whether to return trace information. Defaults to False.
    **kwargs: Additional arguments passed to the base class OnlineChatModuleBase.
''')

add_example('llms.onlinemodule.supplier.ppio.PPIOChat', '''\
>>> import lazyllm
>>> # Set environment variable: export LAZYLLM_PPIO_API_KEY=your_api_key
>>> # Or create config file ~/.lazyllm/config.json: {"ppio_api_key": "your_api_key"}
>>> chat = lazyllm.OnlineChatModule(source='ppio', model='deepseek/deepseek-v3.2')
>>> response = chat('Hello, how are you?')
>>> print(response)
''')

add_chinese_doc('llms.onlinemodule.supplier.doubao.DoubaoMultiModal', '''\
豆包多模态模块，继承自 OnlineMultiModalBase，封装了调用豆包多模态服务的能力。  
可通过指定 API Key、模型名称和服务基础 URL，远程调用豆包接口进行多模态数据处理和特征提取。

Args:
    api_key (Optional[str]): 访问豆包服务的 API Key，若未提供则从 lazyllm 配置中读取。
    model_name (Optional[str]): 使用的豆包多模态模型名称。
    base_url (str): 豆包服务的基础 URL，默认指向北京区域的服务地址。
    return_trace (bool): 是否返回调试追踪信息，默认为 False。
    **kwargs: 其他传递给 OnlineMultiModalBase 的参数。
''')

add_english_doc('llms.onlinemodule.supplier.doubao.DoubaoMultiModal', '''\
Doubao MultiModal module, inheriting from OnlineMultiModalBase, encapsulates the functionality to call Doubao's multimodal service.  
By specifying the API key, model name, and base service URL, it allows remote interaction with Doubao's API for multimodal data processing and feature extraction.

Args:
    api_key (Optional[str]): API key for accessing Doubao service. If not provided, it is read from lazyllm config.
    model_name (Optional[str]): Name of the Doubao multimodal model to use.
    base_url (str): Base URL of the Doubao service, defaulting to the Beijing region endpoint.
    return_trace (bool): Whether to return debug trace information, default is False.
    **kwargs: Additional parameters passed to OnlineMultiModalBase.
''')

add_chinese_doc('llms.onlinemodule.supplier.openai.OpenAIEmbed', '''\
OpenAI 在线嵌入模块。
该类封装了对 OpenAI 嵌入 API 的调用，默认使用模型 `text-embedding-ada-002`，用于将文本编码为向量表示。

Args:
    embed_url (str): OpenAI 嵌入 API 的 URL，默认为 "https://api.openai.com/v1/embeddings"。
    embed_model_name (str): 使用的嵌入模型名称，默认为 "text-embedding-ada-002"。
    api_key (str, optional): OpenAI 的 API Key。若未提供，则从 lazyllm.config 中读取。
''')

add_english_doc('llms.onlinemodule.supplier.openai.OpenAIEmbed', '''\
Online embedding module using OpenAI.
This class wraps the OpenAI Embedding API, defaulting to the `text-embedding-ada-002` model, and converts text into vector representations.

Args:
    embed_url (str): The URL endpoint of the OpenAI embedding API. Default is "https://api.openai.com/v1/embeddings".
    embed_model_name (str): The name of the embedding model to use. Default is "text-embedding-ada-002".
    api_key (str, optional): The OpenAI API key. If not provided, it will be read from `lazyllm.config`.
''')

add_chinese_doc('llms.onlinemodule.supplier.qwen.QwenSTT', '''\
基于千问多模态接口的语音转文本（STT）模块，默认使用 ``paraformer-v2`` 模型。

Args:
    model (str): 模型名称。默认为 ``None``，将依次从 ``lazyllm.config['qwen_stt_model_name']`` 或 ``QwenSTT.MODEL_NAME`` 获取。
    api_key (str): 千问 API 的密钥。默认为 ``None``。
    return_trace (bool): 是否返回推理的中间 trace 信息。默认为 ``False``。
    **kwargs: 传递给父类 ``QwenMultiModal`` 的额外参数。
''')

add_english_doc('llms.onlinemodule.supplier.qwen.QwenSTT', '''\
Speech-to-Text (STT) module based on Qwen's multimodal API, with ``paraformer-v2`` as the default model.

Args:
    model (str): Model name. Defaults to ``None``, in which case it will use ``lazyllm.config['qwen_stt_model_name']`` or ``QwenSTT.MODEL_NAME``.
    api_key (str): API key for Qwen service. Defaults to ``None``.
    return_trace (bool): Whether to return intermediate trace information during inference. Defaults to ``False``.
    **kwargs: Additional parameters passed to the parent class ``QwenMultiModal``.
''')

add_chinese_doc('OnlineEmbeddingModule', '''\
用来管理创建目前市面上的在线Embedding服务模块，目前支持openai、sensenova、glm、qwen、doubao

Args:
    source (str): 指定要创建的模块类型，可选为 ``openai`` /  ``sensenova`` /  ``glm`` /  ``qwen`` / ``doubao``
    embed_url (str): 指定要访问的平台的基础链接，默认是官方链接
    embed_mode_name (str): 指定要访问的模型 (注意使用豆包时需用 Model ID 或 Endpoint ID，获取方式详见 [获取推理接入点](https://www.volcengine.com/docs/82379/1099522)。使用模型前，要先在豆包平台开通对应服务。)，默认为 ``text-embedding-ada-002(openai)`` / ``nova-embedding-stable(sensenova)`` / ``embedding-2(glm)`` / ``text-embedding-v1(qwen)`` / ``doubao-embedding-text-240715(doubao)`` 
''')

add_english_doc('OnlineEmbeddingModule', '''\
Used to manage and create online Embedding service modules currently on the market, currently supporting openai, sensenova, glm, qwen, doubao.

Args:
    source (str): Specify the type of module to create. Options are  ``openai`` /  ``sensenova`` /  ``glm`` /  ``qwen`` / ``doubao``.
    embed_url (str): Specify the base link of the platform to be accessed. The default is the official link.
    embed_mode_name (str): Specify the model to access (Note that you need to use Model ID or Endpoint ID when using Doubao. For details on how to obtain it, see [Getting the Inference Access Point](https://www.volcengine.com/docs/82379/1099522). Before using the model, you must first activate the corresponding service on the Doubao platform.), default is ``text-embedding-ada-002(openai)`` / ``nova-embedding-stable(sensenova)`` / ``embedding-2(glm)`` / ``text-embedding-v1(qwen)`` / ``doubao-embedding-text-240715(doubao)``
''')

add_example('OnlineEmbeddingModule', '''\
>>> import lazyllm
>>> m = lazyllm.OnlineEmbeddingModule(source="sensenova")
>>> emb = m("hello world")
>>> print(f"emb: {emb}")
emb: [0.0010528564, 0.0063285828, 0.0049476624, -0.012008667, ..., -0.009124756, 0.0032043457, -0.051696777]
''')

add_chinese_doc('OnlineChatModuleBase', '''\
OnlineChatModuleBase是管理开放平台的LLM接口的公共组件，具备训练、部署、推理等关键能力。OnlineChatModuleBase本身不支持直接实例化，
                需要子类继承该类，并实现微调相关的上传文件、创建微调任务、查询微调任务以及和部署相关的创建部署服务、查询部署任务等接口。

如果你需要支持新的开放平台的LLM的能力，请让你自定义的类继承自OnlineChatModuleBase：

    1、根据新平台的模型返回参数情况考虑对返回结果进行后处理，如果模型返回的格式和openai一致，可以不用做任何处理
    2、如果新平台支持模型的微调，也需要继承FileHandlerBase类，该类主要是验证文件格式，并在自定义类中把.jsonl格式数据转换为模型支持的数据才能用于后面的模型训练
    3、如果新平台支持模型的微调，则需要实现文件上传、创建微调服务、查询微调服务的接口。即使新平台不用对微调后的模型进行部署，也请实现一个假的创建部署服务和查询部署服务的接口即可
    4、如果新平台支持模型的微调，可以提供一个支持微调的模型列表，有助于在微调服务时进行判断
    5、配置新平台支持的api_key到全局变量，通过lazyllm.config.add(变量名，类型，默认值，环境变量名)进行添加

Args:
    api_key (str): API访问密钥
    base_url (str): API基础URL
    model_name (str): 模型名称
    stream (Union[bool, Dict[str, str]]): 流式输出或流式配置
    return_trace (bool, optional): 返回追踪信息，默认为False
    skip_auth (bool, optional): 跳过认证，默认为False
    static_params (Optional[StaticParams], optional): 静态参数配置，默认为None
    **kwargs: 其他模型参数
''')

add_english_doc('OnlineChatModuleBase', '''\
OnlineChatModuleBase is a public component that manages the LLM interface for open platforms, and has key capabilities such as training, deployment, and inference. OnlineChatModuleBase itself does not support direct instantiation; it requires subclasses to inherit from this class and implement interfaces related to fine-tuning, such as uploading files, creating fine-tuning tasks, querying fine-tuning tasks, and deployment-related interfaces, such as creating deployment services and querying deployment tasks.

If you need to support the capabilities of a new open platform's LLM, please extend your custom class from OnlineChatModuleBase:

    1. Consider post-processing the returned results based on the parameters returned by the new platform's model. If the model's return format is consistent with OpenAI, no processing is necessary.
    2. If the new platform supports model fine-tuning, you must also inherit from the FileHandlerBase class. This class primarily validates file formats and converts .jsonl formatted data into a format supported by the model for subsequent training. 
    3. If the new platform supports model fine-tuning, you must implement interfaces for file upload, creating fine-tuning services, and querying fine-tuning services. Even if the new platform does not require deployment of the fine-tuned model, please implement dummy interfaces for creating and querying deployment services.
    4. If the new platform supports model fine-tuning, provide a list of models that support fine-tuning to facilitate judgment during the fine-tuning service process.
    5. Configure the api_key supported by the new platform as a global variable by using ``lazyllm.config.add(variable_name, type, default_value, environment_variable_name)`` .

Args:
    api_key (str): API access key
    base_url (str): API base URL
    model_name (str): Model name
    stream (Union[bool, Dict[str, str]]): Whether to stream output or stream configuration
    return_trace (bool, optional): Whether to return trace information, defaults to False
    skip_auth (bool, optional): Whether to skip authentication, defaults to False
    static_params (Optional[StaticParams], optional): Static parameter configuration, defaults to None
    **kwargs: Other model parameters
''')

add_example('OnlineChatModuleBase', '''\
>>> import lazyllm
>>> from lazyllm.module import OnlineChatModuleBase
>>> from lazyllm.module.onlineChatModule.fileHandler import FileHandlerBase
>>> class NewPlatformChatModule(OnlineChatModuleBase):
...     def __init__(self,
...                   base_url: str = "<new platform base url>",
...                   model: str = "<new platform model name>",
...                   system_prompt: str = "<new platform system prompt>",
...                   stream: bool = True,
...                   return_trace: bool = False):
...         super().__init__(model_type="new_class_name",
...                          api_key=lazyllm.config['new_platform_api_key'],
...                          base_url=base_url,
...                          system_prompt=system_prompt,
...                          stream=stream,
...                          return_trace=return_trace)
...
>>> class NewPlatformChatModule1(OnlineChatModuleBase, FileHandlerBase):
...     TRAINABLE_MODELS_LIST = ['model_t1', 'model_t2', 'model_t3']
...     def __init__(self,
...                   base_url: str = "<new platform base url>",
...                   model: str = "<new platform model name>",
...                   system_prompt: str = "<new platform system prompt>",
...                   stream: bool = True,
...                   return_trace: bool = False):
...         OnlineChatModuleBase.__init__(self,
...                                       model_type="new_class_name",
...                                       api_key=lazyllm.config['new_platform_api_key'],
...                                       base_url=base_url,
...                                       system_prompt=system_prompt,
...                                       stream=stream,
...                                       trainable_models=NewPlatformChatModule1.TRAINABLE_MODELS_LIST,
...                                       return_trace=return_trace)
...         FileHandlerBase.__init__(self)
...     
...     def _convert_file_format(self, filepath:str) -> str:
...         pass
...         return data_str
...
...     def _upload_train_file(self, train_file):
...         pass
...         return train_file_id
...
...     def _create_finetuning_job(self, train_model, train_file_id, **kw):
...         pass
...         return fine_tuning_job_id, status
...
...     def _query_finetuning_job(self, fine_tuning_job_id):
...         pass
...         return fine_tuned_model, status
...
...     def _create_deployment(self):
...         pass
...         return self._model_name, "RUNNING"
... 
...     def _query_deployment(self, deployment_id):
...         pass
...         return "RUNNING"
...
''')

add_chinese_doc('OnlineChatModuleBase.set_train_tasks', """\
设置模型微调训练任务参数。

配置微调训练所需的训练数据文件和训练超参数，为后续训练任务做准备。

Args:
    train_file: 训练数据文件路径或文件对象
    **kw: 训练超参数，如学习率、训练轮数等
""")

add_english_doc('OnlineChatModuleBase.set_train_tasks', """\
Set model fine-tuning training task parameters.

Configure training data file and training hyperparameters required for fine-tuning, preparing for subsequent training tasks.

Args:
    train_file: Training data file path or file object
    **kw: Training hyperparameters such as learning rate, training epochs, etc.
""")

add_chinese_doc('OnlineChatModuleBase.set_specific_finetuned_model', """\
设置并使用特定的已微调模型。

从已完成的微调模型列表中选择指定模型ID作为当前使用的模型。

Args:
    model_id (str): 要使用的微调模型ID

**异常:** \n
- ValueError: 当提供的model_id不在有效微调模型列表中时抛出
""")

add_english_doc('OnlineChatModuleBase.set_specific_finetuned_model', """\
Set and use specific fine-tuned model.

Select specified model ID from completed fine-tuned model list as current model to use.

Args:
    model_id (str): Fine-tuned model ID to use

**Exceptions:** \n
- ValueError: Raised when provided model_id is not in valid fine-tuned model list
""")

add_chinese_doc('OnlineEmbeddingModuleBase', '''\
OnlineEmbeddingModuleBase是管理开放平台的嵌入模型接口的基类，用于请求文本获取嵌入向量。不建议直接对该类进行直接实例化。需要特定平台类继承该类进行实例化。

如果你需要支持新的开放平台的嵌入模型的能力，请让你自定义的类继承自OnlineEmbeddingModuleBase：

1. 如果新平台的嵌入模型的请求和返回数据格式都和openai一样，可以不用做任何处理，只传url和模型即可
2. 如果新平台的嵌入模型的请求或者返回的数据格式和openai不一样，需要重写_encapsulated_data或_parse_response方法。
3. 配置新平台支持的api_key到全局变量，通过lazyllm.config.add(变量名，类型，默认值，环境变量名)进行添加

Args:
    embed_url (str): 嵌入API的URL地址。
    api_key (str): API访问密钥。
    embed_model_name (str): 嵌入模型名称。
    return_trace (bool, optional): 是否返回追踪信息，默认为False。
''')

add_english_doc('OnlineEmbeddingModuleBase', '''\
OnlineEmbeddingModuleBase is the base class for managing embedding model interfaces on open platforms, used for requesting text to obtain embedding vectors. It is not recommended to directly instantiate this class. Specific platform classes should inherit from this class for instantiation.\n

If you need to support the capabilities of embedding models on a new open platform, please extend your custom class from OnlineEmbeddingModuleBase:

1. If the request and response data formats of the new platform's embedding model are the same as OpenAI's, no additional processing is needed; simply pass the URL and model.
2. If the request or response data formats of the new platform's embedding model differ from OpenAI's, you need to override the _encapsulated_data or _parse_response methods.
3. Configure the api_key supported by the new platform as a global variable by using ``lazyllm.config.add(variable_name, type, default_value, environment_variable_name)`` .

Args:
    embed_url (str): Embedding API URL address.
    api_key (str): API access key.
    embed_model_name (str): Embedding model name.
    return_trace (bool, optional): Whether to return trace information, defaults to False.
''')

add_example('OnlineEmbeddingModuleBase', '''\
>>> import lazyllm
>>> from lazyllm.module import OnlineEmbeddingModuleBase
>>> class NewPlatformEmbeddingModule(OnlineEmbeddingModuleBase):
...     def __init__(self,
...                 embed_url: str = '<new platform embedding url>',
...                 embed_model_name: str = '<new platform embedding model name>'):
...         super().__init__(embed_url, lazyllm.config['new_platform_api_key'], embed_model_name)
...
>>> class NewPlatformEmbeddingModule1(OnlineEmbeddingModuleBase):
...     def __init__(self,
...                 embed_url: str = '<new platform embedding url>',
...                 embed_model_name: str = '<new platform embedding model name>'):
...         super().__init__(embed_url, lazyllm.config['new_platform_api_key'], embed_model_name)
...
...     def _encapsulated_data(self, text:str, **kwargs):
...         pass
...         return json_data
...
...     def _parse_response(self, response: dict[str, any]):
...         pass
...         return embedding
''')

add_chinese_doc('OnlineEmbeddingModuleBase.run_embed_batch', """\
执行批量嵌入处理的内部方法。

此方法负责处理批量文本嵌入请求，支持单线程和多线程两种处理模式。
当遇到请求失败时，会自动调整批处理大小并重试，提供健壮的错误处理机制。

Args:
    input (List): 原始的输入文本列表
    data (List): 封装好的批量请求数据列表
    proxies: 代理设置，如果NO_PROXY为True则设置为None
    url (str, optional): 本次请求使用的完整接口地址，默认为初始传入的 embed_url
    **kwargs: 其他关键字参数

**Returns:**\n
- 嵌入向量列表的列表，每个子列表对应一个输入文本的嵌入向量
""")

add_english_doc('OnlineEmbeddingModuleBase.run_embed_batch', """\
Internal method for executing batch embedding processing.

This method handles batch text embedding requests, supporting both single-threaded 
and multi-threaded processing modes. It automatically adjusts batch size and retries 
on request failures, providing robust error handling mechanisms.

Args:
    input (List): Original input text list
    data (List): Encapsulated batch request data list
    proxies: Proxy settings, set to None if NO_PROXY is True
    url (str, optional): Full endpoint URL used for this request, default to be self._embed_url
    **kwargs: Additional keyword arguments

**Returns:**\n
- A list of embedding vector lists, each sublist corresponds to an input text's embedding vector
""")

add_chinese_doc('llms.onlinemodule.supplier.doubao.DoubaoEmbed', '''\
豆包嵌入类，继承自 OnlineEmbeddingModuleBase，封装了调用豆包在线文本嵌入服务的功能。  
通过指定服务接口 URL、模型名称及 API Key，支持远程获取文本向量表示。

Args:
    embed_url (Optional[str]): 豆包文本嵌入服务的接口 URL，默认指向北京区域的服务地址。
    embed_model_name (Optional[str]): 使用的豆包嵌入模型名称，默认为 "doubao-embedding-text-240715"。
    api_key (Optional[str]): 访问豆包服务的 API Key，若未提供则从 lazyllm 配置中读取。
''')

add_english_doc('llms.onlinemodule.supplier.doubao.DoubaoEmbed', '''\
DoubaoEmbed class inherits from OnlineEmbeddingModuleBase, encapsulating the functionality to call Doubao's online text embedding service.  
It supports remote text vector representation retrieval by specifying the service URL, model name, and API key.

Args:
    embed_url (Optional[str]): URL of the Doubao text embedding service, defaulting to the Beijing region endpoint.
    embed_model_name (Optional[str]): Name of the Doubao embedding model used, default is "doubao-embedding-text-240715".
    api_key (Optional[str]): API key for accessing the Doubao service. If not provided, it is read from lazyllm config.
''')

add_chinese_doc('llms.onlinemodule.supplier.doubao.DoubaoMultimodalEmbed', '''\
豆包多模态嵌入类，继承自 OnlineEmbeddingModuleBase，封装了调用豆包在线多模态（文本+图像）嵌入服务的功能。  
支持将文本和图像输入转换为统一的向量表示，通过指定服务接口 URL、模型名称及 API Key，实现远程获取多模态向量。

Args:
    embed_url (Optional[str]): 豆包多模态嵌入服务的接口 URL，默认指向北京区域的服务地址。
    embed_model_name (Optional[str]): 使用的豆包多模态嵌入模型名称，默认为 "doubao-embedding-vision-241215"。
    api_key (Optional[str]): 访问豆包服务的 API Key，若未提供则从 lazyllm 配置中读取。
''')

add_english_doc('llms.onlinemodule.supplier.doubao.DoubaoMultimodalEmbed', '''\
DoubaoMultimodalEmbed class inherits from OnlineEmbeddingModuleBase, encapsulating the functionality to call Doubao's online multimodal (text + image) embedding service.  
It supports converting text and image inputs into a unified vector representation by specifying the service URL, model name, and API key, enabling remote retrieval of multimodal embeddings.

Args:
    embed_url (Optional[str]): URL of the Doubao multimodal embedding service, defaulting to the Beijing region endpoint.
    embed_model_name (Optional[str]): Name of the Doubao multimodal embedding model used, default is "doubao-embedding-vision-241215".
    api_key (Optional[str]): API key for accessing the Doubao service. If not provided, it is read from lazyllm config.
''')

add_chinese_doc('llms.onlinemodule.supplier.glm.GLMChat', '''\
GLMChat 类，继承自 OnlineChatModuleBase 和 FileHandlerBase，封装了对智谱 GLM 系列模型的在线调用功能。  
支持对话生成、文件处理以及模型微调等能力。默认使用 GLM-4 模型，也可指定其他训练型模型（如 chatglm3-6b、chatglm_12b 等）。

Args:
    base_url (Optional[str]): 智谱 GLM 服务的 API 接口地址，默认为 "https://open.bigmodel.cn/api/paas/v4/"。
    model (Optional[str]): 使用的 GLM 模型名称，默认为 "glm-4"，也可选择 TRAINABLE_MODEL_LIST 中的其他模型。
    api_key (Optional[str]): 访问 GLM 服务的 API Key，若未提供则从 lazyllm 配置中读取。
    stream (Optional[bool]): 是否开启流式输出，默认为 True。
    return_trace (Optional[bool]): 是否返回调试追踪信息，默认为 False。
    **kwargs: 其他传递给 OnlineChatModuleBase 的可选参数。
''')

add_english_doc('llms.onlinemodule.supplier.glm.GLMChat', '''\
GLMChat class inherits from OnlineChatModuleBase and FileHandlerBase, encapsulating the functionality of accessing Zhipu's GLM series models online.  
It supports chat generation, file handling, and fine-tuning. The default model is GLM-4, but other trainable models (e.g., chatglm3-6b, chatglm_12b) are also supported.

Args:
    base_url (Optional[str]): API endpoint for Zhipu GLM service, default is "https://open.bigmodel.cn/api/paas/v4/".
    model (Optional[str]): Name of the GLM model to use. Defaults to "glm-4", or one from the TRAINABLE_MODEL_LIST.
    api_key (Optional[str]): API key for accessing GLM service. If not provided, it is read from lazyllm config.
    stream (Optional[bool]): Whether to enable streaming output. Defaults to True.
    return_trace (Optional[bool]): Whether to return debug trace information. Defaults to False.
    **kwargs: Additional optional parameters passed to OnlineChatModuleBase.
''')

add_chinese_doc('llms.onlinemodule.supplier.glm.GLMText2Image', '''\
GLM文本生成图像模块，继承自 GLMMultiModal，封装了调用 GLM CogView-4 模型生成图像的功能。  
支持根据文本提示（prompt）生成指定数量和分辨率的图像，并可通过 API Key 调用远程服务。

Args:
    model_name (Optional[str]): 使用的 GLM 模型名称，默认使用 "cogview-4-250304" 或配置中的 'glm_text_to_image_model_name'。
    api_key (Optional[str]): API Key，用于访问 GLM 图像生成服务。
    return_trace (bool): 是否返回调试追踪信息，默认为 False。
    **kwargs: 其他传递给 GLMMultiModal 的参数。
''')

add_english_doc('llms.onlinemodule.supplier.glm.GLMText2Image', '''\
GLM Text-to-Image module, inheriting from GLMMultiModal, encapsulates the functionality to generate images using the GLM CogView-4 model.  
It supports generating a specified number of images with given resolution based on a text prompt and can call the remote service via an API key.

Args:
    model_name (Optional[str]): Name of the GLM model to use, defaulting to "cogview-4-250304" or the 'glm_text_to_image_model_name' in config.
    api_key (Optional[str]): API key to access the GLM image generation service.
    return_trace (bool): Whether to return debug trace information, default is False.
    **kwargs: Additional parameters passed to GLMMultiModal.
''')

add_chinese_doc('llms.onlinemodule.supplier.qwen.QwenText2Image', '''\
Qwen文本生成图像模块和图像编辑模块，继承自 QwenMultiModal，封装了调用 Qwen Wanx2.1-t2i-turbo 模型生成图像的能力和调用Qwen-image-edit-plus模型进行图像编辑的能力。  
支持根据文本提示生成指定数量和分辨率的图像，支持图像编辑，并可设置负面提示、随机种子及扩展提示功能，通过 DashScope API 远程调用服务。

Args:
    model (Optional[str]): 使用的 Qwen 模型名称，默认从配置 'qwen_text2image_model_name' 获取，若未设置则使用 "wanx2.1-t2i-turbo"。
    api_key (Optional[str]): 调用 DashScope 服务的 API Key。
    return_trace (bool): 是否返回调试追踪信息，默认为 False。
    **kwargs: 其他传递给 QwenMultiModal 的参数。
''')

add_english_doc('llms.onlinemodule.supplier.qwen.QwenText2Image', '''\
Qwen Text-to-Image module and Image-Edit module, inheriting from QwenMultiModal, encapsulates the functionality to generate images using the Qwen Wanx2.1-t2i-turbo model.  
It supports generating a specified number of images with given resolution based on a text prompt, and allows setting negative prompts, random seeds, and prompt extension. The service is called remotely via DashScope API.

Args:
    model (Optional[str]): Name of the Qwen model to use, default is taken from config 'qwen_text2image_model_name', or "wanx2.1-t2i-turbo" if not set.
    api_key (Optional[str]): API key for accessing DashScope service.
    return_trace (bool): Whether to return debug trace information, default is False.
    **kwargs: Additional parameters passed to QwenMultiModal.
''')

add_chinese_doc('llms.onlinemodule.supplier.kimi.KimiChat', '''\
KimiChat 类，继承自 OnlineChatModuleBase，封装了调用 Moonshot AI 提供的 Kimi 聊天服务的能力。  
可通过指定 API Key、模型名称和服务 URL，支持中文和英文的安全问答交互，并支持图像输入的 base64 格式处理。

Args:
    base_url (str): Kimi 服务的基础 URL，默认为 "https://api.moonshot.cn/"。
    model (str): 使用的 Kimi 模型名称，默认为 "moonshot-v1-8k"。
    api_key (Optional[str]): 访问 Kimi 服务的 API Key，若未提供则从 lazyllm 配置中读取。
    stream (bool): 是否开启流式输出，默认为 True。
    return_trace (bool): 是否返回调试追踪信息，默认为 False。
    **kwargs: 其他传递给 OnlineChatModuleBase 的参数。
''')

add_english_doc('llms.onlinemodule.supplier.kimi.KimiChat', '''\
KimiChat class, inheriting from OnlineChatModuleBase, encapsulates the functionality to call Kimi chat service provided by Moonshot AI.  
By specifying the API key, model name, and service URL, it supports safe and accurate Chinese and English Q&A interactions, as well as image input in base64 format.

Args:
    base_url (str): Base URL of the Kimi service, default is "https://api.moonshot.cn/".
    model (str): Kimi model name to use, default is "moonshot-v1-8k".
    api_key (Optional[str]): API key for accessing Kimi service. If not provided, it is read from lazyllm config.
    stream (bool): Whether to enable streaming output, default is True.
    return_trace (bool): Whether to return debug trace information, default is False.
    **kwargs: Additional parameters passed to OnlineChatModuleBase.
''')

add_chinese_doc('llms.onlinemodule.fileHandler.FileHandlerBase', '''\
FileHandlerBase 是处理微调数据文件的基类，主要用于验证和转换微调数据格式。  
该类不支持直接实例化，需要子类继承并实现特定的文件格式转换逻辑。

功能包括：
    1. 验证微调数据文件格式是否为标准的 `.jsonl`。
    2. 检查每条数据是否包含符合规范的消息格式（包含 `role` 和 `content` 字段）。
    3. 验证角色类型是否在允许范围内（system、knowledge、user、assistant）。
    4. 确保每个对话示例包含至少一条 assistant 回复。
    5. 提供临时文件存储机制，便于后续处理。
''')

add_english_doc('llms.onlinemodule.fileHandler.FileHandlerBase', '''\
FileHandlerBase is a base class for handling fine-tuning data files, mainly used for validating and converting fine-tuning data formats.  
This class cannot be instantiated directly; it must be inherited by a subclass that implements specific file format conversion logic.

Capabilities include:
    1. Validate that the fine-tuning data file is in standard `.jsonl` format.
    2. Check that each data entry contains messages in the correct format (with `role` and `content` fields).
    3. Verify that roles are within the allowed range (system, knowledge, user, assistant).
    4. Ensure each conversation example contains at least one assistant response.
    5. Provide temporary file storage for further processing.
''')

add_example('llms.onlinemodule.fileHandler.FileHandlerBase', '''\
>>> import lazyllm
>>> from lazyllm.module.llms.onlinemodule.fileHandler import FileHandlerBase
>>> import tempfile
>>> import json
>>> sample_data = [
...     {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]},
...     {"messages": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I'm doing well, thank you!"}]}
... ] 
>>> with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
...     for item in sample_data:
...         f.write(json.dumps(item, ensure_ascii=False) + '\n')
...     temp_file_path = f.name
>>> class CustomFileHandler(FileHandlerBase):
...     def _convert_file_format(self, filepath: str) -> str:
...         with open(filepath, 'r', encoding='utf-8') as f:
...             data = [json.loads(line) for line in f]
...         converted_data = []
...         for item in data:
...             messages = item.get('messages', [])
...             conversation = []
...             for msg in messages:
...                 conversation.append(f"{msg['role']}: {msg['content']}")
...             converted_data.append('\n'.join(conversation))
...         return '\n---\n'.join(converted_data)
>>> handler = CustomFileHandler()
>>> try:
...     result = handler.get_finetune_data(temp_file_path)
...     print("数据验证和转换成功")
... except Exception as e:
...     print(f"错误: {e}")
... finally:
...     import os
...     os.unlink(temp_file_path)
''')

add_chinese_doc('llms.onlinemodule.fileHandler.FileHandlerBase.get_finetune_data', '''\
获取并处理微调数据文件，包括验证文件格式和转换为目标平台支持的格式。

Args:
    filepath (str): 微调数据文件的路径，必须是.jsonl格式
''')

add_english_doc('llms.onlinemodule.fileHandler.FileHandlerBase.get_finetune_data', '''\
Get and process fine-tuning data files, including validating file format and converting to the format supported by the target platform.

Args:
    filepath (str): Path to the fine-tuning data file, must be in .jsonl format
''')

add_chinese_doc('llms.onlinemodule.supplier.glm.GLMRerank', '''\
智谱AI的重排序模块，继承自OnlineEmbeddingModuleBase，用于对文档进行相关性重排序。

Args:
    embed_url (str): 重排序API的基础URL，默认为"https://open.bigmodel.cn/api/paas/v4/rerank"。
    embed_model_name (str): 使用的模型名称，默认为"rerank"。
    api_key (str): 智谱AI的API密钥，如果未提供则从lazyllm.config['glm_api_key']读取。

属性：
    type: 返回模型类型，固定为"ONLINE_RERANK"。

主要功能：
    - 对输入的查询和文档列表进行相关性重排序
    - 支持自定义排序参数
    - 返回每个文档的相关性得分
''')

add_english_doc('llms.onlinemodule.supplier.glm.GLMRerank', '''\
Reranking module for Zhipu AI, inheriting from OnlineEmbeddingModuleBase, used for relevance reranking of documents.

Args:
    embed_url (str): Base URL for reranking API, defaults to "https://open.bigmodel.cn/api/paas/v4/rerank".
    embed_model_name (str): Model name to use, defaults to "rerank".
    api_key (str): Zhipu AI API key, if not provided will be read from lazyllm.config['glm_api_key'].

Properties:
    type: Returns model type, fixed as "ONLINE_RERANK".

Main Features:
    - Performs relevance reranking for input query and document list
    - Supports custom ranking parameters
    - Returns relevance scores for each document
''')

add_chinese_doc('llms.onlinemodule.supplier.glm.GLMMultiModal', '''\
智谱AI的多模态基础模块，继承自OnlineMultiModalBase，用于处理多模态任务。

Args:
    model_name (str): 模型名称。
    api_key (str): API密钥，如果未提供则从lazyllm.config['glm_api_key']读取。
    base_url (str): API的基础URL，默认为'https://open.bigmodel.cn/api/paas/v4'。
    return_trace (bool): 是否返回调用追踪信息，默认为False。
    **kwargs: 其他传递给基类的参数。

功能特点：

    1. 支持多模态输入处理
    2. 使用ZhipuAI客户端进行API调用
    3. 提供统一的多模态接口
    4. 可自定义基础URL和API密钥

注意：
    该类作为GLM多模态功能的基础类，通常作为其他具体多模态实现（如语音转文本、文本生成图像等）的父类。
''')

add_english_doc('llms.onlinemodule.supplier.glm.GLMMultiModal', '''\
Zhipu AI's multimodal base module, inheriting from OnlineMultiModalBase, for handling multimodal tasks.

Args:
    model_name (str): Model name.
    api_key (str): API key, if not provided will be read from lazyllm.config['glm_api_key'].
    base_url (str): Base URL for API, defaults to 'https://open.bigmodel.cn/api/paas/v4'.
    return_trace (bool): Whether to return call trace information, defaults to False.
    **kwargs: Additional arguments passed to the base class.

Features:

    1. Supports multimodal input processing
    2. Uses ZhipuAI client for API calls
    3. Provides unified multimodal interface
    4. Customizable base URL and API key

Note:
    This class serves as the base class for GLM multimodal functionality, typically used as the parent class for specific multimodal implementations (such as speech-to-text, text-to-image, etc.).
''')

add_chinese_doc('llms.onlinemodule.supplier.qwen.QwenRerank', '''\
通义千问的重排序模块，继承自OnlineEmbeddingModuleBase，用于对文档进行相关性重排序。

Args:
    embed_url (str): 重排序API的基础URL，默认为"https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"。
    embed_model_name (str): 使用的模型名称，默认为"gte-rerank"。
    api_key (str): 通义千问的API密钥，如果未提供则从lazyllm.config['qwen_api_key']读取。
    **kwargs: 其他传递给基类的参数。

属性：
    type: 返回模型类型，固定为"ONLINE_RERANK"。

主要功能：
    - 对输入的查询和文档列表进行相关性重排序
    - 支持自定义排序参数
    - 返回每个文档的索引和相关性得分
''')

add_english_doc('llms.onlinemodule.supplier.qwen.QwenRerank', '''\
Qwen reranking module, inheriting from OnlineEmbeddingModuleBase, used for relevance reranking of documents.

Args:
    embed_url (str): Base URL for reranking API, defaults to "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank".
    embed_model_name (str): Model name to use, defaults to "gte-rerank".
    api_key (str): Qwen API key, if not provided will be read from lazyllm.config['qwen_api_key'].
    **kwargs: Additional arguments passed to the base class.

Properties:
    type: Returns model type, fixed as "ONLINE_RERANK".

Main Features:
    - Performs relevance reranking for input query and document list
    - Supports custom ranking parameters
    - Returns index and relevance score for each document
''')

add_chinese_doc('llms.onlinemodule.supplier.qwen.QwenMultiModal', '''\
通义千问的多模态基础模块，继承自OnlineMultiModalBase，用于处理多模态任务。

Args:
    api_key (str): API密钥，如果未提供则从lazyllm.config['qwen_api_key']读取。
    model_name (str): 模型名称。
    base_url (str): HTTP API的基础URL，默认为'https://dashscope.aliyuncs.com/api/v1'。
    base_websocket_url (str): WebSocket API的基础URL，默认为'wss://dashscope.aliyuncs.com/api-ws/v1/inference'。
    return_trace (bool): 是否返回调用追踪信息，默认为False。
    **kwargs: 其他传递给基类的参数。

功能特点：
    1. 支持HTTP和WebSocket两种API调用方式
    2. 使用DashScope客户端进行API调用
    3. 提供统一的多模态接口
    4. 可自定义基础URL和API密钥

注意：
    该类作为通义千问多模态功能的基础类，通常作为其他具体多模态实现（如语音转文本、文本生成图像等）的父类。
''')

add_english_doc('llms.onlinemodule.supplier.qwen.QwenMultiModal', '''\
Qwen's multimodal base module, inheriting from OnlineMultiModalBase, for handling multimodal tasks.

Args:
    api_key (str): API key, if not provided will read from lazyllm.config['qwen_api_key'].
    model_name (str): Model name.
    base_url (str): Base URL for HTTP API, defaults to 'https://dashscope.aliyuncs.com/api/v1'.
    base_websocket_url (str): Base URL for WebSocket API, defaults to 'wss://dashscope.aliyuncs.com/api-ws/v1/inference'.
    return_trace (bool): Whether to return call trace information, defaults to False.
    **kwargs: Additional parameters passed to the base class.

Features:
    1. Supports both HTTP and WebSocket API calls
    2. Uses DashScope client for API calls
    3. Provides unified multimodal interface
    4. Customizable base URLs and API key

Note:
    This class serves as the base class for Qwen's multimodal functionality, typically used as the parent class for other specific multimodal implementations (such as speech-to-text, text-to-image, etc.).
''')

add_chinese_doc('llms.onlinemodule.supplier.qwen.QwenTTS', '''\
通义千问的文本转语音模块，继承自QwenMultiModal，提供多种语音合成模型支持。

Args:
    model (str): 模型名称，默认为"qwen-tts"。可选模型包括：
        - cosyvoice-v2
        - cosyvoice-v1
        - sambert
        - qwen-tts
        - qwen-tts-latest
    api_key (str): API密钥，默认为None，将从lazyllm.config['qwen_api_key']读取。
    return_trace (bool): 是否返回调用追踪信息，默认为False。
    **kwargs: 其他传递给基类的参数。

语音合成参数：

    input (str): 要转换的文本内容。
    voice (str): 说话人声音，默认使用模型默认声音。
    speech_rate (float): 语速，默认为1.0。
    volume (int): 音量，默认为50。
    pitch (float): 音高，默认为1.0。

注意：
    - 不同的模型可能支持不同的声音选项
    - 返回的音频数据会被自动编码为文件格式
''')

add_english_doc('llms.onlinemodule.supplier.qwen.QwenTTS', '''\
Qwen's text-to-speech module, inheriting from QwenMultiModal, providing support for multiple speech synthesis models.

Args:
    model (str): Model name, defaults to "qwen-tts". Available models include:
        - cosyvoice-v2
        - cosyvoice-v1
        - sambert
        - qwen-tts
        - qwen-tts-latest
    api_key (str): API key, defaults to None, will be read from lazyllm.config['qwen_api_key'].
    return_trace (bool): Whether to return call trace information, defaults to False.
    **kwargs: Additional arguments passed to the base class.

Synthesis Parameters:

    input (str): Text content to convert.
    voice (str): Speaker voice, defaults to model's default voice.
    speech_rate (float): Speech rate, defaults to 1.0.
    volume (int): Volume, defaults to 50.
    pitch (float): Pitch, defaults to 1.0.

Note:
    - Different models may support different voice options
    - Returned audio data is automatically encoded into file format
''')

add_chinese_doc('llms.onlinemodule.supplier.sensenova.SenseNovaChat', '''\
SenseNovaChat是商汤科技开放平台的LLM接口管理组件，继承自OnlineChatModuleBase和FileHandlerBase，具备对话和文件处理能力。

Args:
    base_url (str): API的基础URL，默认为"https://api.sensenova.cn/compatible-mode/v1/"。
    model (str): 使用的模型名称，默认为"SenseChat-5"。
    api_key (str): 商汤API密钥，如果未提供则从lazyllm.config['sensenova_api_key']读取。
    secret_key (str): 商汤密钥，如果未提供则从lazyllm.config['sensenova_secret_key']读取。
    stream (bool): 是否启用流式输出，默认为True。
    return_trace (bool): 是否返回调用链跟踪信息，默认为False。
    **kwargs: 其他传递给基类的参数。
''')

add_english_doc('llms.onlinemodule.supplier.sensenova.SenseNovaChat', '''\
SenseNovaChat is the LLM interface management component for SenseTime's open platform, inheriting from OnlineChatModuleBase and FileHandlerBase, providing both chat and file handling capabilities.

Args:
    base_url (str): Base URL for the API, defaults to "https://api.sensenova.cn/compatible-mode/v1/".
    model (str): Name of the model to use, defaults to "SenseChat-5".
    api_key (str): SenseTime API key, if not provided will be read from lazyllm.config['sensenova_api_key'].
    secret_key (str): SenseTime secret key, if not provided will be read from lazyllm.config['sensenova_secret_key'].
    stream (bool): Whether to enable streaming output, defaults to True.
    return_trace (bool): Whether to return trace information, defaults to False.
    **kwargs: Additional arguments passed to the base class.
''')

add_chinese_doc('llms.onlinemodule.supplier.sensenova.SenseNovaChat.set_deploy_parameters', '''\
设置模型部署的参数。

Args:
    **kw: 部署参数的键值对，这些参数将在创建部署时使用。
''')

add_english_doc('llms.onlinemodule.supplier.sensenova.SenseNovaChat.set_deploy_parameters', '''\
Set parameters for model deployment.

Args:
    **kw: Key-value pairs of deployment parameters that will be used when creating deployment.
''')

add_chinese_doc('llms.onlinemodule.base.onlineMultiModalBase.OnlineMultiModalBase', '''\
多模态在线模型的基类，继承自LLMBase，提供多模态模型的基础功能实现。

Args:
    model_name (str): 模型名称，默认为None。如果未指定会产生警告。
    return_trace (bool): 是否返回调用追踪信息，默认为False。
    **kwargs: 其他传递给基类的参数。

属性：

    series: 返回模型系列名称。
    type: 返回模型类型，固定为"MultiModal"。

主要方法：

    share(): 创建模块的共享实例。
    forward(input, lazyllm_files, **kwargs): 处理输入和文件的主要方法。
    _forward(input, files, **kwargs): 需要被子类实现的具体前向处理方法。

注意：
    - 子类必须实现_forward方法。
    - 如果未指定模型名称(model_name)，系统会产生警告日志。
''')

add_english_doc('llms.onlinemodule.base.onlineMultiModalBase.OnlineMultiModalBase', '''\
Base class for online multimodal models, inheriting from LLMBase, providing basic functionality for multimodal models.

Args:
    model_name (str): Model name, defaults to None. A warning will be generated if not specified.
    return_trace (bool): Whether to return call trace information, defaults to False.
    **kwargs: Additional arguments passed to the base class.

Properties:

    series: Returns the model series name.
    type: Returns the model type, fixed as "MultiModal".

Main Methods:

    share(): Create a shared instance of the module.
    forward(input, lazyllm_files, **kwargs): Main method for handling input and files.
    _forward(input, files, **kwargs): Forward method to be implemented by subclasses.

Notes:
    - Subclasses must implement the _forward method.
    - A warning log will be generated if model name (model_name) is not specified.
''')

add_chinese_doc('llms.onlinemodule.supplier.openai.OpenAIChat', '''\
OpenAI API集成模块，用于聊天完成和微调操作。

提供与OpenAI聊天模型交互的接口，支持推理和微调功能。继承自OnlineChatModuleBase和FileHandlerBase。

Args:
    base_url (str, optional): OpenAI API基础URL，默认为"https://api.openai.com/v1/"。
    model (str, optional): 用于聊天完成的模型名称，默认为"gpt-3.5-turbo"。
    api_key (str, optional): OpenAI API密钥，默认为lazyllm.config['openai_api_key']。
    stream (bool, optional): 使用流式响应，默认为True。
    return_trace (bool, optional): 返回追踪信息，默认为False。
    **kwargs: 传递给OnlineChatModuleBase的额外参数。
''')

add_english_doc('llms.onlinemodule.supplier.openai.OpenAIChat', '''\
OpenAI API integration module for chat completion and fine-tuning operations.

Provides interface to interact with OpenAI's chat models, supporting both inference
and fine-tuning capabilities. Inherits from OnlineChatModuleBase and FileHandlerBase.

Args:
    base_url (str, optional): OpenAI API base URL, defaults to "https://api.openai.com/v1/".
    model (str, optional): Model name to use for chat completion, defaults to "gpt-3.5-turbo".
    api_key (str, optional): OpenAI API key, defaults to lazyllm.config['openai_api_key'].
    stream (bool, optional): Whether to use streaming response, defaults to True.
    return_trace (bool, optional): Whether to return trace information, defaults to False.
    **kwargs: Additional arguments passed to OnlineChatModuleBase.
''')

add_chinese_doc('llms.onlinemodule.supplier.openai.OpenAIRerank', '''
OpenAIRerank 类用于调用 OpenAI 的 Reranking 接口，对文本列表进行重排序（Re-ranking）。

该类继承自 `OnlineEmbeddingModuleBase`，主要功能包括：

- 设置嵌入（Embedding）模型的 URL 和名称；
- 封装请求数据并调用 OpenAI Rerank API；
- 解析返回的排序结果。

Args:
    embed_url (str): OpenAI API 的基础 URL，默认值为 'https://api.openai.com/v1/'。
    embed_model_name (str): 嵌入模型名称，用于指定 Rerank 模型。
    api_key (str): OpenAI API Key，可选，如果未提供，则使用 lazyllm 配置中的默认值。
    **kw: 其他可选关键字参数，传递给父类构造函数。
''')

add_english_doc('llms.onlinemodule.supplier.openai.OpenAIRerank', '''
The OpenAIRerank class provides functionality to call OpenAI's Reranking API for re-ordering a list of text documents.

This class inherits from `OnlineEmbeddingModuleBase` and mainly provides:

- Setting the embedding model URL and name;
- Encapsulating request data and calling the OpenAI Rerank API;
- Parsing the returned ranking results.

Args:
    embed_url (str): Base URL of the OpenAI API, default is 'https://api.openai.com/v1/'.
    embed_model_name (str): Name of the embedding model used for Rerank.
    api_key (str): OpenAI API Key, optional. If not provided, the default from lazyllm config is used.
    **kw: Additional keyword arguments passed to the parent constructor.
''')

add_chinese_doc('llms.onlinemodule.supplier.sensenova.SenseNovaEmbed', '''\
商汤科技SenseNova嵌入模型模块，用于文本向量化操作。提供与商汤科技SenseNova嵌入模型交互的接口，支持文本到向量的转换功能。继承自OnlineEmbeddingModuleBase和_SenseNovaBase。

Args:
    embed_url (str, optional): 嵌入API的URL地址，默认为"https://api.sensenova.cn/v1/llm/embeddings"。
    embed_model_name (str, optional): 嵌入模型名称，默认为"nova-embedding-stable"。
    api_key (str, optional): API访问密钥，默认为None。
    secret_key (str, optional): API秘密密钥，默认为None。
''')

add_english_doc('llms.onlinemodule.supplier.sensenova.SenseNovaEmbed', '''\
SenseTime SenseNova Embedding module for text vectorization operations.Provides interface to interact with SenseTime's SenseNova embedding models, supporting text-to-vector conversion functionality. Inherits from OnlineEmbeddingModuleBase and _SenseNovaBase.

Args:
    embed_url (str, optional): Embedding API URL, defaults to "https://api.sensenova.cn/v1/llm/embeddings".
    embed_model_name (str, optional): Embedding model name, defaults to "nova-embedding-stable".
    api_key (str, optional): API access key, defaults to None.
    secret_key (str, optional): API secret key, defaults to None.
''')

add_chinese_doc('llms.onlinemodule.supplier.doubao.DoubaoText2Image', '''\
字节跳动豆包文生图模块，支持纯文本生成图像和图像编辑模型。

基于字节跳动豆包多模态模型的文生图、图像编辑功能，继承自DoubaoMultiModal，
提供高质量的文本到图像生成能力。

Args:
    api_key (str, optional): 豆包API密钥，默认为None。
    model_name (str, optional): 模型名称，默认为"doubao-seedream-3-0-t2i-250415"。
    return_trace (bool, optional): 是否返回追踪信息，默认为False。
    **kwargs: 其他传递给父类的参数。
''')

add_english_doc('llms.onlinemodule.supplier.doubao.DoubaoText2Image', '''\
ByteDance Doubao Text-to-Image module supporting text to image generation and image editing.

Based on ByteDance Doubao multimodal model's text-to-image functionality, 
inherits from DoubaoMultiModal, providing high-quality text to image generation capability.

Args:
    api_key (str, optional): Doubao API key, defaults to None.
    model_name (str, optional): Model name, defaults to "doubao-seedream-3-0-t2i-250415".
    return_trace (bool, optional): Whether to return trace information, defaults to False.
    **kwargs: Other parameters passed to parent class.
''')

add_chinese_doc('llms.onlinemodule.supplier.deepseek.DeepSeekChat', """\
DeepSeek大语言模型接口模块。

Args:
    base_url (str): API基础URL，默认为"https://api.deepseek.com"
    model (str): 模型名称，默认为"deepseek-chat"
    api_key (str): API密钥，如果为None则从配置中获取
    stream (bool): 启用流式输出，默认为True
    return_trace (bool): 返回追踪信息，默认为False
    **kwargs: 其他传递给基类的参数
""")

add_english_doc('llms.onlinemodule.supplier.deepseek.DeepSeekChat', """\
DeepSeek large language model interface module.

Args:
    base_url (str): API base URL, defaults to "https://api.deepseek.com"
    model (str): Model name, defaults to "deepseek-chat"
    api_key (str): API key, if None, gets from configuration
    stream (bool): Whether to enable streaming output, defaults to True
    return_trace (bool): Whether to return trace information, defaults to False
    **kwargs: Other parameters passed to base class
""")

add_chinese_doc('llms.onlinemodule.supplier.glm.GLMEmbed', """\
GLM嵌入模型接口类，用于调用智谱AI的文本嵌入服务。

Args:
    embed_url (str): 嵌入服务API地址，默认为"https://open.bigmodel.cn/api/paas/v4/embeddings"
    embed_model_name (str): 嵌入模型名称，默认为"embedding-2"
    api_key (str): API密钥
""")

add_english_doc('llms.onlinemodule.supplier.glm.GLMEmbed', """\
GLM embedding model interface class for calling Zhipu AI's text embedding services.

Args:
    embed_url (str): Embedding service API address, defaults to "https://open.bigmodel.cn/api/paas/v4/embeddings"
    embed_model_name (str): Embedding model name, defaults to "embedding-2"
    api_key (str): API key
""")

add_chinese_doc('llms.onlinemodule.supplier.qwen.QwenEmbed', """\
通义千问在线文本嵌入模块。

该类继承自OnlineEmbeddingModuleBase，提供了与通义千问文本嵌入API的交互能力，支持将文本转换为向量表示。

Args:
    embed_url (str, optional): 嵌入API的URL地址。默认为通义千问官方API地址
    embed_model_name (str, optional): 嵌入模型名称。默认为 'text-embedding-v1'
    api_key (str, optional): API密钥。默认为从配置中获取的 'qwen_api_key'
""")

add_english_doc('llms.onlinemodule.supplier.qwen.QwenEmbed', """\
Qwen online text embedding module.

This class inherits from OnlineEmbeddingModuleBase and provides interaction capabilities with the Qwen text embedding API, supporting conversion of text to vector representations.

Args:
    embed_url (str, optional): Embedding API URL address. Defaults to Qwen official API address
    embed_model_name (str, optional): Embedding model name. Defaults to 'text-embedding-v1'
    api_key (str, optional): API key. Defaults to 'qwen_api_key' from configuration
""")

add_chinese_doc('llms.onlinemodule.supplier.qwen.QwenChat', """\
通义千问模型模块，继承自OnlineChatModuleBase和FileHandlerBase。

提供通义千问大语言模型的API调用、微调训练和部署管理功能，支持阿里云DashScope平台。

Args:
    base_url (str, optional): API基础URL，默认为"https://dashscope.aliyuncs.com/"
    model (str, optional): 模型名称，默认为配置中的模型名或"qwen-plus"
    api_key (str, optional): API密钥，默认为配置中的密钥
    stream (bool, optional): 是否流式输出，默认为True
    return_trace (bool, optional): 是否返回追踪信息，默认为False
    **kwargs: 其他模型参数
""")

add_english_doc('llms.onlinemodule.supplier.qwen.QwenChat', """\
Qwen (Tongyi Qianwen) model module, inherits from OnlineChatModuleBase and FileHandlerBase.

Provides API calls, fine-tuning training and deployment management for Qwen large language model, supports Alibaba Cloud DashScope platform.

Args:
    base_url (str, optional): API base URL, defaults to "https://dashscope.aliyuncs.com/"
    model (str, optional): Model name, defaults to configured model name or "qwen-plus"
    api_key (str, optional): API key, defaults to configured key
    stream (bool, optional): Whether to stream output, defaults to True
    return_trace (bool, optional): Whether to return trace information, defaults to False
    **kwargs: Other model parameters
""")

add_chinese_doc('llms.onlinemodule.supplier.qwen.QwenChat.set_deploy_parameters', """\
设置模型部署参数。

配置部署任务的相关参数，如容量规格等，用于后续模型部署。

Args:
    **kw: 部署参数键值对。
""")

add_english_doc('llms.onlinemodule.supplier.qwen.QwenChat.set_deploy_parameters', """\
Set model deployment parameters.

Configure relevant parameters for deployment tasks, such as capacity specifications, for subsequent model deployment.

Args:
    **kw: Deployment parameter key-value pairs.
""")

add_chinese_doc('llms.onlinemodule.supplier.glm.GLMSTT', """\
GLM语音识别模块，继承自GLMMultiModal。

提供基于智谱AI的语音转文本(STT)功能，支持音频文件的语音识别。

Args:
    model_name (str, optional): 模型名称，默认为配置中的模型名或"glm-asr"
    api_key (str, optional): API密钥，默认为配置中的密钥
    return_trace (bool, optional): 是否返回追踪信息，默认为False
    **kwargs: 其他模型参数
""")

add_english_doc('llms.onlinemodule.supplier.glm.GLMSTT', """\
GLM Speech-to-Text module, inherits from GLMMultiModal.

Provides speech-to-text (STT) functionality based on Zhipu AI, supports audio file speech recognition.

Args:
    model_name (str, optional): Model name, defaults to configured model name or "glm-asr"
    api_key (str, optional): API key, defaults to configured key
    return_trace (bool, optional): Whether to return trace information, defaults to False
    **kwargs: Other model parameters
""")

add_chinese_doc('llms.onlinemodule.supplier.siliconflow.SiliconFlowTTS', """\
SiliconFlow文本转语音模块，继承自OnlineMultiModalBase。

提供基于SiliconFlow的文本转语音(TTS)功能，支持将文本转换为音频文件。

Args:
    api_key (str, optional): API密钥，默认为配置中的siliconflow_api_key
    model_name (str, optional): 模型名称，默认为"fnlp/MOSS-TTSD-v0.5"
    base_url (str, optional): API基础URL，默认为"https://api.siliconflow.cn/v1/"
    return_trace (bool, optional): 是否返回追踪信息，默认为False
    **kwargs: 其他模型参数
""")

add_english_doc('llms.onlinemodule.supplier.siliconflow.SiliconFlowTTS', """\
SiliconFlow Text-to-Speech module, inherits from OnlineMultiModalBase.

Provides text-to-speech (TTS) functionality based on SiliconFlow, supports converting text to audio files.

Args:
    api_key (str, optional): API key, defaults to configured siliconflow_api_key
    model_name (str, optional): Model name, defaults to "fnlp/MOSS-TTSD-v0.5"
    base_url (str, optional): Base API URL, defaults to "https://api.siliconflow.cn/v1/"
    return_trace (bool, optional): Whether to return trace information, defaults to False
    **kwargs: Other model parameters
""")

add_chinese_doc('llms.onlinemodule.base.utils.LazyLLMOnlineBase', '''\
LazyLLM 在线模块基类，继承自 ModuleBase，并使用 LazyLLMRegisterMetaClass， 为所有在线服务模块提供统一的基础功能。  
该类封装了在线模块的通用行为，包括缓存机制和调试追踪功能，是构建各种在线API服务模块的基础类。

功能特性:
    - 继承 ModuleBase 的所有基础功能，包括子模块管理、钩子注册等。
    - 支持在线模块缓存机制，可通过配置控制是否启用缓存。
    - 提供调试追踪功能，便于问题排查和性能分析。
    - 作为所有在线服务模块（如聊天、嵌入、多模态等）的公共基类。

Args:
    return_trace (bool): 是否将推理结果写入 trace 队列，用于调试和追踪。默认为 ``False``。

使用场景:
    1. 作为在线聊天模块（OnlineChatModuleBase）的基类。
    2. 作为在线嵌入模块（OnlineEmbeddingModuleBase）的基类。
    3. 作为在线多模态模块（OnlineMultiModalBase）的基类。
    4. 为自定义在线服务模块提供统一的基础功能。
''')

add_english_doc('llms.onlinemodule.base.utils.LazyLLMOnlineBase', '''\
Base class for online modules, inheriting from ModuleBase and powered by LazyLLMRegisterMetaClass, providing unified basic functionality for all online service modules.  
This class encapsulates common behaviors of online modules, including caching mechanisms and debug tracing functionality, serving as the foundation for building various online API service modules.

Key Features:
    - Inherits all basic functionality from ModuleBase, including submodule management, hook registration, etc.
    - Supports online module caching mechanism, controllable through configuration.
    - Provides debug tracing functionality for troubleshooting and performance analysis.
    - Serves as a common base class for all online service modules (chat, embedding, multimodal, etc.).

Args:
    return_trace (bool): Whether to write inference results into the trace queue for debugging and tracking. Default is ``False``.

Use Cases:
    1. As a base class for online chat modules (OnlineChatModuleBase).
    2. As a base class for online embedding modules (OnlineEmbeddingModuleBase).
    3. As a base class for online multimodal modules (OnlineMultiModalBase).
    4. Providing unified basic functionality for custom online service modules.
''')

add_chinese_doc('module.ModuleCache', '''\
模块缓存管理器，提供统一的缓存存储和检索功能。  
该类封装了多种缓存策略（内存、文件、SQLite、Redis），支持根据配置自动选择缓存存储方式，为模块执行结果提供高效的缓存机制。

功能特性:
    - 支持多种缓存策略：内存缓存、文件缓存、SQLite数据库缓存、Redis缓存。
    - 自动根据配置选择缓存策略，默认为内存缓存。
    - 支持缓存模式控制（读写、只读、只写、禁用）。
    - 提供统一的缓存接口，隐藏底层存储实现细节。
    - 支持参数哈希化，确保缓存键的唯一性。

Args:
    strategy (Optional[str]): 缓存策略，可选值为 'memory'、'file'、'sqlite'、'redis'。默认为 None，将使用配置中的策略。

使用场景:
    1. 为模块执行结果提供缓存，避免重复计算。
    2. 在分布式环境中使用 Redis 缓存实现共享。
    3. 使用文件或数据库缓存实现持久化存储。
    4. 根据性能需求选择不同的缓存策略。
''')

add_english_doc('module.ModuleCache', '''\
Module cache manager providing unified cache storage and retrieval functionality.  
This class encapsulates multiple cache strategies (memory, file, SQLite, Redis), automatically selecting cache storage methods based on configuration, providing efficient caching mechanisms for module execution results.

Key Features:
    - Supports multiple cache strategies: memory cache, file cache, SQLite database cache, Redis cache.
    - Automatically selects cache strategy based on configuration, defaults to memory cache.
    - Supports cache mode control (read-write, read-only, write-only, disabled).
    - Provides unified cache interface, hiding underlying storage implementation details.
    - Supports parameter hashing to ensure uniqueness of cache keys.

Args:
    strategy (Optional[str]): Cache strategy, options include 'memory', 'file', 'sqlite', 'redis'. Defaults to None, will use strategy from configuration.

Use Cases:
    1. Provide caching for module execution results to avoid redundant computation.
    2. Use Redis cache in distributed environments for sharing.
    3. Use file or database cache for persistent storage.
    4. Select different cache strategies based on performance requirements.
''')

add_chinese_doc('module.ModuleCache.get', '''\
从缓存中获取数据。

根据提供的键和参数从缓存中检索数据。如果缓存模式不允许读取或数据不存在，将抛出异常。

Args:
    key: 缓存键，用于标识缓存数据。
    args: 位置参数，用于生成缓存哈希键。
    kw: 关键字参数，用于生成缓存哈希键。

**Returns:**\n
- 任意类型：缓存中存储的数据。

**异常:** \n
- CacheNotFoundError: 当缓存中不存在指定数据时抛出。
- RuntimeError: 当缓存模式设置为只写（WO）时抛出。
''')

add_english_doc('module.ModuleCache.get', '''\
Retrieve data from cache.

Retrieves data from cache based on the provided key and parameters. Throws an exception if cache mode doesn't allow reading or data doesn't exist.

Args:
    key: Cache key used to identify cached data.
    args: Positional arguments used to generate cache hash key.
    kw: Keyword arguments used to generate cache hash key.

**Returns:**\n
- Any: Data stored in cache.

**Exceptions:** \n
- CacheNotFoundError: Raised when specified data doesn't exist in cache.
- RuntimeError: Raised when cache mode is set to write-only (WO).
''')

add_chinese_doc('module.ModuleCache.set', '''\
将数据存储到缓存中。

根据提供的键和参数将数据存储到缓存中。如果缓存模式不允许写入，则直接返回不执行存储操作。

Args:
    key: 缓存键，用于标识缓存数据。
    args: 位置参数，用于生成缓存哈希键。
    kw: 关键字参数，用于生成缓存哈希键。
    value: 要存储的数据。

**注意:** \n
- 如果缓存模式设置为只读（RO）或禁用（NONE），此方法将直接返回而不执行存储操作。
''')

add_english_doc('module.ModuleCache.set', '''\
Store data in cache.

Stores data in cache based on the provided key and parameters. If cache mode doesn't allow writing, returns directly without executing storage operation.

Args:
    key: Cache key used to identify cached data.
    args: Positional arguments used to generate cache hash key.
    kw: Keyword arguments used to generate cache hash key.
    value: Data to be stored.

**Note:** \n
- If cache mode is set to read-only (RO) or disabled (NONE), this method will return directly without executing storage operation.
''')

add_chinese_doc('module.ModuleCache.close', '''\
关闭缓存存储策略。

释放缓存存储策略占用的资源，如关闭数据库连接、清理内存缓存等。调用此方法后，缓存将不再可用。

**注意:** \n
- 调用此方法后，缓存实例将无法继续使用。
- 不同的缓存策略可能有不同的资源清理行为。
''')

add_english_doc('module.ModuleCache.close', '''\
Close cache storage strategy.

Releases resources occupied by the cache storage strategy, such as closing database connections, clearing memory cache, etc. After calling this method, the cache will no longer be available.

**Note:** \n
- After calling this method, the cache instance will no longer be usable.
- Different cache strategies may have different resource cleanup behaviors.
''')

add_chinese_doc('llms.onlinemodule.supplier.siliconflow.SiliconFlowChat', """\
SiliconFlow 模块，继承自 OnlineChatModuleBase 和 FileHandlerBase。

提供基于 SiliconFlow 平台的大语言模型对话能力，支持多种模型（包括视觉语言模型），并具备文件处理功能。

Args:
    base_url (str, optional): API 基础地址，默认为 "https://api.siliconflow.cn/v1/"
    model (str, optional): 使用的模型名称，默认为 "Qwen/QwQ-32B"
    api_key (str, optional): API 密钥，默认从配置项 lazyllm.config['siliconflow_api_key'] 中读取
    stream (bool, optional): 是否启用流式输出，默认为 True
    return_trace (bool, optional): 是否返回追踪信息，默认为 False
    **kwargs: 其他模型参数
""")

add_english_doc('llms.onlinemodule.supplier.siliconflow.SiliconFlowChat', """\
SiliconFlow module, inherits from OnlineChatModuleBase and FileHandlerBase.

Provides large language model chat capabilities via the SiliconFlow platform, supports multiple models (including vision-language models), and includes file handling functionality.

Args:
    base_url (str, optional): Base API URL, defaults to "https://api.siliconflow.cn/v1/"
    model (str, optional): Model name to use, defaults to "Qwen/QwQ-32B"
    api_key (str, optional): API key, defaults to lazyllm.config['siliconflow_api_key']
    stream (bool, optional): Whether to enable streaming output, defaults to True
    return_trace (bool, optional): Whether to return trace information, defaults to False
    **kwargs: Other model parameters
""")

add_chinese_doc('llms.onlinemodule.supplier.siliconflow.SiliconFlowEmbed', """\
SiliconFlow 向量嵌入模块，继承自 OnlineEmbeddingModuleBase。

提供基于 SiliconFlow 平台的文本嵌入（Embedding）功能，支持将文本转换为向量表示。

Args:
    embed_url (str, optional): 嵌入 API 的 URL，默认为 "https://api.siliconflow.cn/v1/embeddings"
    embed_model_name (str, optional): 使用的嵌入模型名称，默认为 "BAAI/bge-large-zh-v1.5"
    api_key (str, optional): API 密钥，默认从配置项 lazyllm.config['siliconflow_api_key'] 中读取
    batch_size (int, optional): 批处理大小，默认为 16
    **kw: 其他嵌入模块参数
""")

add_english_doc('llms.onlinemodule.supplier.siliconflow.SiliconFlowEmbed', """\
SiliconFlow embedding module, inherits from OnlineEmbeddingModuleBase.

Provides text embedding functionality via the SiliconFlow platform, converting text into vector representations.

Args:
    embed_url (str, optional): Embedding API URL, defaults to "https://api.siliconflow.cn/v1/embeddings"
    embed_model_name (str, optional): Name of the embedding model to use, defaults to "BAAI/bge-large-zh-v1.5"
    api_key (str, optional): API key, defaults to lazyllm.config['siliconflow_api_key']
    batch_size (int, optional): Batch size for processing, defaults to 16
    **kw: Additional embedding module parameters
""")

add_chinese_doc('llms.onlinemodule.supplier.siliconflow.SiliconFlowRerank', """\
SiliconFlow 重排序模块，继承自 OnlineEmbeddingModuleBase。

提供基于 SiliconFlow 平台的文本重排序（Reranking）功能，用于对文档列表根据查询相关性进行重新排序。

Args:
    embed_url (str, optional): 重排序 API 的 URL，默认为 "https://api.siliconflow.cn/v1/rerank"
    embed_model_name (str, optional): 使用的重排序模型名称，默认为 "BAAI/bge-reranker-v2-m3"
    api_key (str, optional): API 密钥，默认从配置项 lazyllm.config['siliconflow_api_key'] 中读取
    **kw: 其他重排序模块参数

Returns:
    List[Tuple]: 包含排序结果的列表，每个元素为包含 'index'、'relevance_score' 的元组。
""")

add_english_doc('llms.onlinemodule.supplier.siliconflow.SiliconFlowRerank', """\
SiliconFlow reranking module, inherits from OnlineEmbeddingModuleBase.

Provides text reranking functionality via the SiliconFlow platform, reordering a list of documents based on their relevance to a given query.

Args:
    embed_url (str, optional): Reranking API URL, defaults to "https://api.siliconflow.cn/v1/rerank"
    embed_model_name (str, optional): Name of the reranking model to use, defaults to "BAAI/bge-reranker-v2-m3"
    api_key (str, optional): API key, defaults to lazyllm.config['siliconflow_api_key']
    **kw: Additional reranking module parameters
Returns:
    List[Tuple]: A list of reranking results, each containing 'index' and 'relevance_score'.
""")

add_chinese_doc('llms.onlinemodule.supplier.siliconflow.SiliconFlowText2Image', """\
SiliconFlow文生图模块，继承自OnlineMultiModalBase。

提供基于SiliconFlow的文本生成图像功能，支持根据文本描述生成图像，支持纯文本生成图像和图像编辑。

Args:
    api_key (str, optional): API密钥，默认为配置中的siliconflow_api_key
    model_name (str, optional): 模型名称，默认为"Qwen/Qwen-Image"
    base_url (str, optional): API基础URL，默认为"https://api.siliconflow.cn/v1/"
    return_trace (bool, optional): 是否返回追踪信息，默认为False
    **kwargs: 其他模型参数
""")

add_english_doc('llms.onlinemodule.supplier.siliconflow.SiliconFlowText2Image', """\
SiliconFlow Text-to-Image module, inherits from OnlineMultiModalBase.

Provides text-to-image generation functionality based on SiliconFlow, supports generating images from text descriptions and image editing.

Args:
    api_key (str, optional): API key, defaults to configured siliconflow_api_key
    model_name (str, optional): Model name, defaults to "Qwen/Qwen-Image"
    base_url (str, optional): Base API URL, defaults to "https://api.siliconflow.cn/v1/"
    return_trace (bool, optional): Whether to return trace information, defaults to False
    **kwargs: Other model parameters
""")

add_chinese_doc('llms.onlinemodule.supplier.minimax.MinimaxChat', """\
Minimax 模块，继承自 OnlineChatModuleBase 和 FileHandlerBase。

提供基于 Minimax 平台的大语言模型对话能力。

Args:
    base_url (str, optional): API 基础地址，默认为 "https://api.minimaxi.com/v1/"
    model (str, optional): 使用的模型名称，默认为 "MiniMax-M2"
    api_key (str, optional): API 密钥，默认从配置项 lazyllm.config['minimax_api_key'] 中读取
    stream (bool, optional): 是否启用流式输出，默认为 True；启用时会自动设置请求参数
    return_trace (bool, optional): 是否返回追踪信息，默认为 False
    **kwargs: 其他传递给父类的可选参数
""")

add_english_doc('llms.onlinemodule.supplier.minimax.MinimaxChat', """\
Minimax module, inheriting from OnlineChatModuleBase and FileHandlerBase.

Provides large language model chat capabilities based on the Minimax platform.

Args:
    base_url (str, optional): Base API URL, defaults to "https://api.minimaxi.com/v1/"
    model (str, optional): Model name to use, defaults to "MiniMax-M2"
    api_key (str, optional): API key, defaults to lazyllm.config['minimax_api_key']
    stream (bool, optional): Whether to enable streaming output, defaults to True; automatically configures request parameters when enabled
    return_trace (bool, optional): Whether to return trace information, defaults to False
    **kwargs: Additional optional parameters passed to the parent classes
""")

add_chinese_doc('llms.onlinemodule.supplier.minimax.MinimaxText2Image', """\
Minimax 文生图模块，继承自 OnlineMultiModalBase。

提供基于 Minimax 平台的文本生成图像功能，支持根据文本描述生成图像。

Args:
    api_key (str, optional): API 密钥，默认为配置项 lazyllm.config['minimax_api_key']
    model_name (str, optional): 模型名称，默认为 "image-01"
    base_url (str, optional): API 基础地址，默认为 "https://api.minimaxi.com/v1/"
    return_trace (bool, optional): 是否返回追踪信息，默认为 False
    **kwargs: 其他传递给父类的可选参数
""")

add_english_doc('llms.onlinemodule.supplier.minimax.MinimaxText2Image', """\
Minimax text-to-image module, inheriting from OnlineMultiModalBase.

Provides text-to-image generation functionality based on Minimax, supports generating images from text descriptions.

Args:
    api_key (str, optional): API key, defaults to lazyllm.config['minimax_api_key']
    model_name (str, optional): Model name, defaults to "image-01"
    base_url (str, optional): Base API URL, defaults to "https://api.minimaxi.com/v1/"
    return_trace (bool, optional): Whether to return trace information, defaults to False
    **kwargs: Additional optional parameters passed to the parent classes
""")

add_chinese_doc('llms.onlinemodule.supplier.minimax.MinimaxTTS', """\
Minimax 文本转语音模块，继承自 OnlineMultiModalBase。

提供基于 Minimax 平台的文本转语音(TTS)功能，支持将文本转换为音频文件。

Args:
    api_key (str, optional): API 密钥，默认为配置项 lazyllm.config['minimax_api_key']
    model_name (str, optional): 模型名称，默认为 "speech-2.6-hd"
    base_url (str, optional): API 基础地址，默认为 "https://api.minimaxi.com/v1/"
    return_trace (bool, optional): 是否返回追踪信息，默认为 False
    **kwargs: 其他传递给父类的可选参数
""")

add_english_doc('llms.onlinemodule.supplier.minimax.MinimaxTTS', """\
Minimax text-to-speech module, inheriting from OnlineMultiModalBase.

Provides text-to-speech (TTS) functionality based on Minimax, supports converting text to audio files.

Args:
    api_key (str, optional): API key, defaults to lazyllm.config['minimax_api_key']
    model_name (str, optional): Model name, defaults to "speech-2.6-hd"
    base_url (str, optional): Base API URL, defaults to "https://api.minimaxi.com/v1/"
    return_trace (bool, optional): Whether to return trace information, defaults to False
    **kwargs: Additional optional parameters passed to the parent classes
""")

add_chinese_doc('llms.onlinemodule.supplier.aiping.AipingChat', '''\
AipingChat 是 AIPing 的在线聊天模块，继承自 OnlineChatModuleBase 和 FileHandlerBase。

提供与 AIPing 平台大语言模型交互的接口，支持对话生成、文件处理以及模型微调等功能。支持多种模型，包括视觉语言模型（VLM）如 Qwen2.5-VL、Qwen3-VL、GLM-4.5V、GLM-4.6V 等。

Args:
    base_url (str): API 基础 URL，默认为 "https://aiping.cn/api/v1/"。
    model (str): 使用的模型名称，默认为 "DeepSeek-R1"。
    api_key (Optional[str]): 访问 AIPing 服务的 API Key，若未提供则从 lazyllm 配置中读取。
    stream (bool): 是否开启流式输出，默认为 True。
    return_trace (bool): 是否返回调试追踪信息，默认为 False。
    **kwargs: 其他传递给 OnlineChatModuleBase 的参数。

功能特点:
    1. 支持多种大语言模型，包括通用对话模型和视觉语言模型
    2. 支持流式输出，提升用户体验
    3. 集成文件处理功能，支持微调数据格式验证和转换
    4. 内置系统提示："You are an intelligent assistant developed by AIPing. You are a helpful assistant."
    5. 支持 API Key 验证，确保服务安全性
''')

add_english_doc('llms.onlinemodule.supplier.aiping.AipingChat', '''\
AipingChat is an online chat module for AIPing, inheriting from OnlineChatModuleBase and FileHandlerBase.

Provides an interface to interact with AIPing's large language models, supporting chat generation, file handling, and model fine-tuning. Supports multiple models including Vision-Language Models (VLM) such as Qwen2.5-VL, Qwen3-VL, GLM-4.5V, GLM-4.6V, etc.

Args:
    base_url (str): Base URL for the API, defaults to "https://aiping.cn/api/v1/".
    model (str): Name of the model to use, defaults to "DeepSeek-R1".
    api_key (Optional[str]): API key for accessing AIPing service. If not provided, it is read from lazyllm config.
    stream (bool): Whether to enable streaming output, defaults to True.
    return_trace (bool): Whether to return debug trace information, defaults to False.
    **kwargs: Additional parameters passed to OnlineChatModuleBase.

Features:
    1. Supports multiple large language models, including general chat models and vision-language models
    2. Supports streaming output for better user experience
    3. Integrated file handling functionality, supporting fine-tuning data format validation and conversion
    4. Built-in system prompt: "You are an intelligent assistant developed by AIPing. You are a helpful assistant."
    5. Supports API key validation to ensure service security
''')

add_chinese_doc('llms.onlinemodule.supplier.aiping.AipingEmbed', '''\
 AIPing 文本嵌入模块，继承自 OnlineEmbeddingModuleBase。

提供与 AIPing 文本嵌入服务交互的接口，支持将文本转换为向量表示，支持批量处理。

Args:
    embed_url (str): 嵌入 API 的 URL，默认为 "https://aiping.cn/api/v1/embeddings"。
    embed_model_name (str): 使用的嵌入模型名称，默认为 "text-embedding-v1"。
    api_key (Optional[str]): 访问 AIPing 服务的 API Key，若未提供则从 lazyllm 配置中读取。
    batch_size (int): 批处理大小，默认为 16。
    **kw: 其他传递给基类的参数。

功能特点:
    1. 将文本转换为高维向量表示
    2. 支持批量文本处理，提高效率
    3. 可配置的批处理大小，适应不同性能需求
    4. 与 AIPing  API 无缝集成
''')

add_english_doc('llms.onlinemodule.supplier.aiping.AipingEmbed', '''\
Aiping text embedding module, inheriting from OnlineEmbeddingModuleBase.

Provides an interface to interact with AIPing's text embedding service, supporting conversion of text to vector representations with batch processing support.

Args:
    embed_url (str): Embedding API URL, defaults to "https://aiping.cn/api/v1/embeddings".
    embed_model_name (str): Name of the embedding model to use, defaults to "text-embedding-v1".
    api_key (Optional[str]): API key for accessing AIPing service. If not provided, it is read from lazyllm config.
    batch_size (int): Batch size for processing, defaults to 16.
    **kw: Additional parameters passed to the base class.

Features:
    1. Converts text to high-dimensional vector representations
    2. Supports batch text processing for improved efficiency
    3. Configurable batch size to accommodate different performance requirements
    4. Seamless integration with AIPing API
''')

add_chinese_doc('llms.onlinemodule.supplier.aiping.AipingRerank', '''\
 AIPing 重排序模块，继承自 OnlineEmbeddingModuleBase。

提供与 AIPing 重排序服务交互的接口，用于对文档列表根据查询相关性进行重新排序。该模块返回一个包含文档索引和相关性得分的元组列表。

Args:
    embed_url (str): 重排序 API 的 URL，默认为 "https://aiping.cn/api/v1/rerank"。
    embed_model_name (str): 使用的重排序模型名称，默认为 "Qwen3-Reranker-0.6B"。
    api_key (Optional[str]): 访问 AIPing 服务的 API Key，若未提供则从 lazyllm 配置中读取。
    **kw: 其他传递给基类的参数。

属性:
    type (str): 返回模型类型，固定为 "RERANK"。

功能特点:
    1. 根据查询对文档列表进行相关性重排序
    2. 支持自定义排序参数（top_n 等）
    3. 返回每个文档的索引和相关性得分
    4. 适用于搜索结果优化和文档推荐场景
''')

add_english_doc('llms.onlinemodule.supplier.aiping.AipingRerank', '''\
Aiping reranking module, inheriting from OnlineEmbeddingModuleBase.

Provides an interface to interact with AIPing's reranking service, used for reordering a list of documents based on their relevance to a given query. Returns a list of tuples containing document index and relevance score.

Args:
    embed_url (str): Reranking API URL, defaults to "https://aiping.cn/api/v1/rerank".
    embed_model_name (str): Name of the reranking model to use, defaults to "Qwen3-Reranker-0.6B".
    api_key (Optional[str]): API key for accessing AIPing service. If not provided, it is read from lazyllm config.
    **kw: Additional parameters passed to the base class.

Properties:
    type (str): Returns model type, fixed as "RERANK".

Features:
    1. Reranks documents based on query relevance
    2. Supports custom ranking parameters (e.g., top_n)
    3. Returns index and relevance score for each document
    4. Suitable for search result optimization and document recommendation scenarios
''')

add_chinese_doc('llms.onlinemodule.supplier.aiping.AipingText2Image', '''\
 AIPing 文本生成图像模块，继承自 OnlineMultiModalBase。

提供与 AIPing 图像生成服务交互的接口，支持根据文本描述生成图像。支持负面提示、图像数量、尺寸和随机种子等参数。

Args:
    api_key (Optional[str]): 访问 AIPing 服务的 API Key，若未提供则从 lazyllm 配置中读取。
    model_name (str): 使用的模型名称，默认为 "Qwen-Image"。
    base_url (str): API 基础 URL，默认为 "https://aiping.cn/api/v1/"。
    return_trace (bool): 是否返回调试追踪信息，默认为 False。
    **kwargs: 其他传递给基类的参数。

功能特点:
    1. 根据文本提示生成高质量图像
    2. 支持负面提示，过滤不想要的图像特征
    3. 可配置生成图像的数量（n 参数）
    4. 支持多种图像尺寸规格
    5. 支持随机种子控制，确保结果可重现
    6. 自动下载生成的图像并编码为文件格式
    7. 默认负面提示："模糊，低质量"

注意:
    - 该模块会自动下载生成的图像到本地文件
    - 返回结果会包含文件路径信息，便于后续处理
''')

add_english_doc('llms.onlinemodule.supplier.aiping.AipingText2Image', '''\
Aiping text-to-image module, inheriting from OnlineMultiModalBase.

Provides an interface to interact with AIPing's image generation service, supporting image generation from text descriptions. Supports parameters such as negative prompts, image count, size, and random seeds.

Args:
    api_key (Optional[str]): API key for accessing AIPing service. If not provided, it is read from lazyllm config.
    model_name (str): Name of the model to use, defaults to "Qwen-Image".
    base_url (str): Base URL for the API, defaults to "https://aiping.cn/api/v1/".
    return_trace (bool): Whether to return debug trace information, defaults to False.
    **kwargs: Additional parameters passed to the base class.

Features:
    1. Generates high-quality images from text prompts
    2. Supports negative prompts to filter unwanted image features
    3. Configurable number of images to generate (n parameter)
    4. Supports multiple image size specifications
    5. Supports random seed control for reproducible results
    6. Automatically downloads generated images and encodes them as files
    7. Default negative prompt: "模糊，低质量"

Note:
    - This module automatically downloads generated images to local files
    - The returned result contains file path information for easy subsequent processing
''')

add_chinese_doc('llms.onlinemodule.supplier.gemini.GeminiChat', '''\
GeminiChat 是 Google Gemini系列模型的在线聊天模块，继承自 OnlineChatModuleBase。

提供与Google Gemini系列大语言模型交互的接口，支持对话生成、文件处理以及模型微调等功能。支持多种模型，包括视觉语言模型（VLM）。

Args:
    base_url (str): API 基础 URL，默认为 "https://generativelanguage.googleapis.com/v1beta"。
    model (str): 使用的模型名称，默认为 "gemini-2.5-flash"。
    api_key (Optional[str]): 访问 Gemini 的 API Key，若未提供则从 lazyllm 配置中读取。
    stream (bool): 是否开启流式输出，默认为 True。
    return_trace (bool): 是否返回调试追踪信息，默认为 False。
    **kwargs: 其他传递给 OnlineChatModuleBase 的参数。

功能特点:
    1. 支持多种大语言模型，包括通用对话模型和视觉语言模型
    2. 支持流式输出，提升用户体验
    3. 集成文件处理功能，支持微调数据格式验证和转换
    4. 内置系统提示："You are an intelligent assistant developed by Google. You are a helpful assistant."
    5. 支持 API Key 验证，确保服务安全性
''')

add_english_doc('llms.onlinemodule.supplier.gemini.GeminiChat', '''\
GeminiChat is an online chat module for Google Gemini series models, inheriting from OnlineChatModuleBase.

Provides an interface to interact with Google's large language models, supporting chat generation, file handling, and model fine-tuning. Supports multiple models including Vision-Language Models (VLM).

Args:
    base_url (str): Base URL for the API, defaults to "https://generativelanguage.googleapis.com/v1beta".
    model (str): Name of the model to use, defaults to "gemini-2.5-flash".
    api_key (Optional[str]): API key for accessing Gemini service. If not provided, it is read from lazyllm config.
    stream (bool): Whether to enable streaming output, defaults to True.
    return_trace (bool): Whether to return debug trace information, defaults to False.
    **kwargs: Additional parameters passed to OnlineChatModuleBase.

Features:
    1. Supports multiple large language models, including general chat models and vision-language models
    2. Supports streaming output for better user experience
    3. Integrated file handling functionality, supporting fine-tuning data format validation and conversion
    4. Built-in system prompt: "You are an intelligent assistant developed by Google. You are a helpful assistant."
    5. Supports API key validation to ensure service security
''')

add_chinese_doc('llms.onlinemodule.supplier.gemini.GeminiText2Image', '''\
Qwen文本生成图像模块和图像编辑模块，继承自 LazyLLMOnlineText2ImageModuleBase，封装了调用 gemini-2.5-flash-image 模型生成图像的能力和调用 nano-banana-pro-preview 模型进行图像编辑的能力。  
支持根据文本提示生成指定数量和分辨率的图像，支持图像编辑，并可设置负面提示、随机种子及扩展提示功能。

Args:
    model (Optional[str]): 使用的 Gemini 模型名称，默认从配置 'gemini_text2image_model_name' 获取，若未设置则使用 "gemini-2.5-flash-image"。
    api_key (Optional[str]): 调用 Gemini 模型服务的 API Key。
    return_trace (bool): 是否返回调试追踪信息，默认为 False。
    **kwargs: 其他传递给 GeminiText2Image 的参数。
''')

add_english_doc('llms.onlinemodule.supplier.gemini.GeminiText2Image', '''\
Gemini Text-to-Image module and Image-Edit module, inheriting from LazyLLMOnlineText2ImageModuleBase, encapsulates the functionality to generate images using the Gemini "gemini-2.5-flash-image" model and to edit images using the "nano-banana-pro-preview" model.  
It supports generating a specified number of images with given resolution based on a text prompt, and allows setting negative prompts, random seeds, and prompt extension.

Args:
    model (Optional[str]): Name of the Gemini model to use, default is taken from config 'gemini_text2image_model_name', or "gemini-2.5-flash-image" if not set.
    api_key (Optional[str]): API key for accessing Gemini model service.
    return_trace (bool): Whether to return debug trace information, default is False.
    **kwargs: Additional parameters passed to GeminiText2Image.
''')

add_chinese_doc('llms.onlinemodule.supplier.claude.ClaudeChat', '''\
ClaudeChat 是 Anthropic 公司 Claude系列模型的在线聊天模块，继承自 OnlineChatModuleBase。

提供与Anthropic Claude系列大语言模型交互的接口，支持对话生成、文件处理以及模型微调等功能。支持多种模型，包括视觉语言模型（VLM）。

Args:
    base_url (str): API 基础 URL，默认为 "https://api.anthropic.com/v1"。
    model (str): 使用的模型名称，默认为 "claude-4-5-sonnet-latest"。
    api_key (Optional[str]): 访问 Claude 的 API Key，若未提供则从 lazyllm 配置中读取。
    stream (bool): 是否开启流式输出，默认为 True。
    return_trace (bool): 是否返回调试追踪信息，默认为 False。
    **kwargs: 其他传递给 OnlineChatModuleBase 的参数。

功能特点:
    1. 支持多种大语言模型，包括通用对话模型和视觉语言模型
    2. 支持流式输出，提升用户体验
    3. 集成文件处理功能，支持微调数据格式验证和转换
    4. 内置系统提示："You are an intelligent assistant developed by Anthropic. You are a helpful assistant."
    5. 支持 API Key 验证，确保服务安全性
''')

add_english_doc('llms.onlinemodule.supplier.gemini.ClaudeChat', '''\
ClaudeChat is an online chat module for Anthropic Claude series models, inheriting from OnlineChatModuleBase.

Provides an interface to interact with Anthropic's large language models, supporting chat generation, file handling, and model fine-tuning. Supports multiple models including Vision-Language Models (VLM).

Args:
    base_url (str): Base URL for the API, defaults to "https://api.anthropic.com/v1".
    model (str): Name of the model to use, defaults to "claude-4-5-sonnet-latest".
    api_key (Optional[str]): API key for accessing Claude service. If not provided, it is read from lazyllm config.
    stream (bool): Whether to enable streaming output, defaults to True.
    return_trace (bool): Whether to return debug trace information, defaults to False.
    **kwargs: Additional parameters passed to OnlineChatModuleBase.

Features:
    1. Supports multiple large language models, including general chat models and vision-language models
    2. Supports streaming output for better user experience
    3. Integrated file handling functionality, supporting fine-tuning data format validation and conversion
    4. Built-in system prompt: "You are an intelligent assistant developed by Anthropic. You are a helpful assistant."
    5. Supports API key validation to ensure service security
''')
