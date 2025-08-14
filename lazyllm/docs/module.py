# flake8: noqa E501
from . import utils
import functools
import lazyllm

add_chinese_doc = functools.partial(utils.add_chinese_doc, module=lazyllm.module)
add_english_doc = functools.partial(utils.add_english_doc, module=lazyllm.module)
add_example = functools.partial(utils.add_example, module=lazyllm.module)

add_chinese_doc('ModuleBase', '''\
Module是LazyLLM中的顶层组件，具备训练、部署、推理和评测四项关键能力，每个模块可以选择实现其中的部分或者全部的能力，每项能力都可以由一到多个Component组成。
ModuleBase本身不可以直接实例化，继承并实现 ``forward`` 函数的子类可以作为一个仿函数来使用。
类似pytorch的Moudule，当一个Module A持有了另一个Module B的实例作为成员变量时，会自动加入到submodule中。

如果你需要以下的能力，请让你自定义的类继承自ModuleBase:\n
1. 组合训练、部署、推理和评测的部分或全部能力，例如Embedding模型需要训练和推理\n
2. 持有的成员变量具备训练、部署和评测的部分或全部能力，并且想通过Module的根节点的 ``start``,  ``update``, ``eval`` 等方法对其持有的成员进行训练、部署和评测时。\n
3. 将用户设置的参数从最外层直接传到你自定义的模块中（参考Tools.webpages.WebModule）\n
4. 希望能被参数网格搜索模块使用（参考TrialModule）
''')

add_english_doc('ModuleBase', '''\
Module is the top-level component in LazyLLM, possessing four key capabilities: training, deployment, inference, and evaluation. Each module can choose to implement some or all of these capabilities, and each capability can be composed of one or more components.
ModuleBase itself cannot be instantiated directly; subclasses that inherit and implement the forward function can be used as a functor.
Similar to PyTorch's Module, when a Module A holds an instance of another Module B as a member variable, B will be automatically added to A's submodules.
If you need the following capabilities, please have your custom class inherit from ModuleBase:\n
1. Combine some or all of the training, deployment, inference, and evaluation capabilities. For example, an Embedding model requires training and inference.\n
2. If you want the member variables to possess some or all of the capabilities for training, deployment, and evaluation, and you want to train, deploy, and evaluate these members through the start, update, eval, and other methods of the Module's root node.\n
3. Pass user-set parameters directly to your custom module from the outermost layer (refer to WebModule).\n
4. The desire for it to be usable by the parameter grid search module (refer to TrialModule).
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

add_chinese_doc('ModuleBase.forward', '''\
定义了每次执行的计算步骤，ModuleBase的所有的子类都需要重写这个函数。
''')

add_english_doc('ModuleBase.forward', '''\
Define computation steps executed each time, all subclasses of ModuleBase need to override.
''')

add_example('ModuleBase.forward', '''\
>>> import lazyllm
>>> class MyModule(lazyllm.module.ModuleBase):
...     def forward(self, input):
...         return input + 1
... 
>>> MyModule()(1)
2   
''')

add_chinese_doc('ModuleBase.start', '''\
部署模块及所有的子模块
''')

add_english_doc('ModuleBase.start', '''\
Deploy the module and all its submodules.
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
重新重启模块及所有的子模块
''')

add_english_doc('ModuleBase.restart', '''\
Re-deploy the module and all its submodules.
''')

add_example('ModuleBase.restart', '''\
>>> import lazyllm
>>> m = lazyllm.TrainableModule().deploy_method(lazyllm.deploy.dummy).prompt(None)
>>> m.restart()
<Module type=Trainable mode=None basemodel= target= stream=False return_trace=False>
>>> m(1)
"reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}"
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

add_chinese_doc('ModuleBase.evalset', '''\
为Module设置评测集，设置过评测集的Module在 ``update`` 或 ``eval`` 的时候会进行评测，评测结果会存在eval_result变量中。
''')

add_english_doc('ModuleBase.evalset', '''\
during update or eval, and the results will be stored in the eval_result variable.
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

**Returns**\n
- self: 便于链式调用。
''')

add_english_doc('servermodule.LLMBase.prompt', '''\
Set or switch the prompt. Supports None, PrompterBase subclass, or string/dict to create ChatPrompter.

Args:
    prompt (str/dict/PrompterBase/None): The prompt to set.
    history (list): Conversation history, only valid when prompt is str or dict.

**Returns**\n
- self: For chaining calls.
''')

add_chinese_doc('servermodule.LLMBase.formatter', '''\
设置或切换输出格式化器。支持 None、FormatterBase 子类或可调用对象。

Args:
    format (FormatterBase/Callable/None): 格式化器对象或函数，默认为 None。

**Returns**\n
- self: 便于链式调用。
''')

add_english_doc('servermodule.LLMBase.formatter', '''\
Set or switch the output formatter. Supports None, FormatterBase subclass or callable.

Args:
    format (FormatterBase/Callable/None): Formatter object or function, default is None.

**Returns**\n
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

**Returns**\n
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

**Returns**\n
- LLMBase: The new shared instance.
''')

add_chinese_doc('TrainableModule', '''\
可训练模块，所有模型（包括LLM、Embedding等）都通过TrainableModule提供服务

<span style="font-size: 20px;">**`TrainableModule(base_model='', target_path='', *, stream=False, return_trace=False)`**</span>


Args:
    base_model (str): 基础模型的名称或路径。
    target_path (str): 保存微调任务的路径。
    source (str): 模型来源，如果未设置，将从环境变量LAZYLLM_MODEL_SOURCE读取。
    stream (bool): 输出流式结果。     
    return_trace (bool): 在trace中记录结果。

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
    source (str): Model source. If not set, it will read the value from the environment variable LAZYLLM_MODEL_SOURCE.
    stream (bool): Whether to output stream. 
    return_trace (bool): Record the results in trace.


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

Returns:
- List[str]: A list of fine-tuned model identifiers or directories.
''')

add_chinese_doc("TrainableModule.get_all_models", '''\
get_all_models() -> List[str]

返回当前目标路径下所有微调模型的路径列表。

返回值：
- List[str]：所有微调模型的名称或路径列表。
''')

add_english_doc("TrainableModule.status", '''\
status(task_name: Optional[str] = None) -> str

Returns the current status of a specific task in the module.

Args:
- task_name (Optional[str]): Name of the task (e.g., 'deploy'). Defaults to 'deploy' if not provided.

Returns:
- str: Status string such as 'running', 'finished', or 'stopped'.
''')

add_chinese_doc("TrainableModule.status", '''\
status(task_name: Optional[str] = None) -> str

返回模块中指定任务的当前状态。

参数：
- task_name (Optional[str])：任务名称（如 'deploy'），默认返回 'deploy' 任务的状态。

返回值：
- str：状态字符串，例如 'running'、'finished' 或 'stopped'。
''')

add_english_doc("TrainableModule.set_specific_finetuned_model", '''\
set_specific_finetuned_model(model_path: str) -> None

Sets the model to be used from a specific fine-tuned model path.

Args:
- model_path (str): The path to the fine-tuned model to use.
''')

add_chinese_doc("TrainableModule.set_specific_finetuned_model", '''\
set_specific_finetuned_model(model_path: str) -> None

设置要使用的特定微调模型路径。

参数：
- model_path (str)：要使用的微调模型的路径。
''')

add_english_doc("TrainableModule.set_default_parameters", '''\
set_default_parameters(*, optional_keys: List[str] = [], **kw) -> None

Sets the default parameters to be used during inference or evaluation.

Args:
- optional_keys (List[str]): A list of optional keys to allow additional parameters without error.
- **kw: Key-value pairs for default parameters such as temperature, top_k, etc.

''')

add_chinese_doc("TrainableModule.set_default_parameters", '''\
set_default_parameters(*, optional_keys: List[str] = [], **kw) -> None

设置用于推理或评估的默认参数。

参数：
- optional_keys (List[str])：允许传入额外参数的可选键列表。
- **kw：用于设置默认参数的键值对，如 temperature、top_k 等。

''')

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
借助 fastapi，将任意可调用对象包装成 api 服务，可同时启动一个主服务和多个卫星服务。

Args:
    m (Callable): 被包装成服务的函数，可以是一个函数，也可以是一个仿函数。当启动卫星服务时，需要是一个实现了 ``__call__`` 的对象（仿函数）。
    pre (Callable): 前处理函数，在服务进程执行，可以是一个函数，也可以是一个仿函数，默认为 ``None``。
    post (Callable): 后处理函数，在服务进程执行，可以是一个函数，也可以是一个仿函数，默认为 ``None``。
    stream (bool): 是否流式请求和输出，默认为非流式。
    return_trace (bool): 是否将结果记录在 trace 中，默认为``False``。
    port (int): 指定服务部署后的端口，默认为 ``None`` 会随机生成端口。
    pythonpath(str):传递给子进程的 PYTHONPATH 环境变量，默认为 ``None``。
    launcher (LazyLLMLaunchersBase): 用于选择服务执行的计算节点，默认为是异步远程部署"launchers.remote(sync=False)"。
    url(str):模块服务的地址，默认为"None",使用Redis获取。
''')

add_english_doc('ServerModule', '''\
Using FastAPI, any callable object can be wrapped into an API service, allowing the simultaneous launch of one main service and multiple satellite services.

Args:
    m (Callable): The function to be wrapped as a service. It can be a function or a functor. When launching satellite services, it needs to be an object implementing ``__call__`` (a functor).
    pre (Callable): Preprocessing function executed in the service process. It can be a function or a functor, default is ``None``.
    post (Callable): Postprocessing function executed in the service process. It can be a function or a functor, default is ``None``.
    stream (bool): Whether to request and output in streaming mode, default is non-streaming.
    return_trace (bool): Whether to record the results in trace, default is ``False``.
    port (int): Specifies the port after the service is deployed. The default is ``None``, which will generate a random port.
    pythonpath (str): PYTHONPATH environment variable passed to the subprocess. Defaults to None.
    launcher (LazyLLMLaunchersBase): Specifies the compute node for running the service. Defaults to asynchronous remote deployment via launchers.remote(sync=False).
    url (str): The service URL of the module. Defaults to None, in which case the URL is retrieved from Redis.

**Examples:**\n
```python
>>> def demo(input): return input * 2
... 
>>> s = lazyllm.ServerModule(demo, launcher=launchers.empty(sync=False))
>>> s.start()
INFO:     Uvicorn running on http://0.0.0.0:35485
>>> print(s(1))
2
```

```python
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
>>> print(m(1))
INFO:     Uvicorn running on http://0.0.0.0:32028
>>> print(m(1))
2  
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

add_chinese_doc('AutoModel', '''\
用于部署在线 API 模型或本地模型的模块，支持加载在线推理模块或本地可微调模块。
Args:
    model (str): 指定要加载的模型名称，例如 ``internlm2-chat-7b``，可为空。为空时默认加载 ``internlm2-chat-7b``。
    source (str): 指定要使用的在线模型服务，如需使用在线模型，必须传入此参数。支持 ``qwen`` / ``glm`` / ``openai`` / ``moonshot`` 等。
    framework (str): 指定本地部署所使用的推理框架，支持 ``lightllm`` / ``vllm`` / ``lmdeploy``。将通过 ``TrainableModule`` 与指定框架组合进行部署。
''')

add_english_doc('AutoModel', '''\
A module for deploying either online API-based models or local models, supporting both online inference and locally trainable modules.
Args:
    model (str): The name of the model to load, e.g., ``internlm2-chat-7b``. If None, ``internlm2-chat-7b`` will be loaded by default.
    source (str): Specifies the online model service to use. Required when using online models. Supported values include ``qwen``, ``glm``, ``openai``, ``moonshot``, etc.
    framework (str): The local inference framework to use for deployment. Supported values are ``lightllm``, ``vllm``, and ``lmdeploy``. The model will be deployed via ``TrainableModule`` using the specified framework.
''')

add_chinese_doc('OnlineChatModule', '''\
用来管理创建目前市面上公开的大模型平台访问模块，目前支持openai、sensenova、glm、kimi、qwen、doubao、deekseek(由于该平台暂时不让充值了，暂时不支持访问)。平台的api key获取方法参见 [开始入门](/#platform)

Args:
    model (str): 指定要访问的模型 (注意使用豆包时需用 Model ID 或 Endpoint ID，获取方式详见 [获取推理接入点](https://www.volcengine.com/docs/82379/1099522)。使用模型前，要先在豆包平台开通对应服务。)，默认为 ``gpt-3.5-turbo(openai)`` / ``SenseChat-5(sensenova)`` / ``glm-4(glm)`` / ``moonshot-v1-8k(kimi)`` / ``qwen-plus(qwen)`` / ``mistral-7b-instruct-v0.2(doubao)`` 
    source (str): 指定要创建的模块类型，可选为 ``openai`` /  ``sensenova`` /  ``glm`` /  ``kimi`` /  ``qwen`` / ``doubao`` / ``deepseek(暂时不支持访问)``
    base_url (str): 指定要访问的平台的基础链接，默认是官方链接
    system_prompt (str): 指定请求的system prompt，默认是官方给的system prompt
    stream (bool): 是否流式请求和输出，默认为流式
    return_trace (bool): 是否将结果记录在trace中，默认为False
''')

add_english_doc('OnlineChatModule', '''\
Used to manage and create access modules for large model platforms currently available on the market. Currently, it supports openai, sensenova, glm, kimi, qwen, doubao and deepseek (since the platform does not allow recharges for the time being, access is not supported for the time being). For how to obtain the platform's API key, please visit [Getting Started](/#platform)

Args:
    model (str): Specify the model to access (Note that you need to use Model ID or Endpoint ID when using Doubao. For details on how to obtain it, see [Getting the Inference Access Point](https://www.volcengine.com/docs/82379/1099522). Before using the model, you must first activate the corresponding service on the Doubao platform.), default is ``gpt-3.5-turbo(openai)`` / ``SenseChat-5(sensenova)`` / ``glm-4(glm)`` / ``moonshot-v1-8k(kimi)`` / ``qwen-plus(qwen)`` / ``mistral-7b-instruct-v0.2(doubao)`` .
    source (str): Specify the type of module to create. Options include  ``openai`` /  ``sensenova`` /  ``glm`` /  ``kimi`` /  ``qwen`` / ``doubao`` / ``deepseek (not yet supported)`` .
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

add_chinese_doc('llms.onlinemodule.supplier.doubao.DoubaoModule', '''\
豆包（Doubao）在线对话模块。
该类封装了对字节跳动豆包 API 的调用，用于进行多轮对话。默认使用模型 `doubao-1-5-pro-32k-250115`，支持流式输出和调用链跟踪。
Args:
    model (str): 使用的模型名称，默认为 `doubao-1-5-pro-32k-250115`。
    base_url (str): API 的基础 URL，默认为 "https://ark.cn-beijing.volces.com/api/v3/"。
    api_key (str): 豆包 API Key。若未提供，则从 lazyllm.config['doubao_api_key'] 读取。
    stream (bool): 是否启用流式输出，默认为 True。
    return_trace (bool): 是否返回调用链跟踪信息，默认为 False。
    **kwargs: 其他传递给基类的参数。
''')

add_english_doc('llms.onlinemodule.supplier.doubao.DoubaoModule', '''\
Doubao online chat module.
This class wraps the Doubao API (from ByteDance) for multi-turn chat. It defaults to model `doubao-1-5-pro-32k-250115` and supports streaming and optional trace return.
Args:
    model (str): The name of the model to use. Defaults to `doubao-1-5-pro-32k-250115`.
    base_url (str): The base URL for the API. Defaults to "https://ark.cn-beijing.volces.com/api/v3/".
    api_key (str): Doubao API key. If not provided, it will be read from `lazyllm.config['doubao_api_key']`.
    stream (bool): Whether to use streaming output. Defaults to True.
    return_trace (bool): Whether to return trace information. Defaults to False.
    **kwargs: Additional arguments passed to the base class.
''')

add_chinese_doc('llms.onlinemodule.supplier.openai.OpenAIEmbedding', '''\
OpenAI 在线嵌入模块。
该类封装了对 OpenAI 嵌入 API 的调用，默认使用模型 `text-embedding-ada-002`，用于将文本编码为向量表示。
Args:
    embed_url (str): OpenAI 嵌入 API 的 URL，默认为 "https://api.openai.com/v1/embeddings"。
    embed_model_name (str): 使用的嵌入模型名称，默认为 "text-embedding-ada-002"。
    api_key (str, optional): OpenAI 的 API Key。若未提供，则从 lazyllm.config 中读取。
''')

add_english_doc('llms.onlinemodule.supplier.openai.OpenAIEmbedding', '''\
Online embedding module using OpenAI.
This class wraps the OpenAI Embedding API, defaulting to the `text-embedding-ada-002` model, and converts text into vector representations.
Args:
    embed_url (str): The URL endpoint of the OpenAI embedding API. Default is "https://api.openai.com/v1/embeddings".
    embed_model_name (str): The name of the embedding model to use. Default is "text-embedding-ada-002".
    api_key (str, optional): The OpenAI API key. If not provided, it will be read from `lazyllm.config`.
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

如果你需要支持新的开放平台的LLM的能力，请让你自定义的类继承自OnlineChatModuleBase：\n
1、根据新平台的模型返回参数情况考虑对返回结果进行后处理，如果模型返回的格式和openai一致，可以不用做任何处理\n
2、如果新平台支持模型的微调，也需要继承FileHandlerBase类，该类主要是验证文件格式，并在自定义类中把.jsonl格式数据转换为模型支持的数据才能用于后面的模型训练\n
3、如果新平台支持模型的微调，则需要实现文件上传、创建微调服务、查询微调服务的接口。即使新平台不用对微调后的模型进行部署，也请实现一个假的创建部署服务和查询部署服务的接口即可\n
4、如果新平台支持模型的微调，可以提供一个支持微调的模型列表，有助于在微调服务时进行判断\n
5、配置新平台支持的api_key到全局变量，通过lazyllm.config.add(变量名，类型，默认值，环境变量名)进行添加
''')

add_english_doc('OnlineChatModuleBase', '''\
OnlineChatModuleBase is a public component that manages the LLM interface for open platforms, and has key capabilities such as training, deployment, and inference. OnlineChatModuleBase itself does not support direct instantiation; it requires subclasses to inherit from this class and implement interfaces related to fine-tuning, such as uploading files, creating fine-tuning tasks, querying fine-tuning tasks, and deployment-related interfaces, such as creating deployment services and querying deployment tasks.
If you need to support the capabilities of a new open platform's LLM, please extend your custom class from OnlineChatModuleBase:\n
1. Consider post-processing the returned results based on the parameters returned by the new platform's model. If the model's return format is consistent with OpenAI, no processing is necessary.\n
2. If the new platform supports model fine-tuning, you must also inherit from the FileHandlerBase class. This class primarily validates file formats and converts .jsonl formatted data into a format supported by the model for subsequent training. \n
3. If the new platform supports model fine-tuning, you must implement interfaces for file upload, creating fine-tuning services, and querying fine-tuning services. Even if the new platform does not require deployment of the fine-tuned model, please implement dummy interfaces for creating and querying deployment services.\n
4. If the new platform supports model fine-tuning, provide a list of models that support fine-tuning to facilitate judgment during the fine-tuning service process.\n
5. Configure the api_key supported by the new platform as a global variable by using ``lazyllm.config.add(variable_name, type, default_value, environment_variable_name)`` .
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

add_chinese_doc('OnlineEmbeddingModuleBase', '''\
OnlineEmbeddingModuleBase是管理开放平台的嵌入模型接口的基类，用于请求文本获取嵌入向量。不建议直接对该类进行直接实例化。需要特定平台类继承该类进行实例化。

如果你需要支持新的开放平台的嵌入模型的能力，请让你自定义的类继承自OnlineEmbeddingModuleBase：\n
1、如果新平台的嵌入模型的请求和返回数据格式都和openai一样，可以不用做任何处理，只传url和模型即可\n
2、如果新平台的嵌入模型的请求或者返回的数据格式和openai不一样，需要重写_encapsulated_data或_parse_response方法。\n
3、配置新平台支持的api_key到全局变量，通过lazyllm.config.add(变量名，类型，默认值，环境变量名)进行添加
''')

add_english_doc('OnlineEmbeddingModuleBase', '''
OnlineEmbeddingModuleBase is the base class for managing embedding model interfaces on open platforms, used for requesting text to obtain embedding vectors. It is not recommended to directly instantiate this class. Specific platform classes should inherit from this class for instantiation.
If you need to support the capabilities of embedding models on a new open platform, please extend your custom class from OnlineEmbeddingModuleBase:\n
1. If the request and response data formats of the new platform's embedding model are the same as OpenAI's, no additional processing is needed; simply pass the URL and model.\n
2. If the request or response data formats of the new platform's embedding model differ from OpenAI's, you need to override the _encapsulated_data or _parse_response methods.\n
3. Configure the api_key supported by the new platform as a global variable by using ``lazyllm.config.add(variable_name, type, default_value, environment_variable_name)`` .
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

add_chinese_doc('llms.onlinemodule.supplier.doubao.DoubaoEmbedding', '''\
豆包嵌入类，继承自 OnlineEmbeddingModuleBase，封装了调用豆包在线文本嵌入服务的功能。  
通过指定服务接口 URL、模型名称及 API Key，支持远程获取文本向量表示。

Args:
    embed_url (Optional[str]): 豆包文本嵌入服务的接口 URL，默认指向北京区域的服务地址。
    embed_model_name (Optional[str]): 使用的豆包嵌入模型名称，默认为 "doubao-embedding-text-240715"。
    api_key (Optional[str]): 访问豆包服务的 API Key，若未提供则从 lazyllm 配置中读取。
''')

add_english_doc('llms.onlinemodule.supplier.doubao.DoubaoEmbedding', '''\
DoubaoEmbedding class inherits from OnlineEmbeddingModuleBase, encapsulating the functionality to call Doubao's online text embedding service.  
It supports remote text vector representation retrieval by specifying the service URL, model name, and API key.

Args:
    embed_url (Optional[str]): URL of the Doubao text embedding service, defaulting to the Beijing region endpoint.
    embed_model_name (Optional[str]): Name of the Doubao embedding model used, default is "doubao-embedding-text-240715".
    api_key (Optional[str]): API key for accessing the Doubao service. If not provided, it is read from lazyllm config.
''')

add_chinese_doc('llms.onlinemodule.fileHandler.FileHandlerBase', '''\
FileHandlerBase是用于处理微调数据文件的基类，主要用于验证和转换微调数据格式。该类本身不支持直接实例化，需要子类继承该类并实现特定的文件格式转换逻辑。

FileHandlerBase提供以下功能：\n
1. 验证微调数据文件的格式是否符合标准（.jsonl格式）\n
2. 检查数据内容是否符合预期的消息格式（包含role和content字段）\n
3. 验证角色类型是否在允许的范围内（system、knowledge、user、assistant）\n
4. 确保每个对话示例都包含assistant回复\n
5. 提供临时文件存储机制用于后续处理\n
''')

add_english_doc('llms.onlinemodule.fileHandler.FileHandlerBase', '''\
FileHandlerBase is a base class for handling fine-tuning data files, primarily used for validating and converting fine-tuning data formats. This class itself does not support direct instantiation; it requires subclasses to inherit from this class and implement specific file format conversion logic.

FileHandlerBase provides the following capabilities:\n
1. Validate that the fine-tuning data file format conforms to standards (.jsonl format)\n
2. Check if the data content conforms to the expected message format (containing role and content fields)\n
3. Verify that role types are within the allowed range (system, knowledge, user, assistant)\n
4. Ensure that each conversation example contains an assistant response\n
5. Provide temporary file storage mechanism for subsequent processing\n
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
