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
''')

add_english_doc('ActionModule', '''\
Used to wrap a Module around functions, modules, flows, Module, and other callable objects. The wrapped Module (including the Module within the flow) will become a submodule of this Module.

Args:
    action (Callable|list[Callable]): The object to be wrapped, which is one or a set of callable objects.

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

add_chinese_doc('TrainableModule', '''\
可训练模块，所有模型（包括LLM、Embedding等）都通过TrainableModule提供服务

<span style="font-size: 20px;">**`TrainableModule(base_model='', target_path='', *, stream=False, return_trace=False)`**</span>


Args:
    base_model (str): 基础模型的名称或路径。如果本地没有该模型，将会自动从模型源下载。
    target_path (str): 保存微调任务的路径。如果仅进行推理，可以留空。
    source (str): 模型来源，可选值为huggingface或...。如果未设置，将从环境变量LAZYLLM_MODEL_SOURCE读取。
    stream (bool): 是否输出流式结果。如果使用的推理引擎不支持流式输出，该参数将被忽略。
    return_trace (bool): 是否在trace中记录结果。

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
    base_model (str): Name or path of the base model. If the model is not available locally, it will be automatically downloaded from the model source.
    target_path (str): Path to save the fine-tuning task. Can be left empty if only performing inference.
    source (str): Model source, optional values are huggingface or. If not set, it will read the value from the environment variable LAZYLLM_MODEL_SOURCE.
    stream (bool): Whether to output stream. If the inference engine used does not support streaming, this parameter will be ignored.
    return_trace (bool): Whether to record the results in trace.

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
    url (str): 要包装的服务的Url
    stream (bool): 是否流式请求和输出，默认为非流式
    return_trace (bool): 是否将结果记录在trace中，默认为False
''')

add_english_doc('UrlModule', '''\
The URL obtained from deploying the ServerModule can be wrapped into a Module. When calling ``__call__`` , it will access the service.

Args:
    url (str): The URL of the service to be wrapped.
    stream (bool): Whether to request and output in streaming mode, default is non-streaming.
    return_trace (bool): Whether to record the results in trace, default is False.
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
    launcher (LazyLLMLaunchersBase): 用于选择服务执行的计算节点，默认为`` launchers.remote``。
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
    launcher (LazyLLMLaunchersBase): Used to select the compute node for service execution, default is ``launchers.remote`` .

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
