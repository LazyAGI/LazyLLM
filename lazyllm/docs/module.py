# flake8: noqa E501
from . import utils
import functools
import lazyllm


add_chinese_doc = functools.partial(utils.add_chinese_doc, module=lazyllm.module)
add_english_doc = functools.partial(utils.add_english_doc, module=lazyllm.module)
add_example = functools.partial(utils.add_example, module=lazyllm.module)


add_chinese_doc('ModuleBase', '''\
Module是LazyLLM中的顶层组件，具备训练、部署、推理和评测四项关键能力，每个模块可以选择实现其中的部分或者全部的能力，每项能力都可以由1到多个Component组成。
ModuleBase本身不可以直接实例化，继承并实现`forward`函数的子类可以作为一个仿函数来使用。
类似pytorch的Moudule，当一个Module A持有了另一个Module B的实例作为成员变量时，会自动加入到submodule中。

如果你需要以下的能力，请让你自定义的类继承自ModuleBase:
    1. 组合训练、部署、推理和评测的部分或全部能力，例如Embedding模型需要训练和推理
    2. 持有的成员变量具备训练、部署和评测的部分或全部能力，并且想通过Module的根节点的`start`, `update`, `eval`等方法对其持有的成员进行训练、部署和评测时。
    3. 将用户设置的参数从最外层直接传到你自定义的模块中（参考WebModule）
    4. 希望能被参数网格搜索模块使用（参考TrialModule）
''')

add_english_doc('ModuleBase', '''\
Module is the top-level component in LazyLLM, possessing four key capabilities: training, deployment, inference, and evaluation. Each module can choose to implement some or all of these capabilities, and each capability can be composed of one or more components.
ModuleBase itself cannot be instantiated directly; subclasses that inherit and implement the forward function can be used as a functor.
Similar to PyTorch's Module, when a Module A holds an instance of another Module B as a member variable, B will be automatically added to A's submodules.
If you need the following capabilities, please have your custom class inherit from ModuleBase:
    1. Combine some or all of the training, deployment, inference, and evaluation capabilities. For example, an Embedding model requires training and inference.
    2. If you want the member variables to possess some or all of the capabilities for training, deployment, and evaluation, and you want to train, deploy, and evaluate these members through the start, update, eval, and other methods of the Module's root node.
    3. Pass user-set parameters directly to your custom module from the outermost layer (refer to WebModule).
    4. The desire for it to be usable by the parameter grid search module (refer to TrialModule).
''')

add_example('ModuleBase', '''\
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
[<__main__.Module object at 0x7f3dc3bb16f0>]
>>> m.m3 = Module()
>>> m.submodules
[<__main__.Module object at 0x7f3dc3bb16f0>, <__main__.Module object at 0x7f3dc3bb0be0>]
''')

add_chinese_doc('ModuleBase.forward', '''\
定义了每次执行的计算步骤，ModuleBase的所有的子类都需要重写这个函数。
''')

add_english_doc('ModuleBase.forward', '''\
Define computation steps executed each time, all subclasses of ModuleBase need to override.
''')
                
add_example('ModuleBase.forward', '''\
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
>>> m = lazyllm.module.TrainableModule().deploy_method(deploy.dummy)
>>> m.start()
>>> m(1)
LazyLlmResponse(messages="reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}", trace='', err=(0, ''))
''')

add_chinese_doc('ModuleBase.restart', '''\
重启模块及所有的子模块
''')

add_english_doc('ModuleBase.restart', '''\
Re-deploy the module and all its submodules.
''')

add_example('ModuleBase.restart', '''\
>>> m = lazyllm.module.TrainableModule().deploy_method(deploy.dummy)
>>> m.restart()
>>> m(1)
LazyLlmResponse(messages="reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}", trace='', err=(0, ''))
''')
                
add_chinese_doc('ModuleBase.update', '''\
更新模块（及所有的子模块）。当模块重写了`'_get_train_tasks'`方法后，模块会被更新。更新完后会自动进入部署和推理的流程。
Args:
    recursive (bool): 是否递归更新所有的子模块，默认为True
''')

add_english_doc('ModuleBase.update', '''\
Update the module (and all its submodules). The module will be updated when the `'_get_train_tasks'` method is overridden.

Args:
    recursive (bool): Whether to recursively update all submodules, default is True.
''')

add_example('ModuleBase.update', '''\
>>> import lazyllm
>>> m = lazyllm.module.TrainableModule().finetune_method(finetune.dummy).deploy_method(deploy.dummy).mode('finetune')
>>> m.evalset([1, 2, 3])
>>> m.update()
INFO: (lazyllm.launcher) PID: dummy finetune!, and init-args is {}
>>> m.eval_result
["reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 2, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 3, and parameters is {'do_sample': False, 'temperature': 0.1}"]
''')

add_chinese_doc('ModuleBase.evalset', '''\
为Module设置评测集，设置过评测集的Module在`update`或`eval`的时候会进行评测，评测结果会存在eval_result变量中。

.. function:: evalset(evalset, collect_f=lambda x: x) -> None

Args:
    evalset (list): 评测集
    collect_f (Callable): 评测结果的后处理方式，默认不做后处理

.. function:: evalset(evalset, load_f=None, collect_f=lambda x: x) -> None

Args:
    evalset (str): 评测集的路径
    load_f (Callable): 加载评测集的方式，包含解析文件格式和转换成list
    collect_f (Callable): 评测结果的后处理方式，默认不做后处理
''')

add_english_doc('ModuleBase.evalset', '''\
during update or eval, and the results will be stored in the eval_result variable.

.. function:: evalset(evalset, collect_f=lambda x: x) -> None

Args:
    evalset (list): Evaluation set
    collect_f (Callable): Post-processing method for evaluation results, no post-processing by default

.. function:: evalset(evalset, load_f=None, collect_f=lambda x: x) -> None

Args:
    evalset (str): Path to the evaluation set
    load_f (Callable): Method to load the evaluation set, including parsing file formats and converting to a list
    collect_f (Callable): Post-processing method for evaluation results, no post-processing by default
''')

add_example('ModuleBase.evalset', '''\
>>> import lazyllm
>>> m = lazyllm.module.TrainableModule().deploy_method(deploy.dummy)
>>> m.evalset([1, 2, 3])
>>> m.update()
INFO: (lazyllm.launcher) PID: dummy finetune!, and init-args is {}
>>> m.eval_result
["reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 2, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 3, and parameters is {'do_sample': False, 'temperature': 0.1}"]
''')

add_chinese_doc('ModuleBase.eval', '''\
对模块（及所有的子模块）进行评测。当模块通过`'evalset'`设置了评测集之后，本函数生效。

Args:
    recursive (bool): 是否递归评测所有的子模块，默认为True
''')

add_english_doc('ModuleBase.eval', '''\
Evaluate the module (and all its submodules). This function takes effect after the module has been set with an evaluation set using 'evalset'.

Args:
    recursive (bool): Whether to recursively evaluate all submodules. Defaults to True.
''')

add_example('ModuleBase.eval', '''\
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
''')

add_example('ActionModule', '''\
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
''')

add_chinese_doc('TrainableModule', '''\
可训练的模块，所有的模型(包括llm、Embedding等)均通过TrainableModule提供服务

.. function:: TrainableModule(base_model='', target_path='', *, source=lazyllm.config['model_source'], stream=False, return_trace=False)

Args:
    base_model (str): 基模型的名称或者路径，如果本地没有模型，则会自动尝试从模型源进行下载
    target_path (str): 微调任务的保存路径，如只推理，则可以不填
    source (str): 模型源，可选huggingface或，如果不设置则会读取环境变量LAZYLLM_MODEL_SOURCE的值
    stream (bool): 是否流式输出。如果使用的推理引擎不支持流式，则此参数会被忽略
    return_trace (bool): 是否将结果记录在trace中

.. function:: TrainableModule.trainset(v):
为TrainableModule设置训练集

Args:
    v (str): 训练/微调的数据集路径

.. function:: TrainableModule.train_method(v, **kw):
为TrainableModule设置训练方法，暂不支持继续预训练，预计下个版本上线

Args:
    v (LazyLLMTrainBase): 训练的方法，可选为`train.auto`等
    kw (**dict): 训练的方法所需要的参数，和v对应

.. function:: TrainableModule.finetune_method(v, **kw):
为TrainableModule设置微调方法及其参数

Args:
    v (LazyLLMFinetuneBase): 微调的方法，可选为`finetune.auto` / `finetune.alpacalora` / `finetune.collie`等
    kw (**dict): 微调的方法所需要的参数，和v对应

.. function:: TrainableModule.deploy_method(v, **kw):
为TrainableModule设置推理方法及其参数

Args:
    v (LazyLLMDeployBase): 推理的方法，可选为`deploy.auto` / `deploy.lightllm` / `deploy.vllm`等
    kw (**dict): 推理的方法所需要的参数，和v对应

.. function:: TrainableModule.mode(v):
为TrainableModule设置update时执行训练还是微调

Args:
    v (str): 设置update时执行训练还是微调，可选为'finetune'和'train'，默认为'finetune'
''')

add_english_doc('TrainableModule', '''\
Trainable module, all models (including LLM, Embedding, etc.) are served through TrainableModule

.. function:: TrainableModule(base_model='', target_path='', *, source=lazyllm.config['model_source'], stream=False, return_trace=False)

Args:
    base_model (str): Name or path of the base model. If the model is not available locally, it will be automatically downloaded from the model source.
    target_path (str): Path to save the fine-tuning task. Can be left empty if only performing inference.
    source (str): Model source, optional values are huggingface or. If not set, it will read the value from the environment variable LAZYLLM_MODEL_SOURCE.
    stream (bool): Whether to output stream. If the inference engine used does not support streaming, this parameter will be ignored.
    return_trace (bool): Whether to record the results in trace.

.. function:: TrainableModule.trainset(v):
Set the training set for TrainableModule

Args:
    v (str): Path to the training/fine-tuning dataset.

.. function:: TrainableModule.train_method(v, **kw):
Set the training method for TrainableModule. Continued pre-training is not supported yet, expected to be available in the next version.

Args:
    v (LazyLLMTrainBase): Training method, options include `train.auto` etc.
    kw (**dict): Parameters required by the training method, corresponding to v.

.. function:: TrainableModule.finetune_method(v, **kw):
Set the fine-tuning method and its parameters for TrainableModule.

Args:
    v (LazyLLMFinetuneBase): Fine-tuning method, options include `finetune.auto` / `finetune.alpacalora` / `finetune.collie` etc.
    kw (**dict): Parameters required by the fine-tuning method, corresponding to v.

.. function:: TrainableModule.deploy_method(v, **kw):
Set the deployment method and its parameters for TrainableModule.

Args:
    v (LazyLLMDeployBase): Deployment method, options include `deploy.auto` / `deploy.lightllm` / `deploy.vllm` etc.
    kw (**dict): Parameters required by the deployment method, corresponding to v.

.. function:: TrainableModule.mode(v):
Set whether to execute training or fine-tuning during update for TrainableModule.

Args:
    v (str): Sets whether to execute training or fine-tuning during update, options are 'finetune' and 'train', default is 'finetune'.
''')

add_example('TrainableModule', ['''\
''', '''\
>>> m = lazyllm.module.TrainableModule().finetune_method(finetune.dummy).trainset('/file/to/path').deploy_method(None).mode('finetune')
>>> m.update()
INFO: (lazyllm.launcher) PID: dummy finetune!, and init-args is {}
''', '''\
''', '''\
>>> m = lazyllm.module.TrainableModule().finetune_method(finetune.dummy).mode('finetune')
>>> m.update()
INFO: (lazyllm.launcher) PID: dummy finetune!, and init-args is {}
''', '''\
>>> m = lazyllm.module.TrainableModule().deploy_method(deploy.dummy)
>>> m.evalset([1, 2, 3])
>>> m.update()
INFO: (lazyllm.launcher) PID: dummy finetune!, and init-args is {}
>>> m.eval_result
["reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 2, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 3, and parameters is {'do_sample': False, 'temperature': 0.1}"]
''', '''\
>>> m = lazyllm.module.TrainableModule().finetune_method(finetune.dummy).mode('finetune')
>>> m.update()
INFO: (lazyllm.launcher) PID: dummy finetune!, and init-args is {}
'''])

add_chinese_doc('UrlModule', '''\
可以将ServerModule部署得到的Url包装成一个Module，调用`'__call__'`时会访问该服务。

Args:
    url (str): 要包装的服务的Url
    stream (bool): 是否流式请求和输出，默认为非流式
    return_trace (bool): 是否将结果记录在trace中，默认为False
''')

add_english_doc('UrlModule', '''\
The URL obtained from deploying the ServerModule can be wrapped into a Module. When calling `__call__`, it will access the service.

Args:
    url (str): The URL of the service to be wrapped.
    stream (bool): Whether to request and output in streaming mode, default is non-streaming.
    return_trace (bool): Whether to record the results in trace, default is False.
''')

add_example('UrlModule', '''\
>>> def demo(input): return input * 2
... 
>>> s = lazyllm.ServerModule(demo, launcher=launchers.empty(sync=False))
>>> s.start()
INFO:     Uvicorn running on http://0.0.0.0:35485
>>> u = lazyllm.UrlModule(url=s._url)
>>> print(u(1))
messages=2 trace='' err=(0, '')
''')

add_chinese_doc('ServerModule', '''\
借助fastapi，将任意可调用对象包装成api服务，可同时启动一个主服务和多个卫星服务.

Args:
    m (Callable): 被包装成服务的函数，可以是一个函数，也可以是一个仿函数。当启动卫星服务时，需要是一个实现了__call__的对象（仿函数）。
    pre (Callable): 前处理函数，在服务进程执行，可以是一个函数，也可以是一个仿函数，默认为None.
    post (Callable): 后处理函数，在服务进程执行，可以是一个函数，也可以是一个仿函数，默认为None.
    stream (bool): 是否流式请求和输出，默认为非流式
    return_trace (bool): 是否将结果记录在trace中，默认为False
    launcher (LazyLLMLaunchersBase): 用于选择服务执行的计算节点，默认为launchers.remote
''')

add_english_doc('ServerModule', '''\
Using FastAPI, any callable object can be wrapped into an API service, allowing the simultaneous launch of one main service and multiple satellite services.

Args:
    m (Callable): The function to be wrapped as a service. It can be a function or a functor. When launching satellite services, it needs to be an object implementing `__call__` (a functor).
    pre (Callable): Preprocessing function executed in the service process. It can be a function or a functor, default is None.
    post (Callable): Postprocessing function executed in the service process. It can be a function or a functor, default is None.
    stream (bool): Whether to request and output in streaming mode, default is non-streaming.
    return_trace (bool): Whether to record the results in trace, default is False.
    launcher (LazyLLMLaunchersBase): Used to select the compute node for service execution, default is `launchers.remote`.
''')

add_example('ServerModule', '''\
>>> def demo(input): return input * 2
... 
>>> s = lazyllm.ServerModule(demo, launcher=launchers.empty(sync=False))
>>> s.start()
INFO:     Uvicorn running on http://0.0.0.0:35485
>>> print(s(1))
messages=2 trace='' err=(0, '')

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
messages=2 trace='' err=(0, '')
''')

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
>>> m = lazyllm.TrainableModule(lazyllm.Option(['b1', 'b2', 'b3']), 't').finetune_method(finetune.dummy, **dict(a=lazyllm.Option(['f1', 'f2']))).deploy_method(None).mode('finetune')
>>> s = lazyllm.ServerModule(m, post=lambda x, *, ori: f'post2({x})')
>>> t = lazyllm.TrialModule(s)
>>> t.update()
''')

add_chinese_doc('OnlineChatModule', '''\
这是 OnlineChatModule
''')

add_english_doc('OnlineChatModule', '''\
This is OnlineChatModule Doc
''')

add_chinese_doc('OnlineEmbeddingModule', '''\
这是 OnlineEmbeddingModule
''')

add_english_doc('OnlineEmbeddingModule', '''\
This is OnlineEmbeddingModule Doc
''')
