# Module使用

Module部分的使用主要包含以下内容:

1. ModuleBase的使用
2. 内置Module的使用

本文档主要介绍ModuleBase的使用，内置Module的使用参考文档[内置Module的使用](./modules.md)

## ModuleBase

用户自定义的模块需要继承 ModuleBase，并实现 forward 方法来定义具体的推理逻辑。

参数:

- return_trace (bool, default: False ) – 是否将推理结果写入 trace 队列，用于调试和追踪。默认为 False。

### 自定义模块
```python
import lazyllm
class Module(lazyllm.module.ModuleBase):
...     pass
... 
class Module2(lazyllm.module.ModuleBase):
...     def __init__(self):
...         super(__class__, self).__init__()
...         self.m = Module()
... 
m = Module2()
print(m.submodules)
>>> [<Module type=Module>]
m.m3 = Module()
print(m.submodules)
>>> [<Module type=Module>, <Module type=Module>]
```

### eval方法

对模块（及所有的子模块）进行评测。当模块通过 evalset 设置了评测集之后，本函数生效。

参数:

- recursive (bool, default: True ) – 是否递归评测所有的子模块，默认为True

```python
import lazyllm
class MyModule(lazyllm.module.ModuleBase):
...     def forward(self, input):
...         return f'reply for input'
... 
m = MyModule()
m.evalset([1, 2, 3])
print(m.eval().eval_result)
['reply for input', 'reply for input', 'reply for input']
```

### evalset方法

为模块设置评测集（evaluation set）。模块在调用 update 或 eval 时会使用评测集进行推理，并将评测结果存储在 eval_result 变量中。

参数:

- evalset (Union[list, str]) – 评测数据列表，或者评测数据文件路径。
- load_f (Optional[Callable], default: None ) – 当 evalset 为文件路径时，用于加载文件并返回列表的函数，默认为 None。
- collect_f (Callable, default: lambda x: x ) – 对评测结果进行后处理的函数，默认为 lambda x: x。

```python
import lazyllm
m = lazyllm.module.TrainableModule().deploy_method(lazyllm.deploy.dummy).finetune_method(lazyllm.finetune.dummy).trainset("").mode("finetune").prompt(None)
m.evalset([1, 2, 3])
m.update()
print(m.eval_result)
["reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 2, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 3, and parameters is {'do_sample': False, 'temperature': 0.1}"]
```

### forward方法

前向计算接口，需要子类实现。该方法定义了模块接收输入并返回输出的逻辑，是模块作为仿函数的核心函数。

参数:

- *args – 可变位置参数，子类可根据实际需求定义输入。
- **kw – 可变关键字参数，子类可根据实际需求定义输入。

### start方法

启动模块及所有子模块的部署服务。该方法会确保模块和子模块的 server 功能被执行，适合用于初始化或重新启动服务。

```python
import lazyllm
m = lazyllm.TrainableModule().deploy_method(lazyllm.deploy.dummy).prompt(None)
m.start()
print(m(1))
"reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}"
```

### restart方法

重启模块及其子模块的部署服务。内部会调用 start 方法，实现服务的重新启动。

```python
import lazyllm
m = lazyllm.TrainableModule().deploy_method(lazyllm.deploy.dummy).prompt(None)
m.restart()
m(1)
"reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}"
```

### update方法

更新模块（及所有的子模块）。当模块重写了 _get_train_tasks 方法后，模块会被更新。更新完后会自动进入部署和推理的流程。

```python
import lazyllm
m = lazyllm.module.TrainableModule().finetune_method(lazyllm.finetune.dummy).trainset("").deploy_method(lazyllm.deploy.dummy).mode('finetune').prompt(None)
m.evalset([1, 2, 3])
m.update()
print(m.eval_result)
["reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 2, and parameters is {'do_sample': False, 'temperature': 0.1}", "reply for 3, and parameters is {'do_sample': False, 'temperature': 0.1}"]
```
