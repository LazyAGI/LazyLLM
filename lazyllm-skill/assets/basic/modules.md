# Module的使用

Module部分的使用主要包含以下内容:

1. ModuleBase的使用
2. 内置Module的使用

本文档主要介绍内置Module的使用，ModuleBase的使用参考文档[ModuleBase使用示例](./modulebase.md)

## ActionModule

用于将函数、模块、flow、Module等可调用的对象包装一个Module。被包装的Module（包括flow中的Module）都会变成该Module的submodule

参数:

- action (Callable | list[Callable], default: () ) – 被包装的对象，是一个或一组可执行的对象。
- return_trace (bool, default: False ) – 是否开启 trace 模式，用于记录调用栈，默认为 False。

```python
m1 = MyModule('m1')
m2 = MyModule('m2')
am = lazyllm.ActionModule(m1, m2)
am.submodules
[<Module type=MyModule name=m1>, <Module type=MyModule name=m2>]
```

## TrainableModule

可训练模块，所有模型（包括LLM、Embedding等）都通过TrainableModule提供服务

TrainableModule部分的使用参考文档: [Model使用示例](./model.md)

## OnlineModule

在线模型都通过OnlineModule提供服务

OnlineModule部分的使用参考文档: [Model使用示例](./model.md)

## UrlModule

可以将ServerModule部署得到的Url包装成一个Module，调用 __call__ 时会访问该服务

参数:

- url (str, default: '' ) – 要包装的服务的Url，默认为空字符串
- stream (bool | Dict[str, str], default: False ) – 是否流式请求和输出，默认为非流式
- return_trace (bool, default: False ) – 是否将结果记录在trace中，默认为False
- init_prompt (bool, default: True ) – 是否初始化prompt，默认为True

```python
import lazyllm
def demo(input): return input * 2
... 
s = lazyllm.ServerModule(demo, launcher=lazyllm.launchers.empty(sync=False))
s.start()
INFO:     Uvicorn running on http://0.0.0.0:35485
u = lazyllm.UrlModule(url=s._url)
print(u(1))
>>> 2
```

## ServerModule

ServerModule 类，继承自 UrlModule，封装了将任意可调用对象部署为 API 服务的能力。通过 FastAPI 实现，可以启动一个主服务和多个卫星服务，并支持流式调用、预处理和后处理逻辑。既可以传入本地可调用对象启动服务，也可以通过 URL 直接连接远程服务。

参数:

- m (Optional[Union[str, ModuleBase]], default: None ) – 被包装成服务的模块或其名称。若为字符串则表示 URL，此时 url 必须为 None；若为 ModuleBase 则包装为服务。
- pre (Optional[Callable], default: None ) – 前处理函数，在服务进程执行，默认为 None。
- post (Optional[Callable], default: None ) – 后处理函数，在服务进程执行，默认为 None。
- stream (Union[bool, Dict], default: False ) – 是否开启流式输出。可以是布尔值，或包含流式配置的字典，默认为 False。
- return_trace (Optional[bool], default: False ) – 是否返回调试追踪信息。默认为 False。
- port (Optional[int], default: None ) – 指定服务部署的端口。默认为 None，将自动分配端口。
- pythonpath (Optional[str], default: None ) – 传递给子进程的 PYTHONPATH 环境变量，默认为 None。
- launcher (Optional[LazyLLMLaunchersBase], default: None ) – 启动服务所使用的 Launcher，默认使用异步远程部署。
- url (Optional[str], default: None ) – 已部署服务的 URL 地址。若提供，则 m 必须为 None。

```python
>>> import lazyllm
>>> def demo(input): return input * 2
...
>>> s = lazyllm.ServerModule(demo, launcher=launchers.empty(sync=False))
>>> s.start()
INFO:     Uvicorn running on http://0.0.0.0:35485
>>> print(s(1))
2
```

```python
class MyServe(object):
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
m = lazyllm.ServerModule(MyServe(), launcher=launchers.empty(sync=False))
m.start()
INFO:     Uvicorn running on http://0.0.0.0:32028
>>> print(m(1))
2
```

## TrialModule

参数网格搜索模块，会遍历其所有的submodule，收集所有的可被搜索的参数，遍历这些参数进行微调、部署和评测

参数:

- m (Callable) – 被网格搜索参数的子模块，微调、部署和评测都会基于这个模块进行

```python
import lazyllm
from lazyllm import finetune, deploy
m = lazyllm.TrainableModule('b1', 't').finetune_method(finetune.dummy, **dict(a=lazyllm.Option(['f1', 'f2'])))
m.deploy_method(deploy.dummy).mode('finetune').prompt(None)
s = lazyllm.ServerModule(m, post=lambda x, ori: f'post2({x})')
s.evalset([1, 2, 3])
t = lazyllm.TrialModule(s)
t.update()
>>>
dummy finetune!, and init-args is {a: f1}
dummy finetune!, and init-args is {a: f2}
[["post2(reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1})", "post2(reply for 2, and parameters is {'do_sample': False, 'temperature': 0.1})", "post2(reply for 3, and parameters is {'do_sample': False, 'temperature': 0.1})"], ["post2(reply for 1, and parameters is {'do_sample': False, 'temperature': 0.1})", "post2(reply for 2, and parameters is {'do_sample': False, 'temperature': 0.1})", "post2(reply for 3, and parameters is {'do_sample': False, 'temperature': 0.1})"]]
```

## WebModule

WebModule是LazyLLM为开发者提供的基于Web的交互界面。在初始化并启动一个WebModule之后，开发者可以从页面上看到WebModule背后的模块结构，并将Chatbot组件的输入传输给自己开发的模块进行处理。 模块返回的结果和日志会直接显示在网页的“处理日志”和Chatbot组件上。除此之外，WebModule支持在网页上动态加入Checkbox或Text组件用于向模块发送额外的参数。 WebModule页面还提供“使用上下文”，“流式输出”和“追加输出”的Checkbox，可以用来改变页面和后台模块的交互方式。

参数:

- m (Any) – 要包装的模型对象，可以是lazyllm.FlowBase子类或其他可调用对象。
- components (Dict[Any, Any], default: dict() ) – 额外的UI组件配置，默认为空字典。
- title (str, default: '对话演示终端' ) – Web页面标题，默认为'对话演示终端'。
- port (Optional[Union[int, range, tuple, list]], default: None ) – 服务端口号或端口范围，默认为20500-20799。
- history (List[Any], default: [] ) – 历史会话模块列表，默认为空列表。
- text_mode (Optional[Mode], default: None ) – 文本输出模式（Dynamic/Refresh/Appendix），默认为Dynamic。
- trace_mode (Optional[Mode], default: None ) – 追踪模式参数(已弃用)。
- audio (bool, default: False ) – 是否启用音频输入功能，默认为False。
- stream (bool, default: False ) – 是否启用流式输出，默认为False。
- files_target (Optional[Union[Any, List[Any]]], default: None ) – 文件处理的目标模块，默认为None。
- static_paths (Optional[Union[str, Path, List[Union[str, Path]]]], default: None ) – 静态资源路径，默认为None。
- encode_files (bool, default: False ) – 是否对文件路径进行编码处理，默认为False。
- share (bool, default: False ) – 是否生成可分享的公共链接，默认为False。

```python
import lazyllm
def func2(in_str, do_sample=True, temperature=0.0, *args, **kwargs):
...     return f"func2:{in_str}|do_sample:{str(do_sample)}|temp:{temperature}"
...
m1=lazyllm.ActionModule(func2)
m1.name="Module1"
w = lazyllm.WebModule(m1, port=[20570, 20571, 20572], components={
...         m1:[('do_sample', 'Checkbox', True), ('temperature', 'Text', 0.1)]},
...                       text_mode=lazyllm.tools.WebModule.Mode.Refresh)
>>> w.start()
193703: 2024-06-07 10:26:00 lazyllm SUCCESS: ...
```

## DocWebModule

文档Web界面模块，继承自ModuleBase，提供基于Web的文档管理交互界面。

参数:

- doc_server (ServerModule) – 文档服务模块实例，提供后端API支持
- title (str, default: '文档管理演示终端' ) – 界面标题，默认为"文档管理演示终端"
- port (optional, default: None ) – 服务端口号或端口范围。默认为 None（使用20800-20999范围）
- history (optional, default: None ) – 初始聊天历史记录，默认为 None
- text_mode (optional, default: None ) – 文本处理模式，默认为None(动态模式)
- trace_mode (optional, default: None ) – 追踪模式，默认为None(刷新模式)

Mode: 模式枚举类，包含:
    - Dynamic: 动态模式
    - Refresh: 刷新模式
    - Appendix: 附录模式

```python
import lazyllm
from lazyllm.tools.rag.web import DocWebModule
from lazyllm import
doc_server = ServerModule(url="your_url")
doc_web = DocWebModule(
  doc_server=doc_server,
  title="文档管理演示终端",
  port=range(20800, 20805)  # 自动寻找可用端口)
deploy_task = doc_web._get_deploy_tasks()
deploy_task()
print(doc_web.url)
doc_web.stop()
```
