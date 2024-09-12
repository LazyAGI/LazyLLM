# 绘画大师

我们将基于上一节的 [构建你的第一个聊天机器人](robot.md)，构建一个AI绘画大师。

!!! abstract "通过本节您将学习到 LazyLLM 的以下要点"

    - 如何给模型设置提示词；
    - 如何基于 [Pipeline][lazyllm.flow.Pipeline] 控制流来组装应用；
    - 如何使用 LazyLLM 中的非 LLM 类模型；

## 设计思路

首先要绘制图像，我们需要一个会画图的模型，这里我们选择 Stable Diffusion3；

然后由于绘图模型都需要一些专门的提示词，并且我们还想让它支持中文，所以考虑引入一个单轮聊天机器人，让它做翻译和撰写绘画的提示词；

最后我们将上面的两个模块组合为工作流并套上一个用户界面就好了。

所以设计是这样子的：

![Painting Master](../assets/2_painting_master1.svg)

## 代码实现

让我们基于 LazyLLM 来实现上述设计思路吧。
 
### 设计提示词

我们设计一个提示词，在其中指定它的角色为绘画提示词大师，并且可以进行翻译，可以根据用户的输入进行提示词生成和扩写。具体如下：

```python
prompt = 'You are a drawing prompt word master who can convert any Chinese content entered by the user into English drawing prompt words. In this task, you need to convert any input content into English drawing prompt words, and you can enrich and expand the prompt word content.'
```

### 设置模型

接下来我们基于上一节的 [构建你的第一个聊天机器人](robot.md)，并把刚刚写好的提示词设置给它。

[](){#use-prompt}

```python
llm = lazyllm.TrainableModule('internlm2-chat-7b').prompt(lazyllm.ChatPrompter(prompt))
```

与此同时我们还需要再引入SD3模型：

```python
sd3 = lazyllm.TrainableModule('stable-diffusion-3-medium')
```

这里SD3模型是一个非 LLM 模型，但使用方式与 LLM 模型一致——直接在 [TrainableModule][lazyllm.module.TrainableModule] 中指定模型名即可。

### 组装应用

LazyLLM 中有很多类型的控制流，控制流一般就是用于控制数据的流向。通过控制流将模块组装起来，以构成我们的绘画大师。这里我们选择使用 [Pipeline][lazyllm.flow.Pipeline] 来实现顺序执行：先大模型生成提示词，再将提示词喂给SD3模型来获取图像。

```python
with pipeline() as ppl:
    ppl.llm = lazyllm.TrainableModule('internlm2-chat-7b').prompt(lazyllm.ChatPrompter(prompt))
    ppl.sd3 = lazyllm.TrainableModule('stable-diffusion-3-medium')
```

在上面的代码片段中，我们使用 `pipeline` 上下文管理器来构建一个 LazyLLM 的控制流程。

[](){#use-pipeline}

```python
with pipeline() as ppl:
```

这行代码创建了一个名为 `ppl` 的管道实例，并进入了一个上下文管理器。

!!! Note "注意"

    - 要将模块添加到 `ppl` 中，需要给ppl添加属性：`ppl.llm = xxx` 和 `ppl.sd3 = xxx`
    - 未明确添加到 `ppl` 中的模块是不会经过控制流的；

### 启动应用

最后，我们将控制流 `ppl` 套入一个客户端，并启动部署（`start()`），在部署完后保持客户端不关闭（`wait()`）。

```python
lazyllm.WebModule(ppl, port=23466).start().wait()
```

## 完整代码
<details>
<summary>点击获取import和prompt</summary>

```python
import lazyllm
from lazyllm import pipeline

prompt = 'You are a drawing prompt word master who can convert any Chinese content entered by the user into English drawing prompt words. In this task, you need to convert any input content into English drawing prompt words, and you can enrich and expand the prompt word content.'
```
</details>

```python
with pipeline() as ppl:
    ppl.llm = lazyllm.TrainableModule('internlm2-chat-7b').prompt(lazyllm.ChatPrompter(prompt))
    ppl.sd3 = lazyllm.TrainableModule('stable-diffusion-3-medium')
lazyllm.WebModule(ppl, port=23466).start().wait()
```

效果如下：

![Painting Master](../assets/2_painting_master2.png)
