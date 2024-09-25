# 大作家

本文我们将实现一个写作机器人应用。

!!! abstract "通过本节您将学习到 LazyLLM 的以下要点"

    - 如何给模型设置 [formatter][lazyllm.components.formatter.LazyLLMFormatterBase]；
    - 如何基于 [Warp][lazyllm.flow.Warp] 控制流来实现多输入并发；
    - 如何在函数上使用 [bind](../Best Practice/flow.md#use-bind) 来传入参数；

## 设计思路

为了能实现更长文本的内容生成，我们打算使用两个机器人来实现。第一个机器人用于生成目录大纲以及对每个大纲的简要描述，第二个机器人用于接收每个大纲的信息并输出对这个大纲对应的内容。最后将目录和大纲拼接一起来实现长文本的生成。

综合以上想法，我们进行如下设计：

![Great Writer](../assets/4_great_writer.svg)

UI-Web 接收来自用户的请求发送给目录大纲生成机器人，该机器人生成标题及其描述。接下来每个标题及其描述被发给第二个机器人，第二个机器人生成对应的内容。合成阶段会融合标题及其内容返回给客户端。

## 代码实现

让我们基于 LazyLLM 来实现上述设计思路吧。

### 设计提示词

根据设计，我们需要一个模型进行大纲拟定及其描述，另外一个模型根据提供的信息来写每一个大纲。所以我们需要设计两个提示词。

首先是大纲生成机器人的提示词：

```python
toc_prompt=""" 你现在是一个智能助手。你的任务是理解用户的输入，将大纲以列表嵌套字典的列表。每个字典包含一个 `title` 和 `describe`，其中 `title` 中需要用Markdown格式标清层级，`describe` `describe` 是对该段的描述和写作指导。

请根据以下用户输入生成相应的列表嵌套字典：

输出示例:
[
    {
        "title": "# 一级标题",
        "describe": "请详细描述此标题的内容，提供背景信息和核心观点。"
    },
    {
        "title": "## 二级标题",
        "describe": "请详细描述标题的内容，提供具体的细节和例子来支持一级标题的观点。"
    },
    {
        "title": "### 三级标题",
        "describe": "请详细描述标题的内容，深入分析并提供更多的细节和数据支持。"
    }
]
用户输入如下：
"""
```

然后是内容生成机器人的提示词：

``` python
completion_prompt="""
你现在是一个智能助手。你的任务是接收一个包含 `title` 和 `describe` 的字典，并根据 `describe` 中的指导展开写作
输入示例:
{
    "title": "# 一级标题",
    "describe": "这是写作的描述。"
}

输出:
这是展开写作写的内容
接收如下：

"""

writer_prompt = {"system": completion_prompt, "user": '{"title": {title}, "describe": {describe}}'}
```

### 设置模型

首先是大纲机器人：

```python
outline_writer = lazyllm.TrainableModule('internlm2-chat-7b').formatter(JsonFormatter()).prompt(toc_prompt)
```

这里继续使用了 `internlm2-chat-7b` 模型，并且设置了提示词。类似应用可参见：[绘画大师](painting_master.md#use-prompt)

值得注意的是，这里进行了 `formatter` 的设置，指定了使用 [JsonFormatter][lazyllm.components.JsonFormatter]， 他可以从模型输出的字符串中提取出 json。

由于我们在提示词设计的时候要求大纲机器人输出的就是 json 格式的字符串，所以在这里就需要对输出进行解析。

然后是内容生成机器人：

```python
story_generater = warp(ppl.outline_writer.share(prompt=writer_prompt).formatter())
```

这里使用了 [TrainableModule][lazyllm.module.TrainableModule] 的 `share`，类似应用可见：[多模态机器人](multimodal_robot.md#use_share)，
它可以让同一个模型用不同的提示词模板来作为不同的机器人。

### 组装应用

让我们把上面的模块用控制流把它组装起来吧。

```python
with pipeline() as ppl:
    ppl.outline_writer = lazyllm.TrainableModule('internlm2-chat-7b').formatter(JsonFormatter()).prompt(toc_prompt)
    ppl.story_generater = warp(ppl.outline_writer.share(prompt=writer_prompt).formatter())
    ppl.synthesizer = (lambda *storys, outlines: "\n".join([f"{o['title']}\n{s}" for s, o in zip(storys, outlines)])) | bind(outlines=ppl.output('outline_writer'))
```

上面代码中，除了常用的 [Pipeline][lazyllm.flow.Pipeline] 控制流(类似应用见：[绘画大师](painting_master.md#use-pipeline))，
需要关注的是使用了 [Warp][lazyllm.flow.Warp] 这个控制流（设计图中的红色线条）。

```python
warp(ppl.outline_writer.share(prompt=writer_prompt).formatter())
```

它接受任意多个输入，然后并行地送入到同一个分支上。由于上一步骤大纲机器人输入的 json 对数（即章节的个数）是不确定的，
而且一般是多个不同的输出。作为下一级的内容生成机器人，每个输入都需要进行处理，所以用 [Warp][lazyllm.flow.Warp] 就再合适不过了。

```python
ppl.synthesizer = (lambda *storys, outlines: "\n".join([f"{o['title']}\n{s}" for s, o in zip(storys, outlines)])) | bind(outlines=ppl.output('outline_writer'))
```

[](){#use-bind}

这个代码我们先来关注 `bind`（对应设计图中的蓝色线条）。这里的 `bind` 是将大纲机器人输出的大纲送入给匿名函数中的 `outlines` 上。
与此同时，上一步输出的每段内容被 `*storys` 打包为一个元组。最终合成的内容就是每一章的小标题+内容。 `bind` 用法介绍详见：[参数绑定](../Best Practice/flow.md#use-bind)

### 启动应用

最后，我们将控制流 `ppl` 套入一个客户端，并启动部署（`start()`），在部署完后保持客户端不关闭（`wait()`）。

```python
lazyllm.WebModule(ppl, port=23466).start().wait()
```

## 完整代码

<details>
<summary>点击查看import和prompt</summary>

```python
import lazyllm
from lazyllm import pipeline, warp, bind
from lazyllm.components.formatter import JsonFormatter

toc_prompt=""" 你现在是一个智能助手。你的任务是理解用户的输入，将大纲以列表嵌套字典的列表。每个字典包含一个 `title` 和 `describe`，其中 `title` 中需要用Markdown格式标清层级，`describe` `describe` 是对该段的描述和写作指导。

请根据以下用户输入生成相应的列表嵌套字典：

输出示例:
[
    {
        "title": "# 一级标题",
        "describe": "请详细描述此标题的内容，提供背景信息和核心观点。"
    },
    {
        "title": "## 二级标题",
        "describe": "请详细描述标题的内容，提供具体的细节和例子来支持一级标题的观点。"
    },
    {
        "title": "### 三级标题",
        "describe": "请详细描述标题的内容，深入分析并提供更多的细节和数据支持。"
    }
]
用户输入如下：
"""

completion_prompt="""
你现在是一个智能助手。你的任务是接收一个包含 `title` 和 `describe` 的字典，并根据 `describe` 中的指导展开写作
输入示例:
{
    "title": "# 一级标题",
    "describe": "这是写作的描述。"
}

输出:
这是展开写作写的内容
接收如下：

"""

writer_prompt = {"system": completion_prompt, "user": '{"title": {title}, "describe": {describe}}'}
```
</details>

```python
with pipeline() as ppl:
    ppl.outline_writer = lazyllm.TrainableModule('internlm2-chat-7b').formatter(JsonFormatter()).prompt(toc_prompt)
    ppl.story_generater = warp(ppl.outline_writer.share(prompt=writer_prompt).formatter())
    ppl.synthesizer = (lambda *storys, outlines: "\n".join([f"{o['title']}\n{s}" for s, o in zip(storys, outlines)])) | bind(outlines=ppl.output('outline_writer'))
lazyllm.WebModule(ppl, port=23466).start().wait()
```
