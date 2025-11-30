# 多模态聊天机器人

我们将进一步增强我们的机器人！在上一节的 [绘画大师](painting_master.md) 的基础上，引入更多模型使它成为一个多模态机器人！让我们开始吧！

!!! abstract "通过本节您将学习到 LazyLLM 的以下要点"

    - 如何给同一个模型设置不同的提示词；
    - 如何基于 [Switch][lazyllm.flow.Switch] 控制流来实现路由，我们将基于它结合 LLM 实现一个简单的意图识别机器人；
    - 如何给 [TrainableModule][lazyllm.module.TrainableModule] 指定部署的框架；
    - 如何在控制流上使用 [bind](../Best Practice/flow.md#use-bind) 来传入参数；
    - 如何使用 [IntentClassifier][lazyllm.tools.IntentClassifier] 来实现意图识别。

## 设计思路

为了增加我们机器人的功能，让它不仅会画画，还要让它能语音识别、编曲、图文问答等，让它具有多媒体的能力。这里我们将引入以下模型：

- `ChatTTS`：用于将文本转换为语音；
- `musicgen-small`：用于生成音乐；
- `stable-diffusion-3-medium`: 沿用上一节 [绘画大师](painting_master.md) 的模型，用于生成图像；
- `internvl-chat-2b-v1-5`：用于图文问答；
- `SenseVoiceSmall`: 用于语音识别；

我们注意到引入的模型中有生成图像和生成音乐的模型，他们对提示词的要求都相对较高，
我们需要依靠一个 LLM 来实现提示词的生成和翻译，就像是上一节 [绘画大师](painting_master.md) 那样。

另外由于引入了大量的模型，我们需要一个意图识别机器人来实现对用户意图的转发，把用户的意图路由给对应的模型。

综合以上考虑，我们进行如下设计：

![Multimodal bot](../assets/3_multimodal-bot3.svg)

这里意图识别机器人为核心，通过它将用户的需求路由到六个功能机器人之一上。

值得注意的是 LazyLLM 提供了一个意图识别工具，详见本节的 [代码优化](#code_opt)。

## 代码实现

让我们基于 LazyLLM 来实现上述设计思路吧。

### 设计提示词

这里我们需要设计三个提示词，分别给意图识别机器人、绘画生成机器人、音乐生成机器人用。

对于意图识别机器人，我们如下设计：

```python
chatflow_intent_list = ["聊天", "语音识别", "图片问答", "画图", "生成音乐", "文字转语音"]
agent_prompt = f"""
现在你是一个意图分类引擎，负责根据对话信息分析用户输入文本并确定唯一的意图类别。\n你只需要回复意图的名字即可，不要额外输出其他字段，也不要进行翻译。"intent_list"为所有意图名列表。\n
如果输入中带有attachments，根据attachments的后缀类型以最高优先级确定意图：如果是图像后缀如.jpg、.png等，则输出：图片问答；如果是音频后缀如.mp3、.wav等，则输出：语音识别。
## intent_list:\n{chatflow_intent_list}\n\n## 示例\nUser: 你好啊\nAssistant:  聊天\n
"""
```

对于绘画生成机器人，我们如下设计：

```python
painter_prompt = '现在你是一位绘图提示词大师，能够将用户输入的任意中文内容转换成英文绘图提示词，在本任务中你需要将任意输入内容转换成英文绘图提示词，并且你可以丰富和扩充提示词内容。'
```

类似的还有音乐生成机器人：

```python
musician_prompt = '现在你是一位作曲提示词大师，能够将用户输入的任意中文内容转换成英文作曲提示词，在本任务中你需要将任意输入内容转换成英文作曲提示词，并且你可以丰富和扩充提示词内容。'
```

### 设置模型

在此次应用设计中，我们总共会用到四个不同的 LLM，但实际上它们之间的不同仅仅在于提示词不同，
所以我们只用一个模型 `internlm2-chat-7b` 来作为我们的共享机器人，通过设置不同的提示词模板来实现不同的机器人。

首先是我们的意图识别机器人，将刚设计的提示词给它设置上：

```python
base = TrainableModule('internlm2-chat-7b').prompt(agent_prompt)
```

然后是我们的聊天机器人，这里的提示词设置为空：

```python
chat = base.share().prompt()
```

[](){#use_share}

这里我们用到了 `TrainableModule` 的 `share` 功能。它可以在原来机器人基础上设置不同的模板，它返回的对象也是 [TrainableModule][lazyllm.module.TrainableModule] 的，但和原来的共享模型。


接下来是绘画和音乐生成机器人，也用 `share` 功能来实现不同功能的机器人：

```python
painter = pipeline(base.share().prompt(painter_prompt), TrainableModule('stable-diffusion-3-medium'))
musician = pipeline(base.share().prompt(musician_prompt), TrainableModule('musicgen-small'))
```

这里我们用到了 `pipeline` 的非上下文管理器用法（这里相当于沿用了上一节 [绘画大师](painting_master.md) 的代码，其使用的是 [Pipeline][lazyllm.flow.Pipeline] 上下文管理器用法。

对于剩下的多模态模型，我们都只用指定它的名字即可调用：

```pyhton
stt = TrainableModule('SenseVoiceSmall')
vqa = TrainableModule('internvl-chat-2b-v1-5').deploy_method(deploy.LMDeploy)
tts = TrainableModule('ChatTTS')
```

这里需要说明的是，`internvl-chat-2b-v1-5` 目前仅在 LMDeploy 中支持，所以需要明确指定部署用到的框架，即这里的：`.deploy_method(deploy.LMDeploy)`。

### 组装应用

让我们把上面定义好的模型组装起来：

```python
with pipeline() as ppl:
    ppl.cls = base
    ppl.cls_normalizer = lambda x: x if x in chatflow_intent_list else chatflow_intent_list[0]
    with switch(judge_on_full_input=False).bind(_0, ppl.input) as ppl.sw:
        ppl.sw.case[chatflow_intent_list[0], chat]
        ppl.sw.case[chatflow_intent_list[1], TrainableModule('SenseVoiceSmall')]
        ppl.sw.case[chatflow_intent_list[2], TrainableModule('internvl-chat-2b-v1-5').deploy_method(deploy.LMDeploy)]
        ppl.sw.case[chatflow_intent_list[3], pipeline(base.share().prompt(painter_prompt), TrainableModule('stable-diffusion-3-medium'))]
        ppl.sw.case[chatflow_intent_list[4], pipeline(base.share().prompt(musician_prompt), TrainableModule('musicgen-small'))]
        ppl.sw.case[chatflow_intent_list[5], TrainableModule('ChatTTS')]
```

在上面代码中，我们首先实例化了一个顺序执行的 `ppl`，在这个 `ppl` 中先进行意图识别，设置`ppl.cls`。
然后为了保证意图识别的鲁棒性，在意图识别之后设置一个 `ppl.cls_normalizer` 的匿名函数，
将识别的结果仅映射到预制列表属性项中，即：确保识别的结果只在预制表内。

```python
with switch(judge_on_full_input=False).bind(_0, ppl.input) as ppl.sw:
```

对于这行代码：

- 我们首先关注 `bind(_0, ppl.input)`, 其中 `_0` 是上一步输出的结果第0个参数，即意图列表中的一个意图。`ppl.input`是用户的输入（对应设计图中红色线条）。所以这行代码是给 `switch` 控制流设置了两个参数，第一个参数是意图，第二个参数是用户的输入。更多 `bind` 使用方法参见：[参数绑定](../Best Practice/flow.md#use-bind)

- 然后 `judge_on_full_input=False`，可以将输入分为两部分，第一部分是作为判断条件，剩下部分作为分支的输入，否则如果为 `True` 就会把整个输入作为判断条件和分支输入。

- 最后我们将实例化后的 `switch` 也添加到了 `ppl` 上：`ppl.sw`。更多参见：[Switch][lazyllm.flow.Switch]。

剩下代码基本一致，都是设置条件和对应路由分支，以下面代码为例：

```python
ppl.sw.case[chatflow_intent_list[0], chat]
```

该代码的第一个参数是意图列表的第一个元素，即"聊天"，这个是判断条件的依据。
如果 `switch` 输入的第一个参数是“聊天”，那么就会走向这个分支 `chat`。而 `chat` 的输入就是 `ppl.input`。

### 启动应用

最后，我们将控制流 `ppl` 套入一个客户端，并启动部署（`start()`），在部署完后保持客户端不关闭（`wait()`）。

```python
WebModule(ppl, history=[chat], audio=True, port=8847).start().wait()
```

这里由于需要用到麦克风来捕获声音，所以设置了 `audio=True`。

## 完整代码

<details>
<summary>点击获取import和prompt</summary>

```python
from lazyllm import TrainableModule, WebModule, deploy, pipeline, switch, _0

chatflow_intent_list = ["聊天", "语音识别", "图片问答", "画图", "生成音乐", "文字转语音"]
agent_prompt = f"""
现在你是一个意图分类引擎，负责根据对话信息分析用户输入文本并确定唯一的意图类别。\n你只需要回复意图的名字即可，不要额外输出其他字段，也不要进行翻译。"intent_list"为所有意图名列表。\n
如果输入中带有attachments，根据attachments的后缀类型以最高优先级确定意图：如果是图像后缀如.jpg、.png等，则输出：图片问答；如果是音频后缀如.mp3、.wav等，则输出：语音识别。
## intent_list:\n{chatflow_intent_list}\n\n## 示例\nUser: 你好啊\nAssistant:  聊天\n
"""
painter_prompt = '现在你是一位绘图提示词大师，能够将用户输入的任意中文内容转换成英文绘图提示词，在本任务中你需要将任意输入内容转换成英文绘图提示词，并且你可以丰富和扩充提示词内容。'
musician_prompt = '现在你是一位作曲提示词大师，能够将用户输入的任意中文内容转换成英文作曲提示词，在本任务中你需要将任意输入内容转换成英文作曲提示词，并且你可以丰富和扩充提示词内容。'
```
</details>

```python
base = TrainableModule('internlm2-chat-7b').prompt(agent_prompt)
chat = base.share().prompt()
with pipeline() as ppl:
    ppl.cls = base
    ppl.cls_normalizer = lambda x: x if x in chatflow_intent_list else chatflow_intent_list[0]
    with switch(judge_on_full_input=False).bind(_0, ppl.input) as ppl.sw:
        ppl.sw.case[chatflow_intent_list[0], chat]
        ppl.sw.case[chatflow_intent_list[1], TrainableModule('SenseVoiceSmall')]
        ppl.sw.case[chatflow_intent_list[2], TrainableModule('internvl-chat-2b-v1-5').deploy_method(deploy.LMDeploy)]
        ppl.sw.case[chatflow_intent_list[3], pipeline(base.share().prompt(painter_prompt), TrainableModule('stable-diffusion-3-medium'))]
        ppl.sw.case[chatflow_intent_list[4], pipeline(base.share().prompt(musician_prompt), TrainableModule('musicgen-small'))]
        ppl.sw.case[chatflow_intent_list[5], TrainableModule('ChatTTS')]
WebModule(ppl, history=[chat], audio=True, port=8847).start().wait()
```
效果如下：

![Multimodal Robot](../assets/3_multimodal-bot2.png)

## 代码优化

[](){#code_opt}

上面代码中，我们使用 LazyLLM 的一些基本模块，实现了一个简单的意图识别。然而该意图识别代码比较冗余，使用比较复杂，而且不具有历史记忆等功能。所以 LazyLLM 提供了一个意图识别工具，简化意图识别的实现。我们将设计修改如下：

![Multimodal bot](../assets/3_multimodal-bot4.svg)

在 LazyLLM 中可以将 [IntentClassifier][lazyllm.tools.IntentClassifier] 作为一个上下文管理器来使用，它内置了 Prompt, 我们仅需要指定 LLM 模型即可，它的使用方法和 [Switch][lazyllm.flow.Switch] 类似，指定每个 `case` 分支，并设置其中的意图类别和对应调用的对象即可。完整代码实现如下：

<details>
<summary>点击获取import和prompt</summary>

```python
from lazyllm import TrainableModule, WebModule, deploy, pipeline
from lazyllm.tools import IntentClassifier

painter_prompt = '现在你是一位绘图提示词大师，能够将用户输入的任意中文内容转换成英文绘图提示词，在本任务中你需要将任意输入内容转换成英文绘图提示词，并且你可以丰富和扩充提示词内容。'
musician_prompt = '现在你是一位作曲提示词大师，能够将用户输入的任意中文内容转换成英文作曲提示词，在本任务中你需要将任意输入内容转换成英文作曲提示词，并且你可以丰富和扩充提示词内容。'
```
</details>

```python
base = TrainableModule('internlm2-chat-7b')
with IntentClassifier(base) as ic:
    ic.case['聊天', base]
    ic.case['语音识别', TrainableModule('SenseVoiceSmall')]
    ic.case['图片问答', TrainableModule('InternVL3_5-1B').deploy_method(deploy.LMDeploy)]
    ic.case['画图', pipeline(base.share().prompt(painter_prompt), TrainableModule('stable-diffusion-3-medium'))]
    ic.case['生成音乐', pipeline(base.share().prompt(musician_prompt), TrainableModule('musicgen-small'))]
    ic.case['文字转语音', TrainableModule('ChatTTS')]
WebModule(ic, history=[base], audio=True, port=8847).start().wait()
```
