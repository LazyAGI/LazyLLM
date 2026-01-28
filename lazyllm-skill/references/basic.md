# LazyLLM的基础能力

## 模型使用

### AutoModel使用
LazyLLM的模型使用通过AutoModel进行整合，用于快速创建在线推理模块 OnlineModule 或本地 TrainableModule 的工厂类。会优先采用用户传入的参数，若开启 config 则会根据 auto_model_config_map 中的配置进行覆盖，然后自动判断应当构建在线模块还是本地模块：

    - 当判定为在线模块时，参数会透传给 OnlineModule（自动匹配 OnlineChatModule / OnlineEmbeddingModule / OnlineMultiModalModule）。
    - 当判定为本地模块时，则以 model 与用户参数初始化 TrainableModule，并读取 config map 里的配置参数。

参数：

- model (str) – 指定模型名称。例如 Qwen3-32B。必填。
- config_id (Optional[str]) – 指定配置文件里的id。默认为空。
- source (Optional[str]) – 使用的服务提供方。为在线模块（OnlineModule）指定 qwen / glm / openai 等；若设为 local 则强制创建本地 TrainableModule。
- type (Optional[str]) – 模型类型。若未指定会尝试从 kwargs 中获取或由在线模块自动推断。
- config (Union[str, bool]) – 是否启用 auto_model_config_map 的覆盖逻辑，或者用户指定的 config 文件路径。默认为 True。
- **kwargs – 兼容 model 的同义字段 base_model 和 embed_model_name，不接收其他用户传入的字段。

```python
import lazyllm

llm = lazyllm.AutoModel(
        source="sensenova", 
        model="SenseNova-V6-5-Pro"
    )
response = llm("你好，简单介绍一下你自己。")
print(response)
```

### 在线大模型

内置供应商:

- Doubao(豆包)
- PPIO(派欧云)
- OpenAI(OpenAI)
- Qwen(千问)
- GLM(智谱)
- Kimi(Kimi)
- SenseNova(商汤)
- SiliconFlow(硅基流动)
- AiPing(清程极智)

通过OnlineChatModule和OnlineEmbeddingModule分别调用在线大模型和嵌入模型。

OnlineChatModule:

参数:

- source 指定模型服务来源
- model 指定具体的模型名称

OnlineEmbeddingModule:

- source 指定模型服务来源
- model 指定具体的模型名称

```python
import lazyllm

llm = lazyllm.OnlineChatModule(
        source="sensenova", 
        model="SenseNova-V6-5-Pro"
    )
response = llm("你好，简单介绍一下你自己。")
print(response)
```

### 本地大模型

通过TrainableModule接入和部署本地模型，并提供相应的模型服务。

- 在基础的使用方式下，如果没有下载过模型，则会自动去 huggingface 或者 modelscope 上下载模型。如果不希望自动下载模型，则可以为TrainableModule传入参数 trust_remote_code=False 。当客户端连接lazyllm启动的推理服务时，这个做法通常很有效。
- 模型默认会下载到 ~/.lazyllm/ 目录下。如果想要下载到其他目录下，则可以使用环境变量 LAZYLLM_MODEL_CACHE_DIR=/path/to/model 来指定模型的下载路径
- 由于中国境内的上网策略，因此默认的模型源为 modelscope 。如果希望更换模型源，则可以使用环境变量 LAZYLLM_MODEL_SOURCE=huggingface 来将模型源切换到huggingface。
- 如果本地有下载好的模型权重，则可以用绝对路径指定 model = lazyllm.TrainableModule('/path/to/qwen2-1.5b')
- 如果将模型下载到统一的位置，则可以配置环境变量 LAZYLLM_MODEL_PATH=/path/to 来指定模型的根目录

基础使用：

```python
import lazyllm

model = lazyllm.TrainableModule('qwen2-1.5b').start()
response = model('Hello')
print(response)
```

### 模型共享

多个实例可以共享同一个微调模型，节省资源。

```python
# 创建基础模型
base_model = lazyllm.TrainableModule('qwen2-1.5b').start()

# 创建共享实例，使用相同的模型但不同的 prompt
chat_model = base_model.share(prompt="你是一个聊天机器人")
code_model = base_model.share(prompt="你是一个编程助手")
```

## Prompter提示词设计

借助内置Prompter来实现提示词注入

###内置Prompter:

- EmptyPrompter: 空提示生成器，用于直接返回原始输入。该类不会对输入进行任何处理，适用于无需格式化的调试、测试或占位场景。

参数:

- input (Any) – 任意输入，作为Prompt返回。
- history (Option[List[List | Dict]], default: None ) – 历史对话，可忽略，默认None。
- tools (Option[List[Dict]], default: None ) – 工具参数，可忽略，默认None。
- label (Option[str], default: None ) – 标签，可忽略，默认None。
- show (bool, default: False ) – 是否打印返回内容，默认为False。

```python
from lazyllm.components.prompter import EmptyPrompter
prompter = EmptyPrompter()
prompter.generate_prompt("Hello LazyLLM")
'Hello LazyLLM'
prompter.generate_prompt({"query": "Tell me a joke"})
{'query': 'Tell me a joke'}
# Even with additional parameters, the input is returned unchanged
prompter.generate_prompt("No-op", history=[["Hi", "Hello"]], tools=[{"name": "search"}], label="debug")
'No-op'
```

- Prompter: 用于生成模型输入的Prompt类，支持模板、历史对话拼接与响应抽取。该类支持从字典、模板名称或文件中加载prompt配置，支持历史对话结构拼接（用于Chat类任务）， 可灵活处理有/无history结构的prompt输入，适配非字典类型输入。

参数:

- prompt (Optional[str], default: None ) – 模板Prompt字符串，支持格式化字段。
- response_split (Optional[str], default: None ) – 对模型响应进行切分的分隔符，仅用于抽取模型回答。
- chat_prompt (Optional[str], default: None ) – 多轮对话使用的Prompt模板，必须包含history字段。
- history_symbol (str, default: 'llm_chat_history' ) – 表示历史对话字段的名称，默认为'llm_chat_history'。
- eoa (Optional[str], default: None ) – 对话中 assistant/user 分隔符。
- eoh (Optional[str], default: None ) – 多轮history中 user-assistant 分隔符。
- show (bool, default: False ) – 是否打印最终生成的Prompt，默认False。

```python
from lazyllm import Prompter
p = Prompter(prompt="Answer the following: {question}")
p.generate_prompt("What is AI?")
'Answer the following: What is AI?'
p.generate_prompt({"question": "Define machine learning"})
'Answer the following: Define machine learning'
p = Prompter(
...     prompt="Instruction: {instruction}",
...     chat_prompt="Instruction: {instruction}\nHistory:\n{llm_chat_history}",
...     history_symbol="llm_chat_history",
...     eoa="</s>",
...     eoh="|"
... )
p.generate_prompt(
...     input={"instruction": "Translate this."},
...     history=[["hello", "你好"], ["how are you", "你好吗"]]
... )
'Instruction: Translate this.\nHistory:\nhello|你好</s>how are you|你好吗'
prompt_conf = {
...     "prompt": "Task: {task}",
...     "response_split": "---"
... }
p = Prompter.from_dict(prompt_conf)
p.generate_prompt("Summarize this article.")
'Task: Summarize this article.'
```

- AlpacaPrompter: Alpaca格式的Prompter，支持工具调用，不支持历史对话。

参数:

- instruction (Option[str], default: None ) – 大模型的任务指令，至少带一个可填充的槽位(如 {instruction})。或者使用字典指定 system 和 user 的指令。
- extra_keys (Option[List], default: None ) – 额外的字段，用户的输入会填充这些字段。
- show (bool, default: False ) – 标志是否打印生成的Prompt，默认为False
- tools (Option[list], default: None ) – 大模型可以使用的工具集合，默认为None

```python
from lazyllm import AlpacaPrompter
p = AlpacaPrompter('hello world {instruction}')
p.generate_prompt('this is my input')
'You are an AI-Agent developed by LazyLLM.\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\n\n ### Instruction:\nhello world this is my input\n\n\n### Response:\n'
p.generate_prompt('this is my input', return_dict=True)
{'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\n\n ### Instruction:\nhello world this is my input\n\n'}, {'role': 'user', 'content': ''}]}
p = AlpacaPrompter('hello world {instruction}, {input}', extra_keys=['knowledge'])
p.generate_prompt(dict(instruction='hello world', input='my input', knowledge='lazyllm'))
'You are an AI-Agent developed by LazyLLM.\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\n\n ### Instruction:\nhello world hello world, my input\n\nHere are some extra messages you can referred to:\n\n### knowledge:\nlazyllm\n\n\n### Response:\n'
p.generate_prompt(dict(instruction='hello world', input='my input', knowledge='lazyllm'), return_dict=True)
{'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\n\n ### Instruction:\nhello world hello world, my input\n\nHere are some extra messages you can referred to:\n\n### knowledge:\nlazyllm\n\n'}, {'role': 'user', 'content': ''}]}
p = AlpacaPrompter(dict(system="hello world", user="this is user instruction {input}"))
p.generate_prompt(dict(input="my input"))
'You are an AI-Agent developed by LazyLLM.\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\n\n ### Instruction:\nhello word\n\n\n\nthis is user instruction my input### Response:\n'
p.generate_prompt(dict(input="my input"), return_dict=True)
{'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\n\n ### Instruction:\nhello world'}, {'role': 'user', 'content': 'this is user instruction my input'}]}
```

- ChatModel: 用于多轮对话的大模型Prompt构造器，支持工具调用、历史对话与自定义指令模版。支持传入 system/user 拆分的指令结构，自动合并为统一模板。支持额外字段注入和打印提示信息。

参数:

- instruction (Option[str | Dict[str, str]], default: None ) – Prompt模板指令，可为字符串或包含 system 和 user 的字典。若为字典，将自动拼接并注入特殊标记分隔符。
- extra_keys (Option[List[str]], default: None ) – 额外的字段列表，用户输入中的内容会被插入对应槽位，用于丰富上下文。
- show (bool, default: False ) – 是否打印生成的Prompt，默认False。
- tools (Option[List], default: None ) – 可选的工具列表，用于FunctionCall任务，默认None。
- history (Option[List[List[str]]], default: None ) – 可选的历史对话，用于对话记忆，格式为[[user, assistant], ...]，默认None。

```python
from lazyllm import ChatPrompter
# Simple instruction string
p = ChatPrompter('hello world')
p.generate_prompt('this is my input')
'You are an AI-Agent developed by LazyLLM.hello world\nthis is my input\n'
# Using extra_keys
p = ChatPrompter('hello world {instruction}', extra_keys=['knowledge'])
p.generate_prompt({
...     'instruction': 'this is my ins',
...     'input': 'this is my inp',
...     'knowledge': 'LazyLLM-Knowledge'
... })
'You are an AI-Agent developed by LazyLLM.hello world this is my ins\nHere are some extra messages you can referred to:\n\n### knowledge:\nLazyLLM-Knowledge\nthis is my inp\n'
# With conversation history
p.generate_prompt({
...     'instruction': 'this is my ins',
...     'input': 'this is my inp',
...     'knowledge': 'LazyLLM-Knowledge'
... }, history=[['s1', 'e1'], ['s2', 'e2']])
'You are an AI-Agent developed by LazyLLM.hello world this is my ins\nHere are some extra messages you can referred to:\n\n### knowledge:\nLazyLLM-Knowledge\ns1|e1\ns2|e2\nthis is my inp\n'
# Using dict format for system/user instructions
p = ChatPrompter(dict(system="hello world", user="this is user instruction {input}"))
p.generate_prompt({'input': "my input", 'query': "this is user query"})
'You are an AI-Agent developed by LazyLLM.hello world\nthis is user instruction my input this is user query\n'
p.generate_prompt({'input': "my input", 'query': "this is user query"}, return_dict=True)
{'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\nhello world'}, {'role': 'user', 'content': 'this is user instruction my input this is user query'}]}
```

### 结合大模型

```python
# 使用OnlineChatModule
import lazyllm
instruction = '你是一个由LazyLLM开发的知识问答助手，你的任务是根据提供的上下文信息来回答用户的问题。上下文信息是{context}，用户的问题是{input}, 现在请你做出回答。'
prompter = lazyllm.AlpacaPrompter({"user": instruction})
module = lazyllm.OnlineChatModule('openai').prompt(prompter)
module(dict(context='背景', input='输入'))

# 使用TrainableModule
import lazyllm
instruction = '你是一个由LazyLLM开发的知识问答助手，你的任务是根据提供的上下文信息来回答用户的问题。上下文信息是{context}，用户的问题是{input}, 现在请你做出回答。'
prompter = lazyllm.AlpacaPrompter(instruction)
module = lazyllm.TrainableModule('internlm2-chat-7b').prompt(prompter)
module.start()
module(dict(context='背景', input='输入'))
```
