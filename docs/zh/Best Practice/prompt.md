# Prompter

为了让您在不同的线上模型和不同的本地模型都能获得一致的使用体验，且在微调和推理中也能获得一致的使用体验，我们定义了 Prompter 类，用于屏蔽各个模型和推理框架的实现细节。
Prompter 的 API 文档可以参考 [prompter][lazyllm.components.prompter.LazyLLMPrompterBase]。 接下来，我将会从一个例子开始，逐步介绍 LazyLLM 中 Prompter 的设计思路。

## Prompter 牛刀小试

假设我们正在设计一个文档问答的应用，需要大模型的输入是用户的问题以及检索到的背景知识，此时我们的 Prompt 设计为：

```text
你是一个由LazyLLM开发的知识问答助手，你的任务是根据提供的上下文信息来回答用户的问题。上下文信息是{上下文}，用户的问题是{问题}，现在请你做出回答。
```

[](){#prompter_try}

你可以借助 LazyLLM 提供的内置 Prompter 来实现你需要的能力，示例如下：

```python
import lazyllm
instruction = '你是一个由LazyLLM开发的知识问答助手，你的任务是根据提供的上下文信息来回答用户的问题。上下文信息是{context}，用户的问题是{input}, 现在请你做出回答。'
prompter = lazyllm.AlpacaPrompter({"user": instruction})
module = lazyllm.OnlineChatModule('openai').prompt(prompter)
module(dict(context='背景', input='输入'))
```

上述 ``prompter`` 即可以直接传给大模型去使用。我们可以用如下代码试验 Prompter 的效果。

```python
import lazyllm
# 'You are an AI-Agent developed by LazyLLM.\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\n\n ### Instruction:\n你是一个由LazyLLM开发的知识问答助手，你的任务是根据提供的上下文信息来回答用户的问题。上下文信息是背景，用户的问题是输入，现在请你做出回答。\n\n\n### Response:\n'
prompter.generate_prompt(dict(context='背景', input='输入'))

# {'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\n\n ### Instruction:\n你是一个由LazyLLM开发的知识问答助手，你的任务是根据提供的上下文信息来回答用户的问题。上下文信息是背景，用户的问题是输入，现在请你做出回答。\n\n'}, {'role': 'user', 'content': ''}]}
prompter.generate_prompt(dict(context='背景', input='输入'), return_dict=True)
```

在上面的例子中， ``generate_prompt`` 的输入是一个 ``dict`` ，他会把值依次填入 ``instruction`` 提供的槽位中。

!!! Note "注意"

    - 上面代码中出现了一个值得您关注的参数 ``return_dict`` , 当 ``return_dict`` 为 True 时，会返回 OpenAI 格式的 dict 用于线上模型。
    - 一般情况下，您只需要将Prompter设置给 ``TrainableModule`` 或 ``OnlineChatModule`` 即可，而无需关心 ``generate_prompt`` 这一函数。

## LazyLLM Prompter 的设计思路

### 基本概念说明

**PromptTemplate**: 每个 Prompter 内置的 Template，每个子类均需要实现自己的 PrompterTemplate，用于选择启用部分字段，以及约定拼接规则。

- PrompterTemplate 中可选的字段有：
    - system: 系统提示，一般会读取模型的归属信息并进行设置，如不设置默认为 ``You are an AI-Agent developed by LazyLLM.`` 。
    - instruction: 任务指令，由 ``InstructionTemplate`` 拼接用户的输入得到。这个是应用开发者需要着重了解的字段。如果 instruction 是字符串，则默认是系统指令，如果是字典，且其键值只能是 ``system`` 和 ``user`` 。``system`` 指定的是系统级指令， ``user`` 指定的是用户级指令。
    - history: 历史对话，由用户的输入得到，格式为 ``[[a, b], [c, d]]`` 或 ``[{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]``
    - tools: 可以使用的工具，在构造 ``prompter`` 时传入或者由用户使用时传入，当构造 ``prompter`` 时定义了工具之后，将禁止用户使用时再次传入。格式为 ``[{"type": "function",  "function": {"name": "", "description": "", "parameters": {},  "required": []}]``
    - user: 用户级指令，可选指令，由用户通过 instruction 指定。
    - sos: ``start of system`` , 标志着系统提示的开始，该符号由模型填入，开发者和用户均无需考虑
    - eos: ``end of system`` , 标志着系统提示的结束，该符号由模型填入，开发者和用户均无需考虑
    - soh: ``start of human`` , 标志着用户输入的开始，常用于多轮对话中作为分隔符。该符号由模型填入，开发者和用户均无需考虑
    - eoh: ``end of human`` , 标志着用户输入的结束，常用于多轮对话中作为分隔符。该符号由模型填入，开发者和用户均无需考虑
    - soa: ``start of assistant`` , 标志着模型输出的开始，常用于多轮对话中作为分隔符。该符号由模型填入，开发者和用户均无需考虑
    - eoa: ``end of assistant`` , 标志着模型输出的结束，常用于多轮对话中作为分隔符。该符号由模型填入，开发者和用户均无需考虑
- ``TrainableModule`` 所使用的内置的Prompt的拼接规则如下：
    - AlpacaPrompter: ``{system}\n{instruction}\n{tools}\n{user}### Response:\n``
    - ChatPrompter: ``{sos}{system}{instruction}{tools}{eos}\n\n{history}\n{soh}\n{user}{input}\n{eoh}{soa}\n``
- ``OnlineChatModule`` 的输出格式为: ``dict(messages=[{"role": "system", "content": ""}, {"role": "user", "content": ""}, ...], tools=[])``

!!! Note "注意"

    - 从上面可以看出， ``AlpacaPrompter`` 相比于 ``ChatPrompter`` ，缺少了 ``history`` ，以及 ``soh`` 等多轮对话分割所需要的标识符。
    - 因此我们可以看出， ``AlpacaPrompter`` 不支持多轮对话。

**InstructionTemplate**: 每个 Prompter 内置的，用于结合用户输入的 ``instruction`` ，产生最终的 ``instruction`` 的模板。 ``InstructionTemplate`` 中的用到的 2 个字段是：

- ``instruction`` : 由开发者在构造 ``Prompter`` 时传入，可带若干个待填充的槽位，用于填充用户的输入。或者指定系统级指令和用户级指令，当指定用户级指令时，需要使用字典类型，且键值为 ``user`` 和 ``system`` 。
- ``extra_keys`` : 需要用户调用大模型时额外提供的信息，有开发者在构造 ``Prompter`` 时传入，会自动转换成 ``instruction`` 中的槽位。

!!! Note "注意"

    在内置的 ``Prompter`` 中， ``Alpaca``的 ``InstructionTemplate`` 额外附带了 ``alpaca`` 格式的标准提示词，即 ``Below is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request``

### Prompt 生成过程解析

我们借助 [Prompter 牛刀小试](#prompter_try) 中使用 ``AlpacaPrompter`` 的文档问答的例子，详细介绍一下 Prompt 的生成过程。

1. ``AlpacaPrompter`` 结合构造 ``prompter`` 时传入的 ``instruction`` （及 ``extra_keys``， 如有），结合 ``InstructionTemplate`` ，将 ``instruction`` 设置为:
```python
"Below is an instruction that describes a task, paired with extra messages such as input that provides "
"further context if possible. Write a response that appropriately completes the request.\\n\\n ### "
"Instruction:\\n 你是一个由LazyLLM开发的知识问答助手，你的任务是根据提供的上下文信息来回答用户的问题。上下文信息是{{context}}，"
"用户的问题是{{input}}, 现在请你做出回答。### Response:\\n}"
```

2. 用户的输入为 ``dict(context='背景', input='问题')``
3. 用户的输入与1中得到的 ``instruction`` 进行拼接 ，得到:
```python
"Below is an instruction that describes a task, paired with extra messages such as input that provides "
"further context if possible. Write a response that appropriately completes the request.\\n\\n ### "
"Instruction:\\n 你是一个由LazyLLM开发的知识问答助手，你的任务是根据提供的上下文信息来回答用户的问题。上下文信息是背景，"
"用户的问题是问题, 现在请你做出回答。### Response:\\n}"
```

4. ``AlpacaPrompter`` 读取 ``system`` 和 ``tools`` 字段，其中 ``system`` 字段由 ``Module`` 设置，而 ``tools`` 字段则会在后面的 : [使用工具](#_2) 一节中介绍。
[](){#analysis}
5. 如果 ``prompter`` 的结果用于线上模型（ ``OnlineChatModule`` ），则不会再进一步拼接 ``PromptTemplate`` ，而是会直接得到一个dict，即
```python
"{'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\n\n ### Instruction:\n你是一个由LazyLLM开发的知识问答助手，你的任务是根据提供的上下文信息来回答用户的问题。上下文信息是背景，用户的问题是输入，现在请你做出回答。\n\n'}, {'role': 'user', 'content': ''}]}"
```

6. 如果 ``prompter`` 的结果用于线下模型（ ``TrainableModule`` ），则会通过 ``PromptTemplate`` 得到最终的结果：
```python
"You are an AI-Agent developed by LazyLLM.\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\n\n ### Instruction:\n你是一个由LazyLLM开发的知识问答助手，你的任务是根据提供的上下文信息来回答用户的问题。上下文信息是背景，用户的问题是问题，现在请你做出回答。\n\n\n### Response:\n"
```

## 定义和使用 Prompter

### 定义一个新的 Prompter

参考 API 文档： [prompter][lazyllm.components.prompter.LazyLLMPrompterBase]

### Query为string，而非dict

我们在 [Prompter 牛刀小试](#prompter_try) 中展示了一个基本的用法，并在随后的小节里解释了 ``prompter`` 的工作原理。
但在绝大部分情况下，用户的输入往往是一个 ``string`` ，本小节展示了 ``prompter`` 在输入为 ``string`` 时的用法。

当用户的输入为 ``string`` 时，我们最多允许 ``Prompter`` 的 ``instruction`` 中有一个槽位。我们借助“大模型做加法”这一场景，给出一个示例的代码:

```python
>>> p = lazyllm.AlpacaPrompter('请完成加法运算, 输入为{instruction}')
>>> p.generate_prompt('a+b')
'You are an AI-Agent developed by LazyLLM.\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\n\\n ### Instruction:\\n请完成加法运算, 输入为a+b\\n\\n\\n### Response:\\n'
>>>  p = lazyllm.AlpacaPrompter('请完成加法运算', extra_keys='input')
'You are an AI-Agent developed by LazyLLM.\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\n\\n ### Instruction:\\n请完成加法运算\\n\\nHere are some extra messages you can referred to:\\n\\n### input:\\na+b\\n\\n\\n### Response:\\n'
```

!!! Note "注意"

    当使用 ``AlpacaPrompter`` 时，需要定义一个唯一的槽位，可以任意取一个名字， ``string`` 类型的输入会填充进去。

```python
>>> p = lazyllm.ChatPrompter('请完成加法运算，输入为{input}')
>> p.generate_prompt('a+b')
'<|start_system|>You are an AI-Agent developed by LazyLLM.请完成加法运算，输入为a+b\\n\\n<|end_system|>\\n\\n\\n<|Human|>:\\n\\n<|Assistant|>:\\n'
>>> p = lazyllm.ChatPrompter('请完成加法运算')
>> p.generate_prompt('a+b')
'<|start_system|>You are an AI-Agent developed by LazyLLM.请完成加法运算\n\n<|end_system|>\n\n\n<|Human|>:\na+b\n<|Assistant|>
```

!!! Note "注意"

    - 当使用 ``ChatPrompter`` 时，不同于 ``AlpacaPrompter`` ，在 ``instruction`` 中定义槽位不是必须的。
    - 如果不定义槽位，则输入会放到对话中作为用户的输入，在 ``<soh>`` 和 ``<eoh>`` 之间。
    - 如果像 ``AlpacaPrompter`` 一样定义了槽位，也可以任意取一个名字，此时输入会放到 ``<system>`` 字段中。
    - 如果 ``instruction`` 中指定了系统级指令和用户级指令，则在拼接完成后，系统级指令放在prompt_template中的{instruction}位置，用户级指令放在{user}位置。

### 使用工具

一般来说，大模型在进行 ``function-call`` 时，需要按照约定的格式定义好一系列工具，然后按照一定的格式传给大模型去使用。工具可以在构造 ``prompter`` 时传入，也可以由用户使用时传入。当构造 ``prompter`` 时定义了工具之后，将禁止用户使用时再次传入。
工具的格式一般为:

```python
    [
        {
            "type": "function",
            "function": {
                "name": "",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "arg-1": {
                            "type": "",
                            "description": "",
                        },
                        "arg-2": {}
                        "arg-3": {}
                    }
                },
                "required": ['arg-1', 'arg-2', 'arg-3']
            }
        },
    ]
```

下面我们借助一个很简单的工具 ``tools=[dict(type='function', function=dict(name='example'))]`` 来演示 ``Prompter`` 是如何使用工具的。

1. 应用开发者定义工具

    ```python
    >>> import lazyllm
    >>> tools=[dict(type='function', function=dict(name='example'))]
    >>> prompter = lazyllm.AlpacaPrompter('你是一个工具调用的Agent，我会给你提供一些工具，请根据用户输入，帮我选择最合适的工具并使用', extra_keys='input', tools=tools)
    >>> prompter.generate_prompt('帮我查询一下今天的天气')
    'You are an AI-Agent developed by LazyLLM.\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\n\\n ### Instruction:\\n你是一个工具调用的Agent，我会给你提供一些工具，请根据用户输入，帮我选择最合适的工具并使用\\n\\nHere are some extra messages you can referred to:\\n\\n### input:\\n帮我查询一下今天的天气\\n\\n\\n### Function-call Tools. \\n\\n[{"type": "function", "function": {"name": "example"}}]\\n\\n### Response:\\n'
    >>>
    >>> prompter = lazyllm.ChatPrompter('你是一个工具调用的Agent，我会给你提供一些工具，请根据用户输入，帮我选择最合适的工具并使用', tools=tools)
    >>> prompter.generate_prompt('帮我查询一下今天的天气')
    '<|start_system|>You are an AI-Agent developed by LazyLLM.你是一个工具调用的Agent，我会给你提供一些工具，请根据用户输入，帮我选择最合适的工具并使用\\n\\n### Function-call Tools. \\n\\n[{"type": "function", "function": {"name": "example"}}]\\n\\n<|end_system|>\\n\\n\\n<|Human|>:\\n帮我查询一下今天的天气\\n<|Assistant|>:\\n'
    ```

2. 用户定义工具

    ```python
    >>> import lazyllm
    >>> tools=[dict(type='function', function=dict(name='example'))]
    >>> prompter = lazyllm.AlpacaPrompter('你是一个工具调用的Agent，我会给你提供一些工具，请根据用户输入，帮我选择最合适的工具并使用', extra_keys='input')
    >>> prompter.generate_prompt('帮我查询一下今天的天气', tools=tools)
    'You are an AI-Agent developed by LazyLLM.\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\n\\n ### Instruction:\\n你是一个工具调用的Agent，我会给你提供一些工具，请根据用户输入，帮我选择最合适的工具并使用\\n\\nHere are some extra messages you can referred to:\\n\\n### input:\\n帮我查询一下今天的天气\\n\\n\\n### Function-call Tools. \\n\\n[{"type": "function", "function": {"name": "example"}}]\\n\\n### Response:\\n'
    >>>
    >>> prompter = lazyllm.ChatPrompter('你是一个工具调用的Agent，我会给你提供一些工具，请根据用户输入，帮我选择最合适的工具并使用')
    >>> prompter.generate_prompt('帮我查询一下今天的天气', tools=tools)
    '<|start_system|>You are an AI-Agent developed by LazyLLM.你是一个工具调用的Agent，我会给你提供一些工具，请根据用户输入，帮我选择最合适的工具并使用\\n\\n### Function-call Tools. \\n\\n[{"type": "function", "function": {"name": "example"}}]\\n\\n<|end_system|>\\n\\n\\n<|Human|>:\\n帮我查询一下今天的天气\\n<|Assistant|>:\\n'
    ```

工具会在 [Prompt 生成过程解析](#analysis) 中的步骤4，转换为 json 后被读取。

!!! Note "注意"

    如果是使用线上模型，工具会变成和 ``messages`` 并列的一个字段，示例如下：

        >>> import lazyllm
        >>> tools=[dict(type='function', function=dict(name='example'))]
        >>> prompter = lazyllm.AlpacaPrompter('你是一个工具调用的Agent，我会给你提供一些工具，请根据用户输入，帮我选择最合适的工具并使用', extra_keys='input', tools=tools)
        >>> prompter.generate_prompt('帮我查询一下今天的天气', return_dict=True)
        {'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\n\\n ### Instruction:\\n你是一个工具调用的Agent，我会给你提供一些工具，请根据用户输入，帮我选择最合适的工具并使用\\n\\nHere are some extra messages you can referred to:\\n\\n### input:\\n帮我查询一下今天的天气\\n\\n'}, {'role': 'user', 'content': ''}],
        'tools': [{'type': 'function', 'function': {'name': 'example'}}]}


### 使用历史对话

如果我们想让模型具备多轮对话的能力，就需要将对话上下文拼接到 ``prompt`` 当中。上下文是由用户传入的，但需要以键值对的形式传入。下面给出一个例子：

```python
>>> import lazyllm
>>> prompter = lazyllm.ChatPrompter('你是一个对话机器人，现在你要和用户进行友好的对话')
>>> prompter.generate_prompt('我们聊会儿天吧', history=[['你好', '你好，我是一个对话机器人，有什么能为您服务的']])
'<|start_system|>You are an AI-Agent developed by LazyLLM.你是一个对话机器人，现在你要和用户进行友好的对话\\n\\n<|end_system|>\\n\\n<|Human|>:你好<|Assistant|>:你好，我是一个对话机器人，有什么能为您服务的\\n<|Human|>:\\n我们聊会儿天吧\\n<|Assistant|>:\\n'
>>> prompter.generate_prompt('我们聊会儿天吧', history=[['你好', '你好，我是一个对话机器人，有什么能为您服务的']], return_dict=True)
{'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\n你是一个对话机器人，现在你要和用户进行友好的对话\\n\\n'}, {'role': 'user', 'content': '你好'}, {'role': 'assistant', 'content': '你好，我是一个对话机器人，有什么能为您服务的'}, {'role': 'user', 'content': '我们聊会儿天吧'}]}
>>> prompter.generate_prompt('我们聊会儿天吧', history=[dict(role='user', content='你好'), dict(role='assistant', content='你好，我是一个对话机器人，有什么能为您服务的')], return_dict=True)
{'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\n你是一个对话机器人，现在你要和用户进行友好的对话\\n\\n'}, {'role': 'user', 'content': '你好'}, {'role': 'assistant', 'content': '你好，我是一个对话机器人，有什么能为您服务的'}, {'role': 'user', 'content': '我们聊会儿天吧'}]}
```

历史对话会在 [Prompt 生成过程解析](#analysis) 中的步骤4，做简单的格式转换后被读取。

!!! Note "注意"

    - 只有 ``ChatPrompter`` 支持传入历史对话
    - 当输入是 ``[[a, b], ...]`` 格式时，同时支持 ``return_dict`` 为 ``True`` 或 ``False`` ， 而当输入为  ``[dict, dict]`` 格式时，仅支持 ``return_dict`` 为 ``True``

### 和 OnlineChatModule 一起使用

当 ``Prompter`` 和 ``OnlineChatModule`` 一起使用时， ``OnlineChatModule.__call__`` 会调用 ``Prompter.generate_prompt`` ，并且将 ``__input``,
``history`` 和 ``tools`` 传给 ``generate_prompt`` ，此时 ``generate_prompt`` 的 ``return_dict`` 会被设置为 ``True``。下面给出一个例子：

```python
import lazyllm
instruction = '你是一个由LazyLLM开发的知识问答助手，你的任务是根据提供的上下文信息来回答用户的问题。上下文信息是{context}，用户的问题是{input}, 现在请你做出回答。'
prompter = lazyllm.AlpacaPrompter(instruction)
module = lazyllm.OnlineChatModule('openai').prompt(prompter)
module(dict(context='背景', input='输入'))
```

### 和 TrainableModule 一起使用

当 ``Prompter`` 和 ``TrainableModule`` 一起使用时， ``TrainableModule.__call__`` 会调用 ``Prompter.generate_prompt`` ，并且将 ``__input``,
``history`` 和 ``tools`` 传给 ``generate_prompt`` ，此时 ``generate_prompt`` 的 ``return_dict`` 会被设置为 ``True``。下面给出一个例子：

```python
import lazyllm
instruction = '你是一个由LazyLLM开发的知识问答助手，你的任务是根据提供的上下文信息来回答用户的问题。上下文信息是{context}，用户的问题是{input}, 现在请你做出回答。'
prompter = lazyllm.AlpacaPrompter(instruction)
module = lazyllm.TrainableModule('internlm2-chat-7b').prompt(prompter)
module.start()
module(dict(context='背景', input='输入'))
```

!!! Note "注意"

    - 我们保证了 ``Prompter`` 在 ``TrainableModule`` 和 ``OnlineChatModule`` 具有一致的使用体验，您可以方便的更换模型以进行效果的尝试。
    - ``TrainableModule`` 需要手动调用 ``start`` 以启动服务，想了解更多关于 ``TrainableModule`` 的用法，可以参考 : [module][lazyllm.module.ModuleBase]

### LazyLLM 中内置的场景 Prompt

敬请期待。
