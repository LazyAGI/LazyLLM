## Prompter

To ensure a consistent user experience across different online models and various local models, as well as between fine-tuning and inference, we have defined the Prompter class. This class abstracts away the implementation details of different models and inference frameworks.
You can refer to the Prompter API documentation at [prompter][lazyllm.components.prompter.LazyLLMPrompterBase]. Next, I will introduce the design concept of the Prompter in LazyLLM step by step, starting with an example.

### Prompter Quick Start

Suppose we are designing a document question-answering application that requires the input to a large model to be the user's question along with the retrieved background knowledge. In this case, our prompt is designed as follows:

```text
你是一个由LazyLLM开发的知识问答助手，你的任务是根据提供的上下文信息来回答用户的问题。上下文信息是{上下文}，用户的问题是{问题}，现在请你做出回答。
```

[](){#prompter_try}

You can utilize the built-in Prompter provided by LazyLLM to achieve the required functionality. Here is an example:

```python
import lazyllm
instruction = '你是一个由LazyLLM开发的知识问答助手，你的任务是根据提供的上下文信息来回答用户的问题。上下文信息是{context}，用户的问题是{input}, 现在请你做出回答。'
prompter = lazyllm.AlpacaPrompter({"user": instruction})
module = lazyllm.OnlineChatModule('openai').prompt(prompter)
module(dict(context='背景', input='输入'))
```

The prompter generated above can be directly fed to a large model for use. We can test the effectiveness of the Prompter with the following code:

```python
import lazyllm
# 'You are an AI-Agent developed by LazyLLM.\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\n\n ### Instruction:\n你是一个由LazyLLM开发的知识问答助手，你的任务是根据提供的上下文信息来回答用户的问题。上下文信息是背景，用户的问题是输入，现在请你做出回答。\n\n\n### Response:\n'
prompter.generate_prompt(dict(context='背景', input='输入'))

# {'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\n\n ### Instruction:\n你是一个由LazyLLM开发的知识问答助手，你的任务是根据提供的上下文信息来回答用户的问题。上下文信息是背景，用户的问题是输入，现在请你做出回答。\n\n'}, {'role': 'user', 'content': ''}]}
prompter.generate_prompt(dict(context='背景', input='输入'), return_dict=True)
```

In the example above, the ``generate_prompt`` function accepts a ``dict`` as input, filling in the slots in the ``instruction`` template with the provided values.

!!! Note

    - In the code above, there's a parameter ``return_dict`` worth noting. When ``return_dict`` is set to True, it returns a dictionary in the OpenAI format for online models.
    - Typically, you just need to assign the Prompter to a ``TrainableModule`` or ``OnlineChatModule`` without worrying about the ``generate_prompt`` function.

### Design Concept of LazyLLM Prompter

#### Basic

**PromptTemplate**: A built-in template for each Prompter. Each subclass must implement its own PromptTemplate to select certain fields and define concatenation rules.

- The optional fields in PrompterTemplate include::
    - system: System prompt typically reads the model's ownership information and sets it accordingly. If not set, it defaults to: ``You are an AI-Agent developed by LazyLLM.`` .
    - instruction: Task instruction, obtained by concatenating the user's input with the ``InstructionTemplate``. This is an important field for application developers to understand. If the instruction is a string, it is considered a system instruction by default. If it is a dictionary, the keys can only be ``system`` and ``user``. ``system`` specifies system-level instructions, while ``user`` specifies user-level instructions.
    - history: Historical conversation, derived from user input, formatted as ``[[a, b], [c, d]]`` or ``[{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]``
    - tools: Tools that can be used, passed in when constructing the ``prompter`` or by the user when using it. Defining tools when constructing the ``prompter`` will prohibit users from passing them in again when using it. The format is  ``[{"type": "function",  "function": {"name": "", "description": "", "parameters": {},  "required": []}]``
    - user: User-level instructions, optional instructions specified by the user through instruction.
    - sos: ``start of system``, signifies the beginning of system prompts. This symbol is filled in by the model, and developers and users don't need to consider it.
    - eos: ``end of system``, signifies the end of system prompts. This symbol is filled in by the model, and developers and users don't need to consider it.
    - soh: ``start of human``, signifies the beginning of user input, often used as a separator in multi-turn dialogues. This symbol is filled in by the model, and developers and users don't need to consider it.
    - eoh: ``end of human``, signifies the end of user input, often used as a separator in multi-turn dialogues. This symbol is filled in by the model, and developers and users don't need to consider it.
    - soa: ``start of assistant``, signifies the beginning of the model's output, often used as a separator in multi-turn dialogues. This symbol is filled in by the model, and developers and users don't need to consider it.
    - eoa: ``end of assistant``, signifies the end of the model's output, often used as a separator in multi-turn dialogues. This symbol is filled in by the model, and developers and users don't need to consider it.
- The built-in Prompt concatenation rules used by ``TrainableModule`` are as follows::
    - AlpacaPrompter: ``{system}\n{instruction}\n{tools}\n{user}### Response:\n``
    - ChatPrompter: ``{sos}{system}{instruction}{tools}{eos}\n\n{history}\n{soh}\n{user}{input}\n{eoh}{soa}\n``
- The output format of ``OnlineChatModule`` is:``dict(messages=[{"role": "system", "content": ""}, {"role": "user", "content": ""}, ...], tools=[])``

!!! Note

    - From the above, we can see that ``AlpacaPrompter`` lacks the ``history`` field as well as the identifiers like ``soh`` necessary for multi-turn dialogue segmentation compared to ``ChatPrompter``.
    - Therefore, it is evident that ``AlpacaPrompter`` does not support multi-turn dialogue.

**InstructionTemplate**: Each Prompter has a built-in template used to combine user input with the ``instruction`` to generate the final ``instruction``. The two fields used in the ``InstructionTemplate`` are:

- ``instruction``:Provided by the developer when constructing the ``Prompter``, it may contain several placeholders to be filled with user input. Alternatively, it can specify system-level and user-level instructions. When specifying user-level instructions, a dictionary type must be used with keys ``user`` and ``system``.
- ``extra_keys`` : Additional information required when the user calls the model, provided by the developer during the construction of the Prompter. These will automatically be converted into placeholders within the ``instruction``.

!!! Note

    In the built-in ``Prompter``, the ``Alpaca's`` ``InstructionTemplate`` additionally includes the standard prompt in the ``alpaca`` format, which is: ``Below is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request``

#### Prompt Generation Process Analysis

Taking the example of a document question and answer task using ``AlpacaPrompter`` from: [Prompter Quick Start](#prompter_try), let's go through the prompt generation process in detail.

1. ``AlpacaPrompter`` combines the ``instruction`` (and ``extra_keys``, if any) provided during the construction of the ``prompter`` with the ``InstructionTemplate``. The ``instruction`` is set as:
```python
"Below is an instruction that describes a task, paired with extra messages such as input that provides "
"further context if possible. Write a response that appropriately completes the request.\\n\\n ### "
"Instruction:\\n 你是一个由LazyLLM开发的知识问答助手，你的任务是根据提供的上下文信息来回答用户的问题。上下文信息是{{context}}，"
"用户的问题是{{input}}, 现在请你做出回答。### Response:\\n}"
```

2. Given the user's input as ``dict(context='背景', input='问题')``
3. Concatenate the user's input with the instruction obtained in ’1‘ to get:
```python
"Below is an instruction that describes a task, paired with extra messages such as input that provides "
"further context if possible. Write a response that appropriately completes the request.\\n\\n ### "
"Instruction:\\n 你是一个由LazyLLM开发的知识问答助手，你的任务是根据提供的上下文信息来回答用户的问题。上下文信息是背景，"
"用户的问题是问题, 现在请你做出回答。### Response:\\n}"
```

4. ``AlpacaPrompter`` reads the system and ``tools`` fields, where the ``system`` field is set by the ``Module``, and the ``tools`` field will be introduced in the later section: [Use the tools](#use-the-tools).
[](){#analysis}
5. If the ``prompter`` result is used for the online model (``OnlineChatModule``), the ``PromptTemplate`` will not be concatenated further. Instead, a dict will be directly obtained, namely:
```python
"{'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\n\n ### Instruction:\nYou are a knowledge assistant developed by LazyLLM. Your task is to answer the user's question based on the provided context information. The context information is background, and the user's question is question. Please provide an answer.\n\n'}, {'role': 'user', 'content': ''}]}"
```

6. If the ``prompter`` result is used for the offline model (``TrainableModule``), the final result will be obtained through the PromptTemplate:
```python
"You are an AI-Agent developed by LazyLLM.\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\n\n ### Instruction:\nYou are a knowledge assistant developed by LazyLLM. Your task is to answer the user's question based on the provided context information. The context information is background, and the user's question is question. Please provide an answer.\n\n\n### Response:\n"
```

### Defining and Using a Prompter

#### Defining a New Prompter

Refer to the API documentation: [prompter][lazyllm.components.prompter.LazyLLMPrompterBase]

#### Query as a string, not a dict

In [Prompter Quick Start](#prompter_try), we demonstrated a basic usage and explained the working principle of the ``prompter`` in the following sections.
However, in most cases, the user's input is often a ``string``. This section demonstrates how to use the ``prompter`` when the input is a ``string``.

When the user's input is a string, we allow at most one slot in the ``Prompter``'s ``instruction``. Using the scenario of "large models doing arithmetic", we provide an example code:

```python
>>> p = lazyllm.AlpacaPrompter('请完成加法运算, 输入为{instruction}')
>>> p.generate_prompt('a+b')
'You are an AI-Agent developed by LazyLLM.\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\n\\n ### Instruction:\\n请完成加法运算, 输入为a+b\\n\\n\\n### Response:\\n'
>>>  p = lazyllm.AlpacaPrompter('请完成加法运算', extra_keys='input')
'You are an AI-Agent developed by LazyLLM.\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\n\\n ### Instruction:\\n请完成加法运算\\n\\nHere are some extra messages you can referred to:\\n\\n### input:\\na+b\\n\\n\\n### Response:\\n'
```

!!! Note

     When using ``AlpacaPrompter``, you need to define a unique slot that can be given any name, and the input of type ``string`` will be filled into it.

```python
>>> p = lazyllm.ChatPrompter('请完成加法运算，输入为{input}')
>> p.generate_prompt('a+b')
'<|start_system|>You are an AI-Agent developed by LazyLLM.请完成加法运算，输入为a+b\\n\\n<|end_system|>\\n\\n\\n<|Human|>:\\n\\n<|Assistant|>:\\n'
>>> p = lazyllm.ChatPrompter('请完成加法运算')
>> p.generate_prompt('a+b')
'<|start_system|>You are an AI-Agent developed by LazyLLM.请完成加法运算\n\n<|end_system|>\n\n\n<|Human|>:\na+b\n<|Assistant|>
```

!!! Note

    - When using ``ChatPrompter``, unlike ``AlpacaPrompter``, defining a slot in ``instruction`` is not mandatory.
    - If a slot is not defined, the input will be placed into the conversation as the user's input between ``<soh>`` and ``<eoh>``.
    - If a slot is defined, similar to ``AlpacaPrompter``, it can be given any name, and the input will be placed in the ``<system>`` field.
    - If ``instruction`` specifies both system-level and user-level instructions, after concatenation, the system-level instruction will be placed in the {instruction} position of the prompt template, and the user-level instruction will be placed in the {user} position.

#### Use the tools

In general, when using large models for ``function-call``, it is necessary to define a set of tools according to a specific format and pass them to the large model in a certain format. Tools can be passed in when constructing the ``prompter``, or they can be passed in by the user during usage. Once tools are defined when constructing the ``prompter``, it will prevent the user from passing them in again during usage.
The format of tools is generally as follows:

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

let's demonstrate how ``Prompter`` uses tools through a simple tool tools=[dict(type='function', function=dict(name='example'))].

1. Application Developer Defines the Tool

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

2. User Defines the Tool

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

The tool will be read after step 4 in [Prompt Generation Process Analysis](#analysis) once it's converted to JSON.

!!! Note

    If using an online model, the tool will become a field parallel to ``messages``, as shown in the example below:

        >>> import lazyllm
        >>> tools=[dict(type='function', function=dict(name='example'))]
         >>> prompter = lazyllm.AlpacaPrompter('你是一个工具调用的Agent，我会给你提供一些工具，请根据用户输入，帮我选择最合适的工具并使用', extra_keys='input', tools=tools)
        >>> prompter.generate_prompt('帮我查询一下今天的天气', return_dict=True)
        {'messages': [{'role': 'system', 'content': 'You are an AI-Agent developed by LazyLLM.\\nBelow is an instruction that describes a task, paired with extra messages such as input that provides further context if possible. Write a response that appropriately completes the request.\\n\\n ### Instruction:\\n你是一个工具调用的Agent，我会给你提供一些工具，请根据用户输入，帮我选择最合适的工具并使用\\n\\nHere are some extra messages you can referred to:\\n\\n### input:\\n帮我查询一下今天的天气\\n\\n'}, {'role': 'user', 'content': ''}],
        'tools': [{'type': 'function', 'function': {'name': 'example'}}]}

#### Using conversation history

If we want the model to have multi-turn conversation capabilities, we need to concatenate the conversation context with the ``prompt``. The context is provided by the user but needs to be passed in as key-value pairs. Here's an example:

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

The conversation history will be read in step 4 of [Prompt Generation Process Analysis](#analysis) after a simple format conversion.

!!! Note

    - Only ``ChatPrompter`` supports passing in conversation history.
    - When the input is in the format ``[[a, b], ...]``, both ``return_dict`` set to ``True`` or ``False`` are supported, whereas when the input is in the format ``[dict, dict]``, only ``return_dict`` set to ``True`` is supported.

#### Used with OnlineChatModule

When the ``Prompter`` is used with the ``OnlineChatModule``, ``OnlineChatModule.__call__`` will call ``Prompter.generate_prompt`` and ``pass __input``,
``history``, and ``tools`` to ``generate_prompt``. At this time, the ``return_dict`` of ``generate_prompt`` will be set to ``True``. Below is an example:

```python
import lazyllm
instruction = '你是一个由LazyLLM开发的知识问答助手，你的任务是根据提供的上下文信息来回答用户的问题。上下文信息是{context}，用户的问题是{input}, 现在请你做出回答。'
prompter = lazyllm.AlpacaPrompter(instruction)
module = lazyllm.OnlineChatModule('openai').prompt(prompter)
module(dict(context='背景', input='输入'))
```

#### Used with TrainableModule

When the ``Prompter`` is used with the ``TrainableModule``, ``TrainableModule.__call__`` will call ``Prompter.generate_prompt`` and ``pass __input``,
``history``, and ``tools`` to ``generate_prompt``. At this time, the ``return_dict`` of ``generate_prompt`` will be set to ``True``. Below is an example:

```python
import lazyllm
instruction = '你是一个由LazyLLM开发的知识问答助手，你的任务是根据提供的上下文信息来回答用户的问题。上下文信息是{context}，用户的问题是{input}, 现在请你做出回答。'
prompter = lazyllm.AlpacaPrompter(instruction)
module = lazyllm.TrainableModule('internlm2-chat-7b').prompt(prompter)
module.start()
module(dict(context='背景', input='输入'))
```

!!! Note

    - We have ensured that the ``Prompter`` has a consistent usage experience with both ``TrainableModule`` and ``OnlineChatModule``, allowing you to easily switch models for experimentation.
    - ``TrainableModule`` requires manually calling ``start`` to initiate the service. For more information on how to use ``TrainableModule``, refer to [module][lazyllm.module.ModuleBase].

#### Built-in Scenario Prompt in LazyLLM

Stay tuned.
