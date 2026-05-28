# flake8: noqa E501
import importlib
import functools
from .. import utils
add_chinese_doc = functools.partial(utils.add_chinese_doc, module=importlib.import_module('lazyllm.tools'))
add_english_doc = functools.partial(utils.add_english_doc, module=importlib.import_module('lazyllm.tools'))
add_example = functools.partial(utils.add_example, module=importlib.import_module('lazyllm.tools'))
add_agent_chinese_doc = functools.partial(utils.add_chinese_doc, module=importlib.import_module('lazyllm.tools.agent'))
add_agent_english_doc = functools.partial(utils.add_english_doc, module=importlib.import_module('lazyllm.tools.agent'))
add_agent_example = functools.partial(utils.add_example, module=importlib.import_module('lazyllm.tools.agent'))
add_toolsmgr_chinese_doc = functools.partial(utils.add_chinese_doc, module=importlib.import_module('lazyllm.tools.agent.toolsManager'))
add_toolsmgr_english_doc = functools.partial(utils.add_english_doc, module=importlib.import_module('lazyllm.tools.agent.toolsManager'))
add_toolsmgr_example = functools.partial(utils.add_example, module=importlib.import_module('lazyllm.tools.agent.toolsManager'))

add_chinese_doc('IntentClassifier', '''\
意图分类模块，用于根据输入文本在给定的意图列表中进行分类。
支持中英文自动选择提示模板，并可通过示例、提示、约束和注意事项增强分类效果。

Args:
    llm: 用于意图分类的大语言模型实例。
    intent_list (list): 可选，意图类别列表，例如 ["聊天", "天气", "问答"]。
    prompt (str): 可选，自定义提示语，插入到系统提示模板中。
    constrain (str): 可选，分类约束条件说明。
    attention (str): 可选，提示注意事项。
    examples (list[list[str, str]]): 可选，分类示例列表，每个元素为 [输入文本, 标签]。
    return_trace (bool): 是否返回执行过程的 trace，默认为 False。
''')

add_english_doc('IntentClassifier', '''\
Intent classification module that classifies input text into a given intent list.
Supports automatic selection of Chinese or English prompt templates, and allows enhancement through examples, prompt text, constraints, and attention notes.

Args:
    llm: The large language model instance used for intent classification.
    intent_list (list): Optional, list of intent categories, e.g., ["chat", "weather", "QA"].
    prompt (str): Optional, custom prompt inserted into the system prompt template.
    constrain (str): Optional, classification constraint description.
    attention (str): Optional, attention notes for classification.
    examples (list[list[str, str]]): Optional, classification examples, each element is [input text, label].
    return_trace (bool): Whether to return execution trace. Default is False.
''')


add_example(
    "IntentClassifier",
    """\
    >>> import lazyllm
    >>> from lazyllm.tools import IntentClassifier
    >>> classifier_llm = lazyllm.OnlineChatModule(source="openai")
    >>> chatflow_intent_list = ["Chat", "Financial Knowledge Q&A", "Employee Information Query", "Weather Query"]
    >>> classifier = IntentClassifier(classifier_llm, intent_list=chatflow_intent_list)
    >>> classifier.start()
    >>> print(classifier('What is the weather today'))
    Weather Query
    >>>
    >>> with IntentClassifier(classifier_llm) as ic:
    >>>     ic.case['Weather Query', lambda x: '38.5°C']
    >>>     ic.case['Chat', lambda x: 'permission denied']
    >>>     ic.case['Financial Knowledge Q&A', lambda x: 'Calling Financial RAG']
    >>>     ic.case['Employee Information Query', lambda x: 'Beijing']
    ...
    >>> ic.start()
    >>> print(ic('What is the weather today'))
    38.5°C
""",
)


add_chinese_doc('IntentClassifier.intent_promt_hook', '''\
意图分类的预处理 Hook。
将输入文本与意图列表打包为 JSON，并生成历史对话信息字符串。

Args:
    input (str | List | Dict | None): 输入文本，仅支持字符串类型。
    history (List): 历史对话记录，默认为空列表。
    tools (List[Dict] | None): 工具信息，可选。
    label (str | None): 标签，可选。

**Returns:**\n
- tuple: 输入数据字典, 历史记录列表, 工具信息, 标签
''')

add_english_doc('IntentClassifier.intent_promt_hook', '''\
Pre-processing hook for intent classification.
Packages the input text and intent list into JSON and generates a string of conversation history.

Args:
    input (str | List | Dict | None): The input text, only string type is supported.
    history (List): Conversation history, default empty list.
    tools (List[Dict] | None): Optional tool information.
    label (str | None): Optional label.

**Returns:**\n
- tuple: input data dict, history list, tools, label
''')

add_chinese_doc('IntentClassifier.post_process_result', '''\
意图分类结果的后处理。
如果结果在意图列表中则直接返回，否则返回意图列表的第一个元素。

Args:
    input (str): 分类模型输出结果。

**Returns:**\n
- str: 最终的分类标签。
''')

add_english_doc('IntentClassifier.post_process_result', '''\
Post-processing of intent classification result.
Returns the result directly if it is in the intent list, otherwise returns the first element of the intent list.

Args:
    input (str): Output result from the classification model.

**Returns:**\n
- str: The final classification label.
''')

# rag/document.py

add_chinese_doc('ToolManager', '''\
ToolManager是一个工具管理类，用于提供工具信息和工具调用给function call。

此管理类构造时需要传入工具列表。工具可以是以下几种形式：

- 工具名字符串（已通过 ``fc_register`` 注册的工具）
- 可调用对象（函数或 ``ModuleTool`` 实例）
- 带有 ``__public_apis__`` 的对象实例（自动包装为 ``InstanceToolGroup``）
- ``(instance, key_source)`` 元组（带凭据来源的实例工具组）
- ``dict``：直接定义一个工具组，格式为 ``dict(name='grp', desc='...', tools=[...], lazy=True)``，其中 ``name`` 和 ``tools`` 为必填字段。

工具组（``ToolGroup``）支持两种模式：

- **lazy 模式**（默认）：初始只向 LLM 暴露一个 ``get_<name>_methods`` 工具；LLM 调用该工具后，该组的子工具描述会动态注入 system prompt，LLM 在下一轮中再选择并调用具体工具。适合工具数量多、希望减少上下文长度的场景。
- **eager 模式**（``lazy=False``）：直接把所有子工具描述展开注入 system prompt，与旧行为一致。

工具组支持多级嵌套，子节点可以是普通工具或另一个工具组（通过嵌套 ``dict`` 定义）。

Args:
    tools (List): 工具列表，每个元素支持字符串、Callable、ModuleTool、带 ``__public_apis__`` 的实例、``(instance, key_source)`` 元组，或 ``dict`` 工具组。
    return_trace (bool): 是否返回中间步骤和工具调用信息。
    sandbox (LazyLLMSandboxBase | None): 沙箱实例。若提供，则当工具的 ``execute_in_sandbox`` 为 True 时，工具将在此沙箱中执行，并自动处理文件上传/下载。
''')

add_english_doc('ToolManager', '''\
ToolManager is a tool management class used to provide tool information and tool calls to function call.

When constructing this management class, you pass in a list of tools. Each element can be one of the following:

- A tool name string (a tool registered via ``fc_register``)
- A callable (plain function or ``ModuleTool`` instance)
- An object instance with ``__public_apis__`` (automatically wrapped as ``InstanceToolGroup``)
- A ``(instance, key_source)`` tuple (instance tool group with a runtime credential source)
- A ``dict``: defines a tool group inline, with the format ``dict(name='grp', desc='...', tools=[...], lazy=True)``. ``name`` and ``tools`` are required fields.

Tool groups (``ToolGroup``) support two modes:

- **lazy mode** (default): Initially only a ``get_<name>_methods`` gateway tool is exposed to the LLM. After the LLM calls it, the child tool descriptions are dynamically injected into the system prompt, and the LLM selects and calls the actual tool in the next turn. Suitable when there are many tools and you want to reduce context length.
- **eager mode** (``lazy=False``): All child tool descriptions are expanded and injected into the system prompt immediately, matching the previous behavior.

Tool groups support multi-level nesting; child nodes can be plain tools or another tool group (defined via a nested ``dict``).

Args:
    tools (List): Tool list. Each element can be a string, Callable, ModuleTool, an instance with ``__public_apis__``, a ``(instance, key_source)`` tuple, or a ``dict`` tool group.
    return_trace (bool): If True, return intermediate steps and tool calls.
    sandbox (LazyLLMSandboxBase | None): A sandbox instance. When provided, tools with ``execute_in_sandbox`` set to True will be executed inside this sandbox, with automatic file upload/download handling.

''')

add_example('ToolManager', """\
>>> from lazyllm.tools import ToolManager, fc_register
>>> import json
>>> from typing import Literal
>>> @fc_register("tool")
>>> def get_current_weather(location: str, unit: Literal["fahrenheit", "celsius"]="fahrenheit"):
...     '''
...     Get the current weather in a given location
...
...     Args:
...         location (str): The city and state, e.g. San Francisco, CA.
...         unit (str): The temperature unit to use. Infer this from the users location.
...     '''
...     if 'tokyo' in location.lower():
...         return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius'})
...     elif 'san francisco' in location.lower():
...         return json.dumps({'location': 'San Francisco', 'temperature': '72', 'unit': 'fahrenheit'})
...     elif 'paris' in location.lower():
...         return json.dumps({'location': 'Paris', 'temperature': '22', 'unit': 'celsius'})
...     elif 'beijing' in location.lower():
...         return json.dumps({'location': 'Beijing', 'temperature': '90', 'unit': 'fahrenheit'})
...     else:
...         return json.dumps({'location': location, 'temperature': 'unknown'})
...
>>> @fc_register("tool")
>>> def get_n_day_weather_forecast(location: str, num_days: int, unit: Literal["celsius", "fahrenheit"]='fahrenheit'):
...     '''
...     Get an N-day weather forecast
...
...     Args:
...         location (str): The city and state, e.g. San Francisco, CA.
...         num_days (int): The number of days to forecast.
...         unit (Literal['celsius', 'fahrenheit']): The temperature unit to use. Infer this from the users location.
...     '''
...     if 'tokyo' in location.lower():
...         return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius', "num_days": num_days})
...     elif 'san francisco' in location.lower():
...         return json.dumps({'location': 'San Francisco', 'temperature': '75', 'unit': 'fahrenheit', "num_days": num_days})
...     elif 'paris' in location.lower():
...         return json.dumps({'location': 'Paris', 'temperature': '25', 'unit': 'celsius', "num_days": num_days})
...     elif 'beijing' in location.lower():
...         return json.dumps({'location': 'Beijing', 'temperature': '85', 'unit': 'fahrenheit', "num_days": num_days})
...     else:
...         return json.dumps({'location': location, 'temperature': 'unknown'})
...
>>> tools = ["get_current_weather", "get_n_day_weather_forecast"]
>>> tm = ToolManager(tools)
>>> print(tm([{'name': 'get_n_day_weather_forecast', 'arguments': {'location': 'Beijing', 'num_days': 3}}])[0])
'{"location": "Beijing", "temperature": "85", "unit": "fahrenheit", "num_days": 3}'

>>> # Using dict to define a lazy tool group (reduces initial context length)
>>> def search_web(query: str) -> str:
...     '''Search the web.
...
...     Args:
...         query (str): Search query.
...
...     Returns:
...         str: Search results.
...     '''
...     return f'results for {query}'
...
>>> def search_news(query: str) -> str:
...     '''Search news articles.
...
...     Args:
...         query (str): News search query.
...
...     Returns:
...         str: News results.
...     '''
...     return f'news for {query}'
...
>>> tm2 = ToolManager([
...     dict(name='search', desc='Web and news search tools', tools=[search_web, search_news]),
... ])
>>> # Initially only the gateway tool is visible
>>> [d['function']['name'] for d in tm2.tools_description]
['get_search_methods']
""")

add_agent_chinese_doc('register', '''\
工具注册器，用于将函数注册为可供 FunctionCall/Agent 调用的工具。

Args:
    group (str): 工具分组，建议使用 'tool'。
    execute_in_sandbox (bool): 是否在沙箱中执行，默认 True；若不希望在沙箱执行，请设置为 False。
    input_files_parm (str): 指定函数中哪个参数包含输入文件路径，沙箱会在执行前上传这些文件。该参数指向的函数参数类型必须为 ``str`` 或 ``List[str]``。
    output_files_parm (str): 指定函数中哪个参数包含输出文件路径，沙箱执行完成后会下载这些文件。该参数指向的函数参数类型必须为 ``str`` 或 ``List[str]``。
    output_files (List[str]): 额外的输出文件路径列表，用于工具中硬编码的输出文件名（不通过函数参数传递），沙箱执行后也会下载这些文件。
''')

add_agent_english_doc('register', '''\
Tool registrar for registering functions as tools callable by FunctionCall/Agent.

Args:
    group (str): tool group, recommend using 'tool'.
    execute_in_sandbox (bool): whether to execute in sandbox, default True; set False to disable sandbox execution.
    input_files_parm (str): the name of the function parameter that holds input file paths; the sandbox uploads these files before execution. The parameter it points to must be of type ``str`` or ``List[str]``.
    output_files_parm (str): the name of the function parameter that holds output file paths; the sandbox downloads these files after execution. The parameter it points to must be of type ``str`` or ``List[str]``.
    output_files (List[str]): additional output file paths for the sandbox to download, for cases where output filenames are hardcoded in the tool rather than passed as parameters.
''')

add_agent_example('register', """\
>>> from lazyllm.tools import fc_register
>>> @fc_register("tool")
>>> def my_tool(text: str):
...     '''Simple tool.
...
...     Args:
...         text (str): input text.
...     '''
...     return text.upper()

>>> from typing import List, Optional
>>> @fc_register("tool", input_files_parm="input_paths", output_files_parm="output_paths")
>>> def file_tool(input_paths: Optional[List[str]] = None, output_paths: Optional[List[str]] = None):
...     '''Process files in sandbox.
...
...     Args:
...         input_paths (List[str] | None): input file paths.
...         output_paths (List[str] | None): output file paths.
...     '''
...     return "done"
""")

add_agent_chinese_doc('code_interpreter', '''\
内置代码解释工具，基于沙箱执行代码并返回结果。默认使用本地沙箱（DummySandbox），也可通过配置切换为远程沙箱（SandboxFusion）。

沙箱选择：
- config['sandbox_type'] == 'dummy'：使用 DummySandbox，仅支持 python。
- config['sandbox_type'] == 'sandbox_fusion'：使用 SandboxFusion，支持 python / bash。

环境变量：
- LAZYLLM_SANDBOX_TYPE: 设置为 "dummy" 或 "sandbox_fusion"。
- LAZYLLM_SANDBOX_FUSION_BASE_URL: 远程沙箱服务地址（仅 sandbox_fusion 模式需要）。

Args:
    code (str): 待执行的代码。
    language (str): 代码语言，默认 'python'。

**Returns:**\n
    dict 或 str：成功时为执行结果字典（包含 stdout/stderr/returncode 等字段）；失败时为错误信息字符串。
''')

add_agent_english_doc('code_interpreter', '''\
Built-in code interpreter tool that executes code inside a sandbox and returns the result.
It uses DummySandbox by default, and can be switched to SandboxFusion via configuration.

Sandbox selection:
- config['sandbox_type'] == 'dummy': DummySandbox, python only.
- config['sandbox_type'] == 'sandbox_fusion': SandboxFusion, python / bash.

Environment variables:
- LAZYLLM_SANDBOX_TYPE: set to "dummy" or "sandbox_fusion".
- LAZYLLM_SANDBOX_FUSION_BASE_URL: remote sandbox base URL (sandbox_fusion only).

Args:
    code (str): code to execute.
    language (str): code language, default 'python'.

**Returns:**\n
    dict or str: a result dict on success (stdout/stderr/returncode, etc.); error message string on failure.
''')

add_agent_example('code_interpreter', """\
>>> from lazyllm.tools.agent import code_interpreter
>>> result = code_interpreter("print('hello')")
>>> print(result['stdout'].strip())
hello
""")

add_chinese_doc('ModuleTool', '''\
用于构建工具模块的基类。

该类封装了函数签名和文档字符串的自动解析逻辑，可生成标准化的参数模式（基于 pydantic），并对输入进行校验和工具调用的标准封装。

`__init__(self, verbose=False, return_trace=True, execute_in_sandbox=True)`
初始化工具模块。

Args:
    verbose (bool): 是否在执行过程中输出详细日志。
    return_trace (bool): 是否在结果中保留中间执行痕迹。
    execute_in_sandbox (bool): 是否在沙箱中执行，默认 True。当 ToolManager 配置了沙箱且此值为 True 时，工具将在沙箱中执行。
''')

add_english_doc('ModuleTool', '''\
Base class for defining tools using callable Python functions.

This class automatically parses function signatures and docstrings to build a parameter schema using `pydantic`. It also performs input validation and handles standardized tool execution.

`__init__(self, verbose=False, return_trace=True, execute_in_sandbox=True)`
Initializes a tool wrapper module.

Args:
    verbose (bool): Whether to print verbose logs during execution.
    return_trace (bool): Whether to keep intermediate execution trace in the result.
    execute_in_sandbox (bool): Whether to execute in sandbox, default True. When ToolManager has a sandbox configured and this is True, the tool will be executed inside the sandbox.
''')

add_example('ModuleTool', """
>>> from lazyllm.components import ModuleTool
>>> class AddTool(ModuleTool):
...     def apply(self, a: int, b: int) -> int:
...         '''Add two integers.
...
...         Args:
...             a (int): First number.
...             b (int): Second number.
...
...         Returns:
...             int: The sum of a and b.
...         '''
...         return a + b
>>> tool = AddTool()
>>> result = tool({'a': 3, 'b': 5})
>>> print(result)
8
""")

add_chinese_doc("ModuleTool.apply", '''
工具函数的具体实现方法。

这是一个抽象方法，需要在子类中具体实现工具的核心功能。

Args:
    *args (Any): 位置参数
    **kwargs (Any): 关键字参数

**Returns:**\n
- 工具执行的结果

**Raises:**\n
    NotImplementedError: 如果未在子类中重写该方法。
''')

add_english_doc("ModuleTool.apply", '''
Concrete implementation method of the tool function.

This is an abstract method that needs to be implemented in subclasses to provide the core functionality of the tool.

Args:
    *args (Any): Positional arguments
    **kwargs (Any): Keyword arguments

**Returns:**\n
- Result of tool execution

**Raises:**\n
    NotImplementedError: If the method is not overridden in a subclass.
''')

add_chinese_doc("ModuleTool.validate_parameters", '''
验证参数是否满足所需条件。

此方法会检查参数字典是否包含所有必须字段，并尝试进一步进行格式验证。

Args:
    arguments (Dict[str, Any]): 传入的参数字典。

**Returns:**\n
- bool: 若参数合法且完整，返回 True；否则返回 False。
''')

add_english_doc("ModuleTool.validate_parameters", '''
Validate whether the provided arguments meet the required criteria.

This method checks if all required keys are present in the input dictionary and attempts format validation.

Args:
    arguments (Dict[str, Any]): Dictionary of input arguments.

**Returns:**\n
- bool: True if valid and complete; False otherwise.
''')

add_chinese_doc("ModuleTool.to_sandbox_code", '''
生成用于在沙箱中执行的代码字符串。

该方法会序列化当前工具与传入参数，返回一段可在沙箱环境中反序列化并执行的 Python 代码。

Args:
    tool_arguments (Dict[str, Any]): 以字典形式提供的工具参数。

**Returns:**\n
- str: 可在沙箱中执行的 Python 代码字符串。
''')

add_english_doc("ModuleTool.to_sandbox_code", '''
Generate a sandbox-executable code string.

This method serializes the tool instance and arguments, and returns a Python code snippet
that can be deserialized and executed inside a sandbox environment.

Args:
    tool_arguments (Dict[str, Any]): Tool arguments as a dict.

**Returns:**\n
- str: A Python code string executable in a sandbox environment.
''')

add_chinese_doc('FunctionCall', '''\
FunctionCall是单轮工具调用类。当LLM自身信息不足以回答用户问题，需要结合外部工具获取辅助信息时，调用此类。
若LLM输出需要调用工具，则执行工具调用并返回调用结果；输出结果为List类型，包含当前轮的输入、模型输出和工具输出。
若不需工具调用，则直接返回LLM输出结果，输出为字符串类型。

Args:
    llm (ModuleBase): 使用的LLM实例，支持TrainableModule或OnlineChatModule。
    tools (List[Union[str, Callable]]): LLM可调用的工具名称或Callable对象列表。
    return_trace (Optional[bool]): 是否返回调用轨迹，默认为False。
    stream (Optional[bool]): 是否启用流式输出，默认为False。
    _prompt (Optional[str]): 自定义工具调用提示语，默认根据llm类型自动设置。

注意：tools中的工具需包含`__doc__`字段，且须遵循[Google Python Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)规范说明用途与参数。
''')

add_english_doc('FunctionCall', '''\
FunctionCall is a single-turn tool invocation class. It is used when the LLM alone cannot answer user queries and requires external knowledge through tool calls.
If the LLM output requires tool calls, the tools are invoked and the combined results (input, model output, tool output) are returned as a list.
If no tool calls are needed, the LLM output is returned directly as a string.

Args:
    llm (ModuleBase): The LLM instance to use, which can be either a TrainableModule or OnlineChatModule.
    tools (List[Union[str, Callable]]): A list of tool names or callable objects that the LLM can use.
    return_trace (Optional[bool]): Whether to return the invocation trace, defaults to False.
    stream (Optional[bool]): Whether to enable streaming output, defaults to False.
    _prompt (Optional[str]): Custom prompt for function call, defaults to automatic selection based on llm type.

Note: Tools in `tools` must include a `__doc__` attribute and describe their purpose and parameters according to the [Google Python Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
''')

add_example('FunctionCall', """\
>>> import lazyllm
>>> from lazyllm.tools import fc_register, FunctionCall
>>> import json
>>> from typing import Literal
>>> @fc_register("tool")
>>> def get_current_weather(location: str, unit: Literal["fahrenheit", "celsius"] = 'fahrenheit'):
...     '''
...     Get the current weather in a given location
...
...     Args:
...         location (str): The city and state, e.g. San Francisco, CA.
...         unit (str): The temperature unit to use. Infer this from the users location.
...     '''
...     if 'tokyo' in location.lower():
...         return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius'})
...     elif 'san francisco' in location.lower():
...         return json.dumps({'location': 'San Francisco', 'temperature': '72', 'unit': 'fahrenheit'})
...     elif 'paris' in location.lower():
...         return json.dumps({'location': 'Paris', 'temperature': '22', 'unit': 'celsius'})
...     else:
...         return json.dumps({'location': location, 'temperature': 'unknown'})
...
>>> @fc_register("tool")
>>> def get_n_day_weather_forecast(location: str, num_days: int, unit: Literal["celsius", "fahrenheit"] = 'fahrenheit'):
...     '''
...     Get an N-day weather forecast
...
...     Args:
...         location (str): The city and state, e.g. San Francisco, CA.
...         num_days (int): The number of days to forecast.
...         unit (Literal['celsius', 'fahrenheit']): The temperature unit to use. Infer this from the users location.
...     '''
...     if 'tokyo' in location.lower():
...         return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius', "num_days": num_days})
...     elif 'san francisco' in location.lower():
...         return json.dumps({'location': 'San Francisco', 'temperature': '72', 'unit': 'fahrenheit', "num_days": num_days})
...     elif 'paris' in location.lower():
...         return json.dumps({'location': 'Paris', 'temperature': '22', 'unit': 'celsius', "num_days": num_days})
...     else:
...         return json.dumps({'location': location, 'temperature': 'unknown'})
...
>>> tools=["get_current_weather", "get_n_day_weather_forecast"]
>>> llm = lazyllm.TrainableModule("internlm2-chat-20b").start()  # or llm = lazyllm.OnlineChatModule("openai", stream=False)
>>> query = "What's the weather like today in celsius in Tokyo."
>>> fc = FunctionCall(llm, tools)
>>> ret = fc(query)
>>> print(ret)
["What's the weather like today in celsius in Tokyo.", {'role': 'assistant', 'content': '
', 'tool_calls': [{'id': 'da19cddac0584869879deb1315356d2a', 'type': 'function', 'function': {'name': 'get_current_weather', 'arguments': {'location': 'Tokyo', 'unit': 'celsius'}}}]}, [{'role': 'tool', 'content': '{"location": "Tokyo", "temperature": "10", "unit": "celsius"}', 'tool_call_id': 'da19cddac0584869879deb1315356d2a', 'name': 'get_current_weather'}]]
>>> query = "Hello"
>>> ret = fc(query)
>>> print(ret)
'Hello! How can I assist you today?'
""")

add_chinese_doc('FunctionCallAgent', '''\
(FunctionCallAgent 已被废弃，将在未来版本中移除。请使用 ReactAgent 代替。) FunctionCallAgent是一个使用工具调用方式进行完整工具调用的代理，即回答用户问题时，LLM如果需要通过工具获取外部知识，就会调用工具，并将工具的返回结果反馈给LLM，最后由LLM进行汇总输出。

Args:
    llm (ModuleBase): 要使用的LLM，可以是TrainableModule或OnlineChatModule。
    tools (List[str]): LLM 使用的工具名称列表。
    max_retries (int): 工具调用迭代的最大次数。默认值为5。
    return_trace (bool): 是否返回执行追踪信息，默认为False。
    stream (bool): 是否启用流式输出，默认为False。
    return_last_tool_calls (bool): 若为True，在模型结束且存在工具调用记录时返回最后一次的工具调用轨迹。
    skills (bool | str | List[str]): Skills 配置。True 启用 Skills 并自动筛选；传入 str/list 启用指定技能。
    desc (str): Agent 能力描述，可为空。
    workspace (str): Agent 默认工作目录，默认是 `config['home']/agent_workspace`。
''')

add_english_doc('FunctionCallAgent', '''\
(FunctionCallAgent is deprecated and will be removed in a future version. Please use ReactAgent instead.) FunctionCallAgent is an agent that uses the tool calling method to perform complete tool calls. That is, when answering user questions, if LLM needs to obtain external knowledge through the tool, it will call the tool and feed back the return results of the tool to LLM, which will finally summarize and output them.

Args:
    llm (ModuleBase): The LLM to be used can be either TrainableModule or OnlineChatModule.
    tools (List[str]): A list of tool names for LLM to use.
    max_retries (int): The maximum number of tool call iterations. The default value is 5.
    return_trace (bool): Whether to return execution trace information, defaults to False.
    stream (bool): Whether to enable streaming output, defaults to False.
    return_last_tool_calls (bool): If True, return the last tool-call trace when the model finishes.
    skills (bool | str | List[str]): Skills config. True enables Skills with auto selection; pass a str/list to enable specific skills.
    desc (str): Optional agent capability description.
    workspace (str): Default agent workspace path. Defaults to `config['home']/agent_workspace`.
''')

add_example('FunctionCallAgent', """\
>>> import lazyllm
>>> from lazyllm.tools import fc_register, FunctionCallAgent
>>> import json
>>> from typing import Literal
>>> @fc_register("tool")
>>> def get_current_weather(location: str, unit: Literal["fahrenheit", "celsius"]='fahrenheit'):
...     '''
...     Get the current weather in a given location
...
...     Args:
...         location (str): The city and state, e.g. San Francisco, CA.
...         unit (str): The temperature unit to use. Infer this from the users location.
...     '''
...     if 'tokyo' in location.lower():
...         return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius'})
...     elif 'san francisco' in location.lower():
...         return json.dumps({'location': 'San Francisco', 'temperature': '72', 'unit': 'fahrenheit'})
...     elif 'paris' in location.lower():
...         return json.dumps({'location': 'Paris', 'temperature': '22', 'unit': 'celsius'})
...     elif 'beijing' in location.lower():
...         return json.dumps({'location': 'Beijing', 'temperature': '90', 'unit': 'Fahrenheit'})
...     else:
...         return json.dumps({'location': location, 'temperature': 'unknown'})
...
>>> @fc_register("tool")
>>> def get_n_day_weather_forecast(location: str, num_days: int, unit: Literal["celsius", "fahrenheit"]='fahrenheit'):
...     '''
...     Get an N-day weather forecast
...
...     Args:
...         location (str): The city and state, e.g. San Francisco, CA.
...         num_days (int): The number of days to forecast.
...         unit (Literal['celsius', 'fahrenheit']): The temperature unit to use. Infer this from the users location.
...     '''
...     if 'tokyo' in location.lower():
...         return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius', "num_days": num_days})
...     elif 'san francisco' in location.lower():
...         return json.dumps({'location': 'San Francisco', 'temperature': '75', 'unit': 'fahrenheit', "num_days": num_days})
...     elif 'paris' in location.lower():
...         return json.dumps({'location': 'Paris', 'temperature': '25', 'unit': 'celsius', "num_days": num_days})
...     elif 'beijing' in location.lower():
...         return json.dumps({'location': 'Beijing', 'temperature': '85', 'unit': 'fahrenheit', "num_days": num_days})
...     else:
...         return json.dumps({'location': location, 'temperature': 'unknown'})
...
>>> tools = ['get_current_weather', 'get_n_day_weather_forecast']
>>> llm = lazyllm.TrainableModule("internlm2-chat-20b").start()  # or llm = lazyllm.OnlineChatModule(source="sensenova")
>>> agent = FunctionCallAgent(llm, tools)
>>> query = "What's the weather like today in celsius in Tokyo and Paris."
>>> res = agent(query)
>>> print(res)
'The current weather in Tokyo is 10 degrees Celsius, and in Paris, it is 22 degrees Celsius.'
>>> query = "Hello"
>>> res = agent(query)
>>> print(res)
'Hello! How can I assist you today?'
""")

add_chinese_doc('LazyLLMAgentBase', '''\
LazyLLMAgentBase 是所有内置 Agent 的公共基类，负责统一的工具管理、技能启用、提示词注入与执行流程封装。

Args:
    llm: 大语言模型实例。
    tools (List[str]): 工具名称列表。
    max_retries (int): 工具调用最大迭代次数，默认 5。
    return_trace (bool): 是否返回中间执行轨迹。
    stream (bool): 是否启用流式输出。
    return_last_tool_calls (bool): 若为True，在模型结束且存在工具调用记录时返回最后一次的工具调用轨迹。
    skills (bool | str | List[str]): Skills 配置。True 启用 Skills 并自动筛选；传入 str/list 启用指定技能。
    memory: 预留的记忆/上下文对象。
    desc (str): Agent 能力描述。
    workspace (str): Agent 默认工作目录，默认是 `config['home']/agent_workspace`。
''')

add_english_doc('LazyLLMAgentBase', '''\
LazyLLMAgentBase is the common base class for built-in agents. It unifies tool management, skills enablement,
system-prompt injection, and execution flow.

Args:
    llm: Large language model instance.
    tools (List[str]): List of tool names.
    max_retries (int): Maximum tool-call iterations. Default is 5.
    return_trace (bool): Whether to return execution traces.
    stream (bool): Whether to enable streaming output.
    return_last_tool_calls (bool): If True, return the last tool-call trace when the model finishes.
    skills (bool | str | List[str]): Skills config. True enables Skills with auto selection; pass a str/list to enable specific skills.
    memory: Reserved memory/context object.
    desc (str): Optional agent capability description.
    workspace (str): Default agent workspace path. Defaults to `config['home']/agent_workspace`.
''')

add_chinese_doc('SkillManager', '''\
SkillManager 用于发现、加载与管理 Skills。

Args:
    dir (str, optional): Skills 目录路径，支持逗号分隔的多个路径。
    skills (Iterable[str], optional): 期望使用的技能名称列表。
    max_skill_md_bytes (int, optional): 单个 SKILL.md 最大读取大小。
    llm: 预留参数，目前不强制使用。
''')

add_english_doc('SkillManager', '''\
SkillManager discovers, loads, and manages Skills.

Args:
    dir (str, optional): Skills directory paths, comma-separated is supported.
    skills (Iterable[str], optional): Expected skill name list.
    max_skill_md_bytes (int, optional): Maximum SKILL.md size to load.
    llm: Reserved parameter, not required currently.
''')

add_chinese_doc('SkillManager.list_skill', '''\
列出当前 skills 目录中的可用技能，返回 Markdown 字符串。

**Returns:**\n
- str: 技能列表（名称/描述/路径）。
''')

add_english_doc('SkillManager.list_skill', '''\
List available skills under configured directories and return a Markdown string.

**Returns:**\n
- str: Skill list with name/description/path.
''')

add_chinese_doc('SkillManager.build_prompt', '''\
构建 Skills 引导提示词。

**Returns:**\n
- str: 拼接后的系统提示词。
''')

add_english_doc('SkillManager.build_prompt', '''\
Build a skills guide prompt.

**Returns:**\n
- str: Composed system prompt.
''')

add_chinese_doc('SkillManager.get_skill', '''\
读取指定技能的 SKILL.md 全量内容。

Args:
    name (str): 技能名称。
    allow_large (bool): 是否允许读取超过大小限制的文件。

**Returns:**\n
- dict: 包含状态、路径与内容的结果。
''')

add_english_doc('SkillManager.get_skill', '''\
Load the full SKILL.md content for a skill.

Args:
    name (str): Skill name.
    allow_large (bool): Whether to allow loading oversized files.

**Returns:**\n
- dict: Result with status, path, and content.
''')

add_chinese_doc('SkillManager.read_file', '''\
读取技能目录下指定相对路径文件内容。

Args:
    name (str): 技能名称。
    rel_path (str): 相对路径。

**Returns:**\n
- dict: 读取结果。
''')

add_english_doc('SkillManager.read_file', '''\
Read a file under a skill directory by relative path.

Args:
    name (str): Skill name.
    rel_path (str): Relative path.

**Returns:**\n
- dict: Read result.
''')

add_chinese_doc('SkillManager.read_reference', '''\
读取技能参考文件内容（别名封装）。

Args:
    name (str): 技能名称。
    rel_path (str): 相对路径。

**Returns:**\n
- dict: 读取结果。
''')

add_english_doc('SkillManager.read_reference', '''\
Read a reference file in a skill directory (alias wrapper).

Args:
    name (str): Skill name.
    rel_path (str): Relative path.

**Returns:**\n
- dict: Read result.
''')

add_chinese_doc('SkillManager.run_script', '''\
执行技能目录下的脚本文件。

Args:
    name (str): 技能名称。
    rel_path (str): 脚本相对路径。
    args (List[str], optional): 脚本参数。
    allow_unsafe (bool): 是否允许执行潜在风险脚本。
    cwd (str, optional): 工作目录。

**Returns:**\n
- dict: 执行结果。
''')

add_english_doc('SkillManager.run_script', '''\
Run a script under a skill directory.

Args:
    name (str): Skill name.
    rel_path (str): Script relative path.
    args (List[str], optional): Script arguments.
    allow_unsafe (bool): Whether to allow potentially unsafe execution.
    cwd (str, optional): Working directory.

**Returns:**\n
- dict: Execution result.
''')


add_chinese_doc('SkillManager.get_skill_tools', '''\
返回 Skills 工具列表（可调用对象）。

**Returns:**\n
- List[Callable]: Skills 工具列表。
''')

add_english_doc('SkillManager.get_skill_tools', '''\
Return the skill tool callables exposed by SkillManager.

**Returns:**\n
- List[Callable]: Skill tool callables.
''')

add_chinese_doc('LazyLLMAgentBase.build_agent', '''\
构建内部执行流程的工厂方法。

说明：
    该方法由子类实现，用于构建该 Agent 的内部工作流。
    基类会在首次执行时调用它完成初始化。
''')

add_english_doc('LazyLLMAgentBase.build_agent', '''\
Factory method for constructing the internal execution workflow.

Notes:
    This method should be implemented by subclasses to build the agent workflow.
    The base class invokes it lazily on first use.
''')

add_chinese_doc('ReactAgent', '''\
ReactAgent是按照 `Thought->Action->Observation->Thought...->Finish` 的流程一步一步的通过LLM和工具调用来显示解决用户问题的步骤，以及最后给用户的答案。

Args:
    llm: 大语言模型实例，用于生成推理和工具调用决策
    tools (List): 可用工具列表，每个元素支持以下几种形式：

        - ``str``：已注册工具的名称，如 ``"multiply_tool"``。
        - ``Callable``：直接传入函数，会被临时注册为工具。
        - ``ModuleTool`` 实例：直接使用已构造好的工具对象。
        - 带有 ``__public_apis__`` 的对象实例：直接传入，框架自动将 ``__public_apis__`` 中的每个方法展开为独立工具。
          若该类定义了 ``__key_source__`` 字段，则自动用作凭据来源；否则工具始终可用。
        - ``(instance, key_source)`` 元组：将带有 ``__public_apis__`` 的对象实例注册为一组工具，并绑定运行时凭据来源。
          ``instance`` 需声明 ``__public_apis__: List[str]``，其中每个方法名都会被展开为一个独立工具。
          ``key_source`` 支持字符串（``'env.XXX'``、``'config.xxx'``、``'globals.config.xxx'``，无 ``.`` 时等价于 ``globals.config.xxx``）、
          callable（如 ``lambda inst: inst._key``）或上述类型的列表（任一满足即可）；
          凭据不存在时，该实例的所有工具均从 LLM 可见列表中隐藏，LLM 不会尝试调用它们。
        - ``dict``：直接定义一个工具组，格式为 ``dict(name='grp', desc='描述', tools=[...], lazy=True)``。
          ``name`` 和 ``tools`` 为必填字段；``lazy=True``（默认）时初始只暴露 ``get_<name>_methods`` gateway 工具，
          LLM 调用后子工具描述动态注入 system prompt；``lazy=False`` 时直接展开所有子工具。
          支持多级嵌套，子 ``tools`` 列表中可以再嵌套 ``dict`` 工具组。

    max_retries (int): 工具调用循环的最大轮次，超出后若 `force_summarize=True` 则触发强制总结，否则抛出异常，默认为5
    return_trace (bool): 是否返回完整的执行轨迹，用于调试和分析，默认为False
    prompt (str): 自定义提示词模板，如果为None则使用内置模板
    stream (bool): 是否启用流式输出，用于实时显示生成过程，默认为False
    return_last_tool_calls (bool): 若为True，在模型结束且存在工具调用记录时返回最后一次的工具调用轨迹。
    skills (bool | str | List[str]): Skills 配置。True 启用 Skills 并自动筛选；传入 str/list 启用指定技能。
    desc (str): Agent 能力描述，可为空。
    workspace (str): Agent 默认工作目录，默认是 `config['home']/agent_workspace`。
    force_summarize (bool): 是否在达到 max_retries 仍未输出最终答案时，强制追加一次 LLM 调用以获取总结输出。
        为 True 时触发强制总结；为 False（默认）时直接抛出 ValueError。
    force_summarize_context (str): 强制总结时注入的额外上下文（如原始任务描述），默认为空字符串。
    keep_full_turns (int): 保留最近 N 轮完整工具结果不截断，其余旧结果压缩至 200 字符，默认为 0（全部压缩）。
''')

add_english_doc('ReactAgent', '''\
ReactAgent follows the `Thought->Action->Observation->Thought...->Finish` loop to solve user tasks step by step through LLM reasoning and tool calls, then delivers a final answer.

Args:
    llm: The large language model instance used for reasoning and tool-call decisions.
    tools (List): List of available tools. Each element can be one of the following:

        - ``str``: Name of a registered tool, e.g. ``"multiply_tool"``.
        - ``Callable``: A plain function passed directly; it is temporarily registered as a tool.
        - ``ModuleTool`` instance: A pre-constructed tool object used as-is.
        - Object instance with ``__public_apis__``: Passed directly; each method in ``__public_apis__`` is expanded into a separate tool automatically.
          If the class defines a ``__key_source__`` attribute, it is used as the credential source; otherwise the tools are always available.
        - ``(instance, key_source)`` tuple: Registers an object instance that declares ``__public_apis__: List[str]`` as a group of tools, with a runtime credential source attached.
          Each method listed in ``__public_apis__`` is expanded into a separate tool.
          ``key_source`` accepts a string (``'env.XXX'``, ``'config.xxx'``, ``'globals.config.xxx'``; no dot means ``globals.config.xxx``),
          a callable (e.g. ``lambda inst: inst._key``), or a list of the above (available if any resolves to a non-empty value).
          When the credential is absent, all tools from that instance are hidden from the LLM's visible tool list and will not be called.
        - ``dict``: Defines a tool group inline, with the format ``dict(name='grp', desc='...', tools=[...], lazy=True)``.
          ``name`` and ``tools`` are required fields. When ``lazy=True`` (default), only a ``get_<name>_methods`` gateway tool is initially
          exposed; after the LLM calls it, child tool descriptions are dynamically injected into the system prompt. When ``lazy=False``,
          all child tools are expanded immediately. Multi-level nesting is supported by embedding ``dict`` groups inside ``tools``.

    max_retries (int): Maximum number of tool-call loop iterations. When exceeded, the force-summarize fallback is triggered (if enabled) or an exception is raised. Defaults to 5.
    return_trace (bool): Whether to return the full execution trace for debugging and analysis. Defaults to False.
    prompt (str): Custom prompt template. If None, the built-in ReAct instruction template is used.
    stream (bool): Whether to enable streaming output for real-time display. Defaults to False.
    return_last_tool_calls (bool): If True, returns the last tool-call trace when the model finishes with pending tool calls.
    skills (bool | str | List[str]): Skills configuration. True enables Skills with automatic selection; a str/list enables the specified skills.
    desc (str): Description of the agent capabilities. Can be empty.
    workspace (str): Default working directory for the agent. Defaults to `config['home']/agent_workspace`.
    force_summarize (bool): When True, if the agent has not produced a final answer after max_retries iterations, one additional LLM call is made with the full conversation history plus a force-summarize instruction, asking the model to stop tool calls and output its final answer immediately. If False (default), a ValueError is raised instead.
        Useful when the task involves many tool-call steps and the LLM struggles to stop on its own.
    force_summarize_context (str): Extra context injected into the force-summarize prompt (e.g. the original task description). Defaults to empty string.
    keep_full_turns (int): Number of most-recent tool results to keep intact during history compaction. Older results are truncated to 200 chars. Defaults to 0 (all results compacted).

''')

add_chinese_doc('ReactAgent.build_agent', '''\
构建 ReactAgent 的内部推理与工具调用闭环。
''')

add_english_doc('ReactAgent.build_agent', '''\
Build the internal reasoning and tool-calling loop for ReactAgent.
''')

add_example('ReactAgent', """\
>>> import lazyllm
>>> from lazyllm.tools import fc_register, ReactAgent
>>> @fc_register("tool")
>>> def multiply_tool(a: int, b: int) -> int:
...     '''
...     Multiply two integers and return the result integer
...
...     Args:
...         a (int): multiplier
...         b (int): multiplier
...     '''
...     return a * b
...
>>> @fc_register("tool")
>>> def add_tool(a: int, b: int):
...     '''
...     Add two integers and returns the result integer
...
...     Args:
...         a (int): addend
...         b (int): addend
...     '''
...     return a + b
...
>>> tools = ["multiply_tool", "add_tool"]
>>> llm = lazyllm.TrainableModule("internlm2-chat-20b").start()   # or llm = lazyllm.OnlineChatModule(source="sensenova")
>>> agent = ReactAgent(llm, tools)
>>> query = "What is 20+(2*4)? Calculate step by step."
>>> res = agent(query)
>>> print(res)
'Answer: The result of 20+(2*4) is 28.'

>>> # Using (instance, key_source) tuple to register a class with __public_apis__
>>> class BingSearch:
...     __public_apis__ = ["search"]
...     def __init__(self, subscription_key: str = ""):
...         self._key = subscription_key
...     def search(self, query: str) -> str:
...         '''Search the web using Bing.
...
...         Args:
...             query (str): The search query string.
...
...         Returns:
...             str: Search results.
...         '''
...         return f"bing results for: {query}"
...
>>> bing = BingSearch()
>>> agent2 = ReactAgent(llm, tools=[(bing, 'globals.bing_key')])
>>> # Without a key, the tool is hidden from the LLM
>>> len(agent2._tools_manager.tools_description)
0
>>> lazyllm.globals['bing_key'] = 'my-subscription-key'
>>> # Now the tool becomes visible
>>> len(agent2._tools_manager.tools_description)
1
>>> res2 = agent2("Search for the latest LazyLLM release.")
""")

add_chinese_doc('PlanAndSolveAgent', '''\
PlanAndSolveAgent由两个组件组成，首先，由planner将整个任务分解为更小的子任务，然后由solver根据计划执行这些子任务，其中可能会涉及到工具调用，最后将答案返回给用户。

Args:
    llm (ModuleBase): 要使用的LLM，可以是TrainableModule或OnlineChatModule。和plan_llm、solve_llm互斥，要么设置llm(planner和solver公用一个LLM)，要么设置plan_llm和solve_llm，或者只指定llm(用来设置planner)和solve_llm，其它情况均认为是无效的。
    tools (List[str]): LLM使用的工具名称列表。
    plan_llm (ModuleBase): planner要使用的LLM，可以是TrainableModule或OnlineChatModule。
    solve_llm (ModuleBase): solver要使用的LLM，可以是TrainableModule或OnlineChatModule。
    max_retries (int): 工具调用迭代的最大次数。默认值为5。
    return_trace (bool): 是否返回中间步骤和工具调用信息。
    stream (bool): 是否以流式方式输出规划和解决过程。
    return_last_tool_calls (bool): 若为True，在模型结束且存在工具调用记录时返回最后一次的工具调用轨迹。
    skills (bool | str | List[str]): Skills 配置。True 启用 Skills 并自动筛选；传入 str/list 启用指定技能。
    desc (str): Agent 能力描述，可为空。
    workspace (str): Agent 默认工作目录，默认是 `config['home']/agent_workspace`。
''')

add_english_doc('PlanAndSolveAgent', '''\
PlanAndSolveAgent consists of two components. First, the planner breaks down the entire task into smaller subtasks, then the solver executes these subtasks according to the plan, which may involve tool calls, and finally returns the answer to the user.

Args:
    llm (ModuleBase): The LLM to be used can be TrainableModule or OnlineChatModule. It is mutually exclusive with plan_llm and solve_llm. Either set llm(the planner and solver share the same LLM), or set plan_llm and solve_llm,or only specify llm(to set the planner) and solve_llm. Other cases are considered invalid.
    tools (List[str]): A list of tool names for LLM to use.
    plan_llm (ModuleBase): The LLM to be used by the planner, which can be either TrainableModule or OnlineChatModule.
    solve_llm (ModuleBase): The LLM to be used by the solver, which can be either TrainableModule or OnlineChatModule.
    max_retries (int): The maximum number of tool call iterations. The default value is 5.
    return_trace (bool): If True, return intermediate steps and tool calls.
    stream (bool): Whether to stream the planning and solving process.
    return_last_tool_calls (bool): If True, return the last tool-call trace when the model finishes.
    skills (bool | str | List[str]): Skills config. True enables Skills with auto selection; pass a str/list to enable specific skills.
    desc (str): Optional agent capability description.
    workspace (str): Default agent workspace path. Defaults to `config['home']/agent_workspace`.
''')

add_chinese_doc('PlanAndSolveAgent.build_agent', '''\
构建 PlanAndSolveAgent 的规划与求解执行流程。
''')

add_english_doc('PlanAndSolveAgent.build_agent', '''\
Build the planning and solving execution workflow for PlanAndSolveAgent.
''')

add_example('PlanAndSolveAgent', """\
>>> import lazyllm
>>> from lazyllm.tools import fc_register, PlanAndSolveAgent
>>> @fc_register("tool")
>>> def multiply(a: int, b: int) -> int:
...     '''
...     Multiply two integers and return the result integer
...
...     Args:
...         a (int): multiplier
...         b (int): multiplier
...     '''
...     return a * b
...
>>> @fc_register("tool")
>>> def add(a: int, b: int):
...     '''
...     Add two integers and returns the result integer
...
...     Args:
...         a (int): addend
...         b (int): addend
...     '''
...     return a + b
...
>>> tools = ["multiply", "add"]
>>> llm = lazyllm.TrainableModule("internlm2-chat-20b").start()  # or llm = lazyllm.OnlineChatModule(source="sensenova")
>>> agent = PlanAndSolveAgent(llm, tools)
>>> query = "What is 20+(2*4)? Calculate step by step."
>>> res = agent(query)
>>> print(res)
'The final answer is 28.'
""")

add_chinese_doc('ReWOOAgent', '''\
ReWOOAgent包含三个部分：Planner、Worker和Solver。其中，Planner使用可预见推理能力为复杂任务创建解决方案蓝图；Worker通过工具调用来与环境交互，并将实际证据或观察结果填充到指令中；Solver处理所有计划和证据以制定原始任务或问题的解决方案。

Args:
    llm (ModuleBase): 要使用的LLM，可以是TrainableModule或OnlineChatModule。和plan_llm、solve_llm互斥，要么设置llm(planner和solver公用一个LLM)，要么设置plan_llm和solve_llm，或者只指定llm(用来设置planner)和solve_llm，其它情况均认为是无效的。
    tools (List[str]): LLM使用的工具名称列表。
    plan_llm (ModuleBase): planner要使用的LLM，可以是TrainableModule或OnlineChatModule。
    solve_llm (ModuleBase): solver要使用的LLM，可以是TrainableModule或OnlineChatModule。
    return_trace (bool): 是否返回中间步骤和工具调用信息。
    stream (bool): 是否以流式方式输出规划和解决过程。
    return_last_tool_calls (bool): 若为True，在模型结束且存在工具调用记录时返回最后一次的工具调用轨迹。
    skills (bool | str | List[str]): Skills 配置。True 启用 Skills 并自动筛选；传入 str/list 启用指定技能。
    desc (str): Agent 能力描述，可为空。
    workspace (str): Agent 默认工作目录，默认是 `config['home']/agent_workspace`。

''')

add_english_doc('ReWOOAgent', '''\
ReWOOAgent consists of three parts: Planer, Worker and Solver. The Planner uses predictive reasoning capabilities to create a solution blueprint for a complex task; the Worker interacts with the environment through tool calls and fills in actual evidence or observations into instructions; the Solver processes all plans and evidence to develop a solution to the original task or problem.

Args:
    llm (ModuleBase): The LLM to be used can be TrainableModule or OnlineChatModule. It is mutually exclusive with plan_llm and solve_llm. Either set llm(the planner and solver share the same LLM), or set plan_llm and solve_llm,or only specify llm(to set the planner) and solve_llm. Other cases are considered invalid.
    tools (List[str]): A list of tool names for LLM to use.
    plan_llm (ModuleBase): The LLM to be used by the planner, which can be either TrainableModule or OnlineChatModule.
    solve_llm (ModuleBase): The LLM to be used by the solver, which can be either TrainableModule or OnlineChatModule.
    return_trace (bool): If True, return intermediate steps and tool calls.
    stream (bool): Whether to stream the planning and solving process.
    return_last_tool_calls (bool): If True, return the last tool-call trace when the model finishes.
    skills (bool | str | List[str]): Skills config. True enables Skills with auto selection; pass a str/list to enable specific skills.
    desc (str): Optional agent capability description.
    workspace (str): Default agent workspace path. Defaults to `config['home']/agent_workspace`.
''')

add_chinese_doc('ReWOOAgent.build_agent', '''\
构建 ReWOOAgent 的 Planner/Worker/Solver 执行流程。
''')

add_english_doc('ReWOOAgent.build_agent', '''\
Build the Planner/Worker/Solver workflow for ReWOOAgent.
''')

add_chinese_doc('FunctionCallAgent.build_agent', '''\
构建 FunctionCallAgent 的工具调用迭代流程。
''')

add_english_doc('FunctionCallAgent.build_agent', '''\
Build the tool-calling iteration workflow for FunctionCallAgent.
''')

add_example(
    "ReWOOAgent", """\
>>> import lazyllm
>>> import wikipedia
>>> from lazyllm.tools import fc_register, ReWOOAgent
>>> @fc_register("tool")
>>> def WikipediaWorker(input: str):
...     '''
...     Worker that search for similar page contents from Wikipedia. Useful when you need to get holistic knowledge about people, places, companies, historical events, or other subjects. The response are long and might contain some irrelevant information. Input should be a search query.
...
...     Args:
...         input (str): search query.
...     '''
...     try:
...         evidence = wikipedia.page(input).content
...         evidence = evidence.split("\\\\n\\\\n")[0]
...     except wikipedia.PageError:
...         evidence = f"Could not find [{input}]. Similar: {wikipedia.search(input)}"
...     except wikipedia.DisambiguationError:
...         evidence = f"Could not find [{input}]. Similar: {wikipedia.search(input)}"
...     return evidence
...
>>> @fc_register("tool")
>>> def LLMWorker(input: str):
...     '''
...     A pretrained LLM like yourself. Useful when you need to act with general world knowledge and common sense. Prioritize it when you are confident in solving the problem yourself. Input can be any instruction.
...
...     Args:
...         input (str): instruction
...     '''
...     llm = lazyllm.OnlineChatModule(source="glm")
...     query = f"Respond in short directly with no extra words.\\\\n\\\\n{input}"
...     response = llm(query, llm_chat_history=[])
...     return response
...
>>> tools = ["WikipediaWorker", "LLMWorker"]
>>> llm = lazyllm.TrainableModule("GLM-4-9B-Chat").deploy_method(lazyllm.deploy.vllm).start()  # or llm = lazyllm.OnlineChatModule(source="sensenova")
>>> agent = ReWOOAgent(llm, tools)
>>> query = "What is the name of the cognac house that makes the main ingredient in The Hennchata?"
>>> res = agent(query)
>>> print(res)
'\nHennessy '
""")


#eval/eval_base.py
add_chinese_doc('BaseEvaluator', '''\
评估模块的抽象基类。

该类定义了模型评估的标准接口，支持并发处理、输入校验和评估结果的自动保存，同时内置了重试机制。

Args:
    concurrency (int): 评估过程中使用的并发线程数。
    retry (int): 每个样本的最大重试次数。
    log_base_name (Optional[str]): 用于保存结果文件的日志文件名前缀（可选）。
''')

add_english_doc('BaseEvaluator', '''\
Abstract base class for evaluation modules.

This class defines the standard interface and retry logic for evaluating model outputs. It supports concurrent processing, input validation, and automatic result saving.

Args:
    concurrency (int): Number of concurrent threads used during evaluation.
    retry (int): Number of retry attempts for each evaluation item.
    log_base_name (Optional[str]): Optional log file name prefix for saving results.
''')

add_example('BaseEvaluator', ['''\
>>> from lazyllm.components import BaseEvaluator
>>> class SimpleAccuracyEvaluator(BaseEvaluator):
...     def _process_one_data_impl(self, data):
...         return {
...             "final_score": float(data["pred"] == data["label"])
...         }
>>> evaluator = SimpleAccuracyEvaluator()
>>> score = evaluator([
...     {"pred": "yes", "label": "yes"},
...     {"pred": "no", "label": "yes"}
... ])
>>> print(score)
... 0.5
'''])

add_chinese_doc('BaseEvaluator.process_one_data', '''\
处理单条数据。

Args:
    data: 要处理的数据项。
    progress_bar (Optional[tqdm]): 进度条对象，默认为None。

**Returns:**\n
- Any: 返回处理结果。

注意：
    该方法会在处理数据时自动更新进度条，并使用线程锁确保线程安全。
''')

add_english_doc('BaseEvaluator.process_one_data', '''\
Process a single data item.

Args:
    data: Data item to process.
    progress_bar (Optional[tqdm]): Progress bar object, defaults to None.

**Returns:**\n
- Any: Returns processing result.

Note:
    This method automatically updates the progress bar during processing and uses thread lock to ensure thread safety.
''')

add_chinese_doc('BaseEvaluator.validate_inputs_key', '''\
验证输入数据的格式和必要键。

Args:
    data: 要验证的数据。

Raises:
    RuntimeError: 当数据格式不正确或缺少必要键时抛出。
        - 如果data不是列表
        - 如果列表中的项不是字典
        - 如果字典中缺少必要的键
''')

add_english_doc('BaseEvaluator.validate_inputs_key', '''\
Validate input data format and required keys.

Args:
    data: Data to validate.

Raises:
    RuntimeError: Raised when data format is incorrect or missing required keys.
        - If data is not a list
        - If items in the list are not dictionaries
        - If dictionaries are missing required keys
''')

add_chinese_doc('BaseEvaluator.batch_process', '''\
批量处理数据。

Args:
    data: 要处理的数据列表。
    progress_bar (tqdm): 进度条对象。

**Returns:**\n
- List: 返回处理结果列表。

流程：
    1. 验证输入数据的格式和必要键
    2. 使用并发处理器处理数据
    3. 保存处理结果
''')

add_english_doc('BaseEvaluator.batch_process', '''\
Process data in batch.

Args:
    data: List of data to process.
    progress_bar (tqdm): Progress bar object.

**Returns:**\n
- List: Returns list of processing results.

Flow:
    1. Validates input data format and required keys
    2. Processes data using concurrent processor
    3. Saves processing results
''')

add_chinese_doc('BaseEvaluator.save_res', '''\
保存评估结果。

Args:
    data: 要保存的数据。
    eval_res_save_name (Optional[str]): 保存文件的基础名称，默认使用类名。

保存格式：
    - 文件名格式：{filename}_{timestamp}.json
    - 时间戳格式：YYYYMMDDHHmmSS
    - 保存路径：lazyllm.config['eval_result_dir']
    - JSON格式，使用4空格缩进
''')

add_english_doc('BaseEvaluator.save_res', '''\
Save evaluation results.

Args:
    data: Data to save.
    eval_res_save_name (Optional[str]): Base name for the save file, defaults to class name.

Save Format:
    - Filename format: {filename}_{timestamp}.json
    - Timestamp format: YYYYMMDDHHmmSS
    - Save path: lazyllm.config['eval_result_dir']
    - JSON format with 4-space indentation
''')

add_chinese_doc('ResponseRelevancy', '''\
用于评估用户问题与模型生成问题之间语义相关性的指标类。

该评估器使用语言模型根据回答生成问题，并通过 Embedding 与余弦相似度度量其与原始问题之间的相关性。

Args:
    llm (ModuleBase): 用于根据回答生成问题的语言模型模块。
    embedding (ModuleBase): 用于编码问题向量的嵌入模块。
    prompt (str, 可选): 自定义的生成提示词，若不提供将使用默认提示。
    prompt_lang (str): 默认提示词的语言，可选 `'en'`（默认）或 `'zh'`。
    num_infer_questions (int): 每条数据生成和评估的问题数量。
    retry (int): 失败时的重试次数。
    concurrency (int): 并发评估的数量。
''')

add_english_doc('ResponseRelevancy', '''\
Evaluator for measuring the semantic relevancy between a user-generated question and a model-generated one.

This evaluator uses a language model to generate possible questions from an answer, and measures their semantic similarity to the original question using embeddings and cosine similarity.

Args:
    llm (ModuleBase): A language model used to generate inferred questions from the given answer.
    embedding (ModuleBase): An embedding module to encode questions for similarity comparison.
    prompt (str, optional): Custom prompt to guide the question generation. If not provided, a default will be used.
    prompt_lang (str): Language for the default prompt. Options: `'en'` (default) or `'zh'`.
    num_infer_questions (int): Number of questions to generate and evaluate for each answer.
    retry (int): Number of retry attempts if generation fails.
    concurrency (int): Number of concurrent evaluations.
''')

add_example('ResponseRelevancy', ['''\
>>> from lazyllm.components import ResponseRelevancy
>>> relevancy = ResponseRelevancy(
...     llm=YourLLM(),
...     embedding=YourEmbedding(),
...     prompt_lang="en",
...     num_infer_questions=3
... )
>>> result = relevancy([
...     {"question": "What is the capital of France?", "answer": "Paris is the capital city of France."}
... ])
>>> print(result)
... 0.95  # (a float score between 0 and 1)
'''])

add_chinese_doc('Faithfulness', '''\
评估回答与上下文之间事实一致性的指标类。

该评估器首先使用语言模型将答案拆分为独立事实句，然后基于上下文对每条句子进行支持性判断（0或1分），最终取平均值作为总体一致性分数。

Args:
    llm (ModuleBase): 同时用于生成句子与进行评估的语言模型模块。
    generate_prompt (str, 可选): 用于将答案转换为事实句的自定义提示词。
    eval_prompt (str, 可选): 用于评估句子与上下文匹配度的提示词。
    prompt_lang (str): 默认提示词的语言，可选 'en' 或 'zh'。
    retry (int): 生成或评估失败时的最大重试次数。
    concurrency (int): 并发评估的数据条数。
''')

add_english_doc('Faithfulness', '''\
Evaluator that measures the factual consistency of an answer with the given context.

This evaluator splits the answer into atomic factual statements using a generation model, then verifies each against the context using binary (1/0) scoring. It computes a final score as the average of the individual statement scores.

Args:
    llm (ModuleBase): A language model capable of both generating statements and evaluating them.
    generate_prompt (str, optional): Custom prompt to generate factual statements from the answer.
    eval_prompt (str, optional): Custom prompt to evaluate statement support within the context.
    prompt_lang (str): Language of the default prompt, either 'en' or 'zh'.
    retry (int): Number of retry attempts when generation or evaluation fails.
    concurrency (int): Number of concurrent evaluations to run in parallel.
''')

add_example('Faithfulness', ['''\
>>> from lazyllm.components import Faithfulness
>>> evaluator = Faithfulness(llm=YourLLM(), prompt_lang="en")
>>> data = {
...     "question": "What is the role of ATP in cells?",
...     "answer": "ATP stores energy and transfers it within cells.",
...     "context": "ATP is the energy currency of the cell. It provides energy for many biochemical reactions."
... }
>>> result = evaluator([data])
>>> print(result)
... 1.0  # Average binary score of all factual statements
'''])

add_chinese_doc('LLMContextRecall', '''\
用于评估回答中的每一句话是否可以归因于检索到的上下文的指标类。

该模块使用语言模型判断回答中的每个句子是否得到上下文的支持，通过二元值进行评分（1 表示支持，0 表示不支持或矛盾），最终计算平均回忆得分。

Args:
    llm (ModuleBase): 用于执行上下文一致性判断的语言模型。
    eval_prompt (str, 可选): 指导模型评估的自定义提示词。
    prompt_lang (str): 默认提示词语言，'en' 表示英文，'zh' 表示中文。
    retry (int): 评估失败时的最大重试次数。
    concurrency (int): 并发评估的任务数量。
''')

add_english_doc('LLMContextRecall', '''\
Evaluator that measures whether each sentence in the answer can be attributed to the retrieved context.

This module uses a language model to analyze the factual alignment between each statement in the answer and the provided context. It scores each sentence with binary values (1 = supported, 0 = unsupported/contradictory) and computes an average recall score.


Args:
    llm (ModuleBase): A language model capable of evaluating answer-context consistency.
    eval_prompt (str, optional): Custom prompt used to instruct the evaluator model.
    prompt_lang (str): Language of the default prompt. Choose 'en' for English or 'zh' for Chinese.
    retry (int): Number of retry attempts if the evaluation fails.
    concurrency (int): Number of parallel evaluations to perform concurrently.
''')

add_example('LLMContextRecall', ['''\
>>> from lazyllm.components import LLMContextRecall
>>> evaluator = LLMContextRecall(llm=YourLLM(), prompt_lang="en")
>>> data = {
...     "question": "What is Photosynthesis?",
...     "answer": "Photosynthesis was discovered in the 1780s. It occurs in chloroplasts.",
...     "context_retrieved": [
...         "Photosynthesis occurs in chloroplasts.",
...         "Light reactions produce ATP using sunlight."
...     ]
... }
>>> result = evaluator([data])
>>> print(result)
... 0.5  # Final recall score averaged over statement evaluations
'''])

add_chinese_doc('NonLLMContextRecall', '''\
基于字符串模糊匹配的非LLM上下文回忆指标类。

该模块通过 Levenshtein 距离计算检索到的上下文与参考上下文的相似度，并给出回忆得分。可选择输出二值得分（是否存在足够相似的匹配）或平均匹配度得分。

Args:
    th (float): 相似度阈值（范围为0到1），值越高表示匹配越严格。
    binary (bool): 若为True，则只判断是否有任一匹配超过阈值；若为False，则输出所有匹配的平均得分。
    retry (int): 失败时最大重试次数。
    concurrency (int): 并发执行的任务数量。
''')

add_english_doc('NonLLMContextRecall', '''\
A non-LLM evaluator that measures whether retrieved contexts match the reference context using fuzzy string matching.

This module compares each retrieved context against a reference using Levenshtein distance and computes a recall score. It can return binary scores (whether any retrieved context is similar enough) or an averaged similarity score.

Args:
    th (float): Similarity threshold (between 0 and 1). A higher value means stricter matching.
    binary (bool): If True, output is binary (1 if any match exceeds threshold), otherwise returns average match score.
    retry (int): Number of retries for evaluation in case of failure.
    concurrency (int): Number of parallel evaluations to run.
''')

add_example('NonLLMContextRecall', ['''\
>>> from lazyllm.components import NonLLMContextRecall
>>> evaluator = NonLLMContextRecall(th=0.8, binary=True)
>>> data = {
...     "context_retrieved": [
...         "Photosynthesis uses sunlight to produce sugar.",
...         "It takes place in chloroplasts."
...     ],
...     "context_reference": [
...         "Photosynthesis occurs in chloroplasts."
...     ]
... }
>>> result = evaluator([data])
>>> print(result)
... 1.0  # At least one retrieved context is similar enough
'''])

add_chinese_doc('ContextRelevance', '''\
基于句子级匹配的非LLM上下文相关性评估器。

该模块将检索到的上下文与参考上下文分别按句子划分，并统计检索内容中与参考完全一致的句子数量，从而计算相关性得分。

Args:
    splitter (str): 句子分隔符，默认为中文句号 "。"，英文可设置为 "."。
    retry (int): 失败时最大重试次数。
    concurrency (int): 并发执行的任务数量。
''')

add_english_doc('ContextRelevance', '''\
A non-LLM evaluator that measures the overlap between retrieved and reference contexts at the sentence level.

This evaluator splits both retrieved and reference contexts into sentences, then counts how many retrieved sentences exactly match those in the reference. It outputs a relevance score as the fraction of overlapping sentences.


Args:
    splitter (str): Sentence splitter. Default is '。' for Chinese. Use '.' for English contexts.
    retry (int): Number of retries for evaluation in case of failure.
    concurrency (int): Number of parallel evaluations to run.
''')

add_example('ContextRelevance', ['''\
>>> from lazyllm.components import ContextRelevance
>>> evaluator = ContextRelevance(splitter='.')
>>> data = {
...     "context_retrieved": [
...         "Photosynthesis occurs in chloroplasts. It produces glucose."
...     ],
...     "context_reference": [
...         "Photosynthesis occurs in chloroplasts. It requires sunlight. It produces glucose."
...     ]
... }
>>> result = evaluator([data])
>>> print(result)
... 0.6667  # 2 of 3 retrieved sentences match
'''])



#http_request/http_request.py
add_chinese_doc('HttpRequest', '''\
通用 HTTP 请求执行器。

该类用于构建并发送 HTTP 请求，支持变量替换、API Key 注入、JSON 或表单编码、文件类型响应识别等功能。

Args:
    method (str): HTTP 方法，如 'GET'、'POST' 等。
    url (str): 请求目标的 URL。
    api_key (str): 可选的 API Key，会被加入请求参数。
    headers (dict): HTTP 请求头。
    params (dict): URL 查询参数。
    body (Union[str, dict]): 请求体，支持字符串或 JSON 字典格式。
    timeout (int): 请求超时时间（秒）。
    proxies (dict, optional): 可选的代理设置。
''')

add_english_doc('HttpRequest', '''\
General HTTP request executor.

This class builds and sends HTTP requests with support for dynamic variable substitution, API key injection, JSON or form data encoding, and file-aware response parsing.

Args:
    method (str): HTTP method, such as 'GET', 'POST', etc.
    url (str): The target URL for the HTTP request.
    api_key (str): Optional API key, inserted into query parameters.
    headers (dict): HTTP request headers.
    params (dict): URL query parameters.
    body (Union[str, dict]): HTTP request body (raw string or JSON-formatted dict).
    timeout (int): Timeout duration for the request (in seconds).
    proxies (dict, optional): Proxy settings for the request, if needed.
''')

add_example('HttpRequest', ['''\
>>> from lazyllm.components import HttpRequest
>>> request = HttpRequest(
...     method="GET",
...     url="https://api.github.com/repos/openai/openai-python",
...     api_key="",
...     headers={"Accept": "application/json"},
...     params={},
...     body=None
... )
>>> result = request()
>>> print(result["status_code"])
... 200
>>> print(result["content"][:100])
... '{"id":123456,"name":"openai-python", ...}'
'''])

add_chinese_doc('DBManager', '''\
数据库管理器的抽象基类。

该类定义了构建数据库连接器的通用接口，包括 `execute_query` 抽象方法和 `desc` 描述属性。

Args:
    db_type (str): 数据库类型标识符，例如 'mysql'、'mongodb'。
''')

add_english_doc('DBManager', '''\
Abstract base class for database managers.

This class defines the standard interface and helpers for building database connectors, including a required `execute_query` method and description property.

Args:
    db_type (str): Type identifier of the database (e.g., 'mysql', 'mongodb').
''')

add_example('DBManager', ['''\
>>> from lazyllm.components import DBManager
>>> class DummyDB(DBManager):
...     def __init__(self):
...         super().__init__(db_type="dummy")
...     def execute_query(self, statement):
...         return f"Executed: {statement}"
...     @property
...     def desc(self):
...         return "Dummy database for testing."
>>> db = DummyDB()
>>> print(db("SELECT * FROM test"))
... Executed: SELECT * FROM test
'''])

add_chinese_doc('DBManager.execute_query', '''\
执行数据库查询语句的抽象方法。此方法需要由具体的数据库管理器子类实现，用于执行各种数据库操作。

Args:
    statement: 要执行的数据库查询语句，可以是 SQL 语句或其他数据库特定的查询语言

此方法的特点：

- **抽象方法**: 需要在子类中实现具体的数据库操作逻辑
- **统一接口**: 为不同的数据库类型提供统一的查询接口
- **错误处理**: 子类实现应该包含适当的错误处理和状态报告
- **结果格式化**: 返回格式化的字符串结果，便于后续处理

**注意**: 此方法是数据库管理器的核心方法，所有具体的数据库操作都通过此方法执行。

''')

add_english_doc('DBManager.execute_query', '''\
Abstract method for executing database query statements. This method needs to be implemented by specific database manager subclasses to execute various database operations.

Args:
    statement: The database query statement to execute, which can be SQL statements or other database-specific query languages

Features of this method:

- **Abstract Method**: Requires implementation of specific database operation logic in subclasses
- **Unified Interface**: Provides a unified query interface for different database types
- **Error Handling**: Subclass implementations should include appropriate error handling and status reporting
- **Result Formatting**: Returns formatted string results for subsequent processing

**Note**: This method is the core method of the database manager, and all specific database operations are executed through this method.

''')

add_chinese_doc("SqlManager","""\
SqlManager是与数据库进行交互的专用工具。它提供了连接数据库，设置、创建、检查数据表，插入数据，执行查询的方法。

Args:
    db_type (str): 数据库类型，支持: postgresql, mysql, mssql, sqlite, mysql+pymysql
    user (str): 数据库用户名
    password (str): 数据库密码
    host (str): 数据库主机地址
    port (int): 数据库端口号
    db_name (str): 数据库名称
    options_str (str, optional): 连接选项字符串，默认为None
    tables_info_dict (Dict, optional): 表结构信息字典，用于初始化表结构，默认为None
""")

add_english_doc("SqlManager","""\
SqlManager is a specialized tool for interacting with databases.
It provides methods for creating tables, executing queries, and performing updates on databases.

Args:
    db_type (str): Database type, supports: postgresql, mysql, mssql, sqlite, mysql+pymysql
    user (str): Database username
    password (str): Database password
    host (str): Database host address
    port (int): Database port number
    db_name (str): Database name
    options_str (str, optional): Connection options string, defaults to None
    tables_info_dict (Dict, optional): Table structure information dictionary for initializing table structure, defaults to None
""")

add_chinese_doc("SqlManager.get_session", """\
一个数据库会话上下文管理器。

默认（``session=None``）会创建一个新的 SQLAlchemy 会话并在上下文退出时自动提交；若上下文内部抛出异常则会自动回滚；无论是否成功，会话最终都会被关闭。

当传入一个外部 ``session`` 时，本上下文是透明的：会话被原样 yield，提交/回滚/关闭由最初打开它的 ``get_session`` 负责。上下文内部抛出的异常仍会向上传播，由外层 ``get_session`` 触发回滚，因此辅助方法可以通过可选的 ``session=None`` 参数参与调用方驱动的多步事务，而无需关心自己是否拥有该会话。

Args:
    session (Optional[Session]): 外部已打开的 SQLAlchemy 会话。为 ``None`` 时创建并自管理一个新会话，否则透传使用。
""")

add_english_doc("SqlManager.get_session", """\
A database session context manager.

When called with the default ``session=None`` it opens a new SQLAlchemy session, commits on success, rolls back on exception, and unconditionally closes the session on exit.

When an external ``session`` is passed in, this context manager is transparent: the session is yielded as-is without commit / rollback / close — ownership stays with whoever originally opened it. Exceptions raised inside the ``with`` block still propagate and trigger rollback at the owning ``get_session`` up the call chain. This allows helper methods to participate in a caller-driven transaction via an optional ``session=None`` parameter without having to know whether they own the session.

Args:
    session (Optional[Session]): An already-open SQLAlchemy session from an outer scope. When ``None``, a new session is opened and managed by this context; otherwise the passed session is yielded as-is.
""")

add_chinese_doc("SqlManager.paginate", """\
对一个 SQLAlchemy ``Query`` 应用基于 page 的分页。

会先将 ``page`` 与 ``page_size`` 截断到不小于 1，再对传入的查询做一次 ``COUNT`` 拿到不分页的总数，然后用 ``OFFSET``/``LIMIT`` 取当前页。返回形如 ``{'items', 'total', 'page', 'page_size'}`` 的字典。

返回的 ``items`` 是原始 SQLAlchemy 行对象——不做任何业务转换。调用方自行负责把行转换成所需格式（例如通过 ``_orm_to_dict``），因此本工具可以被任意列表接口复用。

Args:
    query (Query): 已经应用好所需 ``filter``/``order_by`` 的 SQLAlchemy ``Query``。
    page (int): 页码，从 1 开始；小于 1 时会被截断为 1。
    page_size (int): 每页条数；小于 1 时会被截断为 1。
""")

add_english_doc("SqlManager.paginate", """\
Apply page-based pagination to a SQLAlchemy ``Query``.

Clamps ``page`` and ``page_size`` to at least ``1``, issues a ``COUNT`` against the unpaginated query to obtain the total row count, then fetches the current page using ``OFFSET`` / ``LIMIT``. Returns a dict shaped ``{'items', 'total', 'page', 'page_size'}``.

The returned ``items`` are raw SQLAlchemy row results — no business conversion is performed. Callers are responsible for converting rows to their desired shape (for example via ``_orm_to_dict``), so this helper stays domain-agnostic and reusable across any listing endpoint.

Args:
    query (Query): A SQLAlchemy ``Query`` with the desired ``filter`` / ``order_by`` already applied.
    page (int): 1-based page number; values below 1 are clamped to 1.
    page_size (int): Number of rows per page; values below 1 are clamped to 1.
""")

add_chinese_doc("SqlManager.check_connection", """\
检查数据库连接状态。

测试与数据库的连接是否正常建立。

**Returns:**\n
- DBResult: DBResult.status 连接成功(True), 连接失败(False)。DBResult.detail 包含失败信息
""")

add_english_doc("SqlManager.check_connection", """\
Check database connection status.

Tests whether the connection to the database is successfully established.

**Returns:**\n
- DBResult: DBResult.status True if the connection is successful, False if it fails. DBResult.detail contains failure information.
""")

add_chinese_doc("SqlManager.set_desc", """\
对于SqlManager搭配LLM使用自然语言查询的表项设置其描述，尤其当其表名、列名及取值不具有自解释能力时。
例如：
数据表Document的status列取值包括: "waiting", "working", "success", "failed"，tables_desc_dict参数应为 {"Document": "status列取值包括: waiting, working, success, failed"}

Args:
    tables_desc_dict (dict): 表项的补充说明
""")

add_english_doc("SqlManager.set_desc", """\
When using SqlManager with LLM to query table entries in natural language, set descriptions for better results, especially when table names, column names, and values are not self-explanatory.

Args:
    tables_desc_dict (dict): descriptive comment for tables
""")

add_chinese_doc("SqlManager.get_all_tables", """\
获取数据库中所有表的列表。

刷新元数据后返回当前数据库中的所有表名。

**Returns:**\n
- List[str]: 数据库中所有表名的列表
""")

add_english_doc("SqlManager.get_all_tables", """\
Get list of all tables in the database.

Refreshes metadata and returns all table names in the current database.

**Returns:**\n
- List[str]: List of all table names in the database
""")

add_chinese_doc("SqlManager.get_table_orm_class", """\
根据表名获取对应的ORM类。

通过表名反射获取SQLAlchemy自动映射的ORM类。

Args:
    table_name (str): 要获取的表名

**Returns:**\n
- sqlalchemy.ext.automap.Class: 对应的ORM类，如果表不存在返回None
""")

add_english_doc("SqlManager.get_table_orm_class", """\
Get corresponding ORM class by table name.

Reflects and gets SQLAlchemy automapped ORM class through table name.

Args:
    table_name (str): Table name to retrieve

**Returns:**\n
- sqlalchemy.ext.automap.Class: Corresponding ORM class, returns None if table doesn't exist
""")

add_chinese_doc("SqlManager.execute_commit", """\
执行SQL提交语句。

执行DDL或DML语句并自动提交事务，适用于CREATE、ALTER、INSERT、UPDATE、DELETE等操作。

Args:
    statement (str): 要执行的SQL语句
""")

add_english_doc("SqlManager.execute_commit", """\
Execute SQL commit statements.

Executes DDL or DML statements and automatically commits transactions. Suitable for CREATE, ALTER, INSERT, UPDATE, DELETE operations.

Args:
    statement (str): SQL statement to execute
""")

add_chinese_doc("SqlManager.execute_query", """\
执行sql查询脚本并以JSON字符串返回结果。
""")

add_english_doc("SqlManager.execute_query", """\
Execute the SQL query script and return the result as a JSON string.
""")

add_chinese_doc("SqlManager.create_table", """\
创建数据表

Args:
    table (str/Type[DeclarativeBase]/DeclarativeMeta): 数据表schema。支持三种参数类型：类型为str的sql语句，继承自DeclarativeBase或继承自declarative_base()的ORM类
""")

add_english_doc("SqlManager.create_table", """\
Create a table

Args:
    table (str/Type[DeclarativeBase]/DeclarativeMeta): table schema。Supports three types of parameters: SQL statements with type str, ORM classes that inherit from DeclarativeBase or declarative_base().
""")

add_chinese_doc("SqlManager.drop_table", """\
删除数据表

Args:
    table (str/Type[DeclarativeBase]/DeclarativeMeta): 数据表schema。支持三种参数类型：类型为str的数据表名，继承自DeclarativeBase或继承自declarative_base()的ORM类
""")

add_english_doc("SqlManager.drop_table", """\
Delete a table

Args:
    table (str/Type[DeclarativeBase]/DeclarativeMeta): table schema。Supports three types of parameters: Table name with type str, ORM classes that inherit from DeclarativeBase or declarative_base().
""")

add_chinese_doc("SqlManager.insert_values", """\
批量数据插入

Args:
    table_name (str): 数据表名
    vals (List[dict]): 待插入数据，格式为[{"col_name1": v01, "col_name2": v02, ...}, {"col_name1": v11, "col_name2": v12, ...}, ...]
""")

add_english_doc("SqlManager.insert_values", """\
Bulk insert data

Args:
    table_name (str): Table name
    vals (List[dict]): data to be inserted, format as [{"col_name1": v01, "col_name2": v02, ...}, {"col_name1": v11, "col_name2": v12, ...}, ...]
""")

add_chinese_doc("HttpTool", """
用于访问第三方服务和执行自定义代码的模块。参数中的 `params` 和 `headers` 的 value，以及 `body` 中可以包含形如 `{{variable}}` 这样用两个花括号标记的模板变量，然后在调用的时候通过参数来替换模板中的值。参考 [[lazyllm.tools.HttpTool.forward]] 中的使用说明。

Args:
    method (str, optional): 指定 http 请求方法，参考 `https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods`。
    url (str, optional): 要访问的 url。如果该字段为空，则表示该模块不需要访问第三方服务。
    params (Dict[str, str], optional): 请求 url 需要填充的 params 字段。如果 url 为空，该字段会被忽略。
    headers (Dict[str, str], optional): 访问 url 需要填充的 header 字段。如果 url 为空，该字段会被忽略。
    body (Dict[str, str], optional): 请求 url 需要填充的 body 字段。如果 url 为空，该字段会被忽略。
    timeout (int): 请求超时时间，单位是秒，默认值是 10。
    proxies (Dict[str, str], optional): 指定请求 url 时所使用的代理。代理格式参考 `https://www.python-httpx.org/advanced/proxies`。
    code_str (str, optional): 一个字符串，包含用户定义的函数。如果参数 `url` 为空，则直接执行该函数，执行时所有的参数都会转发给该函数；如果 `url` 不为空，该函数的参数为请求 url 返回的结果，此时该函数作为 url 返回后的后处理函数。
    vars_for_code (Dict[str, Any]): 一个字典，传入运行 code 所需的依赖及变量。
    outputs (Optional[List[str]]): 期望提取的输出字段名。
    extract_from_result (Optional[bool]): 是否从响应字典中直接提取指定字段。
""")

add_english_doc("HttpTool", """
Module for accessing third-party services and executing custom code. The values in `params` and `headers`, as well as in body, can include template variables marked with double curly braces like `{{variable}}`, which are then replaced with actual values through parameters when called. Refer to the usage instructions in [[lazyllm.tools.HttpTool.forward]].

Args:
    method (str, optional): Specifies the HTTP request method, refer to `https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods`.
    url (str, optional): The URL to access. If this field is empty, it indicates that the module does not need to access third-party services.
    params (Dict[str, str], optional): Params fields to be filled when requesting the URL. If the URL is empty, this field will be ignored.
    headers (Dict[str, str], optional): Header fields to be filled when accessing the URL. If the URL is empty, this field will be ignored.
    body (Dict[str, str], optional): Body fields to be filled when requesting the URL. If the URL is empty, this field will be ignored.
    timeout (int): Request timeout in seconds, default value is 10.
    proxies (Dict[str, str], optional): Specifies the proxies to be used when requesting the URL. Proxy format refer to `https://www.python-httpx.org/advanced/proxies`.
    code_str (str, optional): A string containing a user-defined function. If the parameter url is empty, execute this function directly, forwarding all arguments to it; if url is not empty, the parameters of this function are the results returned from the URL request, and in this case, the function serves as a post-processing function for the URL response.
    vars_for_code (Dict[str, Any]): A dictionary that includes dependencies and variables required for running the code.
    outputs (Optional[List[str]]): Names of expected output fields.
    extract_from_result (Optional[bool]): Whether to extract fields directly from response dict using `outputs`.
""")

add_example("HttpTool", """
from lazyllm.tools import HttpTool

code_str = "def identity(content): return content"
tool = HttpTool(method='GET', url='http://www.sensetime.com/', code_str=code_str)
ret = tool()
""")

add_chinese_doc("HttpTool.forward", """
用于执行初始化时指定的操作：请求指定的 url 或者执行传入的函数。一般不直接调用，而是通过基类的 `__call__` 来调用。如果构造函数的 `url` 参数不为空，则传入的所有参数都会作为变量，用于替换在构造函数中使用 `{{}}` 标记的模板参数；如果构造函数的参数 `url` 为空，并且 `code_str` 不为空，则传入的所有参数都会作为 `code_str` 中所定义函数的参数。
""")

add_english_doc("HttpTool.forward", """
Used to perform operations specified during initialization: request the specified URL or execute the passed function. Generally not called directly, but through the base class's `__call__`. If the `url` parameter in the constructor is not empty, all passed parameters will be used as variables to replace template parameters marked with `{{}}` in the constructor; if the `url` parameter in the constructor is empty and `code_str` is not empty, all passed parameters will be used as arguments for the function defined in `code_str`.
""")

add_example("HttpTool.forward", """
from lazyllm.tools import HttpTool

code_str = "def exp(v, n): return v ** n"
tool = HttpTool(code_str=code_str)
assert tool(v=10, n=2) == 100
""")

add_agent_chinese_doc('functionCall.StreamResponse', '''\
StreamResponse类用于封装带有前缀和颜色配置的流式输出行为。
当启用流式模式时，调用实例会将带颜色的文本推送到文件系统队列中，用于异步处理或显示。

Args:
    prefix (str): 输出内容前的前缀文本，通常用于标识信息来源或类别。
    prefix_color (Optional[str]): 前缀文本的颜色，支持终端颜色代码，默认无颜色。
    color (Optional[str]): 主体内容文本颜色，支持终端颜色代码，默认无颜色。
    stream (bool): 是否启用流式输出模式，启用后会将文本推送至文件系统队列，默认关闭。
''')

add_agent_english_doc('functionCall.StreamResponse', '''\
StreamResponse class encapsulates streaming output behavior with configurable prefix and colors.
When streaming is enabled, calling the instance enqueues colored text to a filesystem queue for asynchronous processing or display.

Args:
    prefix (str): Prefix text before the output, typically used to indicate the source or category.
    prefix_color (Optional[str]): Color of the prefix text, supports terminal color codes, defaults to None.
    color (Optional[str]): Color of the main content text, supports terminal color codes, defaults to None.
    stream (bool): Whether to enable streaming output mode, which enqueues text to the filesystem queue, defaults to False.
''')

add_agent_example('functionCall.StreamResponse', '''\
>>> from lazyllm.tools.agent.functionCall import StreamResponse
>>> resp = StreamResponse(prefix="[INFO]", prefix_color="green", color="white", stream=True)
>>> resp("Hello, world!")
Hello, world!
''')


add_toolsmgr_chinese_doc('InstanceToolGroup', '''\
内部类，通常不需要直接使用。将 ``(instance, key_source)`` 元组或带 ``__public_apis__`` 的裸实例传给 ToolManager 时，框架会自动创建该对象。

将一个对象实例的所有 ``__public_apis__`` 方法统一封装为可注册到 ToolManager 的工具组。
支持可选的 key_source 参数，用于在运行时检测凭据是否存在；当凭据不存在时，该组内所有工具会从 tools_description 中自动隐藏，LLM 不会感知到其存在。

主要用于将 SearchBase、LazyLLMFSBase 等带有 __public_apis__ 的对象注册为 Agent 工具。

Args:
    instance (Any): 持有目标方法的对象实例。
    key_source (Union[str, Callable, List[Union[str, Callable]], None]): 运行时凭据来源，用于 should_skip() 判断。

        - None（默认）：若实例的类定义了 ``__key_source__`` 字段，则自动使用该字段作为凭据来源；否则不检查凭据，工具始终可用。
        - 字符串格式：

          - ``env.KEY``：从 ``os.environ`` 读取环境变量 ``KEY``。
          - ``config.key``：从 ``lazyllm.config`` 读取静态全局配置 ``key``。
          - ``globals.config.key``：从 ``lazyllm.globals.config`` 读取会话级动态配置 ``key``。
          - 无 ``.``（如 ``key``）：等价于 ``globals.config.key``。

        - callable：接收 instance 作为参数，返回凭据字符串或空值，如 ``lambda inst: inst._key``。
        - list：多个来源，任一非空即视为凭据可用（逻辑 OR）。
''')

add_toolsmgr_english_doc('InstanceToolGroup', '''\
Internal class; direct use is not normally required. The framework creates instances automatically when a ``(instance, key_source)`` tuple or a bare instance with ``__public_apis__`` is passed to ToolManager.

Wraps all ``__public_apis__`` methods of an object instance into a tool group that can be registered with ToolManager.
Accepts an optional key_source parameter to detect credential availability at runtime; when the credential is absent, all tools in the group are automatically hidden from tools_description so the LLM is unaware of them.

Primarily used to register objects with __public_apis__ (such as SearchBase or LazyLLMFSBase subclasses) as Agent tools.

Args:
    instance (Any): The object instance holding the target methods.
    key_source (Union[str, Callable, List[Union[str, Callable]], None]): Runtime credential source used by should_skip().

        - None (default): If the instance's class defines a ``__key_source__`` attribute, it is used automatically; otherwise no credential check is performed and the tools are always available.
        - String format:

          - ``env.KEY``: reads environment variable ``KEY`` from ``os.environ``.
          - ``config.key``: reads static global config ``key`` from ``lazyllm.config``.
          - ``globals.config.key``: reads session-level dynamic config ``key`` from ``lazyllm.globals.config``.
          - No dot (e.g. ``key``): equivalent to ``globals.config.key``.

        - callable: Receives instance as argument and returns the credential string or an empty value, e.g. ``lambda inst: inst._key``.
        - list: Multiple sources; the tools are considered available if any source resolves to a non-empty value (logical OR).
''')

add_toolsmgr_example('InstanceToolGroup', '''\
>>> import lazyllm
>>> from lazyllm.tools.agent.toolsManager import InstanceToolGroup
>>> class MockSearch:
...     __public_apis__ = ["search"]
...     def __init__(self):
...         self._key = ""
...     def search(self, query: str) -> str:
...         return f"results for {query}"
>>> inst = MockSearch()
>>> grp = InstanceToolGroup(inst, lambda i: i._key)
>>> grp.should_skip()
True
>>> inst._key = "my-key"
>>> grp.should_skip()
False
>>> tm = ToolManager([(inst, lambda i: i._key)])
>>> len(tm.tools_description)
1

>>> # Class-level __key_source__: no need to pass key_source when registering
>>> class BingSearch:
...     __public_apis__ = ["search"]
...     __key_source__ = lambda inst: inst._subscription_key
...     def __init__(self, subscription_key=""):
...         self._subscription_key = subscription_key
...     def search(self, query: str) -> str:
...         return f"bing: {query}"
>>> bing = BingSearch()
>>> tm2 = ToolManager([bing])
>>> len(tm2.tools_description)
0
>>> bing._subscription_key = "valid-key"
>>> len(tm2.tools_description)
1
''')

add_toolsmgr_chinese_doc('InstanceToolGroup.should_skip', '''\
根据 key_source 判断该工具组当前是否应被跳过（即凭据不可用）。

当 key_source 为列表时，任一来源解析出非空值即视为凭据可用（逻辑 OR）。

Returns:
    bool: 若 key_source 为 None 且类未定义 __key_source__ 则返回 False；否则尝试解析凭据，若所有来源均为空或不存在则返回 True。
''')

add_toolsmgr_english_doc('InstanceToolGroup.should_skip', '''\
Determines whether this tool group should currently be skipped based on key_source (i.e., the credential is unavailable).

When key_source is a list, the tools are considered available if any source resolves to a non-empty value (logical OR).

Returns:
    bool: Returns False if key_source is None and the class has no __key_source__; otherwise resolves all sources and returns True only if every source is empty or missing.
''')

add_toolsmgr_chinese_doc('InstanceToolGroup.get_description', '''\
返回该工具组的 OpenAI function calling 格式描述列表。

若 should_skip() 返回 True（凭据不可用），则返回空列表，LLM 不会感知到该组内的任何工具；
否则以 lazy 模式返回 gateway 工具描述（``get_<name>_methods``），LLM 调用后子工具描述动态注入 system prompt。

Returns:
    List[Dict]: 工具描述字典列表。当凭据不可用时返回空列表。
''')

add_toolsmgr_english_doc('InstanceToolGroup.get_description', '''\
Returns the list of tool descriptions in OpenAI function calling format for this tool group.

If should_skip() returns True (credential unavailable), an empty list is returned so the LLM is unaware of any tools in this group;
otherwise returns the gateway tool description (``get_<name>_methods``) in lazy mode; child tool descriptions are dynamically injected into the system prompt after the LLM calls the gateway.

Returns:
    List[Dict]: List of tool description dicts. Returns an empty list when the credential is unavailable.
''')

add_toolsmgr_chinese_doc('ToolGroup', '''\
工具组类，用于将多个工具（或子工具组）组织为一个有名称的集合，并以 lazy 或 eager 模式向 LLM 暴露。

支持两种模式：

- **lazy 模式**（默认，``lazy=True``）：初始只向 LLM 暴露一个 ``get_<name>_methods`` gateway 工具。
  LLM 调用该工具后，子工具描述动态注入 system prompt，LLM 在下一轮中再选择并调用具体工具。
  适合工具数量多、希望减少初始上下文长度的场景。
- **eager 模式**（``lazy=False``）：直接把所有子工具描述展开注入 system prompt，与旧行为一致。

支持多级嵌套：子 ``tools`` 列表中可以包含普通工具（函数、``ModuleTool``）或另一个 ``ToolGroup``（通过嵌套 ``dict`` 定义）。

通常不需要直接实例化 ``ToolGroup``，而是通过向 ``ToolManager``（或 Agent 的 ``tools`` 参数）传入 ``dict`` 来隐式创建：

.. code-block:: python

    tools = [
        tool1,
        dict(name='search', desc='搜索工具集', tools=[search_web, search_news]),
        dict(name='advanced', desc='高级工具', tools=[
            tool2,
            dict(name='sub_ops', desc='子工具集', lazy=False, tools=[tool3, tool4]),
        ]),
    ]

Args:
    tools (List): 子工具列表，每个元素可以是函数、``ModuleTool`` 实例、``ToolGroup`` 实例，或嵌套 ``dict``（格式同上）。
    name (str): 工具组名称，必填。用于生成 gateway 工具名 ``get_<name>_methods``，以及在激活后标识该组。
    desc (str): 工具组描述，会作为 gateway 工具的 description 展示给 LLM。
    lazy (bool): 是否使用 lazy 模式，默认 True。
    key_source: 凭据来源，格式与 ``InstanceToolGroup`` 的 ``key_source`` 参数相同。凭据不可用时整个工具组从 LLM 可见列表中隐藏。
''')

add_toolsmgr_english_doc('ToolGroup', '''\
Tool group class that organizes multiple tools (or sub-groups) into a named collection and exposes them to the LLM in lazy or eager mode.

Two modes are supported:

- **lazy mode** (default, ``lazy=True``): Initially only a ``get_<name>_methods`` gateway tool is exposed to the LLM.
  After the LLM calls it, child tool descriptions are dynamically injected into the system prompt, and the LLM selects and calls the actual tool in the next turn.
  Suitable when there are many tools and you want to reduce the initial context length.
- **eager mode** (``lazy=False``): All child tool descriptions are expanded and injected into the system prompt immediately, matching the previous behavior.

Multi-level nesting is supported: the ``tools`` list can contain plain tools (functions, ``ModuleTool``) or another ``ToolGroup`` (defined via a nested ``dict``).

Typically you do not instantiate ``ToolGroup`` directly; instead, pass a ``dict`` to ``ToolManager`` (or the Agent's ``tools`` parameter) to create one implicitly:

.. code-block:: python

    tools = [
        tool1,
        dict(name='search', desc='Search tools', tools=[search_web, search_news]),
        dict(name='advanced', desc='Advanced tools', tools=[
            tool2,
            dict(name='sub_ops', desc='Sub-tools', lazy=False, tools=[tool3, tool4]),
        ]),
    ]

Args:
    tools (List): Child tool list. Each element can be a function, ``ModuleTool`` instance, ``ToolGroup`` instance, or a nested ``dict`` (same format as above).
    name (str): Tool group name. Required. Used to generate the gateway tool name ``get_<name>_methods`` and to identify the group after activation.
    desc (str): Tool group description, shown to the LLM as the gateway tool's description.
    lazy (bool): Whether to use lazy mode. Defaults to True.
    key_source: Credential source, same format as the ``key_source`` parameter of ``InstanceToolGroup``. When the credential is unavailable, the entire tool group is hidden from the LLM.
''')

add_toolsmgr_example('ToolGroup', '''\
>>> import lazyllm
>>> from lazyllm.tools import ToolManager
>>> from lazyllm import init_session, locals as lazyllm_locals
>>>
>>> def search_web(query: str) -> str:
...     """Search the web.
...
...     Args:
...         query (str): Search query.
...
...     Returns:
...         str: Search results.
...     """
...     return f"web results for {query}"
...
>>> def search_news(query: str) -> str:
...     """Search news articles.
...
...     Args:
...         query (str): News search query.
...
...     Returns:
...         str: News results.
...     """
...     return f"news for {query}"
...
>>> # Lazy mode (default): only gateway tool is visible initially
>>> tm = ToolManager([
...     dict(name='search', desc='Web and news search tools', tools=[search_web, search_news]),
... ])
>>> [d['function']['name'] for d in tm.tools_description]
['get_search_methods']
>>>
>>> # After the LLM calls the gateway tool, child tools become visible
>>> init_session()
>>> lazyllm_locals['_lazyllm_agent'] = {'workspace': {}}
>>> gateway = tm._tool_call['get_search_methods']
>>> result = gateway({})
>>> print(result)
Activated tool group "search". Available tools: search_web, search_news
>>> [d['function']['name'] for d in tm.tools_description]
['get_search_methods', 'search_web', 'search_news']
>>>
>>> # Eager mode: all tools visible immediately
>>> tm2 = ToolManager([
...     dict(name='search', desc='Search tools', lazy=False, tools=[search_web, search_news]),
... ])
>>> [d['function']['name'] for d in tm2.tools_description]
['search_web', 'search_news']
>>>
>>> # Multi-level nesting
>>> def calc_add(a: str, b: str) -> str:
...     """Add two numbers.
...
...     Args:
...         a (str): First number.
...         b (str): Second number.
...
...     Returns:
...         str: Sum.
...     """
...     return str(int(a) + int(b))
...
>>> tm3 = ToolManager([
...     dict(name='outer', desc='Outer group', tools=[
...         search_web,
...         dict(name='inner', desc='Inner group', tools=[search_news, calc_add]),
...     ]),
... ])
>>> [d['function']['name'] for d in tm3.tools_description]
['get_outer_methods']
''')

add_toolsmgr_chinese_doc('MethodModuleTool', '''\
内部类，通常不需要直接使用。InstanceToolGroup 在初始化时会自动为实例的每个 ``__public_apis__`` 方法创建对应的 MethodModuleTool。

将对象实例的某个绑定方法封装为 ModuleTool，使其可被 ToolManager 管理和调用。
工具名称由实例类名和方法名拼接而成（``ClassName_method_name``）；若方法名为 ``__call__``，则工具名直接使用类名。
工具描述和参数 schema 从绑定方法的 docstring 和类型注解中自动解析。

Args:
    instance (Any): 持有目标方法的对象实例。
    method_name (str): 要封装的方法名称。
    key_source (Union[str, Callable, List[Union[str, Callable]], None]): 保留参数，当前未使用，默认为 None。
''')

add_toolsmgr_english_doc('MethodModuleTool', '''\
Internal class; direct use is not normally required. InstanceToolGroup automatically creates a MethodModuleTool for each ``__public_apis__`` method of the instance during initialization.

Wraps a bound method of an object instance as a ModuleTool so it can be managed and invoked by ToolManager.
The tool name is formed by concatenating the class name and method name (``ClassName_method_name``); if the method name is ``__call__``, the tool name is the class name alone.
The tool description and parameter schema are automatically parsed from the bound method's docstring and type annotations.

Args:
    instance (Any): The object instance holding the target method.
    method_name (str): The name of the method to wrap.
    key_source (Union[str, Callable, List[Union[str, Callable]], None]): Reserved parameter, currently unused. Defaults to None.
''')
