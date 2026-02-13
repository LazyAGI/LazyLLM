# flake8: noqa E501
from . import utils
import importlib
import functools
import lazyllm

add_chinese_doc = functools.partial(utils.add_chinese_doc, module=importlib.import_module('lazyllm.prompt_templates'))
add_english_doc = functools.partial(utils.add_english_doc, module=importlib.import_module('lazyllm.prompt_templates'))
add_example = functools.partial(utils.add_example, module=lazyllm.prompt_templates)

add_chinese_doc('prompt_template.PromptTemplate.validate_variables', """\
验证模板变量的完整性。

验证规则：
1. partial_vars中的所有键必须存在于模板变量中
2. required_vars和partial_vars的键必须完全等于所有模板变量

Raises:
    ValueError: 当变量验证失败时抛出，包括：
        - partial_vars包含模板中不存在的变量
        - required_vars和partial_vars有重叠变量
        - required_vars和partial_vars的并集不等于所有模板变量

Returns:
    PromptTemplate: 验证后的实例自身
""")

add_english_doc('prompt_template.PromptTemplate.validate_variables', """\
Validate the completeness of template variables.

Validation rules:
1. All keys in partial_vars must exist in template variables
2. required_vars + partial_vars keys must exactly equal all template variables

Raises:
    ValueError: Raised when variable validation fails, including:
        - partial_vars contains variables not found in template
        - required_vars and partial_vars have overlap
        - The union of required_vars and partial_vars does not equal all template variables

Returns:
    PromptTemplate: Validated instance itself
""")

add_chinese_doc('prompt_template.PromptTemplate.format', """\
格式化提示模板。

使用提供的变量值替换模板中的占位符，支持部分变量的函数调用。

Args:
    **kwargs: 模板变量名和对应的值

Returns:
    str: 格式化后的提示字符串

Raises:
    KeyError: 当缺少必需变量或模板变量不存在时
    TypeError: 当部分变量函数调用出错时
    ValueError: 当模板格式化出错时
""")

add_english_doc('prompt_template.PromptTemplate.format', """\
Format the prompt template.

Replace placeholders in template with provided variable values, supports function calls for partial variables.

Args:
    **kwargs: Template variable names and corresponding values

Returns:
    str: Formatted prompt string

Raises:
    KeyError: When required variables are missing or template variables not found
    TypeError: When partial variable function call fails
    ValueError: When template formatting fails
""")

add_chinese_doc('prompt_template.PromptTemplate.partial', """\
创建部分填充的模板副本。

为指定的变量设置固定值，生成一个新的模板实例，这些变量将不再需要提供。

Args:
    **partial_kwargs: 要设置为部分变量的变量名和值

Returns:
    PromptTemplate: 新的部分填充模板实例

Raises:
    KeyError: 当指定的变量不存在于模板中时
""")

add_english_doc('prompt_template.PromptTemplate.partial', """\
Create a partially filled copy of the template.

Set fixed values for specified variables, generating a new template instance where these variables no longer need to be provided.

Args:
    **partial_kwargs: Variable names and values to set as partial variables

Returns:
    PromptTemplate: New partially filled template instance

Raises:
    KeyError: When specified variables are not found in template
""")

add_chinese_doc('prompt_template.PromptTemplate.from_template', """\
从模板字符串创建PromptTemplate实例。

类方法，根据模板字符串自动提取所有变量并设置为必需变量。

Args:
    template (str): 包含{variable}占位符的模板字符串

Returns:
    PromptTemplate: 新创建的PromptTemplate实例
""")

add_english_doc('prompt_template.PromptTemplate.from_template', """\
Create PromptTemplate instance from template string.

Class method, automatically extracts all variables from template string and sets them as required variables.

Args:
    template (str): Template string containing {variable} placeholders

Returns:
    PromptTemplate: Newly created PromptTemplate instance
""")

add_chinese_doc('base.BasePromptTemplate.get_template_variables', """\
从给定的模板字符串中提取所有占位符变量名。

使用 Python 内置的 string.Formatter 解析模板，识别占位符，
并返回一个按字母顺序排序的唯一变量名列表。

Args:
    template (str): 包含占位符的提示模板字符串

Returns:
    list[str]: 模板中所有占位符变量名的排序列表

Raises:
    ValueError: 当模板格式非法或解析失败时抛出异常
""")


add_english_doc('base.BasePromptTemplate.get_template_variables', """\
Extracts all placeholder variable names from a given template string.

Uses Python's built-in string.Formatter to parse the template and identify placeholders 
. Returns a sorted list of unique variable names.

Args:
    template (str): A prompt template string containing placeholders
                    

Returns:
    list[str]: A sorted list of placeholder variable names

Raises:
    ValueError: If the template is malformed or parsing fails
""")
add_chinese_doc('few_shot_prompt_template.FewShotPromptTemplate', """\
少样本提示模板类，用于构建包含示例（few-shot examples）的结构化提示。

该模板由三部分组成：前缀（prefix）、多个格式化后的示例（examples）、后缀（suffix），
并通过指定的示例模板（egs_prompt_template）渲染每个示例。支持变量部分绑定（partial binding），
允许预先填充部分变量，剩余变量在最终调用 format 时提供。

模板变量必须被明确划分为 required_vars（运行时提供）和 partial_vars（预绑定），
二者之并集必须恰好等于 prefix 与 suffix 中出现的所有变量。

Attributes:
    prefix (str): 示例前的引导文本，可包含变量占位符
    suffix (str): 示例后的指令或问题文本，可包含变量占位符
    examples (List[Dict]): 示例列表，每个示例为字典，需匹配 egs_prompt_template 的变量
    egs_prompt_template (PromptTemplate): 用于格式化每个示例的子模板
    required_vars (List[str]): 最终 format 时必须提供的变量名列表
    partial_vars (Dict[str, Any]): 预绑定的变量字典，值可以是常量或无参可调用对象
    separator_for_egs (str): 示例之间的分隔符，默认为换行符 '\\n'
""")


add_english_doc('few_shot_prompt_template.FewShotPromptTemplate', """\
A few-shot prompt template class for constructing structured prompts with examples.

The template consists of three parts: a prefix, multiple formatted examples, and a suffix.
Each example is rendered using the provided egs_prompt_template. It supports partial variable binding,
allowing some variables to be pre-filled while others are supplied at final formatting time.

All template variables (from prefix and suffix) must be exactly covered by the union of 
required_vars (provided at runtime) and partial_vars (pre-bound).

Attributes:
    prefix (str): Introductory text before examples, may contain variable placeholders
    suffix (str): Instruction or question after examples, may contain variable placeholders
    examples (List[Dict]): List of example dictionaries, each must match egs_prompt_template's variables
    egs_prompt_template (PromptTemplate): Sub-template used to format each example
    required_vars (List[str]): List of variable names that must be provided in the final format call
    partial_vars (Dict[str, Any]): Pre-bound variables; values can be constants or zero-argument callables
    separator_for_egs (str): Separator between examples, defaults to newline '\\n'
""")


add_chinese_doc('few_shot_prompt_template.FewShotPromptTemplate.format', """\
根据提供的变量值生成完整的少样本提示字符串。

必须提供所有 required_vars 中声明的变量。partial_vars 中的变量会自动应用（若为可调用对象则执行），
并覆盖 kwargs 中同名的值。每个示例通过 egs_prompt_template 格式化后，按 separator_for_egs 拼接，
最终与 prefix 和 suffix 合并并填充变量。

Args:
    **kwargs: 包含所有 required_vars 的关键字参数

Returns:
    str: 完整渲染后的提示文本

Raises:
    KeyError: 缺少 required_vars 中的变量，或模板中存在未绑定的变量
    ValueError: 示例格式化失败或最终模板渲染出错
""")


add_english_doc('few_shot_prompt_template.FewShotPromptTemplate.format', """\
Generates a complete few-shot prompt string by filling in the provided variables.

All variables listed in required_vars must be provided via kwargs. Variables in partial_vars 
are automatically applied (callable values are invoked) and will override any same-named kwargs.
Each example is formatted using egs_prompt_template, joined by separator_for_egs, and combined 
with prefix and suffix to produce the final prompt.

Args:
    **kwargs: Keyword arguments containing all required_vars

Returns:
    str: The fully rendered prompt text

Raises:
    KeyError: If any required variable is missing or an unbound variable exists in the template
    ValueError: If example formatting or final template rendering fails
""")


add_chinese_doc('few_shot_prompt_template.FewShotPromptTemplate.partial', """\
对模板进行部分变量绑定，返回一个新的 FewShotPromptTemplate 实例。

新实例中，传入的变量将从 required_vars 移至 partial_vars，后续调用 format 时无需再提供这些变量。
可用于逐步绑定变量或构建可复用的半成品模板。

Args:
    **kwargs: 要预绑定的变量名和值（值可为常量或无参函数）

Returns:
    FewShotPromptTemplate: 新的模板实例，已绑定指定变量

Raises:
    KeyError: 若 kwargs 中包含模板中不存在的变量名
""")


add_english_doc('few_shot_prompt_template.FewShotPromptTemplate.partial', """\
Partially binds variables to the template and returns a new FewShotPromptTemplate instance.

The provided variables are moved from required_vars to partial_vars in the new instance,
so they no longer need to be supplied in subsequent format calls. Useful for incrementally 
binding variables or creating reusable partially-filled templates.

Args:
    **kwargs: Variable names and values to pre-bind (values can be constants or zero-argument callables)

Returns:
    FewShotPromptTemplate: A new template instance with the specified variables bound

Raises:
    KeyError: If any variable in kwargs is not present in the template
""")


add_chinese_doc('few_shot_prompt_template.FewShotPromptTemplate.validate_variables', """\
模型验证器，在实例创建后自动调用，用于校验模板变量和示例的一致性。

执行以下检查：
1. partial_vars 中的所有键必须存在于 prefix 或 suffix 的模板变量中；
2. required_vars 与 partial_vars 不能有交集；
3. required_vars 和 partial_vars 的并集必须恰好等于 prefix 与 suffix 中出现的所有变量；
4. 每个示例字典必须包含 egs_prompt_template 所需的全部变量。

若任一检查失败，将抛出 ValueError。

此方法确保模板在使用前处于合法、自洽的状态。
""")


add_english_doc('few_shot_prompt_template.FewShotPromptTemplate.validate_variables', """\
A model validator automatically invoked after instance creation to ensure consistency 
between template variables and examples.

Performs the following checks:
1. All keys in partial_vars must appear as variables in the prefix or suffix;
2. required_vars and partial_vars must be disjoint (no overlap);
3. The union of required_vars and partial_vars must exactly match all variables in prefix and suffix;
4. Each example dictionary must contain all variables required by egs_prompt_template.

Raises ValueError if any check fails.

This method guarantees that the template is valid and self-consistent before use.
""")

# ActorPrompt.py

add_chinese_doc('ActorPrompt', '''\
提示语库模块。内置了丰富的预设提示语（Prompts），支持中英文分类，可基于角色（act）名称获取。

Args:
    lang (str): 默认语言，可选 'zh' (中文) 或 'en' (英文)。若不指定，默认为 'zh'。
''')

add_english_doc('ActorPrompt', '''\
Prompt library module. Contains a wide range of preset prompts, supporting Chinese and English categories, which can be retrieved by act (role) names.

Args:
    lang (str): Default language, optional 'zh' (Chinese) or 'en' (English). Defaults to 'zh' if not specified.
''')

add_example('ActorPrompt', '''\
    >>> from lazyllm import ActorPrompt
    >>> lib = ActorPrompt(lang='en')
    >>> # Get all available acts
    >>> acts = lib.get_all_acts()
    >>> # Get prompt for a specific act
    >>> prompt = lib.get_prompt('English Translator and Improver')
    >>> # Also callable directly
    >>> prompt = lib('English Translator and Improver')
    >>> print(prompt[:50])
    I want you to act as an English translator, spelli...
''')

add_chinese_doc('ActorPrompt.get_prompt', '''\
根据指定的角色名称和语言获取提示语。

Args:
    act (str): 角色或场景名称。
    lang (str): 语言代码 ('zh' 或 'en')。若未提供，则使用实例初始化时的语言。

**Returns:**\n
- str: 提示语内容。如果未找到则返回空字符串。
''')

add_english_doc('ActorPrompt.get_prompt', '''\
Get prompt content for a specific act and language.

Args:
    act (str): Name of the act or role.
    lang (str): Language code ('zh' or 'en'). If not provided, uses the language set during initialization.

**Returns:**\n
- str: The prompt content. Returns an empty string if not found.
''')

add_chinese_doc('ActorPrompt.get_all_acts', '''\
获取指定语言下所有支持的角色（act）列表。

Args:
    lang (str): 语言代码 ('zh' 或 'en')。若未提供，则使用实例初始化时的语言。

**Returns:**\n
- list: 包含所有可用角色名称的列表。
''')

add_english_doc('ActorPrompt.get_all_acts', '''\
Get the list of all supported acts for a specific language.

Args:
    lang (str): Language code ('zh' or 'en'). If not provided, uses the language set during initialization.

**Returns:**\n
- list: A list of all available act names.
''')

add_chinese_doc('LazyLLMPromptLibraryBase', '''\
提示语库基类，用于管理多语言的提示语集合并提供统一访问接口。

行为简介：
- 维护按语言划分的提示语字典（_prompts）。
- 提供获取单条提示语和列举某语言全部键（key）的能力。
- 在初始化时可指定实例默认语言。

主要方法：
- get_prompt(key, lang=None): 按 key 与语言获取对应提示语字符串或结构，若未提供语言则使用实例语言或默认语言。
- get_all_keys(lang=None): 列出指定语言下的所有提示语键名，若未提供语言则使用实例语言或默认语言。
''')

add_english_doc('LazyLLMPromptLibraryBase', '''\
Base prompt library class that manages multilingual prompt collections and provides a unified access API.

Overview:
- Maintains a language-keyed dictionary of prompts (_prompts).
- Allows fetching a single prompt and listing all keys for a language.
- Instance may be initialized with a default language.

Main methods:
- get_prompt(key, lang=None): Retrieve the prompt (string or structured) by key and language. Falls back to instance language or class default when lang is omitted.
- get_all_keys(lang=None): Return a list of all prompt keys for the specified language; falls back to instance/default language when omitted.
''')

add_chinese_doc('LazyLLMPromptLibraryBase.get_prompt', '''\
按键名获取指定语言的提示语。

Args:
    key (str): 提示语的键名。
    lang (str): 可选，语言代码 ('zh' 或 'en')。若未提供，使用实例语言或类默认语言。

Returns:
- str 或 dict: 返回对应的提示语（可能是字符串或结构化 dict），若未找到则抛出 ValueError。
''')

add_english_doc('LazyLLMPromptLibraryBase.get_prompt', '''\
Get the prompt for a given key and language.

Args:
    key (str): The prompt key name.
    lang (str): Optional language code ('zh' or 'en'). If omitted, instance language or class default is used.

Returns:
- str or dict: The prompt content (may be a string or structured dict). Raises ValueError if not found.
''')

add_chinese_doc('LazyLLMPromptLibraryBase.get_all_keys', '''\
列出指定语言下所有可用的提示语键名。

Args:
    lang (str): 可选，语言代码 ('zh' 或 'en')。若未提供，使用实例语言或类默认语言。

Returns:
- list: 包含所有键名的列表；若语言不受支持则返回空列表并记录警告。
''')

add_english_doc('LazyLLMPromptLibraryBase.get_all_keys', '''\
List all available prompt keys for a specified language.

Args:
    lang (str): Optional language code ('zh' or 'en'). If omitted, instance language or class default is used.

Returns:
- list: A list of key names. Returns an empty list and logs a warning if the language is unsupported.
''')

add_chinese_doc('DataPrompt', '''\
结构化提示语库(用于数据处理模块)，支持将提示语以字典形式存储（包含 system/user/tools/history/extra_keys 等字段），并通过 ChatPrompter 构建可用的对话提示器。

特点：
- 支持以类方法 add_prompt 动态添加提示语。
- __call__ 可返回 ChatPrompter 或原始字典（return_raw=True）。
''')

add_english_doc('DataPrompt', '''\
Structured prompt library (for data processing modules) that stores prompts as dictionaries (including fields like system/user/tools/history/extra_keys) and builds ChatPrompter instances for use.

Features:
- Prompts can be added dynamically via the class method add_prompt.
- __call__ returns a ChatPrompter by default or the raw prompt dict when return_raw=True.
''')

add_chinese_doc('DataPrompt.__call__', '''\
根据 key 和语言构建并返回 ChatPrompter，或在 return_raw=True 时返回原始提示语字典。

Args:
    key (str): 提示语键名。
    lang (str): 可选，语言代码。
    return_raw (bool): 若为 True，返回原始字典；否则返回 ChatPrompter 实例。

Returns:
- ChatPrompter 或 dict: 根据 return_raw 决定返回类型。若未找到会抛出 ValueError。
''')

add_english_doc('DataPrompt.__call__', '''\
Build and return a ChatPrompter for the given key and language, or return the raw prompt dict when return_raw=True.

Args:
    key (str): Prompt key name.
    lang (str): Optional language code.
    return_raw (bool): If True, return the raw dict; otherwise return a ChatPrompter instance.

Returns:
- ChatPrompter or dict: Type depends on return_raw. Raises ValueError if key not found.
''')

add_chinese_doc('DataPrompt.add_prompt', '''\
以结构化形式添加或覆盖一个提示语条目。

Args:
    act (str): 提示语键名。
    system_prompt (str): 可选，system 消息内容。
    user_prompt (str): 可选，user 消息内容。
    tools (any): 可选，工具描述或配置。
    history (any): 可选，历史上下文。
    extra_keys (any): 可选，额外字段。
    lang (str): 目标语言代码，默认 'zh'。

注意：至少要提供 system_prompt 或 user_prompt。
''')

add_english_doc('DataPrompt.add_prompt', '''\
Add or overwrite a prompt entry in structured form.

Args:
    act (str): Prompt key name.
    system_prompt (str): Optional system message content.
    user_prompt (str): Optional user message content.
    tools (any): Optional tools description/config.
    history (any): Optional history context.
    extra_keys (any): Optional extra fields.
    lang (str): Target language code, default 'zh'.

Note: At least one of system_prompt or user_prompt must be provided.
''')

add_example('DataPrompt', '''\
>>> from lazyllm import DataPrompt
>>> # 动态添加一个结构化提示（也可在加载时来自文件）
>>> DataPrompt.add_prompt(
...     act='simple_summarize',
...     system_prompt='You are a concise summarizer.',
...     user_prompt='Please summarize the following text: {text}',
...     lang='en'
... )
>>> lib = DataPrompt(lang='en')
>>> # 获取 ChatPrompter 实例
>>> prompter = lib('simple_summarize')
>>> print(type(prompter))
<class 'lazyllm.components.prompter.chatPrompter.ChatPrompter'>
>>> # 获取原始字典
>>> raw = lib('simple_summarize', return_raw=True)
>>> print(raw['system'][:20])
You are a concise su
''')
