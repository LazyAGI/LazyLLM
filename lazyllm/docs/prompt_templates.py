# flake8: noqa E501
from . import utils
import functools
import lazyllm

add_chinese_doc = functools.partial(utils.add_chinese_doc, module=lazyllm.prompt_templates)
add_english_doc = functools.partial(utils.add_english_doc, module=lazyllm.prompt_templates)
add_example = functools.partial(utils.add_example, module=lazyllm.prompt_templates)

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