# flake8: noqa E501
from . import utils
import functools
import lazyllm

add_chinese_doc = functools.partial(utils.add_chinese_doc, module=lazyllm.prompt_templates)
add_english_doc = functools.partial(utils.add_english_doc, module=lazyllm.prompt_templates)
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
