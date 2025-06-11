from typing import Union
from ...module import ModuleBase, TrainableModule, OnlineChatModuleBase
import re

en_code_generate_prompt = """
## Task
"You are a Python Programming Master,Please handle programming tasks according to the following requirements:
1. Code must fully comply with PEP8 standards
2. Generate Google-style documentation in code block to describe the function's purpose, inputs, and outputs
3. Prioritize using the python standard library
4. Strictly follow the output format and only output Python code blocks

##User Require
'''{prompt}'''

## output format
```python
import ...

def [function name](param) -> return type:
    \"\"\"documentation\"\"\"
    # code

"""


ch_code_generate_prompt = """
## 任务
你是一个Python编程专家，请按以下要求处理编程任务：
1. 代码完全遵循PEP8规范
2. 在代码块内生成google风格的文档,详细描述函数的功能以及输入和输出
3. 优先使用python标准库
4. 严格遵照输出格式，仅输出python代码块

## 用户要求
'''{prompt}'''

## 输出格式
```python
import ...

def [函数名](参数) -> 返回类型:
    \"\"\"文档\"\"\"
    # 代码

"""

class CodeGenerator(ModuleBase):
    def __init__(
        self,
        base_model: Union[str, TrainableModule, OnlineChatModuleBase],
        prompt: str = "",
    ):
        super().__init__()
        self._prompt = self.choose_prompt(prompt).format(prompt=prompt)
        if isinstance(base_model, str):
            self._m = TrainableModule(base_model).start().prompt(self._prompt)
        else:
            self._m = base_model.share(self._prompt)

    def choose_prompt(self, prompt: str):
        # Use chinese prompt if intent elements have chinese character, otherwise use english version
        for ele in prompt:
            # chinese unicode range
            if "\u4e00" <= ele <= "\u9fff":
                return ch_code_generate_prompt
        return en_code_generate_prompt

    def forward(self, *args, **kw):
        res = self._m(*args, **kw)
        pattern = r"```python(.*?)\n```"
        matches = re.findall(pattern, res, re.DOTALL)
        if len(matches) > 0:
            return matches[0]
        return res
