from typing import Union
from ...module import ModuleBase, TrainableModule, OnlineChatModuleBase

en_qustion_rewrite_prompt = """
## Task
You are a professional RAG retrieval assistant tasked with rewriting user questions
to make them clearer and more understandable, facilitating subsequent RAG data retrieval.
Please rewrite the user's input question according to the following guidelines.

## Rewrite Instructions
Follow these principles when rewriting the above question:
1. Simplify language: remove unnecessary complex words and use simple, direct phrasing.
2. Clarify intent: ensure the core purpose of the question is clearly expressed.
3. Add details: if the original question lacks key information, supply necessary context or constraints.
4. Shift perspective: try posing the same question from different angles to spark new thinking paths.
5. Break into steps: when applicable, decompose a question into smaller sub-questions for
easier comprehension and answering.

User Require
'''{prompt}'''

Do not give me any other content, only output rewritten questions
"""


ch_qustion_rewrite_prompt = """
## 任务
你是一个专业的RAG检索助手，负责将用户的问题进行改写，使其更清晰易懂，便于后续的RAG数据召回。请根据以下指导原则对用户输入的问题进行改写。

## 改写的指令
根据以下指导原则对上述问题进行改写：
1. **简化语言**：去除不必要的复杂词汇，使用简单直接的语言。
2. **明确意图**：确保问题的核心意图被清楚地表达出来。
3. **增加细节**：如果原问题缺少关键信息，请补充必要的上下文或限制条件。
4. **变换视角**：尝试从不同角度提出同样的问题，可能会引发新的思考路径。
5. **分步提问**：如果适用，将一个问题分解为几个小问题，以便更容易理解。

## 用户要求
'''{prompt}'''

不要输出任何其他内容，仅输出改写后的问题
"""
class QustionRewrite(ModuleBase):
    """Question Rewrite Module.

This module rewrites or reformulates a user query using a language model. It supports both string and list output formats based on the formatter.

`__init__(self, base_model, rewrite_prompt="", formatter="str")`
Initializes the question rewrite module with a prompt and model.

Args:
    base_model (Union[str, TrainableModule, OnlineChatModuleBase]): A path string or initialized model for question rewriting.
    rewrite_prompt (str): Custom prompt to guide the rewrite behavior.
    formatter (str): Output format type; either "str" or "list".


Examples:
    >>> from lazyllm.components import QustionRewrite
    >>> rewriter = QustionRewrite(base_model="chatglm", rewrite_prompt="请将问题改写为更适合检索的形式", formatter="list")
    >>> result = rewriter("中国的最高山峰是什么？")
    >>> print(result)
    ... ['中国的最高山峰是哪一座？', '中国海拔最高的山是什么？']
    """
    def __init__(
        self,
        base_model: Union[str, TrainableModule, OnlineChatModuleBase],
        rewrite_prompt: str = "",
        formatter: str = "str",
    ):
        super().__init__()
        self._prompt = self.choose_prompt(rewrite_prompt).format(prompt=rewrite_prompt)
        if isinstance(base_model, str):
            self._m = TrainableModule(base_model).start().prompt(self._prompt)
        else:
            self._m = base_model.share(self._prompt)
        self.formatter = formatter

    def choose_prompt(self, prompt: str):
        """
Choose the appropriate prompt template based on the language of the input prompt.

This method analyzes the input prompt string and determines whether to use the Chinese or English prompt template. It checks each character in the prompt string and if any character falls within the Chinese Unicode range (\u4e00-\u9fff), it returns the Chinese prompt template; otherwise, it returns the English prompt template.

Args:
    prompt (str): The input prompt string to be analyzed for language detection.

**Returns:**

- str: The selected prompt template string (either Chinese or English version).


Examples:
    
    >>> from lazyllm.tools.actors.qustion_rewrite import QustionRewrite
    
    # Example 1: English prompt (no Chinese characters)
    >>> rewriter = QustionRewrite("gpt-3.5-turbo")
    >>> prompt_template = rewriter.choose_prompt("How to implement machine learning?")
    >>> print("Template contains Chinese:", "中文" in prompt_template)
    Template contains Chinese: False
    
    # Example 2: Chinese prompt (contains Chinese characters)
    >>> prompt_template = rewriter.choose_prompt("如何实现机器学习？")
    >>> print("Template contains Chinese:", "中文" in prompt_template)
    Template contains Chinese: True
    
    # Example 3: Mixed language prompt (contains Chinese characters)
    >>> prompt_template = rewriter.choose_prompt("What is 机器学习?")
    >>> print("Template contains Chinese:", "中文" in prompt_template)
    Template contains Chinese: True
    """
        # Use chinese prompt if intent elements have chinese character, otherwise use english version
        for ele in prompt:
            # chinese unicode range
            if "\u4e00" <= ele <= "\u9fff":
                return ch_qustion_rewrite_prompt
        return en_qustion_rewrite_prompt

    def forward(self, *args, **kw):
        res = self._m(*args, **kw)
        if self.formatter == "list":
            return list(filter(None, res.split('\n')))
        else:
            return res
