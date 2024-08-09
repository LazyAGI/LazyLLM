# flake8: noqa E501
from lazyllm.module import ModuleBase
from lazyllm.components import AlpacaPrompter
from lazyllm import pipeline, globals
from lazyllm.tools.utils import chat_history_to_str
from typing import Dict, Union, Any, List
import string
import json


en_prompt_classifier_template = """
## role：Intent Classifier
You are an intent classification engine responsible for analyzing user input text based on dialogue information and determining a unique intent category.

## Constrains:
You only need to reply with the name of the intent. Do not output any additional fields and do not translate it.

## Text Format
The input text is in JSON format, where "human_input" contains the user's raw input and "intent_list" contains a list of all intent names.

## Example
User: {{"human_input": "What’s the weather like in Beijing tomorrow?", "intent_list": ["Check Weather", "Search Engine Query", "View Surveillance", "Report Summary", "Chat"]}}
Assistant: Check Weather

## Conversation History
The chat history between the human and the assistant is stored within the <histories></histories> XML tags below.
<histories>
{history_info}
</histories>

Input text is as follows:
{input}
"""

ch_prompt_classifier_template = """
## role：意图分类器
你是一个意图分类引擎，负责根据对话信息分析用户输入文本并确定唯一的意图类别

## Constrains:
你只需要回复意图的名字即可，不要额外输出其他字段，也不要进行翻译

## 文本格式
输入文本为JSON格式，"human_input"中内容为用户的原始输入，"intent_list"为所有意图名列表

## 示例
User: {{"human_input": "北京明天天气怎么样？", "intent_list": ["查看天气", "搜索引擎检索", "查看监控", "周报总结", "聊天"]}}
Assistant:  查看天气

## 历史对话
人类和助手之间的聊天记录存储在下面的 <histories></histories> XML 标记中。
<histories>
{history_info}
</histories>

输入文本如下:
${input}
"""


class IntentClassifier(ModuleBase):
    def __init__(self, llm, intent_list: list, return_trace: bool = False) -> None:
        super().__init__(return_trace=return_trace)

        def choose_prompt():
            # Use chinese prompt if intent elements have chinese character, otherwise use english version
            for ele in intent_list:
                for ch in ele:
                    # chinese unicode range
                    if "\u4e00" <= ch <= "\u9fff":
                        return ch_prompt_classifier_template
            return en_prompt_classifier_template

        self._intent_list = intent_list
        self._prompter = AlpacaPrompter(choose_prompt()).pre_hook(self.intent_promt_hook)
        self._llm = llm.share(prompt=self._prompter)
        self._impl = pipeline(self._llm, self.post_process_result)

    def intent_promt_hook(
        self,
        input: Union[str, List, Dict[str, str], None] = None,
        history: List[Union[List[str], Dict[str, Any]]] = [],
        tools: Union[List[Dict[str, Any]], None] = None,
        label: Union[str, None] = None,
    ):
        input_json = {}
        if isinstance(input, str):
            input_json = {"human_input": input, "intent_list": self._intent_list}
        else:
            raise ValueError(f"Unexpected type for input: {type(input)}")

        history_info = chat_history_to_str(history)
        history = []
        input_text = json.dumps(input_json, ensure_ascii=False)
        return dict(history_info=history_info, input=input_text), history, tools, label

    def post_process_result(self, input):
        return input if input in self._intent_list else self._intent_list[0]

    def forward(self, input: str, llm_chat_history: List[Dict[str, Any]] = None):
        if llm_chat_history is not None and self._llm._module_id not in globals["chat_history"]:
            globals["chat_history"][self._llm._module_id] = llm_chat_history
        return self._impl(input)
