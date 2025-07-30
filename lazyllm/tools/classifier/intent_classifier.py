from lazyllm.module import ModuleBase
from lazyllm.components import AlpacaPrompter
from lazyllm import pipeline, globals, switch
from lazyllm.tools.utils import chat_history_to_str
from typing import Dict, Union, Any, List
import json


en_prompt_classifier_template = """
## role：Intent Classifier
You are an intent classification engine responsible for analyzing user input text based on dialogue information and determining a unique intent category.{user_prompt}

## Constrains:
You only need to reply with the name of the intent. Do not output any additional fields and do not translate it.{user_constrains}

## Attention:
{attention}

## Text Format
The input text is in JSON format, where "human_input" contains the user's raw input and "intent_list" contains a list of all intent names.

## Example
User: {{"human_input": "What’s the weather like in Beijing tomorrow?", "intent_list": ["Check Weather", "Search Engine Query", "View Surveillance", "Report Summary", "Chat"]}}
Assistant: Check Weather
{user_examples}

## Conversation History
The chat history between the human and the assistant is stored within the <histories></histories> XML tags below.
<histories>
{history_info}
</histories>

Input text is as follows:
"""  # noqa E501

ch_prompt_classifier_template = """
## role：意图分类器
你是一个意图分类引擎，负责根据对话信息分析用户输入文本并确定唯一的意图类别。{user_prompt}

## 限制:
你只需要回复意图的名字即可，不要额外输出其他字段，也不要进行翻译。{user_constrains}

## 注意:
{attention}

## 文本格式
输入文本为JSON格式，"human_input"中内容为用户的原始输入，"intent_list"为所有意图名列表

## 示例
User: {{"human_input": "北京明天天气怎么样？", "intent_list": ["查看天气", "搜索引擎检索", "查看监控", "周报总结", "聊天"]}}
Assistant:  查看天气
{user_examples}

## 历史对话
人类和助手之间的聊天记录存储在下面的 <histories></histories> XML 标记中。
<histories>
{history_info}
</histories>

输入文本如下:
"""  # noqa E501


class IntentClassifier(ModuleBase):
    """IntentClassifier is an intent recognizer based on a language model that identifies predefined intents based on user-provided input text and conversational context.
It can handle intent lists and ensures accurate intent recognition through preprocessing and postprocessing steps.

Arguments:
    llm: A language model object used for intent recognition, which can be of type OnlineChatModule or TrainableModule.
    intent_list (list): A list of strings containing all possible intents. This list can include intents in either Chinese or English.
    prompt (str): User-attached prompt words.
    constrain (str): User-attached constrain words.
    examples (list[list]): extra examples，format is `[[query, intent], [query, intent], ...]`.
    return_trace (bool, optional): If set to True, the results will be recorded in the trace. Defaults to False.


Examples:
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
    """
    def __init__(self, llm, intent_list: list = None,
                 *, prompt: str = '', constrain: str = '', attention: str = '',
                 examples: list[list[str, str]] = [], return_trace: bool = False) -> None:
        super().__init__(return_trace=return_trace)
        self._intent_list = intent_list or []
        self._llm = llm
        self._prompt, self._constrain, self._attention, self._examples = prompt, constrain, attention, examples
        if self._intent_list:
            self._init()

    def _init(self):
        def choose_prompt():
            # Use chinese prompt if intent elements have chinese character, otherwise use english version
            for ele in self._intent_list:
                for ch in ele:
                    # chinese unicode range
                    if "\u4e00" <= ch <= "\u9fff":
                        return ch_prompt_classifier_template
            return en_prompt_classifier_template

        example_template = '\nUser: {{{{"human_input": "{inp}", "intent_list": {intent}}}}}\nAssistant: {label}\n'
        examples = ''.join([example_template.format(
            inp=input, intent=self._intent_list, label=label) for input, label in self._examples])
        prompt = choose_prompt().replace(
            '{user_prompt}', f' {self._prompt}').replace('{attention}', self._attention).replace(
            '{user_constrains}', f' {self._constrain}').replace('{user_examples}', f' {examples}')
        self._llm = self._llm.share(prompt=AlpacaPrompter(dict(system=prompt, user='${input}')
                                                          ).pre_hook(self.intent_promt_hook)).used_by(self._module_id)
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
        input = input.strip()
        return input if input in self._intent_list else self._intent_list[0]

    def forward(self, input: str, llm_chat_history: List[Dict[str, Any]] = None):
        if llm_chat_history is not None and self._llm._module_id not in globals["chat_history"]:
            globals["chat_history"][self._llm._module_id] = llm_chat_history
        return self._impl(input)

    def __enter__(self):
        assert not self._intent_list, 'Intent list is already set'
        self._sw = switch()
        self._sw.__enter__()
        return self

    @property
    def case(self):
        return switch.Case(self)

    @property
    def submodules(self):
        submodule = []
        if isinstance(self._impl, switch):
            self._impl.for_each(lambda x: isinstance(x, ModuleBase), lambda x: submodule.append(x))
        return super().submodules + submodule

    # used by switch.Case
    def _add_case(self, cond, func):
        assert isinstance(cond, str), 'intent must be string'
        self._intent_list.append(cond)
        self._sw.case[cond, func]

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._sw.__exit__(exc_type, exc_val, exc_tb)
        self._init()
        self._sw._set_conversion(self._impl)
        self._impl = self._sw
