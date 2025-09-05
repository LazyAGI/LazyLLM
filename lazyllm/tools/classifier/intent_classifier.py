from lazyllm.module import ModuleBase
from lazyllm.components import AlpacaPrompter
from lazyllm import pipeline, globals, switch
from lazyllm.tools.utils import chat_history_to_str
from lazyllm.prompts import IntentClassifierPrompts
from typing import Dict, Union, Any, List, Optional
import json


class IntentClassifier(ModuleBase):
    def __init__(self, llm, intent_list: list = None,
                 *, prompt: str = '', constrain: str = '', attention: str = '',
                 examples: Optional[list[list[str, str]]] = None,
                 prompts: Optional[IntentClassifierPrompts] = None,
                 return_trace: bool = False) -> None:
        super().__init__(return_trace=return_trace)
        self._intent_list = intent_list or []
        self._llm = llm
        self._prompt, self._constrain, self._attention, self._examples = prompt, constrain, attention, examples or []

        # Initialize prompts
        self._prompts = prompts or IntentClassifierPrompts()

        if self._intent_list:
            self._init()

    def _init(self):
        def choose_prompt():
            # Use chinese prompt if intent elements have chinese character, otherwise use english version
            for ele in self._intent_list:
                for ch in ele:
                    # chinese unicode range
                    if "\u4e00" <= ch <= "\u9fff":
                        return self._prompts.chinese
            return self._prompts.template

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
        history: List[Union[List[str], Dict[str, Any]]] = [],  # noqa B006
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
