from lazyllm.module import ModuleBase
from lazyllm.common import package
from lazyllm.components import ChatPrompter, FunctionCallFormatter
from lazyllm import pipeline, switch, loop, globals, root
from .toolsManager import ToolManager
from typing import List, Any, Dict


def function_call_hook(input: Dict[str, Any], history: List[Dict[str, Any]], tools: List[Dict[str, Any]], label: Any):
    if isinstance(input, list):
        for idx in range(len(input[:-1])):
            history.append(input[idx] if isinstance(input[idx], dict) else {"role": "user", "content": input[idx]})
        input = {"input": input[-1]} if isinstance(input[-1], (dict, list)) else input[-1]
    return input, history, tools, label

class FunctionCall(ModuleBase):
    def __init__(self, llm, tools: List[str], *, return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        if llm._model_type == "QwenModule" and llm._stream is True:
            raise ValueError("The qwen platform does not currently support stream function calls.")
        self._tools_manager = ToolManager(tools)
        self._prompter = ChatPrompter(instruction=("Don't make assumptions about what values to plug into functions."
                                                   "Ask for clarification if a user request is ambiguous.\n"),
                                      tools=self._tools_manager.tools_description)\
            .pre_hook(function_call_hook)
        self._llm = llm.share(prompt=self._prompter, formatter=FunctionCallFormatter())

        with switch(judge_on_full_input=False) as sw:
            sw.case[True, self._tools_manager]
            sw.case[False, lambda out, input: package(True, out[0]['content'])]
        self._impl = pipeline(self._llm, sw.bind(input=root.input))

    def forward(self, input: str, llm_chat_history: List[Dict[str, Any]] = None):
        if llm_chat_history is not None and self._llm._module_id not in globals['chat_history']:
            globals['chat_history'][self._llm._module_id] = llm_chat_history
        return self._impl(input)

class FunctionCallAgent(ModuleBase):
    def __init__(self, llm, tools: List[str], max_retries: int = 5, return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        self._max_retries = max_retries
        assert llm and tools, "llm and tools cannot be empty."
        with pipeline() as self._agent:
            self._agent.lp = loop(FunctionCall(llm, tools), stop_condition=lambda x: x,
                                  count=self._max_retries, judge_on_full_input=False)
            self._agent.post_action = lambda output: output[0] if isinstance(output, tuple) else output

    def forward(self, query: str, llm_chat_history: List[Dict[str, Any]] = None):
        ret = self._agent(query, llm_chat_history)
        return ret if isinstance(ret, str) else (_ for _ in ()).throw(
            ValueError(f"After retrying {self._max_retries} times, the function call agent still "
                       "failed to call successfully."))
