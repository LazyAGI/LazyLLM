from lazyllm.module import ModuleBase
from lazyllm.common import package, bind
from lazyllm.components import ChatPrompter, FunctionCallFormatter
from lazyllm import pipeline, switch
from .toolsManager import ToolManager
from typing import List, Any, Dict


def function_call_hook(input: Dict[str, Any], history: List[Dict[str, Any]], tools: List[Dict[str, Any]], label: Any):
    if isinstance(input, dict):
        item = input.get("history", None)
        if isinstance(item, str):
            history.append({"role": "user", "content": item})
        elif isinstance(item, dict):
            history.append(item)
        elif isinstance(item, list):
            history.extend(item)

        if item:
            input.pop("history")

    return input, history, tools, label

class FunctionCall(ModuleBase):
    def __init__(self, llm, tools: List[str], *, return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        self._prompter = ChatPrompter(instruction="Don't make assumptions about what values to plug into functions.\
                                        Ask for clarification if a user request is ambiguous.\n", show=False)\
            .pre_hook(function_call_hook)
        self._formatter = FunctionCallFormatter()
        self._llm = llm.share(prompt=self._prompter if self._prompter else None,
                              formatter=self._formatter if self._formatter else None)
        self._tools_manager = ToolManager(tools)
        self._agent_pipeline = self._build_pipeline()

    def _build_pipeline(self):
        with pipeline() as ppl:
            ppl.m1 = lambda x, y: package(x, y, self._tools_manager.tools_description)
            ppl.m2 = self._llm
            ppl.m3 = switch((lambda input, llm_output: not llm_output[0]),
                            (lambda input, llm_output: package(not llm_output[0], llm_output[1]['content'], input[1])),
                            (lambda input, llm_output: llm_output[0]),
                            self._tools_manager) | bind(ppl.input, ppl.m2)

        return ppl

    def forward(self, input: str, history: List[Dict[str, Any]]):
        return self._agent_pipeline(input, history)
