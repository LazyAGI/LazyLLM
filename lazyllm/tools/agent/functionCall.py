from lazyllm.module import ModuleBase
from lazyllm.common import package
from lazyllm.components import ChatPrompter, FunctionCallFormatter
from lazyllm import pipeline, switch, loop
from .toolsManager import ToolManager
from typing import List, Any, Dict


def function_call_hook(input: Dict[str, Any], history: List[Dict[str, Any]], tools: List[Dict[str, Any]], label: Any):
    if isinstance(input, dict):
        item = input.pop("history", None)
        if isinstance(item, list):
            history.extend(item)
        else:
            raise TypeError(f"The history field in the input currently only supports the list type, \
                              not supports {type(item)}")

    return input, history, tools, label

class FunctionCall(ModuleBase):
    def __init__(self, llm, tools: List[str], *, return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        self._tools_manager = ToolManager(tools)
        self._prompter = ChatPrompter(instruction="Don't make assumptions about what values to plug into functions.\
                                        Ask for clarification if a user request is ambiguous.\n",
                                      tools=self._tools_manager.tools_description)\
            .pre_hook(function_call_hook)
        self._formatter = FunctionCallFormatter()
        self._impl = self._build_pipeline(llm)

    def _build_pipeline(self, llm):
        if llm.model_type == "QwenModule" and llm.stream is True:
            raise ValueError("The qwen platform does not currently support stream function calls.")
        with pipeline() as ppl:
            ppl.m1 = llm.share(prompt=self._prompter, formatter=self._formatter)

            with switch(judge_on_input=False).bind(input=ppl.input) as ppl.sw:
                ppl.sw.case[(lambda isFC: not isFC),
                            (lambda out, input: package(True, out[0]['content'], input[1]))]
                ppl.sw.case[(lambda isFC: isFC), self._tools_manager]

        return ppl

    def forward(self, input: str, history: List[Dict[str, Any]]):
        return self._impl(input, history)

class FunctionCallAgent(ModuleBase):
    def __init__(self, llm, tools: List[str], iterator_count: int = 5, return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        assert llm and tools, "llm and tools cannot be empty."
        with pipeline() as self._agent:
            self._agent.lp = loop(FunctionCall(llm, tools),
                                  stop_condition=lambda x: x,
                                  count=iterator_count,
                                  judge_on_input=False)
            self._agent.post_action = lambda output, history: output

    def forward(self, query, history):
        return self._agent(query, history)
