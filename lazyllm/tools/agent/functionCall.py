from lazyllm.module import ModuleBase
from lazyllm.common import package
from lazyllm.components import ChatPrompter, FunctionCallFormatter
from lazyllm import pipeline, switch
from .toolsManager import ToolManager
from typing import List, Any, Dict
from collections.abc import Iterator


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
        self._tools_manager = ToolManager(tools)
        self._impl = self._build_pipeline(llm)

    def _process_stream(self, data):
        isFC = False
        if isinstance(data, Iterator):
            output_content = {}
            content = ""
            tool_calls = None
            for line in data:
                isFC = line[0] if line[0] else isFC
                role = line[1].get("role", None)
                if role:
                    output_content['role'] = role
                content += line[1].get('content', "")
                if line[0]:
                    tool_calls = line[2]
                    output_content.update(line[1])
                    if "index" in output_content['tool_calls'][0]:
                        output_content['tool_calls'][0].pop('index')
                output_content['content'] = content
            return package(isFC, output_content, tool_calls)

        else:
            return data

    def _build_pipeline(self, llm):
        with pipeline() as ppl:
            ppl.m1 = lambda x, y: package(x, y, self._tools_manager.tools_description)
            ppl.m2 = llm.share(prompt=self._prompter, formatter=self._formatter)
            ppl.m3 = self._process_stream
            # ppl.m4 = switch((lambda input, llm_output: not llm_output[0]),
            #                 (lambda input, llm_output: package(not llm_output[0], llm_output[1]['content'], input[1])),
            #                 (lambda input, llm_output: llm_output[0]),
            #                 self._tools_manager) | bind(ppl.input, ppl.m3)
            with switch().bind(ppl.input, ppl.m3) as ppl.sw:
                ppl.sw.case[(lambda input, llm_output: not llm_output[0]),
                            (lambda input, llm_output: package(not llm_output[0],
                                                               llm_output[1]['content'], input[1]))]
                ppl.sw.case[(lambda input, llm_output: llm_output[0]), self._tools_manager]

        return ppl

    def forward(self, input: str, history: List[Dict[str, Any]]):
        return self._impl(input, history)
