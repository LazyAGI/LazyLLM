import json
import lazyllm
from lazyllm.module import ModuleBase
from lazyllm.common import package, bind
from lazyllm.components import ChatPrompter, FunctionCallFormatter
from lazyllm import pipeline, switch
from .toolsManager import ToolManager
from typing import List, Any, Dict, Union


class FunctionCall(ModuleBase):
    def __init__(self, llm, tools: List[str], *, return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        self._prompter = ChatPrompter(instruction="Don't make assumptions about what values to plug into functions.\
                                        Ask for clarification if a user request is ambiguous.\n", show=False)
        self._formatter = FunctionCallFormatter()
        self._llm = llm
        self._tools_manager = ToolManager(tools)
        self._agent_pipeline = self._build_pipeline()

    def _encapsulation_input(self, input: Union[str, Dict[str, Any]], history: List[Dict[str, Any]]):
        return package(input, history, self._tools_manager.tools_description)

    def _post_process_result(self, data: tuple, original_input: tuple):
        isFinish = not data[0]
        return package(isFinish, data[1]['content'], original_input[1])

    def _validate_tool(self, tool_name: str, tool_arguments: Dict[str, Any]):
        # Does the tool exists?
        tool = self._tools_manager.tools_info.get(tool_name, None)
        if tool:
            return tool.validate_parameters(tool_arguments)

        return False

    def _fc_process(self, data: tuple, input: tuple, verbose: bool = False):
        lazyllm.LOG.info(f"tool_call: {data},\n input: {input}")
        isFinish = not data[0]
        history = input[1]
        if isinstance(input[0], str):
            history.append({'role': 'user', 'content': input[0]})
        elif isinstance(input[0], dict):
            assert len(input[0]) <= 1, f"Unexpected keys found in input: {list(input.keys())}"
            if input:
                history.append(list(input[0].values())[0])
                # item = list(input[0].values())[0]
                # if item.get("role", "") == "tool":
                #     return package(data[0], data[1]["content"], history)
                # else:
                #     history.append(item)
        else:
            raise TypeError(f"The input type only supports str and dict, not {type(input[0])}")
        history.append(data[1])
        tool_calls = data[2]
        if not tool_calls or len(tool_calls) == 0:
            # single function call
            return package(isFinish, f"{tool_calls} is not a valid parameter.", history)
        elif len(tool_calls) == 1:
            # single function call
            tool_call = tool_calls[0]
            # validate parameters
            isVal = self._validate_tool(tool_call['name'], tool_call['tool_input'])
            if isVal:
                ret = self._tools_manager(tool_call["name"], tool_call["tool_input"])
                tool_call.pop("tool_input")
                tool_call['content'] = json.dumps(ret, ensure_ascii=False) if isinstance(ret, dict) else ret
                tool_call['role'] = 'tool'
                return package(isFinish, tool_call, history)

            else:
                # Parameter error
                return package(isFinish, f"{tool_call} parameters error.", history)
        else:
            # multi function call
            raise TypeError("Multiple function calls are not yet implemented.")

    def _build_pipeline(self):
        with pipeline() as ppl:
            ppl.m1 = self._encapsulation_input
            ppl.m2 = self._llm.prompt(self._prompter if self._prompter else None)\
                              .formatter(self._formatter if self._formatter else None)
            ppl.m3 = switch((lambda x, y: not x[0]),
                            self._post_process_result,
                            (lambda x, y: x[0]),
                            self._fc_process) | bind(ppl.m2, ppl.input)

        return ppl

    def forward(self, input: str, history: List[Dict[str, Any]]):
        return self._agent_pipeline(input, history)
