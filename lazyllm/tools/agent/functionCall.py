from lazyllm.module import ModuleBase, OnlineChatModule
from lazyllm.components import ChatPrompter
from .functionCallFormatter import FunctionCallFormatter
from lazyllm import pipeline, ifs, loop, globals, bind, LOG
import json5 as json
from .toolsManager import ToolManager
from typing import List, Any, Dict, Union


def function_call_hook(input: Dict[str, Any], history: List[Dict[str, Any]], tools: List[Dict[str, Any]], label: Any):
    if isinstance(input, list):
        for idx in range(len(input[:-1])):
            data = input[idx]
            if isinstance(data, str):
                history.append({"role": "user", "content": data})
            elif isinstance(data, dict):
                history.append(data)
            else:
                history.extend(data)
        input = {"input": input[-1]} if isinstance(input[-1], (dict, list)) and "input" not in input[-1] else input[-1]
    return input, history, tools, label

FC_PROMPT_LOCAL = """# Tools

## You have access to the following tools:
## When you need to call a tool, please insert the following command in your reply, \
which can be called zero or multiple times according to your needs:

{tool_start_token}The tool to use, should be one of tools list.
{tool_args_token}The input of the tool. The output format is: {"input1": param1, "input2": param2}. Can only return json.
{tool_end_token}End of tool."""

FC_PROMPT_ONLINE = ("Don't make assumptions about what values to plug into functions."
                    "Ask for clarification if a user request is ambiguous.\n")

class FunctionCall(ModuleBase):
    def __init__(self, llm, tools: List[str], *, return_trace: bool = False, _prompt: str = None):
        super().__init__(return_trace=return_trace)
        if isinstance(llm, OnlineChatModule) and llm.series == "QWEN" and llm._stream is True:
            raise ValueError("The qwen platform does not currently support stream function calls.")
        if _prompt is None:
            _prompt = FC_PROMPT_ONLINE if isinstance(llm, OnlineChatModule) else FC_PROMPT_LOCAL

        self._tools_manager = ToolManager(tools)
        self._prompter = ChatPrompter(instruction=_prompt, tools=self._tools_manager.tools_description)\
            .pre_hook(function_call_hook)
        self._llm = llm.share(prompt=self._prompter, format=FunctionCallFormatter())
        with pipeline() as self._impl:
            self._impl.m1 = self._llm
            self._impl.m2 = self._parser
            self._impl.m3 = ifs(lambda x: isinstance(x, list), self._tools_manager, lambda out: out)
            self._impl.m4 = self._tool_post_action | bind(input=self._impl.input, llm_output=self._impl.m1)

    def _parser(self, llm_output: Union[str, List[Dict[str, Any]]]):
        LOG.info(f"llm_output: {llm_output}")
        if isinstance(llm_output, list):
            res = []
            for item in llm_output:
                if isinstance(item, str):
                    continue
                arguments = item.get('function', {}).get('arguments', '')
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                res.append({"name": item.get('function', {}).get('name', ''), 'arguments': arguments})
            return res
        elif isinstance(llm_output, str):
            return llm_output
        else:
            raise TypeError(f"The {llm_output} type currently is only supports `list` and `str`,"
                            f" and does not support {type(llm_output)}.")

    def _tool_post_action(self, output: Union[str, List[str]], input: Union[str, List],
                          llm_output: List[Dict[str, Any]]):
        if isinstance(output, list):
            ret = []
            if isinstance(input, str):
                ret.append(input)
            elif isinstance(input, list):
                ret.append(input[-1])
            else:
                raise TypeError(f"The input type currently only supports `str` and `list`, "
                                f"and does not support {type(input)}.")

            content = "".join([item for item in llm_output if isinstance(item, str)])
            llm_output = [item for item in llm_output if not isinstance(item, str)]
            ret.append({"role": "assistant", "content": content, "tool_calls": llm_output})
            ret.append([{"role": "tool", "content": out, "tool_call_id": llm_output[idx]["id"],
                         "name": llm_output[idx]["function"]["name"]}
                        for idx, out in enumerate(output)])
            LOG.info(f"functionCall result: {ret}")
            return ret
        elif isinstance(output, str):
            return output
        else:
            raise TypeError(f"The {output} type currently is only supports `list` and `str`,"
                            f" and does not support {type(output)}.")

    def forward(self, input: str, llm_chat_history: List[Dict[str, Any]] = None):
        globals['chat_history'].setdefault(self._llm._module_id, [])
        if llm_chat_history is not None:
            globals['chat_history'][self._llm._module_id] = llm_chat_history
        return self._impl(input)

class FunctionCallAgent(ModuleBase):
    def __init__(self, llm, tools: List[str], max_retries: int = 5, return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        self._max_retries = max_retries
        self._agent = loop(FunctionCall(llm, tools),
                           stop_condition=lambda x: isinstance(x, str), count=self._max_retries)

    def forward(self, query: str, llm_chat_history: List[Dict[str, Any]] = None):
        ret = self._agent(query, llm_chat_history) if llm_chat_history is not None else self._agent(query)
        return ret if isinstance(ret, str) else (_ for _ in ()).throw(
            ValueError(f"After retrying {self._max_retries} times, the function call agent still "
                       "failed to call successfully."))
