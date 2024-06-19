from typing import Dict, Iterator, List, Any, Optional, Union, Tuple
import time

import lazyllm
from lazyllm import LOG as logger
from lazyllm.module import OnlineChatModule, TrainableModule

from .tools import BaseTool, ToolManager
from .protocol import Message, ToolCall, ROLE, ASSISTANT, USER, TOOL
from .utils import trans_chat_module_request, trans_chat_module_response, pre_hook

class FuncCall(lazyllm.ModuleBase):
    def __init__(self, 
                 llm:Union[OnlineChatModule, TrainableModule], 
                 tools:List[BaseTool] = None,
                 *, 
                 return_traces=False):
        super().__init__(return_trace=return_traces)
        self._llm = llm
        self._llm._prompt.pre_hook(pre_hook)
        self._tool_manager = ToolManager(tools) if tools else None

        with lazyllm.pipeline() as fc_ppl:
            fc_ppl.llm_ppl = lazyllm.pipeline(trans_chat_module_request, 
                                              self._call_llm, 
                                              trans_chat_module_response)
            fc_ppl.tool_call = lazyllm.ifs(self._has_tool_calls,
                                           lazyllm.pipeline(self._extract_tool_calls, 
                                                            lazyllm.warp(self._call_tool), 
                                                            self._merge_tool_call_message), 
                                           lambda x: [])
            fc_ppl.merge_message = self._merge_all_message | lazyllm.bind(fc_ppl.input["messages"], 
                                                                          fc_ppl.llm_ppl,
                                                                          lazyllm._0,)
            fc_ppl.recall_request = self._prepare_fc_ppl_request | lazyllm.bind(tools=fc_ppl.input["tools"], kwargs=fc_ppl.input["kwargs"])

        self.fc_loop = lazyllm.loop(fc_ppl, stop_condition=lambda x:x["messages"][-1][ROLE]==ASSISTANT)

    def forward(self, __input:str = None, *, messages:Union[List[Message], List[Dict]]=None, tools:List[str]=None, **kwargs):
        if __input:
            assert isinstance(__input, str), "If you provide __input, you should provide a string."
            assert messages is None, "If you provide __input, you should not provide messages."
            messages = [{"role": USER, "content": __input}]
        if not tools and self._tool_manager:
            tools = self._tool_manager.all_tools_description
        elif tools:
            tools = [self._tool_manager.get_description(tool) for tool in tools]
        return self.fc_loop(messages=messages, tools=tools, kwargs=kwargs)["messages"]

    def _call_llm(self, kwargs):
        return self._llm(kwargs.pop("instruction_kwargs"), kwargs.pop("messages"), kwargs.pop("tools"), **kwargs)
    
    def _has_tool_calls(self, message: Message):
        if self._tool_manager and message.get("tool_calls", None):
            return True
        return False
    
    def _prepare_fc_ppl_request(self, messages:List[Message], tools:List[Dict], kwargs):
        if self._tool_manager:
            return {"messages": messages, "tools": tools, "kwargs": kwargs}
        return {"messages": messages, "kwargs": kwargs}
    
    def _call_tool(self, tool_call:ToolCall, **kwargs) -> str:
        start_time = time.time()
        tool_id = tool_call["id"]
        tool_name, tool_args = tool_call["function"]["name"], tool_call["function"]["arguments"]
        tool_res = self._tool_manager._call_tool(tool_name, tool_args, **kwargs)
        logger.debug(f"Tool {tool_name} takes {time.time() - start_time:.2f}s.")
        return {"id": tool_id, "name": tool_name, "content": tool_res}
        
    @staticmethod
    def _extract_tool_calls(message: Message):
        return tuple(message.get("tool_calls", []))

    @staticmethod
    def _merge_tool_call_message(tool_outputs: Tuple[Dict[str, str]]):
        if not isinstance(tool_outputs, (list, tuple)):
            tool_outputs = [tool_outputs]
        tool_call_messages:List[Message] = []
        for tool_output in tool_outputs:
            tool_id = tool_output["id"]
            tool_content = tool_output["content"]
            tool_name = tool_output["name"]
            tool_call_messages.append(Message(role=TOOL, name=tool_name, tool_call_id=tool_id, content=tool_content))
        return tool_call_messages
    
    @staticmethod
    def _merge_all_message(input_message:Union[Message, List[Message]], llm_message:Message, tool_call_messages:List[Message]):
        if not isinstance(input_message, list):
            input_message = [input_message]
        return input_message+[llm_message]+tool_call_messages

    