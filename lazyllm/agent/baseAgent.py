from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Any, Optional, Literal, Union
import traceback
import json
import copy
try:
    import json5
except ImportError:
    raise ImportError("Please install json5 using `pip install json5`. ")

import lazyllm
from lazyllm.module import OnlineChatModule
from lazyllm.module import TrainableModule

from lazyllm import LOG as logger

from lazyllm.agent.protocol import DEFAULT_SYSTEM_MESSAGE, ASSISTANT, Message, Function, ToolCall
from lazyllm.agent.protocol import USER, TOOL, ROLE
from .prompter import AgentPrompter

class BaseAgent(lazyllm.ModuleBase, ABC):
    """Base class for all agents."""

    def __init__(self, 
                 llm:Union[OnlineChatModule, TrainableModule],
                 system_message:str = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 return_trace: bool = False, # 是否递归评测所有的子模块
                 **kwargs) -> None:
        """
        Initialize the agent.

        Args:
            llm (Union[OnlineChatModule, TrainableModule]): The LLM to use for the agent.
            system_message (str): The role of the agent.
            name (Optional[str]): The name of the agent.
            description (Optional[str]): The description of the agent.
            return_trace (bool): Whether to return the trace of the agent's actions.
        """
        super().__init__(return_trace=return_trace)

        self._system_message = system_message
        self._llm = self._init_llm(llm)
        
        self._module_name = name
        self._description = description
        
    def _init_llm(self, llm:Union[OnlineChatModule, TrainableModule]) -> Union[OnlineChatModule, TrainableModule]:
        prompter = AgentPrompter() # TODO: add more configs
        prompter._set_model_configs(system=self._system_message)
        llm.prompt(prompter)
        return llm

    def forward(self,
                messages:Union[List[Dict], List[Message]],
                tools:List[str] = None,
                *,
                return_message_type:Literal["message", "dict"] = None,
                **kwargs
                ) -> Union[Iterator[Message], Iterator[Dict]]:
        """
        Run the agent with the given input and messages.

        Args:
            messages (Union[List[Dict], List[Message]]): The messages to send to the agent.
            tools (List[str]): The tool's name list.
            return_message_type (Literal["message", "dict"]): The type of the returned message.

        Yields:
            Union[Iterator[Message], Iterator[Dict]]: The agent's response messages.
        """
        assert isinstance(messages, list) and len(messages) > 0, "The input messages must be a list with at least one elements."
        new_messages:List[Message] = []
        _ret_type = "message"
        for message in messages:
            if isinstance(message, dict):
                new_messages.append(Message(**message))
                _ret_type = "dict"
            elif isinstance(message, Message):
                new_messages.append(message)
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")
        
        ret_type = return_message_type or _ret_type
        for response in self._run(new_messages, tools, **kwargs):
            if ret_type == "message":
                yield response
            else:
                yield response.model_dump()
                
        
        
    @abstractmethod
    def _run(self, 
             messages:Union[List[Dict], List[Message]], 
             tools:List[str] = None, 
             **kwargs) -> Iterator[Message]:
        raise NotImplementedError


    def _call_llm(self,
                  messages:Union[List[Dict], List[Message]],
                  tools:List[Dict[str, Any]] = None,
                  *,
                  stream:bool = True,
                  generate_cfg: Dict[str, Any] = None,
                  instruction_kwargs: Dict[str, Any] = None,
                  **kwargs) -> Iterator[Message]:
        """
        Call the LLM with the given messages and tools.

        Args:
            messages (List[Message]): The messages to use for the LLM.
            tools (List[Dict[str, Any]]): Tools description list.
            stream (bool): Whether to stream the output.
            generate_cfg (Dict[str, Any]): The configuration for the LLM generation.
            instruction_kwargs (Dict[str, Any]): The keyword arguments for the instruction.
        Yields:
            Message: The LLM's response messages.
        """
        messages = copy.deepcopy(messages)
        generate_cfg = generate_cfg or {}
        if messages[-1][ROLE] not in (USER, TOOL):
            raise ValueError("The last message must be a user or tool message.")

        # OpenAI API v1 does not allow the following args, must pass by extra_body
        for extra_param in ['top_k', 'repetition_penalty']:
            if extra_param in generate_cfg:
                generate_cfg['extra_body'][extra_param] = generate_cfg.pop(extra_param)

        response = self._llm(instruction_kwargs if isinstance(instruction_kwargs, dict) else dict(),
                              llm_chat_history=[item.model_dump() for item in messages],
                              tools=tools,
                              stream=stream, 
                              **generate_cfg)
        
        return transform_online_chat_module_response(response)



def transform_online_chat_module_response(response:Iterator[str]) -> Iterator[Message]:
    args_cache, tool_name, tool_id = [], None, None
    yield Message(role=ASSISTANT, content=None) # first response, role is not None
    for chunk in response:
        if isinstance(chunk, str):
            chunk = json5.loads(chunk)
        resp_message = chunk["choices"][0]["delta"]
        content = resp_message.get("content", None)
        tool_calls = resp_message.get("tool_calls", [])
        ret_tool_calls = []
        if tool_calls:
            for tool_call in tool_calls:
                tool_name = tool_call["function"].get("name", None)
                if tool_name: # new tool call
                    ret_tool_call = _get_tool_call(tool_name, args_cache, tool_id)
                    if ret_tool_call:
                        ret_tool_calls.append(ret_tool_call)
                    args_cache = [] # ret_tool_call为空表示解析失败，也清空args_cache
                tool_id = tool_call["id"]
                tool_args = tool_call["function"].get("arguments", None)
                if tool_args:
                    args_cache.append(tool_args)
            if content or ret_tool_calls:
                yield Message(role=None, content=content, tool_calls=ret_tool_calls)
        else:
            if tool_name and args_cache:
                ret_tool_call = _get_tool_call(tool_name, args_cache, tool_id)
                if ret_tool_call:
                    ret_tool_calls.append(ret_tool_call)
                args_cache, tool_name, tool_id = [], None, None
            if content or ret_tool_calls:
                yield Message(role=None, content=content, tool_calls=ret_tool_calls if ret_tool_calls else None)
        
    if tool_name and args_cache:
        ret_tool_call = _get_tool_call(tool_name, args_cache, tool_id)
        if ret_tool_call:
            yield Message(role=None, content=None, tool_calls=[ret_tool_call])         


def _get_tool_call(tool_name:str, tool_args_cache:list, tool_id:str) -> str:
    if not tool_args_cache: return
    args = ''.join(tool_args_cache)
    func = _format_func(tool_name, args)
    if func:
        return ToolCall(id=tool_id, function=func, type="function")

def _format_func(func_name:str, args:str):
    try:
        args = json.loads(args)
        return Function(name=func_name, arguments=json.dumps(args, ensure_ascii=False))
    except:
        logger.error(f"The args `{args}` of tool {func_name} is not json format, it will be ignored. ")
        logger.debug(traceback.format_exc())
    return None