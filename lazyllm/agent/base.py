from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Any, Optional, Tuple, Union
import traceback
import json
import time
try:
    import json5
except ImportError:
    raise ImportError("Please install json5 using `pip install json5`. ")

import lazyllm
from lazyllm.module import OnlineChatModule
from lazyllm.module import TrainableModule
from lazyllm import LOG as logger

from .protocol import DEFAULT_SYSTEM_MESSAGE, Message
from .protocol import ASSISTANT, SYSTEM, USER, TOOL, ROLE, CONTENT, NAME
from .tools import BaseTool, TOOLS_MAP


class BaseAgent(lazyllm.ModuleBase, ABC):
    """Base class for all agents."""

    def __init__(self, 
                 llm:Union[OnlineChatModule, TrainableModule],
                 tools_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 system_message:str = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 return_trace: bool = False, # 是否递归评测所有的子模块
                 **kwargs) -> None:
        """
        Initialize the agent.

        Args:
            llm (Union[OnlineChatModule, TrainableModule]): The LLM to use for the agent.
            tools_list (Optional[List[Union[str, Dict, BaseTool]]]): A list of tools the agent can use.
            system_message (str): The system message to use for the agent.
            name (Optional[str]): The name of the agent.
            description (Optional[str]): The description of the agent.
            return_trace (bool): Whether to return the trace of the agent's actions.
            **kwargs: Additional keyword arguments to pass to the agent.
        """
        super().__init__(return_trace=return_trace)

        self.llm = llm
        self.system_message = system_message
        self.name = name
        self.description = description
        self.tools_map = {name:tool for name, tool in self._init_tools(tools_list)} if tools_list else {}

    
    def forward(self,
                input:Optional[str] = None,
                history:Optional[Union[List[Dict], List[Message]]] = None,
                **kwargs
                ) -> Iterator[Union[List[Message], List[Dict]]]:
        """
        Run the agent with the given input and messages.

        Args:
            input (Optional[str]): The last input text of user.
            history (Optional[Union[List[Dict], List[Message]]]): The messages to use for the agent.
            **kwargs: Additional keyword arguments to pass to the agent.

        Yields:
            Iterator[Union[List[Message], List[Dict]]]: The agent's response messages.
        """
        assert input or history, "Either input or history must be provided."
        messages:List[Message] = []
        ret_type = kwargs.pop("return_message_type", "message")
        for message in history:
            if isinstance(message, dict):
                messages.append(Message(**message))
                ret_type = "dict"
            elif isinstance(message, Message):
                messages.append(message)
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")
            
        if input:
            messages.append(Message(role=USER, content=input))
        
        for response in self._run(messages=messages, **kwargs):
            for item in response:
                if ret_type == "message":
                    yield item
                else:
                    yield item.model_dump()
        
        
    @abstractmethod
    def _run(self, messages:List[Message], **kwargs) -> Iterator[Message]:
        raise NotImplementedError


    def _call_llm(self,
                  messages:List[Message],
                  tools_desc_list:List[Dict[str, Any]] = None,
                  stream:bool = True,
                  generate_cfg: Dict[str, Any] = None) -> Iterator[Message]:
        """
        Call the LLM with the given messages and tools.

        Args:
            messages (List[Message]): The messages to use for the LLM.
            tools_desc_list (List[Dict[str, Any]]): The tools description list.
            stream (bool): Whether to stream the output.
            generate_cfg (Dict[str, Any]): The configuration for the LLM generation.
        
        Yields:
            Iterator[Message]: The LLM's response messages.
        """
        if messages[0][ROLE] != SYSTEM:
            messages.insert(0, Message(role=SYSTEM, content=self.system_message))
        if messages[-1][ROLE] != USER:
            raise ValueError("The last message must be a user message.")
        
        if tools_desc_list:
            for i, tool_desc in enumerate(tools_desc_list):
                if "function" in tool_desc:
                    tool_desc = tool_desc["function"]
                    tools_desc_list[i] = tool_desc
                if tool_desc["name"] not in self.tools_map:
                    raise ValueError(f"Tool {tool_desc['name']} not found in tool_map.")

        # OpenAI API v1 does not allow the following args, must pass by extra_body
        for extra_param in ['top_k', 'repetition_penalty']:
            if extra_param in generate_cfg:
                generate_cfg['extra_body'][extra_param] = generate_cfg.pop(extra_param)

        for response in self.llm(messages=messages, 
                                 tools=tools_desc_list,
                                 stream=stream, 
                                 **generate_cfg):
            yield response


    def _call_tool(self, tool_name:str, tool_args:Union[str, dict], **kwargs) -> str:
        """
        Call the tool with the given name.

        Args:
            tool_name (str): The name of the tool to call.
            *args: Additional positional arguments to pass to the tool.
        Returns:
            str: The output of the tool.
        """
        start_time = time.time()
        if tool_name not in self.tools_map:
            raise ValueError(f"Tool {tool_name} not found in tool_map")
        try:
            tool_args = json5.loads(tool_args) if isinstance(tool_args, str) else tool_args
            intersection_keys = tool_args.keys() & kwargs.keys()
            poped_args = {k:tool_args.pop(k) for k in intersection_keys}
            if poped_args:
                logger.debug(f"The following args of tool {tool_name} are overridden: {poped_args}")
            tool_res = self.tools_map[tool_name](**tool_args, **kwargs)
            logger.debug(f"Tool {tool_name} takes {time.time() - start_time:.2f}s.")
            return tool_res if isinstance(tool_res, str) else json.dumps(tool_res, ensure_ascii=False, indent=4)
        except:
            logger.error(f"Error calling tool {tool_name}, args: {tool_args}, kwargs: {kwargs}.")
            logger.debug(traceback.format_exc())


    def _init_tool(self, tool:Union[str, Dict, BaseTool]) -> Tuple[str, BaseTool]:
        if isinstance(tool, str):
            if tool not in TOOLS_MAP:
                raise ValueError(f"Tool {tool} not found in TOOLS_MAP")
            return tool, TOOLS_MAP[tool]
        if isinstance(tool, dict):
            tool_name = tool.get("name", None)
            if tool_name not in TOOLS_MAP:
                raise ValueError(f"Tool {tool_name} not found in TOOLS_MAP")
            return tool_name, TOOLS_MAP[tool_name]
        if isinstance(tool, BaseTool):
            return tool.name, tool
