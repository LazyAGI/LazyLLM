from typing import Iterator, Dict, Any, Dict, Union, List
import json
import traceback
import time
try:
    import json5
except ImportError:
    raise ImportError("Please install json5 using `pip install json5`. ")

from lazyllm import LOG as logger
from lazyllm.agent.protocol import Message, ToolCall, Function, ASSISTANT, USER
from .tools import TOOLS_MAP

def trans_chat_module_request(messages:Union[List[Dict], List[Message]] = None, 
                              tools:List[Dict] = None, 
                              **kwargs):
    new_messages = []
    for message in messages:
        if isinstance(message, dict):
            new_messages.append(message)
        elif isinstance(message, Message):
            new_messages.append(message.model_dump())
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")
    return {"instruction_kwargs": kwargs.pop("instruction_kwargs", {}), "messages": messages, "tools":tools, **kwargs}

def trans_chat_module_response(response:Dict[str, Any]) -> Message:
    rsp_message = response["message"]
    tool_calls = rsp_message.get("tool_calls", None)
    content = rsp_message.get("content", None)
    assert tool_calls or content, "tool_calls or content must not be None"
    return Message(role=ASSISTANT, content=content, tool_calls=tool_calls)


def trans_chat_module_stream_response(response:Iterator[str]) -> Iterator[Message]:
    
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

def _get_tool_call(tool_name:str, tool_args_cache:list, tool_id:str) -> Dict[str, str]:
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


def call_tool(tool_call:ToolCall, **kwargs) -> str:
    """
    Call the tool with the given name.

    Args:
        tool_name (str): The name of the tool to call.
        tool_args (Union[str, dict]): The arguments to pass to the tool.
        kwargs (dict): Additional keyword arguments to pass to the tool.
    Returns:
        str: The output of the tool.
    """
    start_time = time.time()
    tool_id = tool_call["id"]
    tool_name, tool_args = tool_call["function"]["name"], tool_call["function"]["arguments"]
    if tool_name not in TOOLS_MAP:
        raise ValueError(f"Tool {tool_name} not found in tool_map")
    try:
        tool_args = json5.loads(tool_args) if isinstance(tool_args, str) else tool_args
        intersection_keys = tool_args.keys() & kwargs.keys()
        poped_args = {k:tool_args.pop(k) for k in intersection_keys}
        if poped_args:
            logger.debug(f"The following args of tool {tool_name} are overridden: {poped_args}")
        tool_res = TOOLS_MAP[tool_name](**tool_args, **kwargs)
        logger.debug(f"Tool {tool_name} takes {time.time() - start_time:.2f}s.")
        tool_res = tool_res if isinstance(tool_res, str) else json.dumps(tool_res, ensure_ascii=False, indent=4)
        return {"id": tool_id, "name": tool_name, "content": tool_res}
    except:
        logger.error(f"Error calling tool {tool_name}, args: {tool_args}, kwargs: {kwargs}.")
        logger.debug(traceback.format_exc())