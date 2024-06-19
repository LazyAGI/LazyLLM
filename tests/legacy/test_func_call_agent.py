from typing import List, Dict, Union, Tuple

import lazyllm
from lazyllm.module.onlineChatModule.openaiModule import OpenAIModule
from lazyllm.agent.protocol import Message, ROLE, USER, ASSISTANT
from lazyllm.agent.utils import (trans_chat_module_request, 
                                 trans_chat_module_response,
                                 call_tool,)
from lazyllm.agent.tools import query_weather

lazyllm.config.add("openai_api_key", str, "Your OpenAI API key", "OPENAI_API_KEY")
lazyllm.config.add("weather_key", str, "Your amap weather API key", "WEATHER_KEY")

llm = OpenAIModule(base_url="http://localhost:22341/v1", model='qwen2', stream=False)


def is_tool_calls(message: Message):
    if message.get("tool_calls", None):
        return True
    return False

def extract_tool_calls(message: Message):
    return tuple(message.get("tool_calls", []))

def call_llm(kwargs):
    return llm(kwargs.pop("input"), **kwargs)

def merge_tool_call_message(tool_outputs: Tuple[Dict[str, str]]):
    if not isinstance(tool_outputs, (list, tuple)):
        tool_outputs = [tool_outputs]
    tool_call_messages:List[Message] = []
    for tool_output in tool_outputs:
        tool_id = tool_output["id"]
        tool_content = tool_output["content"]
        tool_name = tool_output["name"]
        tool_call_messages.append(Message(role="tool", name=tool_name, tool_call_id=tool_id, content=tool_content))
    return tool_call_messages

def merge_all_message(input_message:Union[Message, List[Message]], llm_message:Message, tool_call_messages:List[Message]):
    if not isinstance(input_message, list):
        input_message = [input_message]
    return input_message+[llm_message]+tool_call_messages

def prepare_fc_ppl_request(messages:List[Message], tools:List[Dict[str, str]], **kwargs):
    return {"messages": messages, "tools": tools, "kwargs": kwargs}

with lazyllm.pipeline() as fc_ppl:
    fc_ppl.request = prepare_fc_ppl_request
    fc_ppl.llm_ppl = lazyllm.pipeline(trans_chat_module_request, call_llm, trans_chat_module_response)
    fc_ppl.tool_call = lazyllm.ifs(is_tool_calls, 
                                   lazyllm.pipeline(extract_tool_calls, lazyllm.warp(call_tool), merge_tool_call_message), 
                                   lambda x: [])
    fc_ppl.merge_message = merge_all_message | lazyllm.bind(fc_ppl.input["messages"], 
                                                            fc_ppl.llm_ppl, 
                                                            lazyllm._0,)
    fc_ppl.recall_request = prepare_fc_ppl_request | lazyllm.bind(tools=fc_ppl.input["tools"], kwargs=fc_ppl.input["kwargs"])


fc_loop = lazyllm.loop(fc_ppl, stop_condition=lambda x:x["messages"][-1][ROLE]==ASSISTANT)

if __name__ == "__main__":
    print(fc_loop)
    res = fc_loop(messages=[Message(role=USER, content="海淀区的天气怎么样？")],
                  tools=[query_weather.description],)
    print(res["messages"])