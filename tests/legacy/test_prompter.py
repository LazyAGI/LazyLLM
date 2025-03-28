import lazyllm
import json
import copy
from lazyllm.components.prompter import PrompterBase
from lazyllm import ModuleBase
from typing import Dict, Union, Callable, List, Any
from collections.abc import Iterator

def get_current_weather(location, unit='fahrenheit'):
    """Get the current weather in a given location"""
    if 'tokyo' in location.lower():
        return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius'})
    elif 'san francisco' in location.lower():
        return json.dumps({'location': 'San Francisco', 'temperature': '72', 'unit': 'fahrenheit'})
    elif 'paris' in location.lower():
        return json.dumps({'location': 'Paris', 'temperature': '22', 'unit': 'celsius'})
    else:
        return json.dumps({'location': location, 'temperature': 'unknown'})
    
def get_n_day_weather_forecast(location, num_days, unit='fahrenheit'):
    """Get the current weather in a given location"""
    if 'tokyo' in location.lower():
        return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius', "num_days": num_days})
    elif 'san francisco' in location.lower():
        return json.dumps({'location': 'San Francisco', 'temperature': '72', 'unit': 'fahrenheit', "num_days": num_days})
    elif 'paris' in location.lower():
        return json.dumps({'location': 'Paris', 'temperature': '22', 'unit': 'celsius', "num_days": num_days})
    else:
        return json.dumps({'location': location, 'temperature': 'unknown'})

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                },
                "required": ["location", "unit"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_n_day_weather_forecast",
            "description": "Get an N-day weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "The number of days to forecast",
                    }
                },
                "required": ["location", "unit", "num_days"]
            },
        }
    },
]

tools_map = {"get_current_weather": get_current_weather,
             "get_n_day_weather_forecast": get_n_day_weather_forecast}


class Agent:
    """
    智能体，负责根据设定的指令和用户的query生成回复或调用函数和工具，并管理历史对话
    """
    def __init__(
            self,
            chatBot: ModuleBase,
            prompter: PrompterBase,
            name:str = "ChatAgent",
            tools:List[Dict[str, Dict[str,Union[str,dict]]]] = [],
            tools_map:Dict[str,Callable] = {}
        ) -> None:
        self._chatBot = chatBot
        self._prompter = prompter
        self. _name = name
        self._tools = tools
        self._tools_map = tools_map
    def run(self, query:str, history:List[Dict[str, Any]] = None, extro_info: Dict[str, str] = None):
        return self._run(query, history=history, extro_info=extro_info)
    
    def _run(self, query:str, history:List[Dict[str, Any]] = None, extro_info: Dict[str, str] = None):
        if extro_info:
            input = {"input": query}
            input.update(extro_info)
        else:
            input = query
        
        response = self._chatBot(copy.deepcopy(input), llm_chat_history=history, tools=self._tools)
        lazyllm.LOG.info(response)
        if isinstance(response, Iterator):
            content = ""
            isTools = False
            for r in response:
                if r == "[DONE]":
                    break
                r = json.loads(r)
                lazyllm.LOG.info(r)
                if r["choices"][0]["type"] != "tool_calls":
                    delta = r["choices"][0]["delta"]
                    content += delta["content"]
                else:
                    resp = r["choices"][0]
                    isTools = True
            if not isTools:
                return content
            else:
                if resp.get("finish_reason", "") == "tool_calls":
                    if self._chatBot.get_model_type() != "SenseNovaModule":
                        resp = resp['message']
                    history.append({"role": "user", "content": query})
                    history.append(resp)
                    tool_calls = resp["tool_calls"]
                    for tool_call in tool_calls:
                        tool_call_id = tool_call["id"]
                        tool_name = tool_call["function"]["name"]
                        if tool_name in self._tools_map.keys():
                            tool_args = tool_call["function"]["arguments"]
                            tool_args = json.loads(tool_args)
                            output = self._tools_map[tool_name](**tool_args)
                        else:
                            raise ValueError(f"Tool {tool_name} not found")
                        
                        content = {"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": output}
                        if isinstance(input, dict):
                            input["input"] = content
                        else:
                            input = {"input": content}
                        resp = m(copy.deepcopy(input), llm_chat_history=history, tools=tools)
                        lazyllm.LOG.info(resp)
                        res = ""
                        if isinstance(resp, Iterator):
                            for r in resp:
                                if r == "[DONE]":
                                    break
                                r = json.loads(r)
                                lazyllm.LOG.info(r)
                                if "type" not in r["choices"][0] or ("type" in r["choices"][0] and r["choices"][0]["type"] != "tool_calls"):
                                    delta = r["choices"][0]["delta"]
                                    res += delta["content"]
                        return res
        else:
            if resp.get("finish_reason", "") == "tool_calls":
                if self._chatBot.get_model_type() != "SenseNovaModule":
                    resp = resp['message']
                history.append({"role": "user", "content": query})
                history.append(resp)
                tool_calls = resp["tool_calls"]
                for tool_call in tool_calls:
                    tool_call_id = tool_call["id"]
                    tool_name = tool_call["function"]["name"]
                    if tool_name in self._tools_map.keys():
                        tool_args = tool_call["function"]["arguments"]
                        tool_args = json.loads(tool_args)
                        output = self._tools_map[tool_name](**tool_args)
                    else:
                        raise ValueError(f"Tool {tool_name} not found")
                    
                    content = {"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": output}
                    input["input"] = content
                    resp = m(copy.deepcopy(input), llm_chat_history=history, tools=tools)
                    lazyllm.LOG.info(resp)
                    return resp
            else:
                return resp

# prompter = lazyllm.ChatPrompter(instruction="Answer the following questions as best as you can. You have access to the following tools:\n", 
#                                 extra_keys=["tools"], 
#                                 show=True)
prompter = lazyllm.ChatPrompter(instruction="Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.\n", 
                                show=False)

# m = lazyllm.OnlineChatModule(source="openai", base_url="http://101.230.144.233:23151/v1", stream=False, prompter=prompter)
# m = lazyllm.OnlineChatModule(source="openai", base_url="https://gf.nekoapi.com/v1", stream=False, prompter=prompter)
# online
m = lazyllm.OnlineChatModule(source="sensenova", stream=True, prompter=prompter)

query = "What's the weather like today in Tokyo"

# agent = Agent(chatBot=m, prompter=prompter, tools = tools, tools_map=tools_map)
# res = agent.run(query, history=[])
# lazyllm.LOG.info(f"res: {res}")


import lazyllm
from lazyllm import launchers, deploy

base_model = 'internlm2-chat-1_8b'
prompter = lazyllm.ChatPrompter(instruction="You are a helpful, respectful and honest assistant.", show=True)
m = lazyllm.TrainableModule(base_model, '').prompt(prompt="you are an AI assistant whose name is InternLM (书生·浦语).\n- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工>智能实验室). It is designed to be helpful, honest, and harmless.\n- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.").deploy_method(deploy.vllm, launcher=launchers.remote(ngpus=1))

dataset = ['介绍一下你自己', '李白和李清照是什么关系', '说个笑话吧']
m.evalset([f'<|im_start|>user\n{x}<|im_end|>\n<|im_start|>assistant\n' for x in dataset])

m.update_server()
m.eval()
print(m.eval_result)
