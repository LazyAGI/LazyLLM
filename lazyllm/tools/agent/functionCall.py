from lazyllm.module import ModuleBase, OnlineChatModule
from lazyllm.components import ChatPrompter
from .functionCallFormatter import FunctionCallFormatter
from lazyllm import pipeline, ifs, loop, globals, bind, LOG, Color, package, FileSystemQueue, colored_text
import json5 as json
from .toolsManager import ToolManager
from typing import List, Any, Dict, Union, Callable


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

class StreamResponse():
    def __init__(self, prefix: str, prefix_color: str = None, color: str = None, stream: bool = False):
        self.stream = stream
        self.prefix = prefix
        self.prefix_color = prefix_color
        self.color = color

    def __call__(self, *inputs):
        if self.stream: FileSystemQueue().enqueue(colored_text(f'\n{self.prefix}\n', self.prefix_color))
        if len(inputs) == 1:
            if self.stream: FileSystemQueue().enqueue(colored_text(f'{inputs[0]}', self.color))
            return inputs[0]
        if self.stream: FileSystemQueue().enqueue(colored_text(f'{inputs}', self.color))
        return package(*inputs)


class FunctionCall(ModuleBase):
    """FunctionCall is a single-round tool call class. If the information in LLM is not enough to answer the uesr's question, it is necessary to combine external knowledge to answer the user's question. If the LLM output required a tool call, the tool call is performed and the tool call result is output. The output result is of List type, including the input, model output, and tool output of the current round. If a tool call is not required, the LLM result is directly output, and the output result is of string type.

Note: The tools used in `tools` must have a `__doc__` field, clearly describing the purpose and parameters of the tool according to the [Google Python Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) requirements.

Args:
    llm (ModuleBase): The LLM to be used can be either TrainableModule or OnlineChatModule.
    tools (List[Union[str, Callable]]): A list of tool names for LLM to use.


Examples:
    >>> import lazyllm
    >>> from lazyllm.tools import fc_register, FunctionCall
    >>> import json
    >>> from typing import Literal
    >>> @fc_register("tool")
    >>> def get_current_weather(location: str, unit: Literal["fahrenheit", "celsius"] = 'fahrenheit'):
    ...     '''
    ...     Get the current weather in a given location
    ...
    ...     Args:
    ...         location (str): The city and state, e.g. San Francisco, CA.
    ...         unit (str): The temperature unit to use. Infer this from the users location.
    ...     '''
    ...     if 'tokyo' in location.lower():
    ...         return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius'})
    ...     elif 'san francisco' in location.lower():
    ...         return json.dumps({'location': 'San Francisco', 'temperature': '72', 'unit': 'fahrenheit'})
    ...     elif 'paris' in location.lower():
    ...         return json.dumps({'location': 'Paris', 'temperature': '22', 'unit': 'celsius'})
    ...     else:
    ...         return json.dumps({'location': location, 'temperature': 'unknown'})
    ...
    >>> @fc_register("tool")
    >>> def get_n_day_weather_forecast(location: str, num_days: int, unit: Literal["celsius", "fahrenheit"] = 'fahrenheit'):
    ...     '''
    ...     Get an N-day weather forecast
    ...
    ...     Args:
    ...         location (str): The city and state, e.g. San Francisco, CA.
    ...         num_days (int): The number of days to forecast.
    ...         unit (Literal['celsius', 'fahrenheit']): The temperature unit to use. Infer this from the users location.
    ...     '''
    ...     if 'tokyo' in location.lower():
    ...         return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius', "num_days": num_days})
    ...     elif 'san francisco' in location.lower():
    ...         return json.dumps({'location': 'San Francisco', 'temperature': '72', 'unit': 'fahrenheit', "num_days": num_days})
    ...     elif 'paris' in location.lower():
    ...         return json.dumps({'location': 'Paris', 'temperature': '22', 'unit': 'celsius', "num_days": num_days})
    ...     else:
    ...         return json.dumps({'location': location, 'temperature': 'unknown'})
    ...
    >>> tools=["get_current_weather", "get_n_day_weather_forecast"]
    >>> llm = lazyllm.TrainableModule("internlm2-chat-20b").start()  # or llm = lazyllm.OnlineChatModule("openai", stream=False)
    >>> query = "What's the weather like today in celsius in Tokyo."
    >>> fc = FunctionCall(llm, tools)
    >>> ret = fc(query)
    >>> print(ret)
    ["What's the weather like today in celsius in Tokyo.", {'role': 'assistant', 'content': '
    ', 'tool_calls': [{'id': 'da19cddac0584869879deb1315356d2a', 'type': 'function', 'function': {'name': 'get_current_weather', 'arguments': {'location': 'Tokyo', 'unit': 'celsius'}}}]}, [{'role': 'tool', 'content': '{"location": "Tokyo", "temperature": "10", "unit": "celsius"}', 'tool_call_id': 'da19cddac0584869879deb1315356d2a', 'name': 'get_current_weather'}]]
    >>> query = "Hello"
    >>> ret = fc(query)
    >>> print(ret)
    'Hello! How can I assist you today?'
    """

    def __init__(self, llm, tools: List[Union[str, Callable]], *, return_trace: bool = False,
                 stream: bool = False, _prompt: str = None):
        super().__init__(return_trace=return_trace)
        if isinstance(llm, OnlineChatModule) and llm.series == "QWEN" and llm._stream is True:
            raise ValueError("The qwen platform does not currently support stream function calls.")
        if _prompt is None:
            _prompt = FC_PROMPT_ONLINE if isinstance(llm, OnlineChatModule) else FC_PROMPT_LOCAL

        self._tools_manager = ToolManager(tools, return_trace=return_trace)
        self._prompter = ChatPrompter(instruction=_prompt, tools=self._tools_manager.tools_description)\
            .pre_hook(function_call_hook)
        self._llm = llm.share(prompt=self._prompter, format=FunctionCallFormatter()).used_by(self._module_id)
        with pipeline() as self._impl:
            self._impl.ins = StreamResponse('Received instruction:', prefix_color=Color.yellow,
                                            color=Color.green, stream=stream)
            self._impl.m1 = self._llm
            self._impl.m2 = self._parser
            self._impl.dis = StreamResponse('Decision-making or result in this round:',
                                            prefix_color=Color.yellow, color=Color.green, stream=stream)
            self._impl.m3 = ifs(lambda x: isinstance(x, list),
                                pipeline(self._tools_manager, StreamResponse('Tool-Call result:',
                                         prefix_color=Color.yellow, color=Color.green, stream=stream)),
                                lambda out: out)
            self._impl.m4 = self._tool_post_action | bind(input=self._impl.input, llm_output=self._impl.m1)

    def _parser(self, llm_output: Union[str, List[Dict[str, Any]]]):
        LOG.debug(f"llm_output: {llm_output}")
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
            LOG.debug(f"functionCall result: {ret}")
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
    """FunctionCallAgent is an agent that uses the tool calling method to perform complete tool calls. That is, when answering uesr questions, if LLM needs to obtain external knowledge through the tool, it will call the tool and feed back the return results of the tool to LLM, which will finally summarize and output them.

Args:
    llm (ModuleBase): The LLM to be used can be either TrainableModule or OnlineChatModule.
    tools (List[str]): A list of tool names for LLM to use.
    max_retries (int): The maximum number of tool call iterations. The default value is 5.


Examples:
    >>> import lazyllm
    >>> from lazyllm.tools import fc_register, FunctionCallAgent
    >>> import json
    >>> from typing import Literal
    >>> @fc_register("tool")
    >>> def get_current_weather(location: str, unit: Literal["fahrenheit", "celsius"]='fahrenheit'):
    ...     '''
    ...     Get the current weather in a given location
    ...
    ...     Args:
    ...         location (str): The city and state, e.g. San Francisco, CA.
    ...         unit (str): The temperature unit to use. Infer this from the users location.
    ...     '''
    ...     if 'tokyo' in location.lower():
    ...         return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius'})
    ...     elif 'san francisco' in location.lower():
    ...         return json.dumps({'location': 'San Francisco', 'temperature': '72', 'unit': 'fahrenheit'})
    ...     elif 'paris' in location.lower():
    ...         return json.dumps({'location': 'Paris', 'temperature': '22', 'unit': 'celsius'})
    ...     elif 'beijing' in location.lower():
    ...         return json.dumps({'location': 'Beijing', 'temperature': '90', 'unit': 'Fahrenheit'})
    ...     else:
    ...         return json.dumps({'location': location, 'temperature': 'unknown'})
    ...
    >>> @fc_register("tool")
    >>> def get_n_day_weather_forecast(location: str, num_days: int, unit: Literal["celsius", "fahrenheit"]='fahrenheit'):
    ...     '''
    ...     Get an N-day weather forecast
    ...
    ...     Args:
    ...         location (str): The city and state, e.g. San Francisco, CA.
    ...         num_days (int): The number of days to forecast.
    ...         unit (Literal['celsius', 'fahrenheit']): The temperature unit to use. Infer this from the users location.
    ...     '''
    ...     if 'tokyo' in location.lower():
    ...         return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius', "num_days": num_days})
    ...     elif 'san francisco' in location.lower():
    ...         return json.dumps({'location': 'San Francisco', 'temperature': '75', 'unit': 'fahrenheit', "num_days": num_days})
    ...     elif 'paris' in location.lower():
    ...         return json.dumps({'location': 'Paris', 'temperature': '25', 'unit': 'celsius', "num_days": num_days})
    ...     elif 'beijing' in location.lower():
    ...         return json.dumps({'location': 'Beijing', 'temperature': '85', 'unit': 'fahrenheit', "num_days": num_days})
    ...     else:
    ...         return json.dumps({'location': location, 'temperature': 'unknown'})
    ...
    >>> tools = ['get_current_weather', 'get_n_day_weather_forecast']
    >>> llm = lazyllm.TrainableModule("internlm2-chat-20b").start()  # or llm = lazyllm.OnlineChatModule(source="sensenova")
    >>> agent = FunctionCallAgent(llm, tools)
    >>> query = "What's the weather like today in celsius in Tokyo and Paris."
    >>> res = agent(query)
    >>> print(res)
    'The current weather in Tokyo is 10 degrees Celsius, and in Paris, it is 22 degrees Celsius.'
    >>> query = "Hello"
    >>> res = agent(query)
    >>> print(res)
    'Hello! How can I assist you today?'
    """
    def __init__(self, llm, tools: List[str], max_retries: int = 5, return_trace: bool = False, stream: bool = False):
        super().__init__(return_trace=return_trace)
        self._max_retries = max_retries
        self._fc = FunctionCall(llm, tools, return_trace=return_trace, stream=stream)
        self._agent = loop(self._fc, stop_condition=lambda x: isinstance(x, str), count=self._max_retries)
        self._fc._llm.used_by(self._module_id)

    def forward(self, query: str, llm_chat_history: List[Dict[str, Any]] = None):
        ret = self._agent(query, llm_chat_history) if llm_chat_history is not None else self._agent(query)
        return ret if isinstance(ret, str) else (_ for _ in ()).throw(
            ValueError(f"After retrying {self._max_retries} times, the function call agent still "
                       "failed to call successfully."))
