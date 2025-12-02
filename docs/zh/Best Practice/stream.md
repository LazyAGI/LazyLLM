# Stream

为了给用户更好的使用体验，LazyLLM 在普通的对话系统和带 [FunctionCall][lazyllm.tools.agent.FunctionCall] 的对话系统中都支持了流式输出以及中间结果输出。这样能减少用户的等待时间和方便查看中间结果。接下来我将会从一个简单的支持 [FunctionCall][lazyllm.tools.agent.FunctionCall] 的流式对话机器人开始，初步介绍 LazyLLM 中流式的设计思路。

## Stream 牛刀小试

我们还是使用 [FunctionCall][lazyllm.tools.agent.FunctionCall] 查天气的例子。先定义两个查天气的函数如下：

```python
from typing import Literal
import json
from lazyllm.tools import fc_register

@fc_register("tool")
def get_current_weather(location: str, unit: Literal["fahrenheit", "celsius"]="fahrenheit"):
    """
    Get the current weather in a given location

    Args:
        location (str): The city and state, e.g. San Francisco, CA.
        unit (str): The temperature unit to use. Infer this from the users location.
    """
    if 'tokyo' in location.lower():
        return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius'})
    elif 'san francisco' in location.lower():
        return json.dumps({'location': 'San Francisco', 'temperature': '72', 'unit': 'fahrenheit'})
    elif 'paris' in location.lower():
        return json.dumps({'location': 'Paris', 'temperature': '22', 'unit': 'celsius'})
    elif 'beijing' in location.lower():
        return json.dumps({'location': 'Beijing', 'temperature': '90', 'unit': 'fahrenheit'})
    else:
        return json.dumps({'location': location, 'temperature': 'unknown'})

@fc_register("tool")
def get_n_day_weather_forecast(location: str, num_days: int, unit: Literal["celsius", "fahrenheit"]='fahrenheit'):
    """
    Get an N-day weather forecast

    Args:
        location (str): The city and state, e.g. San Francisco, CA.
        num_days (int): The number of days to forecast.
        unit (Literal['celsius', 'fahrenheit']): The temperature unit to use. Infer this from the users location.
    """
    if 'tokyo' in location.lower():
        return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius', "num_days": num_days})
    elif 'san francisco' in location.lower():
        return json.dumps({'location': 'San Francisco', 'temperature': '75', 'unit': 'fahrenheit', "num_days": num_days})
    elif 'paris' in location.lower():
        return json.dumps({'location': 'Paris', 'temperature': '25', 'unit': 'celsius', "num_days": num_days})
    elif 'beijing' in location.lower():
        return json.dumps({'location': 'Beijing', 'temperature': '85', 'unit': 'fahrenheit', "num_days": num_days})
    else:
        return json.dumps({'location': location, 'temperature': 'unknown'})
```

这里没什么好介绍的了，具体信息可以参见：[FunctionCall](functionCall.md#define-function)。接下来，我们就需要定义模型和 agent 了。示例如下：

```python
import lazyllm
from lazyllm.tools import ReactAgent

llm = lazyllm.TrainableModule("internlm2-chat-20b", stream=True).start()  # or llm = lazyllm.OnlineChatModule(stream=True)
tools = ["get_current_weather", "get_n_day_weather_forecast"]
agent = ReactAgent(llm, tools)
```

和普通的 [FunctionCall][lazyllm.tools.agent.FunctionCall] 的区别仅仅是在定义模型的时候设置 `stream` 为 `True`。设置完之后，我们就要请求 agent ，以及显示请求结果了，示例如下：

```python
query = "What's the weather like today in celsius in Tokyo and Paris."
with lazyllm.ThreadPoolExecutor(1) as executor:
    future = executor.submit(agent, query)
	while True:
	    if value := lazyllm.FileSystemQueue().dequeue():
			print(f"output: {''.join(value)}")
		elif future.done():
			break
	print(f"ret: {future.result()}")
# output: Thought: The current language of the user is: English. I need to use
# output:  a tool to help answer the question.


# output: Answer: The current
# output:  weather in Tokyo is 10°C, and in Paris, it is 22°C.
# ret: Answer: The current weather in Tokyo is 10°C, and in Paris, it is 22°C.
```

在上面的例子中，如果配置了流式输出，在接收数据的时候需要启动多线程从文件队列中进行读取，以实现流式的输出。

## Stream的设计思路

传统的模型支持流式是在请求时设置 `stream` 参数为 `True` ，然后模型的响应就会以生成器的形式返回，即产生流式输出。但是在 [FunctionCall][lazyllm.tools.agent.FunctionCall] 的应用中，如果应用流式输出，那后面的模块就没办法及时判断当前请求是否是 [FunctionCall][lazyllm.tools.agent.FunctionCall] 调用，如果要判断，就需要把模型的输出都接收完才可以，所以一般的 [FunctionCall][lazyllm.tools.agent.FunctionCall] 应用都是非流式的，即使设置了参数 `stream=True`， 也是内部接收完全部模型响应才做后续处理，并不能真正给用户流式输出。

LazyLLM 在面对这个问题的时候通过文件队列的方式进行解决的。在模型流式输出的时候，是按照两条路径处理数据的，一条是正常接收消息并缓存，直到模型产生的消息全部接收完，然后才进行后面的消息处理。另一条是按照流式方式，不断把模型产生的消息压入文件队列中，直到检测到 [FunctionCall][lazyllm.tools.agent.FunctionCall] 调用的相关特殊 token 才会停止把消息压入文件队列中。接收消息的时候，需要在另一个线程中从文件队列中获取数据，并试试显示给用户。

原理如下：
![stream_principle](../assets/stream_principle.svg)

通过使用文件队列，保证了普通对话应用和 [FunctionCall][lazyllm.tools.agent.FunctionCall] 应用中消息内容的流式输出。在需要生成流式数据的地方，不断的把数据压入到文件队列中，而在需要获取流式数据的地方不断地再把数据从文件队列中取出来。这里写数据到队列和从队列里拿出数据必须是在多线程中进行，并且需要使用 LazyLLM 的线程池，因为 LazyLLM 在多线程中会对文件队列增加标识符，来保证在多线程中操作文件队列时不会混乱。如果不使用多线程或者使用的是 python 自己的库创建的多线程，就不能正确的使用流式操作。

!!! Note "注意"

    - 流式需要设置 `stream=True`。
    - 必须要在多线程中使用文件队列来实现流式输出，并且多线程必须要使用 LazyLLM 提供的线程池来实现。

## 中间日志输出

同时，在 [FunctionCall][lazyllm.tools.agent.FunctionCall] 或者 [Flow](flow.md#use-flow) 应用中，由于用户看不到中间的结果日志，所以对用户的调试等方面的影响还是比较大的。为了支持把 [FunctionCall][lazyllm.tools.agent.FunctionCall] 或者 [Flow](flow.md#use-flow) 中的中间结果给用户展示出来，我们也可以借助于文件队列的思想。我们还是以上面的 [FunctionCall][lazyllm.tools.agent.FunctionCall] 的代码为例，现在需要把中间结果打印出来，我们只需要修改几行代码即可，工具定义不变，我们只修改模型和 agent 的定义，以及显示日志的代码，代码如下：

```python
import lazyllm
from lazyllm.tools import ReactAgent

llm = lazyllm.TrainableModule("internlm2-chat-20b", stream=True, return_trace=True).start()  # or llm = lazyllm.OnlineChatModule(stream=True, return_trace=True)
tools = ["get_current_weather", "get_n_day_weather_forecast"]
agent = ReactAgent(llm, tools, return_trace=True)

query = "What's the weather like today in celsius in Tokyo and Paris."
with lazyllm.ThreadPoolExecutor(1) as executor:
    future = executor.submit(agent, query)
	while True:
	    if value := lazyllm.FileSystemQueue().dequeue():
			print(f"output: {''.join(value)}")
		elif value := lazyllm.FileSystemQueue().get_instance('lazy_trace').dequeue():
			print(f"trace: {''.join(value)}")
		elif future.done():
			break
	print(f"ret: {future.result()}")
# output: Thought: The current language of the user is: English. I need to use a tool to help answer the question.

# trace: {'role': 'assistant', 'content': 'Thought: The current language of the user is: English. I need to use a tool to help answer the question.\n', 'tool_calls': [{'index': 0, 'type': 'function', 'id': 'call_d2415b4b478c412ab7363f', 'function': {'arguments': '{"location": "Tokyo", "unit": "celsius"}', 'name': 'get_current_weather'}}, {'index': 1, 'type': 'function', 'id': 'call_15035422418847629999d9', 'function': {'arguments': '{"location": "Paris", "unit": "celsius"}', 'name': 'get_current_weather'}}]}{"location": "Paris", "temperature": "22", "unit": "celsius"}
# trace: {"location": "Tokyo", "temperature": "10", "unit": "celsius"}('{"location": "Tokyo", "temperature": "10", "unit": "celsius"}', '{"location": "Paris", "temperature": "22", "unit": "celsius"}'){'role': 'assistant', 'content': 'Thought: The current language of the user is: English. I need to use a tool to help answer the question.\n', 'tool_calls': [{'index': 0, 'type': 'function', 'id': 'call_d2415b4b478c412ab7363f', 'function': {'arguments': '{"location": "Tokyo", "unit": "celsius"}', 'name': 'get_current_weather'}}, {'index': 1, 'type': 'function', 'id': 'call_15035422418847629999d9', 'function': {'arguments': '{"location": "Paris", "unit": "celsius"}', 'name': 'get_current_weather'}}], 'tool_calls_results': ('{"location": "Tokyo", "temperature": "10", "unit": "celsius"}', '{"location": "Paris", "temperature": "22", "unit": "celsius"}')}
# output: Answer: The current
# output:  weather in Tokyo is 1
# trace: {'role': 'assistant', 'content': 'Answer: The current weather in Tokyo is 10°C, and in Paris, it is 22°C.'}Answer: The current weather in Tokyo is 10°C, and in Paris, it is 22°C.Answer: The current weather in Tokyo is 10°C, and in Paris, it is 22°C.
# output: 0°C, and in Paris, it is 22°C.
# ret: Answer: The current weather in Tokyo is 10°C, and in Paris, it is 22°C.
```

从上面代码中可以看到对于模型和 agent 定义，只需要加上 `return_trace=True` 即可，后面显示代码只需要加上从文件队列中获取日志和打印日志的语句即可。从最后的结果中可以看出来，LazyLLM 可以同时支持流式输出 `output:` 记录和中间结果日志 `trace:` 记录。`trace` 日志的收集是在 [ModuleBase][lazyllm.module.ModuleBase] 中实现的，如果想要在自己实现的模块上实现这个能力，只需要继承 [ModuleBase][lazyllm.module.ModuleBase] 类即可。

!!! Note "注意"

    - 输出中间结果日志时，需要设置 `return_trace=True`;
    - 如果想要自己实现的功能也有收集中间结果日志的能力，需要继承 [ModuleBase][lazyllm.module.ModuleBase] 类。

对于 LazyLLM 来说，使用流式输出或者收集中间结果日志就是如此简单，但是却能给用户很好的效果体验。
