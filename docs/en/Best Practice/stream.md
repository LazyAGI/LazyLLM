# Stream

In order to provide users with a better user experience, LazyLLM supports streaming output and intermediate result output in both ordinary dialogue systems and dialogue systems with [FunctionCall][lazyllm.tools.agent.FunctionCall]. This can reduce the user's waiting time and facilitate viewing of intermediate results. Next, I will start with a simple streaming dialogue robot that supports [FunctionCall][lazyllm.tools.agent.FunctionCall] to preliminarily introduce the streaming  design ideas in LazyLLM. 

## Stream Quick Start

Let's use [FunctionCall][lazyllm.tools.agent.FunctionCall] to check the weather. First, define two functions for checking the weather as follows:

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

There is nothing much to introduce here. For more information, please refer to: [FunctionCall](functionCall.md#define-function). Next, we need to define the model and agent. The example is as follows:

```python
import lazyllm
from lazyllm.tools import ReactAgent

llm = lazyllm.TrainableModule("internlm2-chat-20b", stream=True).start()  # or llm = lazyllm.OnlineChatModule(stream=True)
tools = ["get_current_weather", "get_n_day_weather_forecast"]
agent = ReactAgent(llm, tools)
```

The only difference between this and the normal [FunctionCall][lazyllm.tools.agent.FunctionCall] is that `stream` is set to `True` when defining the model. After setting, we need to request the agent and display the request result. The example is as follows:

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

In the above example, if streaming output is configured, multiple threads need to be started to read from the file queue when receiving data to achieve streaming output.

## Design Concept of Stream

Traditional models support streaming by setting the `stream` parameter to `True` during the request, and then the model's response will be returned in the form of a generator, that is, streaming output. However, in the [FunctionCall][lazyllm.tools.agent.FunctionCall] application, if streaming output is applied, the subsequent modules will not be able to determine in time whether the current request is a [FunctionCall][lazyllm.tools.agent.FunctionCall] call. If you want to judge, you need to receive all the model's outputs. Therefore, general [FunctionCall][lazyllm.tools.agent.FunctionCall] applications are non-streaming. Even if the parameter `stream=True` is set, the entire model response is received internally before subsequent processing is performed, and streaming output cannot be truly provided to users.

When faced with this problem, LazyLLM solved it by using a file queue. When the model is streamed, data is processed along two plans. One is to receive and cache messages normally until all messages generated by the model are received, and then proceed to the subsequent message processing. The other is to continuously push messages generated by the model into the file queue in a streaming manner until the relevant special token called by [FunctionCall][lazyllm.tools.agent.FunctionCall] is detected, and then the message will be stopped from being pushed into the file queue. When receiving messages, you need to get data from the file queue in another thread and try to display it to the user.

The principle is as follows:
![stream_principle](../assets/stream_principle.svg)

By using file queues, the streaming output of message content in common conversation applications and [FunctionCall][lazyllm.tools.agent.FunctionCall] applications is guaranteed. Where streaming data needs to be generated, data is continuously pushed into the file queue, and where streaming data needs to be obtained, data is continuously token out from the file queue. Here, writing data to the queue and taking data out of the queue must be done in multiple threads, and the LazyLLM thread pool must be used, because LazyLLM will add identifiers to the file queue in multiple threads to ensure that there is no confusion when operating the file queue in multiple threads. If you do not use multiple threads or use multithreading created by Python's own library, you cannot use streaming operations correctly.

!!! Note

    - Streaming requires setting `stream=True`.
    - File queues must be used in multi-threading to implement streaming output, and multi-threading must be implemented using the thread pool provided by LazyLLM.

## Intermediate log output

At the same time, in [FunctionCall][lazyllm.tools.agent.FunctionCall] or [Flow](flow.md#use-flow) applications, since users cannot see the intermediate result logs, the impact on user debugging and other aspects is still relatively large. In order to support the display of intermediate results in [FunctionCall][lazyllm.tools.agent.FunctionCall] or [Flow](flow.md#use-flow) to users, we can also use the idea of file queues. Let's take the above [FunctionCall][lazyllm.tools.agent.FunctionCall] code as an example. Now we need to print out the intermediate results. We only need to modify a few lines of code. The tool definition remains unchanged. We only modify the definition of the model and agent, as well as the code for displaying the log. The code is as follows:

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

From the above code, we can see that for the model and agent definitions, we only need to add `return_trace=True`. The following code only needs to add statements to get logs from the file queue and print logs. From the final result, we can see that LazyLLM can support both streaming output `output:` records and intermediate result log `trace:` records. The collection of `trace` logs is implemented in [ModuleBase][lazyllm.module.ModuleBase]. If you want to implement this capability on your own module, you only need to inherit the [ModuleBase][lazyllm.module.ModuleBase] class.

!!! Note

    - When outputing intermediate result logs, you need to set `return_trace=True`;
    - If you want your own functions to also have the ability to collect intermediate result logs, you need to inherit the [ModuleBase][lazyllm.module.ModuleBase] class.

For LazyLLM, using streaming output or collecting intermediate result logs is so simple, but it can give users a good effect experience.
