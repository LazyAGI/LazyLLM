
# Building a streaming bot

Let's start by building a simple conversational bot that supports streaming.

!!! abstract "Through this section, you will learn about the following key points of LazyLLM"

    - How to use [TrainableModule][lazyllm.module.TrainableModule] and [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule] to build a conversational bot that supports streaming.
    - How to use a streaming chatbot with [FunctionCall][lazyllm.tools.agent.FunctionCall].

## Code Implementation

### Streaming conversational robot with front-end interface

Let's first simply implement a streaming conversational robot with a front-end interface. The code is as follows:

```python
import lazyllm

llm = lazyllm.TrainableModule("internlm2-chat-20b", stream=True)  # or llm = lazyllm.OnlineChatModule(stream=True)
lazyllm.WebModule(llm, port=23333, stream=True).start().wait()
```

Isn't the implementation very simple? You just need to define the model using streaming, and leave the rest of the work to [WebModule][lazyllm.tools.webpages.WebModule] to handle it. Then the messages displayed to the user will be displayed in a streaming manner on the front-end interface.

The effect is as follows:
![Stream_chat_bot](../assets/stream_cookbook_robot.png)

In fact, if you are [WebModule][lazyllm.tools.webpages.WebModule], you can also control whether to use streaming from the interface, that is, select or cancel the option of `流式输出` on the left side of the page. Isn't it simple?

### Streaming conversational robot without front-end interface

Let's first simply implement a streaming conversational robot that only calls the model. The code is as follows:

```python
import lazyllm
from functools import partial

llm = lazyllm.TrainableModule("internlm2-chat-20b", stream=True).start() # or llm = lazyllm.OnlineChatModule(stream=True)

query = "What skills do you have?"

with lazyllm.ThreadPoolExecutor(1) as executor:
    future = executor.submit(partial(llm, llm_chat_history=[]), query)
	while True:
	    if value := lazyllm.FileSystemQueue().dequeue():
			print(f"output: {''.join(value)}")
		elif future.done():
			break
	print(f"ret: {future.result()}")
```

Here, you can use [TrainableModule][lazyllm.module.TrainableModule] or [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule] because they have the same user experience. Here we take [TrainableModule][lazyllm.module.TrainableModule] as an example, and choose to use the `internlm2-chat-20b` model. When initializing the model, you need to specigy the parameter `stream=True`, and when receiveing messages, you need to use multi-threading to ensure that the messages output by the model can be received back in a streaming manner. Here you need to use the thread pool provided by LazyLLM to handle it, because it sets an identifier for the streaming response, so as to ensure that there is no confusion between the production thread and the consumer thread in a multi-threaded environment. If you use the thread pool provided by python to implement multi-threading, then you cannot get the message content from the queue correctly here.

OK, now that we have finished talking about the conversational robot, let's introduce how to use [FunctionCall][lazyllm.tools.agent.FunctionCall] in a streaming manner.

### Streaming chatbot with [FunctionCall][lazyllm.tools.agent.FunctionCall]

We first define the tools used by [FunctionCall][lazyllm.tools.agent.FunctionCall].

```python
import json
import lazyllm
from lazyllm import fc_register
from typing import Literal

@fc_register("tool")
def get_current_weather(location: str, unit: Literal["fahrenheit", "celsius"]='fahrenheit'):
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
        return json.dumps({'location': 'Beijing', 'temperature': '90', 'unit': 'Fahrenheit'})
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

When defining a tool, you need to use the `fc_register` register and add comments to the tool so that LLM can distinguish whether is needs to be called and which tool to call. For specific precautions, see [ToolManager][lazyllm.tools.agent.ToolManager].

After defining the tools, we should define the model. Similarly, we can use [TrainableModule][lazyllm.module.TrainableModule] or [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule]. Here we choose [TrainableModule][lazyllm.module.TrainableModule] and `internlm2-chat-20b` as the model.

```python
import lazyllm
llm = lazyllm.TrainableModule("internlm2-chat-20b", stream=True).start()  # or llm = lazyllm.OnlineChatModule()
```

It should be noted here that when using [TrainableModule][lazyllm.module.TrainableModule], you need to explicitly specify `stream=True`, because it defaults to `stream=False`. However, when using [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule], you do not need to specify it, because [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule] defaults to `stream=True`.

After defining the model, we should define [ReactAgent][lazyllm.tools.agent.ReactAgent]. [ReactAgent][lazyllm.tools.agent.ReactAgent] mainly passes in two parameters: model and toolset.

```python
from lazyllm.tools import ReactAgent
tools = ["get_current_weather", "get_n_day_weather_forecast"]
agent = ReactAgent(llm, tools)
```

Now there is only one last step left. We use [WebModule][lazyllm.tools.webpages.WebModule] to encapsulate the agent into a service with an interface.

```python
import lazyllm
lazyllm.WebModule(agent, port=23333, stream=True).start().wait()
```

Now we have completed a conversational robot that supports streaming output and [FunctionCall][lazyllm.tools.agent.FunctionCall]. When there is information to show to the user, the interface will stream the message content. And [FunctionCall][lazyllm.tools.agent.FunctionCall] will execute normally.

The complete code is as follows:

```python
import json
import lazyllm
from lazyllm import fc_register, ReactAgent
from typing import Literal

@fc_register("tool")
def get_current_weather(location: str, unit: Literal["fahrenheit", "celsius"]='fahrenheit'):
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
        return json.dumps({'location': 'Beijing', 'temperature': '90', 'unit': 'Fahrenheit'})
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

llm = lazyllm.TrainableModule("internlm2-chat-20b", stream=True).start()  # or llm = lazyllm.OnlineChatModule()
tools = ["get_current_weather", "get_n_day_weather_forecast"]
agent = ReactAgent(llm, tools)
lazyllm.WebModule(agent, port=23333, stream=True).start().wait()
```

The effect is as follows:
![stream_agent](../assets/stream_cookbook_agent.png)

The interface will only display the content that the model needs to show to the user, and the tool call information generated by the model will not be printed out. Isn't it very simple? Similarly, other agents can also support streaming, which will not be shown here one by one.

If you need to print out the intermediate results of the agent, whether it is the LLM output results during the execution of [FunctionCall][lazyllm.tools.agent.FunctionCall] or the execution results of [ToolManager][lazyllm.tools.agent.ToolManager], set `return_trace` to `True` when defining LLM and agent. You only need to modify two sentences in the above code:

```python
llm = lazyllm.TrainableModule("internlm2-chat-20b", stream=True, return_trace=True).start()  # or llm = lazyllm.OnlineChatModule(return_trace=True)
agent = ReactAgent(llm, tools, return_trace=True)
```

This will display the intermediate processing results of [FunctionCall][lazyllm.tools.agent.FunctionCall] in the `处理日志` in the lower left corner of the interface.

The complete code is as follows:

```python
import json
import lazyllm
from lazyllm import fc_register, ReactAgent
from typing import Literal

@fc_register("tool")
def get_current_weather(location: str, unit: Literal["fahrenheit", "celsius"]='fahrenheit'):
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
        return json.dumps({'location': 'Beijing', 'temperature': '90', 'unit': 'Fahrenheit'})
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

llm = lazyllm.TrainableModule("internlm2-chat-20b", stream=True, return_trace=True).start()  # or llm = lazyllm.OnlineChatModule(return_trace=True)
tools = ["get_current_weather", "get_n_day_weather_forecast"]
agent = ReactAgent(llm, tools, return_trace=True)
lazyllm.WebModule(agent, port=23333, stream=True).start().wait()
```

The effect is as follows:
![stream_agent_trace](../assets/stream_cookbook_agent_trace.png)

This is the end of how to apply streaming in LazyyLLM. Later we can build our own applications according to our needs.
