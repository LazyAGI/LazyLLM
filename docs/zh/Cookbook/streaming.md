
# 构建流式机器人

我们先从构建一个简单的支持流式对话机器人开始。

> 通过本节您将学习到LazyLLM的以下要点：
>
> - 如何使用 [TrainableModule][lazyllm.module.TrainableModule] 和 [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule] 来构建支持流式的对话机器人。
> - 如何使用带 [FunctionCall][lazyllm.tools.agent.FunctionCall] 的流式对话机器人

## 设计思路

传统的模型支持流式是在请求时设置 stream 参数为 True ，然后模型的响应就会以生成器的形式返回，即产生流式输出。但是在 [FunctionCall][lazyllm.tools.agent.FunctionCall] 的应用中，如果应用流式输出，那后面的模块就没办法及时判断当前请求是否是 [FunctionCall][lazyllm.tools.agent.FunctionCall] 调用，如果要判断，就需要把模型的输出都接收完才可以，所以一般的 [FunctionCall][lazyllm.tools.agent.FunctionCall] 应用都是非流式的，即使设置了参数 `stream=True`， 也是内部接收完全部模型响应才做后续处理，并不能真正给用户流式输出。

LazyLLM 在面对这个问题的时候通过多线程的方式进行解决。在模型流式输出的时候，是按照两条路径处理数据的，一条是按照流式方式不断把模型产生的消息压入 `FileSystemQueue` 中，直到检测到 [FunctionCall][lazyllm.tools.agent.FunctionCall] 调用相关的特殊 token 就停止把消息压入 `FileSystemQueue` 中。另一条是正常接收消息，直到模型产生的消息全部接收完，然后才进行后面的消息处理。接收消息的时候，需要从另一个线程中从 `FileSystemQueue` 中取数据，来获取模型生成的消息。

## 代码实现

### 不带前端界面的流式对话机器人

我们先简单实现一个只调用模型的流式对话机器人。代码如下：

```python
import lazyllm
from functools import partial

llm = lazyllm.TrainableModule("internlm2-chat-20b", stream=True) # or llm = lazyllm.OnlineChatModule(stream=True)

query = "你会哪些技能"

with lazyllm.ThreadPoolExecutor(1) as executor:
    future = executor.submit(partial(llm, llm_chat_history=[]), query)
	while True:
	    if value := lazyllm.FileSystemQueue().dequeue():
			print(f"output: {''.join(value)}")
		elif future.done():
			break
	print(f"ret: {future.result()}")
```

这里使用 [TrainableModule][lazyllm.module.TrainableModule] 或者 [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule] 都可以，因为它们的使用体验一致的。我们这里以 [TrainableModule][lazyllm.module.TrainableModule] 为例，模型选择使用 `internlm2-chat-20b` 模型。初始化模型的时候需要指定参数 `stream=True` ，并且接收消息时，需要使用多线程才能保证模型流式输出的消息能流式接收回来。这里需要使用 LazyLLM 提供的线程池来处理，因为里面针对流式响应设置了标识符，这样才能保证在多线程环境下，生产线程和消费线程之间不会错乱。如果使用 python 提供的线程池来实现多线程，那么这里就不能从 queue 中正确拿到消息内容。

### 带前端界面的流式对话机器人

和不带前端界面的流式对话机器人的主要区别是 [WebModule][lazyllm.tools.webpages.WebModule] 负责流式接收消息，并显示在前端界面上。代码如下：

```python
import lazyllm

llm = lazyllm.TrainableModule("internlm2-chat-20b", stream=True)  # or llm = lazyllm.OnlineChatModule(stream=True)
lazyllm.WebModule(llm, port=23333).start().wait()
```

这样就比较简单了，只需要定义好模型使用流式，然后处理工作交给 [WebModule][lazyllm.tools.webpages.WebModule] 来处理即可。

效果入下：
![Stream_chat_bot](../assets/stream_cookbook_robot.png)

其实使用 [WebModule][lazyllm.tools.webpages.WebModule] 的话，还可以从界面上来控制是否使用流式，即选中或者取消页面左侧 `流式输出` 的选项即可。是不是很简单。

好了，对话机器人我们说完了，下面我们就来介绍 [FunctionCall][lazyllm.tools.agent.FunctionCall] 使用流式的情况。

### 带 [FunctionCall][lazyllm.tools.agent.FunctionCall] 的流式对话机器人

我们先定义 [FunctionCall][lazyllm.tools.agent.FunctionCall] 使用的工具。

```python
import json
import lazyllm
from lazyllm import fc_register

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

定义工具时需要注意使用 `fc_register` 注册器，以及需要给工具加上注释，以便 LLM 区分是否需要调用以及调用哪个工具。具体注意事项参见 [ToolManager][lazyllm.tools.agent.ToolManager] 。

定义完工具，我们就该定义模型了，同样的，这里使用 [TrainableModule][lazyllm.module.TrainableModule] 或者 [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule] 都可以。这里我们选用 [TrainableModule][lazyllm.module.TrainableModule] ，模型选用 `internlm2-chat-20b`。

```python
import lazyllm
llm = lazyllm.TrainableModule("internlm2-chat-20b", stream=True)  # or llm = lazyllm.OnlineChatModule()
```

此处需要注意， 使用 [TrainableModule][lazyllm.module.TrainableModule] 时，需要明确指定 `stream=True`， 因为它默认的是 `stream=False` ，但是使用 [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule] 时，可以不用指定，因为 [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule] 默认的是 `stream=True` 。

定义完模型，我们就该来定义 [FunctionCallAgent][lazyllm.tools.agent.FunctionCallAgent] 了。定义 [FunctionCallAgent][lazyllm.tools.agent.FunctionCallAgent] 主要传入两个参数：模型和工具集。

```python
from lazyllm.tools import FunctionCallAgent
tools = ["get_current_weather", "get_n_day_weather_forecast"]
agent = FunctionCallAgent(llm, tools)
```

现在只差最后一步了，我们用 [WebModule][lazyllm.tools.webpages.WebModule] 把agent封装成一个带界面的服务。

```python
import lazyllm
lazyllm.WebModule(agent, port=23333).start().wait()
```

现在便完成了支持流式输出和 [FunctionCall][lazyllm.tools.agent.FunctionCall] 的对话机器人。当有给用户展示的信息时，界面便会流式的输出消息内容。而 [FunctionCall][lazyllm.tools.agent.FunctionCall] 会正常执行。

完整代码如下：

```python
import json
import lazyllm
from lazyllm import fc_register, FunctionCallAgent

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

llm = lazyllm.TrainableModule("internlm2-chat-20b", stream=True)  # or llm = lazyllm.OnlineChatModule()
tools = ["get_current_weather", "get_n_day_weather_forecast"]
agent = FunctionCallAgent(llm, tools)
lazyllm.WebModule(agent, port=23333).start().wait()
```

效果如下：

![stream_agent](../assets/stream_cookbook_agent.png)

界面上只会流式显示模型生成需要给用户展示的内容，而模型产生的工具调用信息则不会打印出来。是不是很简单。同样的，其他的agent也可以支持流式，这里就不一一展示了。

到这里怎么在 LazyLLM 中应用流式就讲完了。后面我们就可以根据需求搭建自己的应用了。
