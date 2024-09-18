# FunctionCall

为了增加模型的能力，使其不仅能生成文本，还可以执行特定任务、查询数据库、与外部系统交互等。我们定义了 [FunctionCall][lazyllm.tools.agent.FunctionCall] 类，用于实现模型的工具调用的能力。
API 文档可以参考 [FunctionCall][lazyllm.tools.agent.FunctionCall]。接下来我将会从一个简单的例子开始，初步介绍 LazyLLM 中 [FunctionCall][lazyllm.tools.agent.FunctionCall] 的设计思路。

## FunctionCall 牛刀小试

[](){#define-function}
假设我们在开发一个查询天气的应用，由于天气信息具有时效性，所以单纯靠大模型是没有办法生成具体的天气信息的，这就需要模型调用外部查询天气的工具来获取实时的天气消息。现在我们定义两个查询天气函数如下：

```python
from typing import Literal
import json
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

为了使大模型可以调用相应的函数以及生成对应的参数，在定义函数时，需要给函数参数添加注解，以及给函数增加功能描述，以便于让大模型知道该函数的功能及什么时候可以调用该函数。

这是第一步，定义工具。第二步我们需要把定义好的工具注册进 LazyLLM 中，以便后面大模型时候时不用再传输函数了。注册方式如下：

```python
from lazyllm.tools import fc_register
@fc_register("tool")
def get_current_weather(location: str, unit: Literal["fahrenheit", "celsius"]="fahrenheit"):
    ...

@fc_register("tool")
def get_n_day_weather_forecast(location: str, num_days: int, unit: Literal["celsius", "fahrenheit"]='fahrenheit'):
	...
```

注册方式很简单，导入 `fc_register` 之后，直接在定义好的函数名之上按照装饰器的方式进行添加即可。这里需要注意，添加的时候要指定默认分组 `tool`，默认注册的工具名称是所注册的函数名称。

我们也可以把工具注册为不同的名字，在注册的时候填入第二个参数即可，例如：

```python
from lazyllm.tools import fc_register

def get_current_weather(location: str, unit: Literal["fahrenheit", "celsius"]="fahrenheit"):
    ...

fc_register("tool")(get_current_weather, "another_get_current_weather")
```

这样上面的函数 `get_current_weather` 就被注册成了名为 `another_get_current_weather` 的工具。

如果我们不打算把工具注册为全局可见，也可以在调用 FunctionCall 的时候直接传入工具本身，像这样：

```python
import lazyllm
from lazyllm.tools import FunctionCall
llm = lazyllm.OnlineChatModule()
tools = [get_current_weather, get_n_day_weather_forecast]
fc = FunctionCall(llm, tools)
query = "What's the weather like today in celsius in Tokyo and Paris."
ret = fc(query)
print(f"ret: {ret}")
```

上面的代码把我们之前定义的两个函数作为工具直接传入，它们只在生成的 `fc` 实例中可见，如果在 `fc` 外通过名称获取这两个工具将会报错。

然后我们就可以定义模型，并使用 [FunctionCall][lazyllm.tools.agent.FunctionCall] 了，示例如下：

```python
import lazyllm
from lazyllm.tools import FunctionCall
llm = lazyllm.TrainableModule("internlm2-chat-20b").start()  # or llm = lazyllm.OnlineChatModule()
tools = ["get_current_weather", "get_n_day_weather_forecast"]
fc = FunctionCall(llm, tools)
query = "What's the weather like today in celsius in Tokyo and Paris."
ret = fc(query)
print(f"ret: {ret}")
# ["What's the weather like today in celsius in Tokyo and Paris.", {'role': 'assistant', 'content': '', 'tool_calls': [{'id': '93d7e8e8721b4d22b1cb9aa14234ad70', 'type': 'function', 'function': {'name': 'get_current_weather', 'arguments': {'location': 'Tokyo', 'unit': 'celsius'}}}]}, [{'role': 'tool', 'content': '{"location": "Tokyo", "temperature": "10", "unit": "celsius"}', 'tool_call_id': '93d7e8e8721b4d22b1cb9aa14234ad70', 'name': 'get_current_weather'}]]
```

结果输出为一个列表，第一个元素是当前的输入，第二个元素是模型的输出，第三个元素是工具的输出。因为 [FunctionCall][lazyllm.tools.agent.FunctionCall] 是一个单轮的工具调用过程，所以返回的结果中不光有工具的返回结果，还有当前轮的输入和模型结果。如果没有触发工具调用，则直接返回字符串。如果想要执行一个完整的 function call，则需要使用 [FunctionCallAgent][lazyllm.tools.agent.FunctionCallAgent]，示例如下：

```python
import lazyllm
from lazyllm.tools import FunctionCallAgent
llm = lazyllm.TrainableModule("internlm2-chat-20b").start()  # or llm = lazyllm.OnlineChatModule()
tools = ["get_current_weather", "get_n_day_weather_forecast"]
agent = FunctionCallAgent(llm, tools)
query = "What's the weather like today in celsius in Tokyo and Paris."
ret = agent(query)
print(f"ret: {ret}")
# The current weather in Tokyo is 10 degrees Celsius, and in Paris, it is 22 degrees Celsius.
```

在上面的例子中，如果输入的 query 触发了 function call，则 [FunctionCall][lazyllm.tools.agent.FunctionCall] 会返回一个列表对象，而 [FunctionCallAgent][lazyllm.tools.agent.FunctionCallAgent] 会迭代执行模型调用和工具调用，直到模型认为信息足够给出一个结论，或者超出了迭代次数，迭代次数通过 max_retries 设置，默认值为5。

完整代码如下：
```python
from typing import Literal
import json
import lazyllm
from lazyllm.tools import fc_register, FunctionCall, FunctionCallAgent

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

llm = lazyllm.TrainableModule("internlm2-chat-20b").start()  # or llm = lazyllm.OnlineChatModule()
tools = ["get_current_weather", "get_n_day_weather_forecast"]
fc = FunctionCall(llm, tools)
query = "What's the weather like today in celsius in Tokyo and Paris."
ret = fc(query)
print(f"ret: {ret}")
# ["What's the weather like today in celsius in Tokyo and Paris.", {'role': 'assistant', 'content': '', 'tool_calls': [{'id': '93d7e8e8721b4d22b1cb9aa14234ad70', 'type': 'function', 'function': {'name': 'get_current_weather', 'arguments': {'location': 'Tokyo', 'unit': 'celsius'}}}]}, [{'role': 'tool', 'content': '{"location": "Tokyo", "temperature": "10", "unit": "celsius"}', 'tool_call_id': '93d7e8e8721b4d22b1cb9aa14234ad70', 'name': 'get_current_weather'}]]

agent = FunctionCallAgent(llm, tools)
ret = agent(query)
print(f"ret: {ret}")
# The current weather in Tokyo is 10 degrees Celsius, and in Paris, it is 22 degrees Celsius.
```

!!! Note "注意"

    - 注册函数或者工具时，必需指定默认分组 `tool`，否则模型没有办法使用对应的工具。
    - 在使用模型时，不用区分 [TrainableModule][lazyllm.module.TrainableModule] 和 [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule]，因为设计的 [TrainableModule][lazyllm.module.TrainableModule] 和 [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule] 的输出类型是一致的。

## FunctionCall 的设计思路
[FunctionCall][lazyllm.tools.agent.FunctionCall] 的设计流程采用自底向上的方式进行的，首先由于 [FunctionCall][lazyllm.tools.agent.FunctionCall] 是必需要调用 LLM 的，所以必须是模型的输出格式一致，因此，首先保证 [TrainableModule][lazyllm.module.TrainableModule] 和 [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule] 的输出对齐，然后再实现单轮的 [FunctionCall][lazyllm.tools.agent.FunctionCall], 即调用 LLM 和 tools 一次，最后实现完整的 [FunctionCallAgent][lazyllm.tools.agent.FunctionCallAgent]，即多次迭代 [FunctionCall][lazyllm.tools.agent.FunctionCall], 直到模型迭代完成或者超过最大迭代次数。

### TrainableModule 和 OnlineChatModule 输出对齐

1、由于 [TrainableModule][lazyllm.module.TrainableModule] 的输出是 string 类型，而 [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule] 的输出是 json 格式，所以为了使 [FunctionCall][lazyllm.tools.agent.FunctionCall] 使用模型时，对模型的类型无感知，则需要对这两种模型的输出格式进行统一。

2、首先，对于 [TrainableModule][lazyllm.module.TrainableModule] 来说，通过 prompt 指定模型输出 tool_calls 的格式，然后通过对模型的输出进行解析，仅获取模型生成的部分，即模型真正的输出。例如：
```text
'\nI need to use the "get_current_weather" function to get the current weather in Tokyo and Paris. I will call the function twice, once for Tokyo and once for Paris.<|action_start|><|plugin|>\n{"name": "get_current_weather", "parameters": {"location": "Tokyo"}}<|action_end|><|im_end|>'
```

3、然后通过 extractor 对模型输出进行解析，获取 content 字段和 tool_calls 的 `name` 和 `arguments` 字段，然后对结果进行拼接输出，格式为:
```text
content<|tool_calls|>tool_calls
```

> - 其中 `content` 表示模型输出的内容信息，`<|tool_calls|>` 表示分隔符， `tool_calls` 表示工具调用的字符串表示，例如：
```text
'[{"id": "xxxx", "type": "function", "function": {"name": "func_name", "arguments": {"param1": "val1", "param2": "val2"}}}]'
```
> - 因为没有 `id` 字段，所以此处的 `id` 字段为生成的一个唯一的随机数标识，为了和 [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule] 对齐。
>
> - 如果没有触发工具调用，即输出中没有 tool_calls 和分隔符，则只有 content 。如果触发了工具调用，但是没有 content ，则输出不包含 content ，只有 <|tool_calls|>tool_calls 。

例如：
```text
'I need to use the "get_current_weather" function to get the current weather in Tokyo and Paris. I will call the function twice, once for Tokyo and once for Paris.<|tool_calls|>[{"id": "bd75399403224eb8972640eabedd0d46", "type": "function", "function":{"name": "get_current_weather", "arguments": "{\"location\": \"Tokyo\"}"}}]'
```

4、其次，对于 [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule] 来说，由于线上模型是支持流式和非流式输出的，而 [FunctionCall][lazyllm.tools.agent.FunctionCall] 是否触发，只有等拿到全部信息之后才能知道，所以对于 [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule] 的流式输出，需要先做一个流式转非流式，即如果模型是流式输出，则等接收完全部消息之后再做后续处理。例如：
```text
{
  "id": "chatcmpl-bbc37506f904440da85a9bad1a21494e",
  "object": "chat.completion",
  "created": 1718099764,
  "model": "moonshot-v1-8k",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [
          {
            "index": 0,
            "id": "get_current_weather:0",
            "type": "function",
            "function": {
              "name": "get_current_weather",
              "arguments": "{\n  \"location\": \"Tokyo\",\n  \"unit\": \"celsius\"\n}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 219,
    "completion_tokens": 22,
    "total_tokens": 241
  }
}
```

5、接收完模型的输出之后，通过 extractor 对模型输出进行解析，获取 content 字段和 tool_calls 中的除了 `index` 之外的其余字段，这里还保留 `type` 和 `function` 是因为下一轮给模型输入的需要。提取完 content 和 tool_calls 字段之后，按照上面的输出格式进行拼接输出。例如：
```text
'<|tool_calls|>[{"id": "get_current_weather:0","type":"function","function":{"name":"get_current_weather","arguments":"{\n\"location\":\"Tokyo\",\n\"unit\":\"celsius\"\n}"}}]'
```

6、这样就能保证 [TrainableModule][lazyllm.module.TrainableModule] 和 [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule] 的使用是一致的体验了。而为了适配 [FunctionCall][lazyllm.tools.agent.FunctionCall] 的应用，针对模型的输出再通过一下 FunctionCallFormatter ，FunctionCallFormatter 的作用是对模型的输出进行解析，获取 content 和 tool_calls 信息。例如：
```text
[{"id": "bd75399403224eb8972640eabedd0d46", "type": "function", "function":{"name": "get_current_weather", "arguments": {"location": "Tokyo"}}}]或者
[{"id": "get_current_weather:0","type":"function","function":{"name":"get_current_weather","arguments":{"location":"Tokyo","unit":"celsius"}}},{"id": "get_current_weather:1","type":"function","function":{"name":"get_current_weather","arguments":{"location":"Paris","unit":"celsius"}}}]
```

如果没有调用工具，则输出结果为 string 类型，即模型的输出。例如：
```text
今天的东京天气温度为10度 Celsius。
今天东京的天气温度是10摄氏度，而巴黎的天气温度是22摄氏度。
```

!!! Note "注意"

    - 模型的输出结果格式为 `content<|tool_calls|>tool_calls`, 分隔符固定，通过分隔符来判断是否是工具调用。
    - 工具调用的信息里面，除了工具的 `name` 和 `arguments` 之外，还有 `id` 、`type` 和 `function` 字段。


### FunctionCall 的输出流程

[FunctionCall][lazyllm.tools.agent.FunctionCall] 是处理单轮的工具调用。
> - 非 function call 请求
```text
Hello World!
```
> - function call 请求
```text
What's the weather like today in Tokyo.
```

1、输入进来首先调用大模型，例如：
> - 非 function call 请求
```text
Hello! How can I assist you today?
```
> - function call 请求
```text
[{"id": "bd75399403224eb8972640eabedd0d46","type":"function", "function":{"name": "get_current_weather", "arguments": "{"location": "Tokyo"}"}}]
```

2、模型的输出经过解析器进行解析
> - 非 function call 请求
```text
Hello! How can I assist you today?
```
> - function call 请求
```text
[{"name": "get_current_weather", "arguments": {"location": "Tokyo"}}]
```

3、判断解析后的输出是否是工具调用，如果是工具调用，测通过 [ToolManager][lazyllm.tools.agent.ToolManager] 工具管理类来调用相应工具。
> - function call 请求
```text
'{"location": "Tokyo", "temperature": "10", "unit": "celsius"}'
```

4、如果不是工具调用，则直接进行输出，如果是工具调用，则把当前轮次输入、模型输出和工具返回结果封装一起，然后进行输出。
> - 非 function call 请求
```text
Hello! How can I assist you today?
```
> - function call 请求
```text
[{'tool_call_id': 'bd75399403224eb8972640eabedd0d46', 'name': 'get_current_weather', 'content': '{"location": "Tokyo", "temperature": "10", "unit": "celsius"}', 'role': 'tool'}]
```

### Function Call Agent 的输出流程

[FunctionCallAgent][lazyllm.tools.agent.FunctionCallAgent] 是处理完整工具调用的过程
> - agent 输入
```text
What's the weather like today in Tokyo.
```

1、输入进来直接调用 [FunctionCall][lazyllm.tools.agent.FunctionCall] 模块
> - function call 输出结果
```text
[{'tool_call_id': 'get_current_weather:0', 'name': 'get_current_weather', 'content': '{"location": "Tokyo", "temperature": "10", "unit": "celsius"}', 'role': 'tool'}]
```
> - 非 function call 输出结果
```text
今天的东京天气温度是10度 Celsius。
```

2、判断 [FunctionCall][lazyllm.tools.agent.FunctionCall] 的结果是否是工具调用，或者达到最大迭代次数。如果是工具调用，则返回步骤一。如果不是工具调用或者达到最大迭代次数，则继续往下走。

3、如果达到最大迭代次数，则抛出异常，如果模型正常生成结果，则直接输出。
> - 达到最大迭代次数后抛出异常
```text
ValueError: After retrying 5 times, the function call agent still failed to call successfully.
```
> - 正常结果输出
```text
今天的东京天气温度是10度 Celsius。
```

## 高级 Agent
智能体/智能代理 (Agent) 视为能够使用传感器感知周围环境，自主做出决策，然后使用执行器执行相应行动的人工实体。它具有自主性(可以独立运行，不需要人为干预)、反应性(能感知环境变化，并做出反应)、社会性(多个Agent可以相互协调共同完成任务)和适应性(能够不断提升自身性能，更好地完成任务)。
[FunctionCallAgent][lazyllm.tools.agent.FunctionCallAgent] 是最基本的 Agent ，通过大模型生成工具调用参数，然后调用工具并把工具返回结果反馈给大模型，这么不断重复，直到模型生成最终答案或者超出最大迭代次数。这其中也会出现一些问题，比如大模型生成的过程是个黑盒，人们不知道具体的推理流程，或者面对复杂问题，大模型不能直接给出答案等等。针对这些问题，研究者们提出了各种解决方案，形成了各种高级 Agent 。下面我们介绍 LazyLLM 中的几种高级 Agent 的实现。

### React

[论文](https://arxiv.org/abs/2210.03629)

思路：[ReactAgent][lazyllm.tools.agent.ReactAgent] 按照 "Thought->Action->Observation->Thought...->Finish" 的流程进行问题处理。Thought 展示模型是如何一步一步进行问题解决的。Action 表示工具调用的信息。Observation 为工具返回的结果。Finish 为问题最后的答案。

该 agent 执行流程和 [FunctionCallAgent][lazyllm.tools.agent.FunctionCallAgent] 的执行流程一样，唯一区别是 prompt 不同，并且 [ReactAgent][lazyllm.tools.agent.ReactAgent] 每一步都要有 Thought 输出，而普通 [FunctionCallAgent][lazyllm.tools.agent.FunctionCallAgent] 可能只有工具调用的信息输出，没有 content 内容。示例如下：
```python
import lazyllm
from lazyllm.tools import fc_register, ReactAgent
@fc_register("tool")
def multiply_tool(a: int, b: int) -> int:
    '''
    Multiply two integers and return the result integer

    Args:
        a (int): multiplier
        b (int): multiplier
    '''
    return a * b

@fc_register("tool")
def add_tool(a: int, b: int):
    '''
    Add two integers and returns the result integer

    Args:
        a (int): addend
        b (int): addend
    '''
    return a + b
tools = ["multiply_tool", "add_tool"]
llm = lazyllm.TrainableModule("internlm2-chat-20b").start()   # or llm = lazyllm.OnlineChatModule()
agent = ReactAgent(llm, tools)
query = "What is 20+(2*4)? Calculate step by step."
res = agent(query)
print(res)
# 'Answer: The result of 20+(2*4) is 28.'
```

### PlanAndSolve

[论文](https://arxiv.org/abs/2305.04091)

思路：[PlanAndSolveAgent][lazyllm.tools.agent.PlanAndSolveAgent]由两个组件组成：首先，将整个任务分解为更小的子任务，其次，根据计划执行这些子任务。最后结果作为答案进行输出。

1、输入进来之后首先经过 planner 模型，针对问题生成解决的计划
```text
Plan:\n1. Identify the given expression: 20 + (2 * 4)\n2. Perform the multiplication operation inside the parentheses: 2 * 4 = 8\n3. Add the result of the multiplication to 20: 20 + 8 = 28\n4. The final answer is 28.\n\nGiven the above steps taken, the answer to the expression 20 + (2 * 4) is 28. <END_OF_PLAN>
```

2、对生成的计划进行解析，以便后面 solver 模型能按照计划进行执行

3、针对计划的每个步骤分别调用 [FunctionCallAgent][lazyllm.tools.agent.FunctionCallAgent] 进行处理，最后生成的结果作为最终答案进行返回
```text
The final answer is 28.
```

示例如下：
```python
import lazyllm
from lazyllm.tools import fc_register, PlanAndSolveAgent

@fc_register("tool")
def multiply(a: int, b: int) -> int:
    """
    Multiply two integers and return the result integer

    Args:
        a (int): multiplier
        b (int): multiplier
    """
    return a * b

@fc_register("tool")
def add(a: int, b: int):
    """
    Add two integers and returns the result integer

    Args:
        a (int): addend
        b (int): addend
    """
    return a + b

llm = lazyllm.TrainableModule("internlm2-chat-20b").start()  # or llm = lazyllm.OnlineChatModule(stream=False)
tools = ["multiply", "add"]
agent = PlanAndSolveAgent(llm, tools=tools)
query = "What is 20+(2*4)? Calculate step by step."
ret = agent(query)
print(ret)
# The final answer is 28.
```

### ReWOO (Reasoning WithOut Observation)

[论文](https://arxiv.org/abs/2305.18323)

思路：[ReWOOAgent][lazyllm.tools.agent.ReWOOAgent] 包含三个部分：Planner 、 Worker 和 Solver。其中， Planner 使用可预见推理能力为复杂任务创建解决方案蓝图； Worker 通过工具调用来与环境交互，并将实际证据或观察结果填充到指令中； Solver 处理所有计划和证据以制定原始任务或问题的解决方案。

1、输入进来首先调用 planner 模型，生成解决问题的蓝图
```text
Plan: To find out the name of the cognac house that makes the main ingredient in The Hennchata, I will first search for information about The Hennchata on Wikipedia.
#E1 = WikipediaWorker[The Hennchata]

Plan: Once I have the information about The Hennchata, I will look for details about the cognac used in the drink.
#E2 = LLMWorker[What cognac is used in The Hennchata, based on #E1]

Plan: After identifying the cognac, I will search for the cognac house that produces it on Wikipedia.
#E3 = WikipediaWorker[producer of cognac used in The Hennchata]

Plan: Finally, I will extract the name of the cognac house from the Wikipedia page.
#E4 = LLMWorker[What is the name of the cognac house in #E3]
```

2、解析生成的计划蓝图，并调用相应的工具，把工具返回的结果填充到相应指令中
```text
Plan: To find out the name of the cognac house that makes the main ingredient in The Hennchata, I will first search for information about The Hennchata on Wikipedia.
Evidence:
The Hennchata is a cocktail consisting of Hennessy cognac and Mexican rice horchata agua fresca. It was invented in 2013 by Jorge Sánchez at his Chaco's Mexican restaurant in San Jose, California.
Plan: Once I have the information about The Hennchata, I will look for details about the cognac used in the drink.
Evidence:
Hennessy cognac.
Plan: After identifying the cognac, I will search for the cognac house that produces it on Wikipedia.
Evidence:
Drinks are liquids that can be consumed, with drinking water being the base ingredient for many of them. In addition to basic needs, drinks form part of the culture of human society. In a commercial setting, drinks, other than water, may be termed beverages.
Plan: Finally, I will extract the name of the cognac house from the Wikipedia page.
Evidence:
The name of the cognac house is not specified.
```

3、把计划和工具执行的结果拼接在一起，然后调用 solver 模型，生成最终答案。
```text
'\nHennessy '
```

示例如下：
```python
import lazyllm
from lazyllm import fc_register, ReWOOAgent, deploy
import wikipedia
@fc_register("tool")
def WikipediaWorker(input: str):
    """
    Worker that search for similar page contents from Wikipedia. Useful when you need to get holistic knowledge about people, places, companies, historical events, or other subjects. The response are long and might contain some irrelevant information. Input should be a search query.

    Args:
        input (str): search query.
    """
    try:
        evidence = wikipedia.page(input).content
        evidence = evidence.split("\n\n")[0]
    except wikipedia.PageError:
        evidence = f"Could not find [{input}]. Similar: {wikipedia.search(input)}"
    except wikipedia.DisambiguationError:
        evidence = f"Could not find [{input}]. Similar: {wikipedia.search(input)}"
    return evidence
@fc_register("tool")
def LLMWorker(input: str):
    """
    A pretrained LLM like yourself. Useful when you need to act with general world knowledge and common sense. Prioritize it when you are confident in solving the problem yourself. Input can be any instruction.

    Args:
        input (str): instruction
    """
    llm = lazyllm.OnlineChatModule(stream=False)
    query = f"Respond in short directly with no extra words.\n\n{input}"
    response = llm(query, llm_chat_history=[])
    return response
tools = ["WikipediaWorker", "LLMWorker"]
llm = lazyllm.TrainableModule("Qwen2-72B-Instruct-AWQ").deploy_method(deploy.vllm).start()  # or llm = lazyllm.OnlineChatModule()
agent = ReWOOAgent(llm, tools=tools)
query = "What is the name of the cognac house that makes the main ingredient in The Hennchata?"
ret = agent(query)
print(ret)
# '\nHennessy '
```
