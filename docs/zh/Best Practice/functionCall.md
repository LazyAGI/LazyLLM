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
如果工具需要在 sandbox 中执行并且涉及文件上传或下载，必须通过 `input_files` / `output_files` 字段传递文件；如果工具不希望在 sandbox 中执行，可在注册时显式指定 `@fc_register("tool", execute_in_sandbox=False)`。

下面是一个文件上传/下载工具的例子：

```python
from typing import List, Optional
from lazyllm.tools import fc_register

@fc_register("tool")
def count_lines_in_file(
    input_files: Optional[List[str]] = None,
    output_files: Optional[List[str]] = None,
):
    """
    Count lines of the first input file and write to an output file.

    Args:
        input_files (list[str] | None): 输入文件路径列表。
        output_files (list[str] | None): 输出文件路径列表（沙箱会回传这些文件）。
    """
    if not input_files or not output_files:
        return "input_files/output_files required"
    src = input_files[0]
    dst = output_files[0]
    with open(src, "r", encoding="utf-8") as f:
        count = sum(1 for _ in f)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(str(count))
    return {"output_files": output_files, "lines": count}
```

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
# {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'xxx', 'type': 'function', 'function': {'name': 'get_current_weather', 'arguments': '{"location":"Tokyo, Japan","unit":"celsius"}'}, 'code_block': None}, {'id': 'xxx', 'type': 'function', 'function': {'name': 'get_current_weather', 'arguments': '{"location":"Paris, France","unit":"celsius"}'}, 'code_block': None}], 'tool_calls_results': ('{"location": "Tokyo", "temperature": "10", "unit": "celsius"}', '{"location": "Paris", "temperature": "22", "unit": "celsius"}')}
```

`FunctionCall` 会返回模型生成的助手消息。当触发工具调用时，该消息同时包含 `tool_calls`（模型发出的结构化调用）和 `tool_calls_results`（每次工具执行返回的输出）。这些数据也会同步写入 `lazyllm.locals['_lazyllm_agent']['workspace']['tool_calls']` 与 `lazyllm.locals['_lazyllm_agent']['workspace']['tool_call_results']`，方便下一轮使用。当`FunctionCall`判断不需要更多的工具调用时，会将整个执行过程中的工具调用存储到`lazyllm.locals['_lazyllm_agent']['completed']`，用于debug和后续任务。如果模型未选择任何工具，则直接返回字符串。若希望自动完成完整的推理闭环，可以使用 [ReactAgent][lazyllm.tools.agent.ReactAgent]，示例如下：

```python
import lazyllm
from lazyllm.tools import ReactAgent
llm = lazyllm.TrainableModule("internlm2-chat-20b").start()  # or llm = lazyllm.OnlineChatModule()
tools = ["get_current_weather", "get_n_day_weather_forecast"]
agent = ReactAgent(llm, tools)
query = "What's the weather like today in celsius in Tokyo and Paris."
ret = agent(query)
print(f"ret: {ret}")
# The current weather in Tokyo is 10 degrees Celsius, and in Paris, it is 22 degrees Celsius.
```

在上面的例子中，如果输入的 query 触发了 function call，[FunctionCall][lazyllm.tools.agent.FunctionCall] 会返回包含 `tool_calls` 与 `tool_calls_results` 的助手消息字典。[ReactAgent][lazyllm.tools.agent.ReactAgent] 则会反复调用模型和工具，直到模型认定信息足以得出结论，或达到由 `max_retries`（默认 5）控制的最大迭代次数。

完整代码如下：
```python
from typing import Literal
import json
import lazyllm
from lazyllm.tools import fc_register, FunctionCall, ReactAgent

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
# ret: {'role': 'assistant', 'content': None, 'tool_calls': [{'type': 'function', 'id': 'call_486db4f85272407995677d', 'function': {'arguments': '{"location": "Tokyo", "unit": "celsius"}', 'name': 'get_current_weather'}, 'index': 0}, {'type': 'function', 'id': 'call_08e4eebe34a44fee8abdc3', 'function': {'arguments': '{"location": "Paris", "unit": "celsius"}', 'name': 'get_current_weather'}, 'index': 1}], 'tool_calls_results': ('{"location": "Tokyo", "temperature": "10", "unit": "celsius"}', '{"location": "Paris", "temperature": "22", "unit": "celsius"}')}

agent = ReactAgent(llm, tools)
ret = agent(query)
print(f"ret: {ret}")
# ret: Answer: The current weather in Tokyo is 10°C, and in Paris, it is 22°C.
```

!!! Note "注意"

    - 注册函数或者工具时，必需指定默认分组 `tool`，否则模型没有办法使用对应的工具。
    - 在使用模型时，不用区分 [TrainableModule][lazyllm.module.TrainableModule] 和 [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule]，因为设计的 [TrainableModule][lazyllm.module.TrainableModule] 和 [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule] 的输出类型是一致的。

## 内置工具：code_interpreter

LazyLLM 提供内置工具 `code_interpreter`，用于在沙箱内执行代码并返回结果。该工具已通过 `fc_register("tool")` 注册，可直接作为工具名使用。

### 使用示例

```python
import lazyllm
from lazyllm.tools import FunctionCall

llm = lazyllm.OnlineChatModule()
tools = ["code_interpreter"]
fc = FunctionCall(llm, tools)

query = "用 python 计算 1 到 100 的和"
ret = fc(query)
print(ret)
```

### 沙箱配置

`code_interpreter` 默认使用本地沙箱（`LocalSandbox`，仅支持 python）。如需远程执行或 bash 支持，可切换为 `SandboxFusion`：

```python
from lazyllm import config

config['sandbox_type'] = 'sandbox_fusion'
config['sandbox_fusion_base_url'] = 'http://your-sandbox-host:port'
```

对应的环境变量：

- `LAZYLLM_SANDBOX_TYPE`：`local` 或 `sandbox_fusion`
- `LAZYLLM_SANDBOX_FUSION_BASE_URL`：远程沙箱服务地址

## FunctionCall 的设计思路
[FunctionCall][lazyllm.tools.agent.FunctionCall] 的设计流程采用自底向上的方式进行的，首先由于 [FunctionCall][lazyllm.tools.agent.FunctionCall] 是必需要调用 LLM 的，所以必须是模型的输出格式一致，因此，首先保证 [TrainableModule][lazyllm.module.TrainableModule] 和 [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule] 的输出对齐，然后再实现单轮的 [FunctionCall][lazyllm.tools.agent.FunctionCall], 即调用 LLM 和 tools 一次，最后实现完整的 [ReactAgent][lazyllm.tools.agent.ReactAgent]，即多次迭代 [FunctionCall][lazyllm.tools.agent.FunctionCall], 直到模型迭代完成或者超过最大迭代次数。

### TrainableModule 和 OnlineChatModule 输出对齐

1、由于 [TrainableModule][lazyllm.module.TrainableModule] 的输出是 string 类型，而 [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule] 的输出是 json 格式，所以为了使 [FunctionCall][lazyllm.tools.agent.FunctionCall] 使用模型时，对模型的类型无感知，则需要对这两种模型的输出格式进行统一。

2、首先，对于 [TrainableModule][lazyllm.module.TrainableModule] 来说，通过 prompt 指定模型输出 tool_calls 的格式，然后通过对模型的输出进行解析，仅获取模型生成的部分，即模型真正的输出。例如：
```text
'\nI need to use the "get_current_weather" function to get the current weather in Tokyo and Paris. I will call the function twice, once for Tokyo and once for Paris.<|action_start|><|plugin|>\n{"name": "get_current_weather", "parameters": {"location": "Tokyo"}}<|action_end|><|im_end|>'
```

3、然后通过 extractor 对模型输出进行解析，获取 content 字段和 tool_calls 的 `name` 和 `arguments` 字段，并生成一个id，然后将结果组织为dict，例如：
{role: "assistant", content: "I need to use the "get_current_weather" function to get the current weather in Tokyo and Paris. I will call the function twice, once for Tokyo and once for Paris.", tool_calls: [{id: "xxx", type: "function", "function": {name: "get_current_weather", arguments: {location: "Tokyo", unit: "celsius"}}}]}

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
            "id": "xxx",
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

5、接收完模型的输出之后，通过 extractor 对模型输出进行解析，获取message。例如：
```text
{
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {
        "index": 0,
        "id": "xxx",
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "arguments": "{\n  \"location\": \"Tokyo\",\n  \"unit\": \"celsius\"\n}"
            }
        }
    ]
}
```

6、这样就能保证 [TrainableModule][lazyllm.module.TrainableModule] 和 [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule] 的使用是一致的体验了。而为了适配 [FunctionCall][lazyllm.tools.agent.FunctionCall] 的应用，针对模型的输出再通过一下 FunctionCallFormatter ，FunctionCallFormatter 的作用是对模型的输出进行解析，获取 role, content 和 tool_calls 信息，忽略其他字段，输出结果为dict。

!!! Note "注意"

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
{'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'xxx', 'type': 'function', 'function': {'name': 'get_current_weather', 'arguments': {'location': 'Tokyo, Japan', 'unit': 'celsius'}}}]}
```

2、判断输出是否是工具调用，如果是工具调用，测通过 [ToolManager][lazyllm.tools.agent.ToolManager] 工具管理类来调用相应工具。
> - function call 请求
```text
'{"location": "Tokyo", "temperature": "10", "unit": "celsius"}'
```

3、如果不是工具调用，则直接进行输出，如果是工具调用，则把模型输出和工具返回结果封装一起，然后进行输出。
> - 非 function call 请求
```text
Hello! How can I assist you today?
```
> - function call 请求
```text
{'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'xxx', 'type': 'function', 'function': {'name': 'get_current_weather', 'arguments': {'location': 'Tokyo, Japan', 'unit': 'celsius'}}}], 'tool_calls_results': ('{"location": "Tokyo", "temperature": "10", "unit": "celsius"}',)}
```

## 高级 Agent
智能体/智能代理 (Agent) 视为能够使用传感器感知周围环境，自主做出决策，然后使用执行器执行相应行动的人工实体。它具有自主性(可以独立运行，不需要人为干预)、反应性(能感知环境变化，并做出反应)、社会性(多个Agent可以相互协调共同完成任务)和适应性(能够不断提升自身性能，更好地完成任务)。下面我们介绍 LazyLLM 中的几种高级 Agent 的实现。

### React

[论文](https://arxiv.org/abs/2210.03629)

思路：[ReactAgent][lazyllm.tools.agent.ReactAgent] 按照 "Thought->Action->Observation->Thought...->Finish" 的流程进行问题处理。Thought 展示模型是如何一步一步进行问题解决的。Action 表示工具调用的信息。Observation 为工具返回的结果。Finish 为问题最后的答案。示例如下：
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

3、针对计划的每个步骤分别调用 [FunctionCall][lazyllm.tools.agent.FunctionCall] 进行处理，直到不需要调用工具或达到预设的最大调用次数。最后生成的结果作为最终答案进行返回
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
