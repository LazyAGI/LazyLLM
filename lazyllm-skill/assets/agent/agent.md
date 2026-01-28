### ReactAgent

ReactAgent是按照 Thought->Action->Observation->Thought...->Finish 的流程一步一步的通过LLM和工具调用来显示解决用户问题的步骤，以及最后给用户的答案。

参数:

- llm – 大语言模型实例，用于生成推理和工具调用决策
- tools (List[str]) – 可用工具列表，可以是工具函数或工具名称
- max_retries (int, default: 5 ) – 最大重试次数，当工具调用失败时自动重试，默认为5
- return_trace (bool, default: False ) – 是否返回完整的执行轨迹，用于调试和分析，默认为False
- prompt (str, default: None ) – 自定义提示词模板，如果为None则使用内置模板
- stream (bool, default: False ) – 是否启用流式输出，用于实时显示生成过程，默认为False

```python
import lazyllm
from lazyllm.tools import fc_register, ReactAgent
@fc_register("tool")
def multiply_tool(a: int, b: int) -> int:
...     '''
...     Multiply two integers and return the result integer
...
...     Args:
...         a (int): multiplier
...         b (int): multiplier
...     '''
...     return a * b
...
@fc_register("tool")
def add_tool(a: int, b: int):
...     '''
...     Add two integers and returns the result integer
...
...     Args:
...         a (int): addend
...         b (int): addend
...     '''
...     return a + b
...
tools = ["multiply_tool", "add_tool"]
 # or llm = lazyllm.OnlineChatModule(source="sensenova")
llm = lazyllm.TrainableModule("internlm2-chat-20b").start()  
agent = ReactAgent(llm, tools)
query = "What is 20+(2*4)? Calculate step by step."
res = agent(query)
print(res)
'Answer: The result of 20+(2*4) is 28.'
```

### PlanAndSolveAgent

PlanAndSolveAgent由两个组件组成，首先，由planner将整个任务分解为更小的子任务，然后由solver根据计划执行这些子任务，其中可能会涉及到工具调用，最后将答案返回给用户。

参数:

- llm (ModuleBase, default: None ) – 要使用的LLM，可以是TrainableModule或OnlineChatModule。和plan_llm、solve_llm互斥，要么设置llm(planner和solver公用一个LLM)，要么设置plan_llm和solve_llm，或者只指定llm(用来设置planner)和solve_llm，其它情况均认为是无效的。
- tools (List[str], default: [] ) – LLM使用的工具名称列表。
- plan_llm (ModuleBase, default: None ) – planner要使用的LLM，可以是TrainableModule或OnlineChatModule。
- solve_llm (ModuleBase, default: None ) – solver要使用的LLM，可以是TrainableModule或OnlineChatModule。
- max_retries (int, default: 5 ) – 工具调用迭代的最大次数。默认值为5。
- return_trace (bool, default: False ) – 是否返回中间步骤和工具调用信息。
- stream (bool, default: False ) – 是否以流式方式输出规划和解决过程。

```python
import lazyllm
from lazyllm.tools import fc_register, PlanAndSolveAgent
@fc_register("tool")
def multiply(a: int, b: int) -> int:
...     '''
...     Multiply two integers and return the result integer
...
...     Args:
...         a (int): multiplier
...         b (int): multiplier
...     '''
...     return a * b
...
@fc_register("tool")
def add(a: int, b: int):
...     '''
...     Add two integers and returns the result integer
...
...     Args:
...         a (int): addend
...         b (int): addend
...     '''
...     return a + b
...
tools = ["multiply", "add"]
# or llm = lazyllm.OnlineChatModule(source="sensenova")
llm = lazyllm.TrainableModule("internlm2-chat-20b").start()
agent = PlanAndSolveAgent(llm, tools)
query = "What is 20+(2*4)? Calculate step by step."
res = agent(query)
print(res)
'The final answer is 28.'
```

### ReWOOAgent

ReWOOAgent包含三个部分：Planner、Worker和Solver。其中，Planner使用可预见推理能力为复杂任务创建解决方案蓝图；Worker通过工具调用来与环境交互，并将实际证据或观察结果填充到指令中；Solver处理所有计划和证据以制定原始任务或问题的解决方案。

参数:

- llm (ModuleBase, default: None ) – 要使用的LLM，可以是TrainableModule或OnlineChatModule。和plan_llm、solve_llm互斥，要么设置llm(planner和solver公用一个LLM)，要么设置plan_llm和solve_llm，或者只指定llm(用来设置planner)和solve_llm，其它情况均认为是无效的。
- tools (List[str], default: [] ) – LLM使用的工具名称列表。
- plan_llm (ModuleBase, default: None ) – planner要使用的LLM，可以是TrainableModule或OnlineChatModule。
- solve_llm (ModuleBase, default: None ) – solver要使用的LLM，可以是TrainableModule或OnlineChatModule。
- return_trace (bool, default: False ) – 是否返回中间步骤和工具调用信息。
- stream (bool, default: False ) – 是否以流式方式输出规划和解决过程。

```python
import lazyllm
import wikipedia
from lazyllm.tools import fc_register, ReWOOAgent
@fc_register("tool")
def WikipediaWorker(input: str):
...     '''
...     Worker that search for similar page contents from Wikipedia. Useful when you need to get holistic knowledge about people, places, companies, historical events, or other subjects. The response are long and might contain some irrelevant information. Input should be a search query.
...
...     Args:
...         input (str): search query.
...     '''
...     try:
...         evidence = wikipedia.page(input).content
...         evidence = evidence.split("\n\n")[0]
...     except wikipedia.PageError:
...         evidence = f"Could not find [{input}]. Similar: {wikipedia.search(input)}"
...     except wikipedia.DisambiguationError:
...         evidence = f"Could not find [{input}]. Similar: {wikipedia.search(input)}"
...     return evidence
...
@fc_register("tool")
def LLMWorker(input: str):
...     '''
...     A pretrained LLM like yourself. Useful when you need to act with general world knowledge and common sense. Prioritize it when you are confident in solving the problem yourself. Input can be any instruction.
...
...     Args:
...         input (str): instruction
...     '''
...     llm = lazyllm.OnlineChatModule(source="glm")
...     query = f"Respond in short directly with no extra words.\n\n{input}"
...     response = llm(query, llm_chat_history=[])
...     return response
...
tools = ["WikipediaWorker", "LLMWorker"]
llm = lazyllm.TrainableModule("GLM-4-9B-Chat").deploy_method(lazyllm.deploy.vllm).start()
# or llm = lazyllm.OnlineChatModule(source="sensenova")
agent = ReWOOAgent(llm, tools)
query = "What is the name of the cognac house that makes the main ingredient in The Hennchata?"
res = agent(query)
print(res)
'Hennessy '
```

### FunctionCallAgent

Function Call Agent 主要包括以下的流程：

行动（Action）：Agent 收到一个 query 后，它会直接行动，比如去调用某个工具；
观察（Observation）: Agent 观察到行动的反馈，比如工具的输出。
上面过程会不断循环往复，如果观察到行动的反馈没问题，满足了 query 的要求，或者达到了最大的迭代次数，那么 Agent 会退出并返回结果 response。

```python
from typing import Literal
import json
import lazyllm
from lazyllm.tools import fc_register, FunctionCall, FunctionCallAgent
@fc_register("tool")
def get_current_weather(location: str, unit: Literal["fahrenheit", "celsius"] = "fahrenheit"):
    ...
@fc_register("tool")
def get_n_day_weather_forecast(location: str, num_days: int, unit: Literal["celsius", "fahrenheit"] = 'fahrenheit'):
    ...
llm = lazyllm.TrainableModule("internlm2-chat-20b").start()  # or llm = lazyllm.OnlineChatModule()
tools = ["get_current_weather", "get_n_day_weather_forecast"]
fc = FunctionCall(llm, tools)
query = "What's the weather like today in celsius in Tokyo and Paris."
ret = fc(query)
print(f"ret: {ret}")
agent = FunctionCallAgent(llm, tools)
ret = agent(query)
print(f"ret: {ret}")
```
