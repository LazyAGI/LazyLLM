# FunctionCall

In order to increase the capabilities of the model so that it can not only generate text, but also perform specific tasks, query databases, interact with external system, etc. We define the [FunctionCall][lazyllm.tools.agent.FunctionCall] class to implement the tool calling capabilities of the model.
You can refer to the API documentation as [FunctionCall][lazyllm.tools.agent.FunctionCall]. Next, i will start with a simple example to introduce the design ideas of [FunctionCall][lazyllm.tools.agent.FunctionCall] in LazyLLM.

## FunctionCall Quick Start

[](){#define-function}
Suppose we are developing an application for querying the weather. Since weather information is time-sensitive, it is impossible to generate specific weather information simply by relying on a large model. This requires the model to call an external weather query tool to obtain realtime weather information. Now we define two weather query functions as follows:

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

In order for the large model to call the corresponding function and generate the corresponding parameters, when defining the function, it is necessary to add annotations to the function parameters and add a functional description to the function so that the large model knows the function of the function ans when it can be called.
This is the first step, defining the tool. The second step is to register the defined tool into LazyLLM so that you don't have to transfer functions when you use large models later. The registration method is as follows:

```python
from lazyllm.tools import fc_register
@fc_register("tool")
def get_current_weather(location: str, unit: Literal["fahrenheit", "celsius"]="fahrenheit"):
    ...

@fc_register("tool")
def get_n_day_weather_forecast(location: str, num_days: int, unit: Literal["celsius", "fahrenheit"]='fahrenheit'):
	...
```

The registration method is very simple. After importing `fc_register`, you can add it directly above the predefined function name in the manner of a decorator. Note that when adding, you need to specify the default group `tool`, and the default registered tool name is the name of the function being registered.

We can also register the tool under a different name by filling in the second parameter during registration, for example:

```python
from lazyllm.tools import fc_register

def get_current_weather(location: str, unit: Literal["fahrenheit", "celsius"]="fahrenheit"):
    ...

fc_register("tool")(get_current_weather, "another_get_current_weather")
```

Thus, the function `get_current_weather` is registered as a tool named `another_get_current_weather`.

If we do not intend to register the tool as globally visible, we can also pass the tool itself directly when calling FunctionCall, like this:

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

The code above directly passes the two functions we previously defined as tools, which are only visible within the generated `fc` instance. If you try to access these tools by name outside of `fc`, it will result in an error.

Then we can define the model and use [FunctionCall][lazyllm.tools.agent.FunctionCall], as shown below:

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

The result is output as a list, the first element is the current input, the second element is the output of the model, and the third element is the output of the tool. Because [FunctionCall][lazyllm.tools.agent.FunctionCall] is a single-round tool call process, the returned result includes not only the tool's return result, but also the current round's input and model results. If the tool call is not triggered, a string is returned directly. If you want to execute a complete function call, you need to use [FunctionCallAgent][lazyllm.tools.agent.FunctionCallAgent], as shown below:

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

In the above example, if the input query triggers a function call, [FunctionCall][lazyllm.tools.agent.FunctionCall] will return a list object, and [FunctionCallAgent][lazyllm.tools.agent.FunctionCallAgent] will iteratively execute the model call and tool call until the model considers that the information is sufficient to make a conclusion, or the number of iterations is exceeded. The number of iterations is set by max_retries, and the default value is 5.

Complete code is as follows:
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

!!! Note

    - When registering a function or tool, you must specify the default group `tool`, otherwise the model will not be able to use the corresponding tool.
    - When using the model, thers is no need to distinguish between [TrainableModule][lazyllm.module.TrainableModule] and [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule], because the output types of [TrainableModule][lazyllm.module.TrainableModule] and [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule] are designed to the same.

## Design Concept of FunctionCall
The design process of [FunctionCall][lazyllm.tools.agent.FunctionCall] is carried out in a bottom-up manner. First, since [FunctionCall][lazyllm.tools.agent.FunctionCall] must call LLM, the output format of the model must be consistent. Therefore, the outputs of [TrainableModule][lazyllm.module.TrainableModule] and [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule] are aligned. Then a single round of [FunctionCall][lazyllm.tools.agent.FunctionCall] is implemented, that is, LLM and tools are called once. Finally, the complete [FunctionCallAgent][lazyllm.tools.agent.FunctionCallAgent] is implemented, that is, [FunctionCall][lazyllm.tools.agent.FunctionCall] is iterated multiple times until the model iteration is completed or the maximum number of iterations is exceeded.

### TrainableModule and OnlineChatModule output alignment

1. Since the output of [TrainableModule][lazyllm.module.TrainableModule] is of string type, and the output of [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule] is in json format, in order to make [FunctionCall][lazyllm.tools.agent.FunctionCall] unaware of the model type when using the model, the output formats of the two models need to be unified.

2. First, for [TrainableModule][lazyllm.module.TrainableModule], specify the format of the model output tool_calls through prompt, and then parse the model output to obtain only the part generated by the model, that is, the real output of the model. For example:
```text
'\nI need to use the "get_current_weather" function to get the current weather in Tokyo and Paris. I will call the function twice, once for Tokyo and once for Paris.<|action_start|><|plugin|>\n{"name": "get_current_weather", "parameters": {"location": "Tokyo"}}<|action_end|><|im_end|>'
```

3. Then parse the model output through extractor, get the content field and the `name` and `arguments` fields of tool_calls, and then splice the results and output them in the format of:
```text
content<|tool_calls|>tool_calls
```
> - Among them, `content` represents the content information of the model output, `<|tool_calls|>` represents the separator, and `tool_calls` represents the string representation of the tool call, for example:
```text
'[{"id": "xxxx", "type": "function", "function": {"name": "func_name", "arguments": {"param1": "val1", "param2": "val2"}}}]'
```
> - Because there is no `id` field here is a unique random number generated to align with the [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule].
>
> - If no tool call is triggered, that is, there is no tool_cals and separator in the output, only content. If the tool call is triggered, but there is no content, the output does not contain content, only <|tool_calls>tool_calls.

	For example:
	```text
	'I need to use the "get_current_weather" function to get the current weather in Tokyo and Paris. I will call the function twice, once for Tokyo and once for Paris.<|tool_calls|>[{"id": "bd75399403224eb8972640eabedd0d46", "type": "function", "function":{"name": "get_current_weather", "arguments": "{\"location\": \"Tokyo\"}"}}]'
	```

4. Secondly, for [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule], since the online model supports streaming and non-streaming output, and whether [FunctionCall][lazyllm.tools.agent.FunctionCall] is triggered can only be known after receiving all the information, for the streaming output of [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule], it is necessary to convert stream to non-streaming first, that is, if the model is streaming output, then wait until all the messages are received before doing subsequent processing. For example:
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

5. After receiving the output of the model, the model output is parsed through the extractor to obtain the content field and the remaining fields in tool_calls except `index`. Here, `type` and `function` are retained because they are needed for the next round of model input. After extracting the content and tool_calls fields, splice and output them according to the above output format. For example:
```text
'<|tool_calls|>[{"id": "get_current_weather:0","type":"function","function":{"name":"get_current_weather","arguments":"{\n\"location\":\"Tokyo\",\n\"unit\":\"celsius\"\n}"}}]'
```

6. This ensures that the use of [TrainableModule][lazyllm.module.TrainableModule] and [OnlineChatModule][lazyllm.module.onlineChatModule.OnlineChatModule] is consistent. In order to adapt to the application of [FunctionCall][lazyllm.tools.agent.FunctionCall], the output of the model is passed through FunctionCallFormatter. The function of FunctionCallFormatter is to parse the output of the model and obtain content and tool_calls information. For example:
```text
[{"id": "bd75399403224eb8972640eabedd0d46", "type": "function", "function":{"name": "get_current_weather", "arguments": {"location": "Tokyo"}}}]或者
[{"id": "get_current_weather:0","type":"function","function":{"name":"get_current_weather","arguments":{"location":"Tokyo","unit":"celsius"}}},{"id": "get_current_weather:1","type":"function","function":{"name":"get_current_weather","arguments":{"location":"Paris","unit":"celsius"}}}]
```
If the tool is not called, the output is of type str, which is the output of the model. For example:
```text
今天的东京天气温度为10度 Celsius。
今天东京的天气温度是10摄氏度，而巴黎的天气温度是22摄氏度。
```

!!! Note

    - The output format of the model is `content<|tool_calls|>tool_calls`, the delimiter is fixed, and the delimiter is used to determine whether it is a tool call.
    - The tool call information contains the tool's `name` and `arguments` fields as well as the `id`, `type` and `function` fields.


### FunctionCall Output Flow

[FunctionCall][lazyllm.tools.agent.FunctionCall] is a tool call that processes a single round.

> - Non-function call request
```text
Hello World!
```
> - function call request
```text
What's the weather like today in Tokyo.
```

1. The input comes in and first calls the large model, for example:
> - Non-function call request
```text
Hello! How can I assist you today?
```
> - function call request
```text
[{"id": "bd75399403224eb8972640eabedd0d46","type":"function", "function":{"name": "get_current_weather", "arguments": "{"location": "Tokyo"}"}}]
```

2. The output of the model is parsed by the parse
> - Non-function call request
```text
Hello! How can I assist you today?
```
> - function call request
```text
[{"name": "get_current_weather", "arguments": {"location": "Tokyo"}}]
```

3. Determine whether the parsed output is a tool call. If it is a tool call, call the corresponding tool through the [ToolManager][lazyllm.tools.agent.ToolManager] tool management calss.
> - function call request
```text
'{"location": "Tokyo", "temperature": "10", "unit": "celsius"}'
```

4. If it is not a tool call, the output is performed directly. If it is a tool call, the current output round input, model output and tool return result are packaged together and then output.
> - Non-function call request
```text
Hello! How can I assist you today?
```
> - function call request
```text
[{'tool_call_id': 'bd75399403224eb8972640eabedd0d46', 'name': 'get_current_weather', 'content': '{"location": "Tokyo", "temperature": "10", "unit": "celsius"}', 'role': 'tool'}]
```

### Function Call Agent Output Process

[FunctionCallAgent][lazyllm.tools.agent.FunctionCallAgent] is the process that handles the complete tool call.
> - Agent Input
```text
What's the weather like today in Tokyo.
```

1. Input comes in and calls the [FunctionCall][lazyllm.tools.agent.FunctionCall] module directly.
> - FunctionCall output results
```text
[{'tool_call_id': 'get_current_weather:0', 'name': 'get_current_weather', 'content': '{"location": "Tokyo", "temperature": "10", "unit": "celsius"}', 'role': 'tool'}]
```
> - Non-FunctionCall output results
```text
今天的东京天气温度是10度 Celsius。
```

2. Determine whether the result of [FunctionCall][lazyllm.tools.agent.FunctionCall] is a tool call or the maximum number of iterations has been reached. If it is a tool call, return to step 1. If it is not a tool call or the maximum number of iterations has been reached, continue to move on.

3. If the maximum number of iterations is reached, an exception is thrown. If the model generates results normally, the results are output directly.
> - Throws an exception after reaching the maximum number of iterations.
```text
ValueError: After retrying 5 times, the function call agent still failed to call successfully.
```
> - Normal result output
```text
今天的东京天气温度是10度 Celsius。
```

## Advanced Agent
An agent is an artificial entity that can use sensors to sense the surrounding environment, make decisions autonomously, and then use actuators to perform corresponding actions. It has autonomy (can run independently without human intervention), responsiveness (can sense environmental changes and respond), sociality (multiple agents can coordinate with each other to complete tasks together), and adaptability (can continuously improve its own performance to better complete tasks).

[FunctionCallAgent][lazyllm.tools.agent.FunctionCallAgent] is the most basic agent. It generates tool call parameters through the big model, then calls the tool and feeds the tool return results back to the big model. This is repeated until the model generates the final answer or the maximum number of iterations is exceeded. There are also some problems in this process. For example, the process of generating a big model is a black box, and people do not know the specific reasoning process, or when faced with complex problems, the big model cannot directly give answers, etc. In response to these problems, researchers have proposed various solutions and formed various advanced agents. Below we introduce the implementation of several advanced agents is LazyLLM.

### React

[paper](https://arxiv.org/abs/2210.03629)

Idea: [ReactAgent][lazyllm.tools.agent.ReactAgent] handles problems according to the process of "Thought->Action->Observation->Thought...->Finish". Thought shows how the model solves problems step by step. Action represents the information of tool calls. Observation is the result returned by the tool. Finish is the final answer to the problem.

The execution process of this agent is the same as that of [FuncitonCallAgent][lazyllm.tools.agent.FunctionCallAgent]. The Only difference is the prompt, and the [ReactAgent][lazyllm.tools.agent.ReactAgent] must have a thought output at each step, while the ordinary [FunctionCallAgent][lazyllm.tools.agent.FunctionCallAgent] may only output the tool call information without content. The example is as follows:
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

[paper](https://arxiv.org/abs/2305.04091)

Idea: [PlanAndSolveAgent][lazyllm.tools.agent.PlanAndSolveAgent] consists of two components: first, decomposing the whole task into smaller subtasks, and second, executing these subtasks according to the plan. Finally, the results are output as answers.

1、After the input comes in, it first passes through the planner model to generate a solution plan for the problem.
```text
Plan:\n1. Identify the given expression: 20 + (2 * 4)\n2. Perform the multiplication operation inside the parentheses: 2 * 4 = 8\n3. Add the result of the multiplication to 20: 20 + 8 = 28\n4. The final answer is 28.\n\nGiven the above steps taken, the answer to the expression 20 + (2 * 4) is 28. <END_OF_PLAN>
```

2、Parse the generated plan so that the solver model can be executed according to the plan.

3、[FunctionCallAgent][lazyllm.tools.agent.FunctionCallAgent] is called for each step of the plan, and the final result is returned as final answer.
```text
The final answer is 28.
```

The following is an example:
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

[paper](https://arxiv.org/abs/2305.18323)

Idea: [ReWOOAgent][lazyllm.tools.agent.ReWOOAgent] consists of three parts: Planner, Worker and Solver. Among them, Planner uses predictable reasoning ability to create a solution blueprint for complex tasks; Woker interacts with the environment through tool calls and fills actual evidence or observations into instructions; Solver processes all plans and evidence to develop solutions to the original tasks or problems.

1. The input first calls the planner model to generate a blueprint for solving the problem.
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

2. Parse the generated plan blueprint, call the corresponding tool, and fill the results returned by the tool into the corresponding instructions.
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

3. Stitch the results of the plan and tool execution together, then call the solver model to generate the final answer.
```text
'\nHennessy '
```

Here is an example：
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
