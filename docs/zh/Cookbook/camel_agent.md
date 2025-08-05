# 多轮智能体对话系统

本教程将教你如何使用 [LazyLLM](https://github.com/LazyLLM/LazyLLM) 搭建一个多轮对话系统，其中两个智能体（如“股票交易员”和“Python 程序员”）协同完成特定任务。该系统支持任务细化、角色扮演、历史记忆以及工具调用。

!!! abstract "通过本节您将学习到 LazyLLM 的以下要点"

    - 如何使用 `OnlineChatModule` 设置不同温度的语言模型模拟角色风格。
    - 如何通过自定义 Prompt 实现智能体的角色扮演与任务驱动。
    - 如何使用工具函数 `get_history` 实现对话上下文记忆。
    - 如何使用 [ReactAgent][lazyllm.tools.agent.ReactAgent] 搭建具备角色指令与解决能力的 Agent。
    - 如何搭建一个支持多轮交互与任务终止机制的对话系统。

## 项目依赖

安装 `lazyllm`：

```bash
pip install lazyllm
```

导入相关包：

```python
from lazyllm import OnlineChatModule
from lazyllm.tools import fc_register, ReactAgent
from typing import List, Dict

from lazyllm.module.llms.onlinemodule.base.onlineChatModuleBase import StaticParams
```

## 功能简介

* 支持多智能体 **角色扮演与交替指令执行**，模拟真实协作流程
* 引入**对话记忆机制**，可调用工具函数获取历史上下文
* 支持通过 `ReactAgent` 自动选择与执行工具
* 使用高低温度模型分别模拟**创意生成与稳定执行风格**
* 支持通过 `<CAMEL_TASK_DONE>` 实现任务终止判断

## 步骤详情

### Step 1：准备工具函数

我们首先定义一个简单的对话历史缓存，以及一个能被注册为工具的函数 `get_history`，用于智能体查询最近几轮对话。

```python
CHAT_HISTORY: Dict[str, List[str]] = {}

def add_to_history(session_id: str, message: str) -> str:
    '''
    Add a message to the conversation history for a specific session.

    Args:
        session_id (str): Unique identifier for the session.
        message (str): The message content to add.

    Returns:
        str: Confirmation message indicating success.
    '''
    if session_id not in CHAT_HISTORY:
        CHAT_HISTORY[session_id] = []
    CHAT_HISTORY[session_id].append(message)
    return f"Message added to session '{session_id}'."

@fc_register("tool")
def get_history(session_id: str, limit: int = 5) -> str:
    '''
    Retrieve the most recent messages from the conversation history.

    Args:
        session_id (str): Unique identifier for the session.
        limit (int, optional): Number of recent messages to retrieve. Defaults to 5.

    Returns:
        str: Concatenated string of recent messages, or notice if empty.
    '''
    history = CHAT_HISTORY.get(session_id, [])
    if not history:
        return "No history found for this session."
    return "\n".join(history[-limit:])

tools = ['get_history']
```

> 💡 工具函数通过 `@fc_register("tool")` 装饰器注册，允许智能体在对话中调用。

### Step 2：初始化语言模型

我们使用两个不同温度的语言模型对象模拟高创造性（高温度）与稳定执行（低温度）的对话风格。

```python
temp_high = StaticParams(temperature=1.0)
temp_low = StaticParams(temperature=0.2)
llm_temp_high = OnlineChatModule(static_params=temp_high)
llm_temp_low = OnlineChatModule(static_params=temp_low)
```

### Step 3：任务重写（让任务更具体）

我们从一个初始任务出发，利用高温度模型将其细化成更有操作性的子任务。

```python
task = "Develop a trading bot for the stock market"
word_limit = 50

rewrite_prompt = f"""
You can make a task more specific.
Here is a task that Python Programmer will help Stock Trader to complete: {task}.
Please make it more specific. Be creative and imaginative.
Please reply with the specified task in {word_limit} words or less. Do not add anything else.
"""

specified_task = llm_temp_high(rewrite_prompt)
print("🎯 Specified Task:\n", specified_task)
```

### Step 4：定义角色提示词（System Prompt）

我们设置两个系统提示词，分别用于定义用户和助手的行为规范。提示词中嵌入了任务信息、交互格式和行为约束。

```python
assistant_role = "Python Programmer"
user_role = "Stock Trader"
```

#### 用户系统提示词

用户需要以“指令 + 输入”的方式给助手下达明确任务：

```python
user_sys_prompt = f"""
Never forget you are a {user_role} and I am a {assistant_role}. Never flip roles! You will always instruct me.
We share a common interest in collaborating to successfully complete a task.
I must help you to complete the task.
Here is the task: {specified_task}. Never forget our task!
You must instruct me based on my expertise and your needs to complete the task ONLY in the following two ways:

1. Instruct with a necessary input:
Instruction: <YOUR_INSTRUCTION>
Input: <YOUR_INPUT>

2. Instruct without any input:
Instruction: <YOUR_INSTRUCTION>
Input: None

The "Instruction" describes a task or question. The paired "Input" provides further context or information for the requested "Instruction".

You must give me one instruction at a time.
I must write a response that appropriately completes the requested instruction.
I must decline your instruction honestly if I cannot perform the instruction due to physical, moral, legal reasons or my capability and explain the reasons.
You should instruct me, not ask me questions.
Now you must start to instruct me using the two ways described above.
Do not add anything else other than your instruction and the optional corresponding input!
Keep giving me instructions and necessary inputs until you think the task is completed.
When the task is completed, you must only reply with a single word <CAMEL_TASK_DONE>.
Never say <CAMEL_TASK_DONE> unless my responses have solved your task.
"""
```

#### 助手系统提示词

助手需要对用户指令返回完整的解决方案，并始终使用如下格式开始：

```python
assistant_sys_prompt = f"""
Never forget you are a {assistant_role} and I am a {user_role}. Never flip roles! Never instruct me!
We share a common interest in collaborating to successfully complete a task.
You must help me to complete the task.
Here is the task: {specified_task}. Never forget our task!
I must instruct you based on your expertise and my needs to complete the task.

I must give you one instruction at a time.
You must write a specific solution that appropriately completes the requested instruction.
You must decline my instruction honestly if you cannot perform the instruction due to physical, moral, legal reasons or your capability and explain the reasons.
Do not add anything else other than your solution to my instruction.
You are never supposed to ask me any questions; you only answer questions.
You are never supposed to reply with a flake solution. Explain your solutions.
Your solution must be declarative sentences and simple present tense.
Unless I say the task is completed, you should always start with:

Solution: <YOUR_SOLUTION>

<YOUR_SOLUTION> should be specific and provide preferable implementations and examples for task-solving.
Always end <YOUR_SOLUTION> with: Next request.

You can use the tool `get_history` to retrieve recent conversation history if needed.
To use it, call: get_history(session_id="session_1").
"""
```

> 📌 提示词中还说明了如何调用工具，例如 `get_history(session_id="session_1")`。

### Step 5：初始化智能体

我们使用 `ReactAgent` 来创建智能体，每个智能体都包含 LLM、系统提示词和可用工具列表。

```python
from lazyllm.tools import ReactAgent

tools = ['get_history']

user_agent = ReactAgent(llm=llm_temp_low, tools=tools, return_trace=True, prompt=user_sys_prompt)
assistant_agent = ReactAgent(llm=llm_temp_low, tools=tools, return_trace=True, prompt=assistant_sys_prompt)
```

### Step 6：初始化对话并保存历史

对话从助手提示用户开始，用户再作出第一条响应。双方的发言都被存储进历史记录中。

```python
assistant_msg = f"{user_sys_prompt} Now start to give me instructions one by one. Only reply with Instruction and Input."
user_msg = f"{assistant_sys_prompt}"

instruction = user_agent(assistant_msg)
print(f"\n👤 {user_role}:\n\n{instruction}\n")
solution = assistant_agent(instruction)
print(f"\n🤖 {assistant_role}:\n\n{solution}\n")

session_id = "session_1"
add_to_history(session_id, f"{user_role}: {instruction}")
add_to_history(session_id, f"{assistant_role}: {solution}")")
```

### Step 7：启动多轮对话循环

通过循环结构实现多轮互动，用户给出指令，助手给出解决方案，直到任务完成。

```python
max_turns = 5
n = 0

while n < max_turns:
    n += 1

    # user_agent 生成指令
    instruction = user_agent(assistant_msg)
    print(f"\n👤 {user_role}:\n\n{instruction}\n")
    add_to_history(session_id, f"{user_role}: {instruction}")

    if "<CAMEL_TASK_DONE>" in instruction:
        break

    # assistant_agent 生成解决方案
    solution = assistant_agent(instruction)
    print(f"\n🤖 {assistant_role}:\n\n{solution}\n")
    add_to_history(session_id, f"{assistant_role}: {solution}")

    assistant_msg = solution
```

## 示例运行结果

```text
🎯 Specified Task:
 Develop a Python trading bot that uses real-time data to execute trades based on a MACD strategy, with adjustable signal and fast EMA periods, and integrates with a broker API for automated trading.

👤 Stock Trader:

Instruction: Define the structure of the Python trading bot.
Input: None


🤖 Python Programmer:

Solution: The Python trading bot structure for a MACD strategy with adjustable signal and fast EMA periods, integrating with a broker API, includes the following components:

1. **Configuration Module**: 
   - Contains settings for API keys, broker details, and adjustable parameters like fast EMA period, slow EMA period, and signal line period.

2. **Data Handling Module**: 
   - Retrieves real-time market data using the broker API.
   - Manages historical data storage and retrieval.

3. **MACD Calculation Module**: 
   - Computes the MACD line by subtracting the slow EMA from the fast EMA.
   - Calculates the signal line as the EMA of the MACD line.

4. **Signal Generation Module**: 
   - Generates buy or sell signals based on the crossover of the MACD line and the signal line.

5. **Trade Execution Module**: 
   - Sends trade orders to the broker API based on the generated signals.
   - Manages order status and execution feedback.

6. **Logging and Monitoring Module**: 
   - Logs all trading activities, signals, and orders.
   - Monitors the bot's performance and health.

7. **Main Execution Loop**: 
   - Orchestrates the flow of data between modules.
   - Continuously checks for new data, updates MACD calculations, generates signals, and executes trades.

Next request.


👤 Stock Trader:

Instruction: Define the structure of the Python trading bot.
Input: None


🤖 Python Programmer:

Solution: The Python trading bot structure for a MACD strategy with adjustable signal and fast EMA periods, integrating with a broker API, includes the following components:

1. **Configuration Module**: 
   - Contains settings for API keys, broker details, and adjustable parameters like the fast EMA period, slow EMA period, signal line period, and any other strategy-specific settings.

2. **Data Handling Module**: 
   - Responsible for fetching real-time market data using the broker API.
   - Includes functions to retrieve historical data for backtesting and strategy evaluation.

3. **MACD Calculation Module**: 
   - Computes the MACD line and the signal line based on the fast and slow EMA periods.
   - Generates trading signals (buy/sell) when the MACD line crosses the signal line.

4. **Trading Strategy Module**: 
   - Implements the logic for entering and exiting trades based on MACD signals.
   - Manages position sizing, risk management, and trade execution.

5. **Broker Integration Module**: 
   - Connects to the broker API to execute trades.
   - Sends orders to buy or sell based on the trading signals generated by the strategy module.

6. **Logging and Monitoring Module**: 
   - Logs all trading activities, signals, and performance metrics.
   - Monitors the health of the bot and handles any errors or exceptions.

7. **Backtesting Module**: 
   - Evaluates the strategy using historical data to test its effectiveness.
   - Generates performance reports and metrics like Sharpe ratio, maximum drawdown, etc.

8. **User Interface Module**: 
   - Provides a dashboard or console for users to monitor the bot's performance and adjust settings in real-time.

Next request.


👤 Stock Trader:

Instruction: Provide a sample code structure for the Configuration Module of the trading bot.
Input: None


🤖 Python Programmer:

Solution: The Configuration Module of the trading bot should handle the settings and parameters required for the bot's operation, including API keys, trading pairs, and MACD strategy parameters. Below is a sample code structure for the Configuration Module:

```python
# config.py

class Config:
    def __init__(self):
        self.api_key = 'your_api_key_here'
        self.api_secret = 'your_api_secret_here'
        self.base_currency = 'BTC'
        self.quote_currency = 'USDT'
        self.trading_pair = f"{self.base_currency}/{self.quote_currency}"
        self.fast_ema_period = 12
        self.slow_ema_period = 26
        self.signal_ema_period = 9
        self.api_url = 'https://api.broker.com'
        self.order_size = 0.01  # Example order size in base currency

    def get_api_credentials(self):
        return self.api_key, self.api_secret

    def get_trading_pair(self):
        return self.trading_pair

    def get_macd_parameters(self):
        return self.fast_ema_period, self.slow_ema_period, self.signal_ema_period

    def get_order_size(self):
        return self.order_size

    def get_api_url(self):
        return self.api_url
```

Next request.


👤 Stock Trader:

Instruction: Provide an example of how to use the Config class to retrieve the MACD parameters and API URL.
Input: None


🤖 Python Programmer:

Solution: To use the Config class for retrieving the MACD parameters and API URL, you can define a class with attributes for the MACD parameters (fast EMA period, slow EMA period, signal period) and the API URL. Then, create a method to retrieve these values. Here is an example implementation:

```python
class Config:
    def __init__(self, fast_ema_period, slow_ema_period, signal_period, api_url):
        self.fast_ema_period = fast_ema_period
        self.slow_ema_period = slow_ema_period
        self.signal_period = signal_period
        self.api_url = api_url

    def get_macd_parameters(self):
        return {
            "fast_ema_period": self.fast_ema_period,
            "slow_ema_period": self.slow_ema_period,
            "signal_period": self.signal_period
        }

    def get_api_url(self):
        return self.api_url

# Example usage
config = Config(fast_ema_period=12, slow_ema_period=26, signal_period=9, api_url="https://api.broker.com")
macd_params = config.get_macd_parameters()
api_url = config.get_api_url()

print("MACD Parameters:", macd_params)
print("API URL:", api_url)
```

Next request.


👤 Stock Trader:

Instruction: Provide the necessary steps to integrate the Config class with a broker API for automated trading using the MACD strategy.
Input: None


🤖 Python Programmer:

Solution: To integrate the Config class with a broker API for automated trading using the MACD strategy, follow these steps:

1. Define the Config class with necessary attributes for API credentials, MACD parameters, and other trading configurations.
2. Implement a method within the Config class to load API credentials from a secure file or environment variables.
3. Create a method to initialize the broker API connection using the credentials from the Config class.
4. Develop a method to fetch real-time market data using the broker API.
5. Implement the MACD calculation logic within the Config class or a separate TradingStrategy class that references the Config class for parameters.
6. Write a method to execute trades by sending trade orders to the broker API based on MACD signals.
7. Ensure error handling and logging are in place for all API interactions and trading actions.
8. Test the integration with the broker API using historical data or a broker's paper trading feature before going live.

Next request.


👤 Stock Trader:

Instruction: Define the structure of the Config class with necessary attributes for API credentials, MACD parameters, and other trading configurations.
Input: None


🤖 Python Programmer:

Solution: The Config class should encapsulate all necessary configurations for the trading bot, including API credentials, MACD parameters, and other trading settings. Here is a possible structure for the Config class:

```python
class Config:
    def __init__(self, api_key: str, api_secret: str, broker_url: str, 
                 fast_ema_period: int, slow_ema_period: int, signal_ema_period: int, 
                 trading_symbol: str, time_frame: str, order_size: float, 
                 take_profit: float, stop_loss: float):
        self.api_key = api_key
        self.api_secret = api_secret
        self.broker_url = broker_url
        self.fast_ema_period = fast_ema_period
        self.slow_ema_period = slow_ema_period
        self.signal_ema_period = signal_ema_period
        self.trading_symbol = trading_symbol
        self.time_frame = time_frame
        self.order_size = order_size
        self.take_profit = take_profit
        self.stop_loss = stop_loss

# Example of initializing the Config class
config = Config(
    api_key="your_api_key",
    api_secret="your_api_secret",
    broker_url="https://brokerapi.com",
    fast_ema_period=12,
    slow_ema_period=26,
    signal_ema_period=9,
    trading_symbol="BTC/USD",
    time_frame="1h",
    order_size=0.01,
    take_profit=0.02,
    stop_loss=0.01
)
```

Next request.
```
