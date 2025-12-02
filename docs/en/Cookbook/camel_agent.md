# Multi-turn Agent Dialogue System

This tutorial will guide you on how to build a multi-turn dialogue system using [LazyLLM](https://github.com/LazyLLM/LazyLLM), where two agents (e.g., "Stock Trader" and "Python Programmer") collaborate to complete a specific task. The system supports task specification, role-playing, memory retention, and tool usage.

!!! abstract "By the end of this tutorial, you will understand these key LazyLLM concepts"

    - How to use `OnlineChatModule` to simulate role styles by adjusting model temperature.
    - How to implement agent role-playing and task-driven behavior using custom prompts.
    - How to use the utility function `get_history` for maintaining dialogue memory.
    - How to use the [ReactAgent][lazyllm.tools.agent.ReactAgent] to build agents with role directives and problem-solving capabilities.
    - How to build a multi-turn dialogue system with task completion detection.

## Design Rationale
We have designed a dual-agent system based on the CAMEL collaborative paradigm to accomplish complex tasks (e.g., developing a stock trading bot). The design rationale is as follows:

Two distinct roles are introduced:
Stock Trader (user role): responsible for issuing step-by-step instructions;
Python Programmer (assistant role): responsible for implementing concrete solutions.
A high-temperature LLM is used to creatively refine the original vague task into a specific, actionable subtask.
Two ReactAgents, powered by a low-temperature LLM, engage in structured interaction under strict system prompts, ensuring role consistency and stable, deterministic outputs.
A tool registration mechanism (get_history) enables agents to retrieve conversation history, providing context awareness during multi-turn interactions.
Session-based history management and a maximum turn limit prevent infinite loops while supporting iterative, collaborative task progression.
The overall workflow follows a cycle: the Trader issues an instruction, the Programmer responds, and this loop continues until the task is complete—signaled by the Trader outputting <CAMEL_TASK_DONE>.
This design is well-suited for complex task automation scenarios that require role-based collaboration, stepwise decomposition, and tool-augmented reasoning.
![camel agent](../assets/camel.png)
## Project Dependencies

Install `lazyllm`:

```bash
pip install lazyllm
```

Import the relevant packages:

```python
from lazyllm import OnlineChatModule
from lazyllm.tools import fc_register, ReactAgent
from lazyllm.module.llms.onlinemodule.base.onlineChatModuleBase import StaticParams
from typing import List, Dict
```

## Feature Overview

* Supports **multi-agent role-playing and alternating instruction execution**, simulating realistic collaboration
* Introduces a **dialogue memory mechanism**, using utility functions to retrieve historical context
* Enables tool usage via `ReactAgent` for automatic selection and execution
* Uses high- and low-temperature models to simulate **creative generation vs. stable execution styles**
* Allows task completion signaling via `<CAMEL_TASK_DONE>`

## Step-by-Step Guide

### Step 1: Prepare Utility Functions

We first define a simple dialogue history cache and a tool function `get_history` that agents can use to retrieve recent conversation history.

```python
CHAT_HISTORY: Dict[str, List[str]] = {}


def add_to_history(session_id: str, message: str) -> str:
    '''Add a message to the conversation history for a specific session.

    Args:
        session_id (str): Unique identifier for the session.
        message (str): The message content to add.

    Returns:
        str: Confirmation message indicating success.
    '''
    if session_id not in CHAT_HISTORY:
        CHAT_HISTORY[session_id] = []
    CHAT_HISTORY[session_id].append(message)
    return f'Message added to session \'{session_id}\'.'

@fc_register('tool')
def get_history(session_id: str, limit: int = 5) -> str:
    '''Retrieve the most recent messages from the conversation history.

    Args:
        session_id (str): Unique identifier for the session.
        limit (int, optional): Number of recent messages to retrieve. Defaults to 5.

    Returns:
        str: Concatenated string of recent messages, or notice if empty.
    '''
    history = CHAT_HISTORY.get(session_id, [])
    if not history:
        return 'No history found for this session.'
    return '\n'.join(history[-limit:])


tools = ['get_history']
```

> 💡 Tool functions are registered using the `@fc_register("tool")` decorator, allowing agents to invoke them during conversations.

### Step 2: Initialize Language Models

We use two language model instances with different temperature settings to simulate a creative (high temperature) and stable (low temperature) dialogue style.

```python
temp_high = StaticParams(temperature=1.0)
temp_low = StaticParams(temperature=0.2)
llm_temp_high = OnlineChatModule(static_params=temp_high)
llm_temp_low = OnlineChatModule(static_params=temp_low)
```

### Step 3: Task Specification (Make the Task More Concrete)

We start with an initial task and use the high-temperature model to rewrite it into a more specific, actionable subtask.

```python
task = 'Develop a trading bot for the stock market'
word_limit = 50

rewrite_prompt = (
    'You can make a task more specific.\n'
    f'Here is a task that Python Programmer will help Stock Trader to complete: {task}.\n'
    'Please make it more specific. Be creative and imaginative.\n'
    f'Please reply with the specified task in {word_limit} words or less. Do not add anything else.'
)

specified_task = llm_temp_high(rewrite_prompt)
print('🎯 Specified Task:\n', specified_task)
```

### Step 4: Define Role Prompts (System Prompts)

We define two system prompts to constrain the user and assistant behaviors. Prompts include task information, interaction formats, and behavioral rules.

```python
assistant_role = 'Python Programmer'
user_role = 'Stock Trader'
```

#### User System Prompt

The user gives clear instructions to the assistant using the following format:

```python
user_sys_prompt = (
    f'Never forget you are a {user_role} and I am a {assistant_role}. Never flip roles! '
    'You will always instruct me.\n'
    'We share a common interest in collaborating to successfully complete a task.\n'
    'I must help you to complete the task.\n'
    f'Here is the task: {specified_task}. Never forget our task!\n'
    'You must instruct me based on my expertise and your needs to complete the task ONLY in the following two ways:\n\n'
    ...
)
```

#### Assistant System Prompt

The assistant responds to user instructions with complete solutions, always beginning the response like this:

```python
assistant_sys_prompt = (
    f'Never forget you are a {assistant_role} and I am a {user_role}. Never flip roles! Never instruct me!\n'
    'We share a common interest in collaborating to successfully complete a task.\n'
    'You must help me to complete the task.\n'
    f'Here is the task: {specified_task}. Never forget our task!\n'
    'I must instruct you based on your expertise and my needs to complete the task.\n\n'
    ...
)
```

> 📌 The prompt also explains how to use tools like `get_history(session_id="session_1")`.

### Step 5: Initialize Agents

We use `ReactAgent` to create agents, each with its own LLM, system prompt, and available tool list.

```python
from lazyllm.tools import ReactAgent

tools = ['get_history']

user_agent = ReactAgent(llm=llm_temp_low, tools=tools, return_trace=True, prompt=user_sys_prompt)
assistant_agent = ReactAgent(llm=llm_temp_low, tools=tools, return_trace=True, prompt=assistant_sys_prompt)
```

### Step 6: Begin Dialogue and Save History

The assistant starts the dialogue by prompting the user, who responds with the first instruction. Both sides’ messages are saved to history.

```python
assistant_msg = f'{user_sys_prompt} Now start to give me instructions one by one. Only reply with Instruction and Input.'
user_msg = f'{assistant_sys_prompt}'

instruction = user_agent(assistant_msg)
print(f'\n👤 {user_role}:\n\n{instruction}\n')
solution = assistant_agent(instruction)
print(f'\n🤖 {assistant_role}:\n\n{solution}\n')

session_id = 'session_1'
add_to_history(session_id, f'{user_role}: {instruction}')
add_to_history(session_id, f'{assistant_role}: {solution}')
```

### Step 7: Start Multi-turn Dialogue Loop

The agents continue interacting until the task is marked complete.

```python
max_turns = 5
n = 0
while n < max_turns:
    n += 1

    instruction = user_agent(assistant_msg)
    print(f'\n👤 {user_role}:\n\n{instruction}\n')
    add_to_history(session_id, f'{user_role}: {instruction}')

    if '<CAMEL_TASK_DONE>' in instruction:
        break

    solution = assistant_agent(instruction)
    print(f'\n🤖 {assistant_role}:\n\n{solution}\n')
    add_to_history(session_id, f'{assistant_role}: {solution}')

    assistant_msg = solution

```
## View Full Code
<details> 
<summary>Click to expand full code</summary>

```python
from lazyllm import OnlineChatModule
from lazyllm.tools import fc_register, ReactAgent
from lazyllm.module.llms.onlinemodule.base.onlineChatModuleBase import StaticParams
from typing import List, Dict

CHAT_HISTORY: Dict[str, List[str]] = {}


def add_to_history(session_id: str, message: str) -> str:
    '''Add a message to the conversation history for a specific session.

    Args:
        session_id (str): Unique identifier for the session.
        message (str): The message content to add.

    Returns:
        str: Confirmation message indicating success.
    '''
    if session_id not in CHAT_HISTORY:
        CHAT_HISTORY[session_id] = []
    CHAT_HISTORY[session_id].append(message)
    return f'Message added to session \'{session_id}\'.'


@fc_register('tool')
def get_history(session_id: str, limit: int = 5) -> str:
    '''Retrieve the most recent messages from the conversation history.

    Args:
        session_id (str): Unique identifier for the session.
        limit (int, optional): Number of recent messages to retrieve. Defaults to 5.

    Returns:
        str: Concatenated string of recent messages, or notice if empty.
    '''
    history = CHAT_HISTORY.get(session_id, [])
    if not history:
        return 'No history found for this session.'
    return '\n'.join(history[-limit:])


# Initialize LLMs with different temperatures
temp_high = StaticParams(temperature=1.0)
temp_low = StaticParams(temperature=0.2)
llm_temp_high = OnlineChatModule(static_params=temp_high)
llm_temp_low = OnlineChatModule(static_params=temp_low)

# Task specification
task = 'Develop a trading bot for the stock market'
word_limit = 50

rewrite_prompt = (
    'You can make a task more specific.\n'
    f'Here is a task that Python Programmer will help Stock Trader to complete: {task}.\n'
    'Please make it more specific. Be creative and imaginative.\n'
    f'Please reply with the specified task in {word_limit} words or less. Do not add anything else.'
)

specified_task = llm_temp_high(rewrite_prompt)
print('🎯 Specified Task:\n', specified_task)

# Roles
assistant_role = 'Python Programmer'
user_role = 'Stock Trader'

# System prompts
user_sys_prompt = (
    f'Never forget you are a {user_role} and I am a {assistant_role}. Never flip roles! '
    'You will always instruct me.\n'
    'We share a common interest in collaborating to successfully complete a task.\n'
    'I must help you to complete the task.\n'
    f'Here is the task: {specified_task}. Never forget our task!\n'
    'You must instruct me based on my expertise and your needs to complete the task ONLY in the following two ways:\n\n'
    '1. Instruct with a necessary input:\n'
    'Instruction: <YOUR_INSTRUCTION>\n'
    'Input: <YOUR_INPUT>\n\n'
    '2. Instruct without any input:\n'
    'Instruction: <YOUR_INSTRUCTION>\n'
    'Input: None\n\n'
    'The "Instruction" describes a task or question. The paired "Input" provides further context or information '
    'for the requested "Instruction".\n\n'
    'You must give me one instruction at a time.\n'
    'I must write a response that appropriately completes the requested instruction.\n'
    'I must decline your instruction honestly if I cannot perform the instruction due to physical, moral, legal '
    'reasons or my capability and explain the reasons.\n'
    'You should instruct me, not ask me questions.\n'
    'Now you must start to instruct me using the two ways described above.\n'
    'Do not add anything else other than your instruction and the optional corresponding input!\n'
    'Keep giving me instructions and necessary inputs until you think the task is completed.\n'
    'When the task is completed, you must only reply with a single word <CAMEL_TASK_DONE>.\n'
    'Never say <CAMEL_TASK_DONE> unless my responses have solved your task.'
)

assistant_sys_prompt = (
    f'Never forget you are a {assistant_role} and I am a {user_role}. Never flip roles! Never instruct me!\n'
    'We share a common interest in collaborating to successfully complete a task.\n'
    'You must help me to complete the task.\n'
    f'Here is the task: {specified_task}. Never forget our task!\n'
    'I must instruct you based on your expertise and my needs to complete the task.\n\n'
    'I must give you one instruction at a time.\n'
    'You must write a specific solution that appropriately completes the requested instruction.\n'
    'You must decline my instruction honestly if you cannot perform the instruction due to physical, moral, legal '
    'reasons or your capability and explain the reasons.\n'
    'Do not add anything else other than your solution to my instruction.\n'
    'You are never supposed to ask me any questions; you only answer questions.\n'
    'You are never supposed to reply with a flake solution. Explain your solutions.\n'
    'Your solution must be declarative sentences and simple present tense.\n'
    'Unless I say the task is completed, you should always start with:\n\n'
    'Solution: <YOUR_SOLUTION>\n\n'
    '<YOUR_SOLUTION> should be specific and provide preferable implementations and examples for task-solving.\n'
    'Always end <YOUR_SOLUTION> with: Next request.\n\n'
    'You can use the tool `get_history` to retrieve recent conversation history if needed.\n'
    'To use it, call: get_history(session_id="session_1").'
)

# Initialize agents
tools = ['get_history']
user_agent = ReactAgent(llm=llm_temp_low, tools=tools, return_trace=True, prompt=user_sys_prompt)
assistant_agent = ReactAgent(llm=llm_temp_low, tools=tools, return_trace=True, prompt=assistant_sys_prompt)

# Initial interaction
assistant_msg = f'{user_sys_prompt} Now start to give me instructions one by one. Only reply with Instruction and Input.'
user_msg = f'{assistant_sys_prompt}'

instruction = user_agent(assistant_msg)
print(f'\n👤 {user_role}:\n\n{instruction}\n')
solution = assistant_agent(instruction)
print(f'\n🤖 {assistant_role}:\n\n{solution}\n')

session_id = 'session_1'
add_to_history(session_id, f'{user_role}: {instruction}')
add_to_history(session_id, f'{assistant_role}: {solution}')

# Multi-turn loop
max_turns = 5
n = 0
while n < max_turns:
    n += 1

    instruction = user_agent(assistant_msg)
    print(f'\n👤 {user_role}:\n\n{instruction}\n')
    add_to_history(session_id, f'{user_role}: {instruction}')

    if '<CAMEL_TASK_DONE>' in instruction:
        break

    solution = assistant_agent(instruction)
    print(f'\n🤖 {assistant_role}:\n\n{solution}\n')
    add_to_history(session_id, f'{assistant_role}: {solution}')

    assistant_msg = solution
```
</details>

## Example Output

<pre><code>
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
</code></pre>
