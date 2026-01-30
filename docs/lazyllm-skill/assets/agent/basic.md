# 基础组件

LazyLLM提供的用于构成Agent的基础组件

## 1. 工具注册与定义

使用 `fc_register` 装饰器注册工具函数。

### 基础工具定义

```python
from lazyllm.tools import fc_register

@fc_register('tool')
def my_tool(param: str) -> str:
    """
    工具描述，用于指导 Agent 何时使用该工具

    Args:
        param (str): 参数描述

    Returns:
        str: 返回值描述
    """
    return f"处理结果: {param}"
```

### 注册装饰器参数

- `'tool'`: 工具组名称，用于分类管理工具

### 工具最佳实践

- 函数必须有明确的类型注解
- docstring 需要详细描述功能、参数和返回值
- 返回值应该是可序列化的基本类型

## 2. ChatPrompter (提示词模板)

使用 `ChatPrompter` 构建结构化提示词。

### 基本用法

```python
import lazyllm

llm = lazyllm.OnlineChatModule()

instruction = """你是一个专业的助手，请根据以下信息完成任务。
请遵循以下规则：
1. 准确理解用户需求
2. 提供有用的回答
"""

prompter = lazyllm.ChatPrompter(
    instruction=instruction,
    extra_keys=['context_str', 'history']
)

llm.prompt(prompter)
```

### 参数说明

- `instruction`: 系统指令
- `extra_keys`: 额外的提示词字段名列表

### 使用提示词

```python
result = llm({
    "query": "用户问题",
    "context_str": "上下文信息",
    "history": "历史对话"
})
```

## 3. FunctionCallAgent (函数调用 Agent)

`FunctionCallAgent` 是 LazyLLM 提供的函数调用 Agent，能够自动选择并调用工具。

### 基本用法

```python
from lazyllm.tools import FunctionCallAgent, fc_register

# 定义工具
@fc_register('tool')
def search_tool(query: str) -> str:
    """搜索工具，用于查找信息"""
    return f"搜索结果: {query}"

@fc_register('tool')
def calculate_tool(num1: float, num2: float) -> float:
    """计算工具，用于数学运算"""
    return num1 + num2

# 创建 Agent
llm = lazyllm.OnlineChatModule()
agent = FunctionCallAgent(llm, tools=['search_tool', 'calculate_tool'])

# 使用 Agent
result = agent("帮我搜索关于 AI 的信息")
print(result)
```

### 参数说明

- `llm`: 语言模型
- `tools`: 工具名称列表


## 4. MCPClient
MCP客户端，用于连接MCP服务器。同时支持本地服务器和sse服务器。
如果传入的 'command_or_url' 是一个 URL 字符串（以 'http' 或 'https' 开头），则将连接到远程服务器；否则，将启动并连接到本地服务器。

参数:

- command_or_url (str) – 用于启动本地服务器或连接远程服务器的命令或 URL 字符串。
- args (list[str], default: None ) – 用于启动本地服务器的参数列表；如果要连接远程服务器，则无需此参数。（默认值为[]）
- env (dict[str, str], default: None ) – 工具中使用的环境变量，例如一些 API 密钥。（默认值为None）
- headers (dict[str, Any], default: None ) – 用于sse客户端连接的HTTP头。（默认值为None）
- timeout (float, default: 5 ) – sse客户端连接的超时时间，单位为秒。(默认值为5)

```python
from lazyllm.tools import MCPClient
mcp_server_configs = {
...     "filesystem": {
...         "command": "npx",
...         "args": [
...             "-y",
...             "@modelcontextprotocol/server-filesystem",
...             "./",
...         ]
...     }
... }
file_sys_config = mcp_server_configs["filesystem"]
file_client = MCPClient(
...     command_or_url=file_sys_config["command"],
...     args=file_sys_config["args"],
... )
from lazyllm import OnlineChatModule
from lazyllm.tools.agent.reactAgent import ReactAgent
llm=OnlineChatModule(source="deepseek", stream=False)
agent = ReactAgent(llm.share(), file_client.get_tools())
print(agent("Write a Chinese poem about the moon, and save it to a file named 'moon.txt".))
```
