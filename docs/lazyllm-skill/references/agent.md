# Agent (智能体)

Agent 是 LazyLLM 的核心功能之一，用于构建能够自主规划、调用工具、执行任务的智能体系统。  
Agent 与工具均来自 **lazyllm.tools** 模块：通过 `fc_register` 注册自定义工具，内置多种 Agent 类型与大量内置工具，也可通过 MCPClient 从 MCP 服务器获取工具。

## 核心概念

| 内容名称 | 内容功能 | 参考文档 |
|---------|---------|---------|
| 基础组件 | 工具注册 (fc_register)、提示词模版、FunctionCallAgent、MCPClient | [基础组件使用](../assets/agent/basic.md) |
| 内置 Agent | ReactAgent、PlanAndSolveAgent、ReWOOAgent、FunctionCallAgent | [内置 Agent 使用](../assets/agent/agent.md) |
| 内置与扩展工具 | 搜索、Http、SQL、CodeGenerator、MCP 等（均来自 lazyllm.tools） | [内置工具的使用](../assets/agent/tools.md) |

## 基础组件

### 1. 工具注册与定义

使用 `fc_register` 装饰器注册工具函数，Agent 可调用已注册的工具。

### 2. ChatPrompter（提示词模板）

使用 `ChatPrompter` 构建结构化提示词。

### 3. FunctionCallAgent（函数调用 Agent）

`FunctionCallAgent` 能够根据用户意图自动选择并调用工具。

### 4. MCPClient

MCP 客户端，用于连接 MCP 服务器，支持本地进程与 SSE 远程。传入 `command_or_url` 为 URL 时连接远程，否则启动本地服务器。可从 MCP 获取工具列表并交给 Agent 使用。

基础组件的使用参考: [基础组件使用](../assets/agent/basic.md)

## 常见 Agent 类型

### ReactAgent

按 Thought → Action → Observation → Thought … → Finish 流程，通过 LLM 与工具逐步解决问题并给出答案。

### PlanAndSolveAgent

由 Planner 将任务分解为子任务，Solver 按计划执行（含工具调用），最后汇总答案。

### ReWOOAgent

Planner 生成解决方案蓝图，Worker 通过工具与环境交互并填充证据，Solver 根据计划与证据给出最终答案。

### FunctionCallAgent

根据 query 直接选择并调用工具，根据工具返回进行后续推理。适合与 CodeGenerator 等组合做代码生成类应用。

具体使用参考: [内置 Agent 使用](../assets/agent/agent.md)

## 工具（lazyllm.tools）

Agent 可使用的工具均来自 **lazyllm.tools**，包括：

- **搜索类**: GoogleSearch、TencentSearch、BingSearch、WikipediaSearch、ArxivSearch、BochaSearch、StackOverflowSearch、SemanticScholarSearch、GoogleBooksSearch 等，用于信息检索。
- **通用工具**: HttpTool（HTTP 请求）、Weather、Calculator、JsonExtractor、JsonConcentrator 等。
- **SQL / 数据**: SqlManager、SqlCall，用于自然语言转 SQL 与表格问答，常与 ReactAgent 组合为 SQL Agent。
- **代码与能力**: CodeGenerator（与 LLM 配合生成代码）、MCPClient（从 MCP 服务器获取工具）。
- **自定义工具**: 使用 `fc_register('tool')` 注册任意函数供 Agent 调用。

工具的具体用法、参数与示例见 [内置工具的使用](../assets/agent/tools.md)。

## 记忆管理

### 对话历史

```python
chat = lazyllm.OnlineChatModule()
history = []

query1 = "什么是机器学习？"
answer1 = chat(query1, llm_chat_history=history)
history.append([query1, answer1])

query2 = "它有哪些应用？"
answer2 = chat(query2, llm_chat_history=history)
history.append([query2, answer2])
```

## 最佳实践

- 选择合适的 Agent: 从 ReactAgent、PlanAndSolveAgent、ReWOOAgent、FunctionCallAgent 中按任务复杂度选择。
- 设置合适的 prompter: 使用 ChatPrompter 传入清晰的指令与 extra_keys。
- 实现与注册工具: 用 fc_register 注册自定义工具，或直接使用 lazyllm.tools 中的搜索、SQL、Http、CodeGenerator 等。
- 通过 Flow 或函数编程组装 Agent、工具与上下游模块。

## 使用场景

- 自动代码生成与执行（CodeGenerator + FunctionCallAgent）
- 智能搜索与信息提取（搜索类工具 + ReactAgent）
- SQL / 表格问答（SqlManager、SqlCall + ReactAgent）
- 多领域专家协作、任务自动化、交互式 Web 应用、多轮对话系统
