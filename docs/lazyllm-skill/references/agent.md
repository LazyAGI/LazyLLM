# Agent (智能体)

Agent 是 LazyLLM 的核心功能之一，用于构建能够自主规划、调用工具、执行任务的智能体系统。

## 核心概念

| 内容名称 | 内容功能 | 参考文档 |
|---------|---------|---------|
| 基础组件 | 工具注册，提示词模版, FunctionCallAgent, MCP | [基础组件使用](../assets/agent/basic.md) |
| 内置Agent | 内置的Agent: ReactAgent, PlanAndSolveAgent, ReWOOAgent, FunctionCallAgent | [内置Agent使用](../assets/agent/agent.md) |
| 工具 | 内置的工具: GoogleSearch, TencentSearch | [内置工具的使用](../assets/agent/tools.md) |

## 基础组件

### 1. 工具注册与定义

使用 `fc_register` 装饰器注册工具函数， Agent可以调用传入已经注册的工具函数。

### 2. ChatPrompter (提示词模板)

使用 `ChatPrompter` 构建结构化提示词。

### 3. FunctionCallAgent (函数调用 Agent)

`FunctionCallAgent` 是 LazyLLM 提供的函数调用 Agent，能够自动选择并调用工具。

### 4. MCPClient
MCP客户端，用于连接MCP服务器。同时支持本地服务器和sse服务器。
如果传入的 'command_or_url' 是一个 URL 字符串（以 'http' 或 'https' 开头），则将连接到远程服务器；否则，将启动并连接到本地服务器。

基础组件的使用参考: [基础组件使用](../assets/agent/basic.md)

## 常见 Agent 类型

内置的Agent类:

### ReactAgent

ReactAgent是按照 Thought->Action->Observation->Thought...->Finish 的流程一步一步的通过LLM和工具调用来显示解决用户问题的步骤，以及最后给用户的答案。

### PlanAndSolveAgent

PlanAndSolveAgent由两个组件组成，首先，由planner将整个任务分解为更小的子任务，然后由solver根据计划执行这些子任务，其中可能会涉及到工具调用，最后将答案返回给用户。

### ReWOOAgent

ReWOOAgent包含三个部分：Planner、Worker和Solver。其中，Planner使用可预见推理能力为复杂任务创建解决方案蓝图；Worker通过工具调用来与环境交互，并将实际证据或观察结果填充到指令中；Solver处理所有计划和证据以制定原始任务或问题的解决方案。

### FunctionCallAgent
行动（Action）：Agent 收到一个 query 后，它会直接行动，比如去调用某个工具；
观察（Observation）: Agent 观察到行动的反馈，比如工具的输出。

具体使用参考: [内置Agent使用](../assets/agent/agent.md)

## 工具

内置的工具：

### GoogleSearch

通过 Google 搜索指定的关键词。

### TencentSearch

腾讯搜索接口封装类，用于调用腾讯云的内容搜索服务。
提供对腾讯云搜索API的封装，支持关键词搜索和结果处理。

工具的具体使用方式参考: [内置工具的使用](../assets/agent/tools.md)

## 记忆管理

### 对话历史

```python
chat = lazyllm.OnlineChatModule()
history = []

# 第一轮
query1 = "什么是机器学习？"
answer1 = chat(query1, llm_chat_history=history)
history.append([query1, answer1])

# 第二轮
query2 = "它有哪些应用？"
answer2 = chat(query2, llm_chat_history=history)
history.append([query2, answer2])
```

## 最佳实践

- 选择合适的Agent: 从ReactAgent, PlanAndSolveAgent, ReWOOAgent中选择合适的Agent
- 设置合适的prompter: 使用ChatPrompter转入合适的提示词模版
- 实现和注册需要的工具: 使用fc_register进行工具注册，或者自行实现相应的工具
- 通过Flow或者函数编程将各部分组件进行组装

## 使用场景

- 自动代码生成与执行
- 智能搜索与信息提取
- 多领域专家协作
- 任务自动化执行
- 交互式 Web 应用
- 多轮对话系统
