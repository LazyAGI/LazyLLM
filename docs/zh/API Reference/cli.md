# CLI

LazyLLM 提供了一个命令行接口（CLI），用于模型部署、依赖安装和运行服务等操作。本文档介绍了通过 `lazyllm.cli` 模块可用的核心命令及其用法示例。

---

## `lazyllm deploy`

根据输入命令执行模型部署或启动 MCP（Model Context Protocol）服务器。

### 函数定义

```python
def deploy(commands: list[str]) -> None
```

当命令以 `mcp_server` 开头时，将启动一个 MCP 服务器，支持环境变量注入、SSE 服务端口配置等功能。

否则，将基于指定的部署框架（如 `vllm`、`lightllm` 等）部署模型。也可以通过设置 `--chat=true` 开启 Web 聊天服务模式。

### 参数

* `commands (list[str])`：CLI 风格的字符串参数列表。

### 使用示例（Bash）

```bash
# 使用 uvx 和 mcp-server-fetch 启动 MCP 服务
lazyllm deploy mcp_server uvx mcp-server-fetch

# 启动 MCP 服务器，并配置环境变量和 SSE 端口
lazyllm deploy mcp_server -e GITHUB_TOKEN your_token --sse-port 8080 npx -- -y @modelcontextprotocol/server-github

# 使用 vllm 部署 LLaMA3 模型，并开启聊天模式
lazyllm deploy llama3-chat --framework vllm --chat=true --top_p=0.9 --max_tokens=2048
```

---

## `lazyllm install`

用于安装额外功能组件组（extras groups）或指定的第三方 Python 包。

### 函数定义

```python
def install(commands: list[str]) -> None
```

你可以安装：

* 预定义的组件组（如 `embedding`、`chat`、`finetune`）
* 明确指定的 Python 包（如 `openai`、`transformers`）

安装逻辑会自动处理版本依赖关系和兼容性问题，例如 `flash-attn` 与 PyTorch 的适配。

### 参数

* `commands (list[str])`：组件组名称或包名称的字符串列表。

### 使用示例（Bash）

```bash
# 安装 embedding 和 chat 组件组
lazyllm install embedding chat

# 安装具体的第三方 Python 包
lazyllm install openai sentence-transformers
```

---

## `lazyllm run`

根据传入子命令执行对应的服务或流程。

### 函数定义

```python
def run(commands: list[str]) -> None
```

支持以下子命令：

* `chatbot`：启动基于指定模型和后端框架的聊天服务。
* `rag`：运行一个基于 RAG（检索增强生成）的问答系统。
* `.json`：执行一个基于 JSON 工作流图的计算流程。
* `training_service`：启动模型训练服务。
* `infer_service`：启动模型推理服务。

### 参数

* `commands (list[str])`：CLI 子命令及其参数。

### 使用示例（Bash）

```bash
# 启动聊天服务
lazyllm run chatbot --model chatglm3-6b --framework vllm

# 启动 RAG 问答服务
lazyllm run rag --model bge-base --framework lightllm --documents /path/to/docs

# 执行 JSON 格式的计算图
lazyllm run workflow.json

# 启动训练服务
lazyllm run training_service

# 启动推理服务
lazyllm run infer_service
```
