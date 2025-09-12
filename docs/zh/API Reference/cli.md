# CLI

LazyLLM 提供了一个命令行接口（CLI），用于模型部署、依赖安装和运行服务等操作。本文档介绍了通过 `lazyllm.cli` 模块可用的核心命令及其用法示例。

---

## `lazyllm deploy`

根据输入命令执行模型部署或启动 MCP（Model Context Protocol）服务器。

### 功能 1：启动MCP服务器

当命令以 `mcp_server` 开头时，将启动一个 MCP 服务器，支持环境变量注入、SSE 服务端口配置等功能。

```bash
# 使用 uvx 和 mcp-server-fetch 启动 MCP 服务
lazyllm deploy mcp_server uvx mcp-server-fetch
```

其中：

- `mcp_server`：触发 MCP 服务器部署模式。
- `uvx`：用于执行 MCP 服务的命令（例如 node/npm/npx 或可执行程序）。
- `mcp-server-fetch`：MCP 服务器具体执行的包或模块名称。

```bash
# 启动 MCP 服务器，并配置环境变量和 SSE 端口
lazyllm deploy mcp_server -e GITHUB_TOKEN your_token --sse-port 8080 npx -- -y @modelcontextprotocol/server-github
```

其中：

- `mcp_server`：以 MCP 模式运行。
- `-e GITHUB_TOKEN your_token`：设置环境变量（可重复使用），这里设置名为 `GITHUB_TOKEN` 的变量。
- `sse-port 8080`：指定 SSE 服务监听的端口号为 `8080`。
- `--`：将后续参数传递给外部命令（如 npx）。
- `-y @modelcontextprotocol/server-github`：实际执行的 MCP server 模块及其参数。

可选参数说明：

- `sse-host`：SSE 服务监听的地址，默认值为 `127.0.0.1`。
- `allow-origin`：允许跨域请求的来源列表，可指定多个。
- `pass-environment`：是否传递所有本地环境变量（默认为 false）。

### 功能2：模型部署

当命令不以 `mcp_server` 开头时，默认以模型部署模式运行，支持多个框架（如 `vllm`、`lightllm` 等），并可启用 Web 聊天接口。

```bash
# 使用 vllm 部署 LLaMA3 模型，并开启聊天模式
lazyllm deploy llama3-chat --framework vllm --chat=true --top_p=0.9 --max_tokens=2048
```

其中：

- `llama3-chat`：要部署的模型名称。
- `framework=vllm`：指定部署使用的框架，支持：

    - `vllm`：高性能推理引擎。
    - `lightllm`：轻量化模型部署。
    - `lmdeploy`、`infinity`、`embedding`、`mindie`：其他特定部署框架。
    - `auto`：自动识别推荐框架。

- `chat=true`：是否开启 Web 聊天服务。等价写法还包括 `chat=1`, `chat=on`。
- `top_p=0.9`：设置推理时的 nucleus sampling 截断概率。
- `max_tokens=2048`：生成文本的最大 token 数。

补充说明：

- 其它参数可通过 `key=value` 的形式自定义传入，用于传递框架支持的推理配置。
- 如果不启用 `chat=true`，部署后将以后台服务形式持续运行。

---

## `lazyllm install`

用于安装额外功能组件组（extras groups）或指定的第三方 Python 包。

你可以安装：

- 预定义的组件组（如 `embedding`、`chat`、`finetune`）
- 明确指定的 Python 包（如 `openai`、`transformers`）

安装逻辑会自动处理版本依赖关系和兼容性问题，例如 `flash-attn` 与 PyTorch 的适配。

### 功能 1：安装组件组

```bash
# 安装 embedding 和 chat 组件组
lazyllm install embedding chat
```

其中：

- `embedding`、`chat`：预定义的功能组件组，分别用于嵌入模型和对话模型相关功能。

### 功能 2：安装第三方 Python 包

```bash
# 安装具体的第三方 Python 包
lazyllm install openai sentence-transformers
```

其中：

- `openai`、`sentence-transformers`：Python 包名称，可用于调用 OpenAI API 或加载向量模型等功能。

---

## `lazyllm run`

根据传入子命令执行对应的服务或流程。

### 功能 1. 启动聊天服务

```bash
lazyllm run chatbot --model chatglm3-6b --framework vllm
```

其中：

- `chatbot`：启动聊天机器人服务。
- `model`：指定要使用的模型名称，如 `chatglm3-6b`。
- `framework`：指定后端推理框架，支持 `lightllm`、`vllm`、`lmdeploy`。

### 功能 2. 启动 RAG 问答服务

```bash
lazyllm run rag --model bge-base --framework lightllm --documents /path/to/docs
```

其中：

- `rag`：启动基于检索增强生成的问答系统。
- `model`：指定模型名称，如 `bge-base`。
- `framework`：指定后端推理框架。
- `documents`：必填，指定包含知识文档的绝对路径。

### 功能 3. 执行 JSON 格式的计算图

```bash
lazyllm run workflow.json
```

其中：

- `workflow.json`：指定 JSON 工作流文件路径，运行对应计算流程。

### 功能 4. 启动训练服务

```bash
lazyllm run training_service
```

其中：

- `training_service`：启动模型训练服务，无需额外参数。

### 功能 5. 启动推理服务

```bash
lazyllm run infer_service
```

其中：

- `infer_service`：启动模型推理服务，无需额外参数。

> ❗ 注意事项：对于 `chatbot` 和 `rag`，`source` 和 `framework` 互斥，且只能从预设选项中选择。如果传入未知命令或参数不正确，会报错提示。
