# CLI

LazyLLM provides a Command Line Interface (CLI) for model deployment, dependency installation, and running various services. This document introduces the core commands available via the `lazyllm.cli` module and usage examples.

---

## `lazyllm deploy`

Executes model deployment or starts an MCP (Model Context Protocol) server based on the input commands.

### Function 1: Start MCP Server

When the command starts with `mcp_server`, it will launch an MCP server supporting environment variable injection, SSE server port configuration, and other features.

```bash
# Start MCP service using uvx and mcp-server-fetch
lazyllm deploy mcp_server uvx mcp-server-fetch
```

Args:

- `mcp_server`: triggers MCP server deployment mode.
- `uvx`: command used to run the MCP service (e.g., node/npm/npx or an executable).
- `mcp-server-fetch`: specific package or module name to run the MCP server.

```bash
# Start MCP server with environment variables and SSE port configured
lazyllm deploy mcp_server -e GITHUB_TOKEN your_token --sse-port 8080 npx -- -y @modelcontextprotocol/server-github
```

Args:

- `mcp_server`: run in MCP server mode.
- `-e GITHUB_TOKEN your_token`: set environment variables (can be used multiple times); here setting `GITHUB_TOKEN`.
- `sse-port 8080`: specify the SSE server listening port as `8080`.
- `--`: passes subsequent parameters to the external command (like npx).
- `-y @modelcontextprotocol/server-github`: the actual MCP server module and its parameters.

Optional parameters:

- `sse-host`: SSE server listening address, default is `127.0.0.1`.
- `allow-origin`: list of allowed origins for CORS; can specify multiple.
- `pass-environment`: whether to pass all local environment variables (default is false).

### Function 2: Model Deployment

If the command does not start with `mcp_server`, it runs in model deployment mode, supporting multiple frameworks (e.g., `vllm`, `lightllm`) and optional web chat interface.

```bash
# Deploy LLaMA3 model using vllm and enable chat mode
lazyllm deploy llama3-chat --framework vllm --chat=true --top_p=0.9 --max_tokens=2048
```

Args:

- `llama3-chat`: the model name to deploy.

- `framework=vllm`: specifies the deployment framework; supports:
    - `vllm`: high-performance inference engine.
    - `lightllm`: lightweight model deployment.
    - `lmdeploy`, `infinity`, `embedding`, `mindie`: other specialized frameworks.
    - `auto`: automatically detect and recommend framework.

- `chat=true`: enable web chat service. Equivalent forms include `chat=1`, `chat=on`.

- `top_p=0.9`: nucleus sampling truncation probability during inference.

- `max_tokens=2048`: maximum number of tokens generated.

Additional notes:

- Other parameters can be passed as `key=value` to customize framework-supported inference configurations.
- If `chat=true` is not enabled, the deployment runs as a background service.

---

## `lazyllm install`

Used to install extra feature groups (extras groups) or specified third-party Python packages.

You can install:

- Predefined component groups (e.g., `embedding`, `chat`, `finetune`)
- Specific Python packages (e.g., `openai`, `transformers`)

The installation logic automatically manages version dependencies and compatibility issues, such as adapting `flash-attn` for PyTorch.

### Function 1: Install Component Groups

```bash
# Install embedding and chat component groups
lazyllm install embedding chat
```

Args:

- `embedding`, `chat`: predefined feature groups for embedding models and chat-related functions.

### Function 2: Install Third-party Python Packages

```bash
# Install specific third-party Python packages
lazyllm install openai sentence-transformers
```

Args:

- `openai`, `sentence-transformers`: Python package names, used for calling OpenAI APIs or loading vector models.

---

## `lazyllm run`

Executes corresponding services or workflows based on the passed subcommands.

### Function 1: Start Chatbot Service

```bash
lazyllm run chatbot --model chatglm3-6b --framework vllm
```

Args:

- `chatbot`: starts the chatbot service.
- `model`: specifies the model name to use, e.g., `chatglm3-6b`.
- `framework`: backend inference framework, supporting `lightllm`, `vllm`, `lmdeploy`.

### Function 2: Start RAG QA Service

```bash
lazyllm run rag --model bge-base --framework lightllm --documents /path/to/docs
```

Args:

- `rag`: starts a retrieval-augmented generation (RAG) question answering system.
- `model`: specifies the model name, e.g., `bge-base`.
- `framework`: backend inference framework.
- `documents`: required; absolute path to knowledge documents.

### Function 3: Run JSON-based Workflow

```bash
lazyllm run workflow.json
```

Args:

- `workflow.json`: JSON workflow file path to run the specified computational graph.

### Function 4: Start Training Service

```bash
lazyllm run training_service
```

Args:

- `training_service`: starts the model training service; no additional parameters required.

### Function 5: Start Inference Service

```bash
lazyllm run infer_service
```

Args:

- `infer_service`: starts the model inference service; no additional parameters required.

> ❗ Note: For `chatbot` and `rag`, `source` and `framework` are mutually exclusive and must be chosen from predefined options. Invalid commands or parameters will result in error messages.
