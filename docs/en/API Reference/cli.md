# CLI

LazyLLM provides a command-line interface (CLI) for model deployment, installation, and runtime operations. This document describes the core commands available via `lazyllm.cli` with usage examples.

---

## `lazyllm deploy`

Launches either a model deployment or an MCP (Model Context Protocol) server, depending on the input command.

### Function

```python
def deploy(commands: list[str]) -> None
```

If the command starts with `mcp_server`, an MCP server is launched, supporting environment variable injection, SSE server configuration, and more.

Otherwise, a model is deployed using the specified backend framework (e.g., `vllm`, `lightllm`). Web chat mode can also be enabled with options like `--chat=true`.

### Arguments

* `commands (list[str])`: A list of CLI-style string arguments.

### Examples (Bash)

```bash
# Launch an MCP server using uvx and mcp-server-fetch
lazyllm deploy mcp_server uvx mcp-server-fetch

# Launch MCP server with environment variables and SSE port
lazyllm deploy mcp_server -e GITHUB_TOKEN your_token --sse-port 8080 npx -- -y @modelcontextprotocol/server-github

# Deploy a LLaMA3 model using vllm backend with chat mode
lazyllm deploy llama3-chat --framework vllm --chat=true --top_p=0.9 --max_tokens=2048
```

---

## `lazyllm install`

Installs extras groups or Python packages required for various components.

### Function

```python
def install(commands: list[str]) -> None
```

You can install:

* Predefined extras groups (e.g., `embedding`, `chat`, `finetune`)
* Specific third-party Python packages (e.g., `openai`, `transformers`)

The installer handles version resolution and compatibility issues, such as `flash-attn` with PyTorch.

### Arguments

* `commands (list[str])`: A list of extras group names or Python package names.

### Examples (Bash)

```bash
# Install extras groups
lazyllm install embedding chat

# Install specific Python packages
lazyllm install openai sentence-transformers
```

---

## `lazyllm run`

Executes services or workflows based on the subcommand provided.

### Function

```python
def run(commands: list[str]) -> None
```

Supported subcommands:

* `chatbot`: Launch a web-based chatbot using a specified model and backend.
* `rag`: Run a Retrieval-Augmented Generation (RAG) Q\&A service.
* `.json`: Execute a JSON-based computational graph.
* `training_service`: Start a model training server.
* `infer_service`: Start a model inference server.

### Arguments

* `commands (list[str])`: A list of subcommand arguments.

### Examples (Bash)

```bash
# Launch chatbot service with a specific model
lazyllm run chatbot --model chatglm3-6b --framework vllm

# Run a RAG Q&A service
lazyllm run rag --model bge-base --framework lightllm --documents /path/to/docs

# Execute a workflow defined in a JSON graph file
lazyllm run workflow.json

# Start the training service
lazyllm run training_service

# Start the inference service
lazyllm run infer_service
```
