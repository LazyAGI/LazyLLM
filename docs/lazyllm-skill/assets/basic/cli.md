# CLI 使用

LazyLLM 命令行用于依赖安装、模型部署与运行服务。完整参数见项目 `docs/zh/API Reference/cli.md` 或 `docs/en/API Reference/cli.md`。

## lazyllm deploy

- **模型部署**（默认）：`lazyllm deploy <模型名> [--framework vllm|lightllm|lmdeploy|...] [--chat=true] [key=value ...]`
  - 支持框架：vllm、lightllm、lmdeploy、infinity、embedding、mindie、auto
  - 其他参数以 `key=value` 形式传入
- **MCP 服务**：`lazyllm deploy mcp_server <cmd> [args...]`，可选 `-e VAR value`、`--sse-port`、`--sse-host`、`--allow-origin`、`--pass-environment`

## lazyllm install

- **组件组**：`lazyllm install embedding chat finetune`（可多选）
- **指定包**：`lazyllm install openai sentence-transformers`（可多选）

## lazyllm run

- **聊天服务**：`lazyllm run chatbot --model <模型> [--framework vllm|lightllm|lmdeploy]`
- **RAG 服务**：`lazyllm run rag --model <模型> [--framework ...] --documents <文档路径>`
- **工作流**：`lazyllm run <workflow.json>`
- **训练/推理服务**：`lazyllm run training_service`、`lazyllm run infer_service`

注意：`chatbot` 与 `rag` 的 `source` 与 `framework` 互斥，仅能从预设选项中选择。
