# LazyLLM Deploy

LazyLLM Deploy is a command-line tool for deploying large language models.

## Installation

```bash
pip install lazyllm
```

## Usage

```bash
lazyllm deploy {model_name} [options]
```

## Parameters

| Parameter | Type | Description | Notes |
|-----------|------|-------------|-------|
| `--port` | int | Service port number |
| `--host` | str | Service host address |
| `--tp` | int | Tensor parallelism degree |
| `--max_batch_size` | int | Maximum batch size |
| `--chat_template` | str | Chat template | Only supported by lmdeploy |
| `--max_input_token_len` | int | Maximum input token length |
| `--max_prefill_tokens` | int | Maximum prefill token count |
| `--max_seq_len` | int | Maximum sequence length |

## Examples

```bash
# Start a basic service
lazyllm deploy llama2 --port=8000

# Use tensor parallelism
lazyllm deploy llama2 --tp=2
```

## Notes

- Ensure all required dependencies are installed
- Adjust parameter configurations based on actual hardware resources
- Ensure the model is available in LAZYLLM_MODEL_PATH