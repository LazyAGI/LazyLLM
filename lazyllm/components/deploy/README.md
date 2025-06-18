# LazyLLM Deploy

LazyLLM Deploy 是一个用于部署大语言模型的命令行工具。

## 安装

```bash
pip install lazyllm
```

## 使用方法

```bash
lazyllm deploy {model_name} [options]
```

## 参数说明

| 参数 | 类型 | 说明 | 备注
|------|------|------|
| `--port` | int | 服务端口号 |
| `--host` | str | 服务主机地址 |
| `--tp` | int | 张量并行度 |
| `--max_batch_size` | int | 最大批处理大小 |
| `--chat_template` | str | 聊天模板 | 仅lmdeploy支持
| `--max_input_token_len` | int | 最大输入token长度 |
| `--max_prefill_tokens` | int | 最大预填充token数 |
| `--max_seq_len` | int | 最大序列长度 |

## 示例

```bash
# 启动一个基础服务
lazyllm deploy llama2 --port 8000 --host localhost

# 使用张量并行
lazyllm deploy llama2 --tp 2 --max_batch_size 32
```

## 注意事项

- 确保已安装所需的依赖包
- 根据实际硬件资源调整参数配置
- 确保模型在LAZYLLM_MODEL_PATH