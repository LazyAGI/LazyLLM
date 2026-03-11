# 环境依赖（本地参考）

本文档为 skill 本地摘要，完整安装与系统差异见项目根目录下 `docs/zh/Home/environment.md` 或 `docs/en/Home/environment.md`。

## 安装 LazyLLM

```bash
pip install lazyllm
```

按需安装功能组件（推荐使用 CLI）:

```bash
lazyllm install embedding   # 嵌入相关
lazyllm install chat        # 对话相关
lazyllm install finetune    # 微调相关
```

## 依赖与场景

- **微调（alpaca-lora）**: datasets, deepspeed, faiss-cpu, fire, gradio, numpy, peft, torch, transformers
- **微调（collie）**: collie-lm, numpy, peft, torch, transformers, datasets, deepspeed, fire
- **推理（lightllm）**: lightllm
- **推理（vllm）**: vllm

## 基础依赖（核心）

fastapi, loguru, pydantic, requests, uvicorn, cloudpickle, gradio, gradio_client, protobuf, setuptools 等。具体版本以项目 `pyproject.toml` / `requirements` 为准。

## 使用在线模型前

请配置对应平台的 API Key 环境变量，见 [api_key_platforms.md](api_key_platforms.md)。
