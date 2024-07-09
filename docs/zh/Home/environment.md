环境依赖

依赖及场景说明
~~~~~~~~~~~~~

- 微调（基于alpaca-lora）: datasets, deepspeed, faiss-cpu, fire, gradio, numpy, peft, torch, transformers
- 微调（基于collie）: collie-lm, numpy, peft, torch, transformers, datasets, deepspeed, fire
- 推理（基于lightllm）: lightllm
- 推理（基于vllm）: vllm
- RAG: llama_index, llama-index-retrievers-bm25, llama-index-storage-docstore-redis, llama-index-vector-stores-redis, rank_bm25, redisvl, llama_index, llama-index-embeddings-huggingface

基础依赖
~~~~~~~~~~

- fastapi: FastAPI 是一个现代、快速（高性能）的Web框架，用于构建API，与Python 3.6+类型提示一起使用。
- loguru: Loguru 是一个Python日志库，旨在通过简洁、易用的API提供灵活的日志记录功能。
- pydantic: Pydantic 是一个数据验证和设置管理工具，它使用Python的类型注解来验证数据。
- Requests: Requests 是一个Python HTTP库，用于发送HTTP请求。它简单易用，是Python中发起Web请求的常用库。
- uvicorn: Uvicorn 是一个轻量级、快速的ASGI服务器，用于运行Python 3.6+的Web应用程序。
- cloudpickle: 一个Python序列化库，能够序列化Python对象到字节流，以便跨Python程序和解释器传输。
- flake8: 一个代码风格检查工具，用于检测Python代码中的错误，并遵守PEP 8编码标准。
- gradio: 一个用于快速创建简单Web界面的库，允许用户与Python模型进行交互。
- gradio_client: Gradio的客户端库，允许用户从远程服务器加载和使用Gradio界面。
- protobuf: Google的Protocol Buffers的Python实现，用于序列化结构化数据。
- setuptools: 一个Python包安装和分发工具，用于打包和分发Python应用程序和库。