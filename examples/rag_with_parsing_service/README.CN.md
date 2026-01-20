RAG 解析服务示例

本目录展示如何使用独立的解析服务，并将其接入 Document / Retriever 流程。相关脚本：
- server_with_worker.py：服务端与 worker 一体运行
- server_and_separate_workers.py：服务端与 worker 分离启动
- document.py：Document 连接解析服务并开启服务模式
- retriever_using_url.py：通过 Document URL 远程检索

1) 独立解析服务配置

先启动解析服务，可选择一体模式或分离模式。

- 一体模式：
  - 运行 `server_with_worker.py`，启动服务端并包含本地 worker。
- 分离模式：
  - 运行 `server_and_separate_workers.py` 启动服务端。
  - 使用 `DocumentProcessorWorker` 启动一个或多个独立 worker。

2) Document 注册与远程接入

创建 Document 时设置解析服务 URL，用于注册与更新算法信息。Document
可切换为服务模式，对外暴露 URL，Retriever 可通过该 URL 远程检索。

参考 `document.py` 的配置：
- `manager=DocumentProcessor(url="http://0.0.0.0:9966")` 指向解析服务。
- `server=9977` 将 Document 作为服务暴露。

然后在 `retriever_using_url.py` 中使用：
`Document(url="http://127.0.0.1:9977", name="doc_example")` 创建远程 Retriever。

注意：使用该模式，需要使用独立部署的存储服务，如 OpenSearch、
Milvus standalone 等。

3) Server-Worker 架构与扩展

解析服务采用 server-worker 架构，worker 可独立部署并支持扩展，
包括基于 Ray 的弹性机制。注意 server 与所有 worker 的数据库配置
需要保持一致（`db_config`），否则任务无法正确协同。
