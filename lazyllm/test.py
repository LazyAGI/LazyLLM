import lazyllm
from lazyllm import pipeline, parallel, bind, OnlineEmbeddingModule, SentenceSplitter, Document, Retriever, Reranker


def rag_pipeline(prompt, source="qwen", stream=True, dataset_path="rag_master"):
    # 加载文档并构建节点组
    documents = Document(dataset_path=dataset_path, embed=OnlineEmbeddingModule(), manager=False)
    documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)

    # 构建流水线
    with pipeline() as ppl:
        with parallel().sum as ppl.prl:
            ppl.prl.retriever1 = Retriever(documents, group_name="sentences", similarity="cosine", topk=3)
            ppl.prl.retriever2 = Retriever(documents, "CoarseChunk", "bm25_chinese", 0.003, topk=3)

        ppl.reranker = Reranker("ModuleReranker", model=OnlineEmbeddingModule(type="rerank"),
                                topk=1, output_format='content', join=True) | bind(query=ppl.input)

        ppl.formatter = (lambda nodes, query: dict(context_str=nodes, query=query)) | bind(query=ppl.input)

        ppl.llm = lazyllm.OnlineChatModule(source=source, stream=stream).prompt(
            lazyllm.ChatPrompter(prompt, extra_keys=["context_str"])
        )

    return ppl


if __name__ == "__main__":
    # 自定义提示词
    prompt = "你是一个知识问答助手，请结合上下文正确回答用户的问题。"

    # 构建流水线
    ppl = rag_pipeline(
        prompt=prompt,
        source="qwen",
        stream=True,
        dataset_path="/home/mnt/zhangzhiqi/LazyLLM/LazyAgents_rag_demo/docs"
    )

    # 启动 Web 服务
    lazyllm.WebModule(ppl, port=range(23471, 23480)).start().wait()
