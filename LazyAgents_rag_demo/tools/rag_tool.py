# rag_tool.py
from lazyllm import pipeline, parallel, bind, OnlineEmbeddingModule, SentenceSplitter, Document, Retriever, Reranker, OnlineChatModule, ChatPrompter

def create_rag_tool(prompt: str, dataset_path: str = "rag_master", source: str = "qwen", stream: bool = False):
    # 创建文档对象
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

        ppl.llm = OnlineChatModule(source=source, stream=stream).prompt(
            ChatPrompter(prompt, extra_keys=["context_str"])
        )

    return ppl
