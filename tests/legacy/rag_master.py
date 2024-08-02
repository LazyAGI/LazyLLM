import lazyllm
from lazyllm import pipeline, Document, bind, parallel
from lazyllm.tools.rag.doc_impl import RetrieverV2
from lazyllm.tools.rag.rerank import RerankerV2
from lazyllm.tools.rag.transform import SentenceSplitter


prompt = (
    "作为国学大师，你将扮演一个人工智能国学问答助手的角色，完成一项对话任务。在这个任务中，你需要根据给定的已知国学篇章以及问题，给出你的结论。请注意，你的回答应基于给定的国学篇章，而非你的先验知识，且注意你回答的前后逻辑不要出现"
    "重复，且不需要提到具体文件名称。\n任务示例如下：\n示例国学篇章：《礼记 大学》大学之道，在明明德，在亲民，在止于至善。\n问题：什么是大学？\n回答：“大学”在《礼记》中代表的是一种理想的教育和社会实践过程，旨在通过个人的"
    "道德修养和社会实践达到最高的善治状态。\n注意以上仅为示例，禁止在下面任务中提取或使用上述示例已知国学篇章。\n现在，请对比以下给定的国学篇章和给出的问题。如果已知国学篇章中有该问题相关的原文，请提取相关原文出来。\n"
    "已知国学篇章：{context_str}\n问题: {query}\n回答：\n"
)
embed_model = lazyllm.TrainableModule("bge-large-zh-v1.5")
documents = Document(
    dataset_path="rag_master",
    embed=embed_model,
    create_ui=False,
)
documents._impl._impl.create_node_group(
    name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100
)

with pipeline() as ppl:
    with parallel().sum as ppl.prl:
        prl.retriever1 = RetrieverV2(
            documents,
            group_name="FineChunk",
            similarity="bm25_chinese",
            similarity_cut_off=0.003,
            topk=3,
        )
        prl.retriever2 = RetrieverV2(
            documents, group_name="sentences", similarity="cosine", topk=3
        )

    ppl.reranker = RerankerV2(
        "ModuleReranker", model="bge-reranker-large", topk=1
    ) | bind(query=ppl.input)
    ppl.formatter = (lambda nodes, query: dict(context_str=nodes[0].get_content(), query=query)) | bind(query=ppl.input)
    ppl.llm = lazyllm.TrainableModule('internlm2-chat-7b').prompt(lazyllm.ChatPrompter(prompt, extro_keys=['context_str']))

mweb = lazyllm.WebModule(ppl, port=23456).start().wait()
