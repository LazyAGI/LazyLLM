from typing import List, Tuple

import lazyllm
from lazyllm import pipeline, parallel
from lazyllm.tools.rag import Document, Retriever, ChatServer

def create_document(kb_name: str) -> Document:
    document = Document(kb_name=kb_name, embed=lazyllm.OnlineEmbeddingModule())
    document.create_node_group(name="sentences", transform=lambda s: '。'.split(s))
    return document

def create_pudong_pipeline(doc_list: List[Document]) -> Tuple[pipeline, pipeline]:
    retriever_pipelines = []
    for doc in doc_list:
        with parallel().sum as retriever_pipeline:
            retriever_pipeline.retriever1 = lazyllm.Retriever(
                doc=doc,
                group_name="CoarseChunk",
                similarity="bm25_chinese",
                topk=3
            )
            retriever_pipeline.retriever2 = lazyllm.Retriever(
                doc=doc,
                group_name="sentences",
                similarity="cosine",
                topk=3
            )
        retriever_pipelines.append(retriever_pipeline)

    with pipeline() as search_pipeline:
        search_pipeline.retriever = parallel(*retriever_pipelines).sum
        search_pipeline.reranker = lazyllm.Reranker(
            name='ModuleReranker',
            model="bge-reranker-large",
            topk=1
        )
    
    llm = lazyllm.OnlineChatModule()
    prompt = '你将扮演一个人工智能问答助手的角色，完成一项对话任务。在这个任务中，你需要根据给定的上下文以及问题，给出你的回答。'
    llm.prompt(lazyllm.ChatPrompter(instruction=prompt, extro_keys=['context_str']))
    
    query_pipeline = pipeline(search_pipeline, llm)
    return search_pipeline, query_pipeline

def create_default_pipeline(doc_list: List[Document]) -> Tuple[pipeline, pipeline]:
    assert len(doc_list) == 1, "Only support one document now"
    document = doc_list[0]

    with pipeline() as search_pipeline:
        with parallel().sum as search_pipeline.prl:
            prl.retriever1 = lazyllm.Retriever(
                doc=document,
                group_name="CoarseChunk",
                similarity="bm25_chinese",
                topk=3
            )
            prl.retriever2 = lazyllm.Retriever(
                doc=document,
                group_name="sentences",
                similarity="cosine",
                topk=3
            )
        search_pipeline.reranker = lazyllm.Reranker(
            name='ModuleReranker',
            model="bge-reranker-large",
            topk=1
        )

    llm = lazyllm.OnlineChatModule()
    prompt = '你将扮演一个人工智能问答助手的角色，完成一项对话任务。在这个任务中，你需要根据给定的上下文以及问题，给出你的回答。'
    llm.prompt(lazyllm.ChatPrompter(instruction=prompt, extro_keys=['context_str']))
    
    query_pipeline = pipeline(search_pipeline, llm)
    return search_pipeline, query_pipeline

def create_pipeline(project_name: str, doc_list: List[Document]) -> Tuple[pipeline, pipeline]:
    if project_name == "pudong":
        return create_pudong_pipeline(doc_list)
    return create_default_pipeline(doc_list)

ChatServer.start_server(doc_creater=create_document, pipeline_creater=create_pipeline)