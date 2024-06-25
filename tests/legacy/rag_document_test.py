import lazyllm
from lazyllm import pipeline, parallel, Identity, Document, Retriever, Reranker, deploy, launchers
from lazyllm.components.embedding.embed import LazyHuggingFaceEmbedding

template = (
    '<|im_start|>user\n你将扮演一个人工智能问答助手的角色，完成一项对话任务。在这个任务中，你需要根据给定的上下文以及问题，给出你的回答。'
    '上下文：{context_str}\n问题: {query_str}\n回答：\n<|im_end|>\n<|im_start|>assistant\n'
)

llm = lazyllm.TrainableModule('internlm2-chat-7b', ''
        ).deploy_method(deploy.vllm, launcher=launchers.remote(ngpus=1)
        ).prompt(template, response_split='<|im_start|>assistant\n')

documents = Document(dataset_path='rag_master',
                     embed=lazyllm.ServerModule(LazyHuggingFaceEmbedding('bge-large-zh-v1.5')))

rm = Retriever(documents, similarity='chinese_bm25', parser='SentenceDivider', similarity_top_k=6)
reranker = Reranker(types='MoudleReranker', model='bge-reranker-large')
m = lazyllm.ActionModule(
    parallel.sequential(
        context_str=pipeline(parallel.sequential(Identity, rm), reranker),
        query_str=Identity
    ).asdict,
    llm
)

mweb = lazyllm.WebModule(m, port=12345)
mweb.start()