import os
import lazyllm
from lazyllm import pipeline, parallel, Identity, launchers, Document, Retriever, Rerank, deploy
from lazyllm.llms.embedding.embed import LazyHuggingFaceEmbedding

# If use redis, please set 'export LAZYLLM_RAG_STORE=Redis', and export LAZYLLM_REDIS_URL=redis://{IP}:{PORT}

template = (
    '<|im_start|>user\n作为国学大师，你将扮演一个人工智能国学问答助手的角色，完成一项对话任务。在这个任务中，你需要根据给定的已知国学篇章以及问题，给出你的结论。请注意，你的回答应基于给定的国学篇章，而非你的先验知识，且注意你回答的前后逻辑不要出现'
    '重复，且不需要提到具体文件名称。\n任务示例如下：\n示例国学篇章：《礼记 大学》大学之道，在明明德，在亲民，在止于至善。\n问题：什么是大学？\n回答：“大学”在《礼记》中代表的是一种理想的教育和社会实践过程，旨在通过个人的'
    '道德修养和社会实践达到最高的善治状态。\n注意以上仅为示例，禁止在下面任务中提取或使用上述示例已知国学篇章。\n现在，请对比以下给定的国学篇章和给出的问题。如果已知国学篇章中有该问题相关的原文，请提取相关原文出来。\n'
    '已知国学篇章：{context_str}\n问题: {query_str}\n回答：\n<|im_end|>\n<|im_start|>assistant\n')

base_model = '/mnt/lustrenew/share_data/sunxiaoye/Models/internlm2-chat-7b'
base_embed = '/mnt/lustrenew/share_data/sunxiaoye/Models/BAAI--bge-large-zh-v1.5'
base_rerank = '/mnt/lustrenew/share_data/sunxiaoye/Models/BAAI--bge-reranker-large'
launcher = launchers.slurm(partition='pat_rd', ngpus=1, sync=False)

llm = lazyllm.TrainableModule(base_model
        ).deploy_method((deploy.vllm, {'launcher': launcher, 'max-model-len': 12000})
        ).prompt(template, response_split='<|im_start|>assistant\n')

documents = Document(doc_path='/mnt/lustre/share_data/sunxiaoye.vendor/MyWorks/LazyLLM/RAG/docs2', 
                     embed=lazyllm.ServerModule(LazyHuggingFaceEmbedding(base_embed), launcher=launcher))

rma1 = Retriever(documents, parser='FineChunk', similarity_top_k=3)
rma2 = Retriever(documents, algo='chinese_bm25', parser='SentenceDivider', similarity_top_k=6)
rerank1 = Rerank(types='Reranker', model=base_rerank)
rerank2 = Rerank(types='SimilarityFilter', threshold=0.003)

rmf = lazyllm.ServerModule(pipeline(
        parallel.sequential(Identity, pipeline(parallel.sequential(x=rma1, y=rma2), lambda x, y : x + y)),
        rerank1, rerank2, lambda nodes: '《'+nodes[0].metadata["file_name"].split('.')[0] + '》 ' + nodes[0].get_content() if len(nodes)>0 else '未找到'))
m = lazyllm.ActionModule(parallel(context_str=rmf, query_str=Identity).asdict, llm)

m.evalset(['介绍五行。','什么是色？','什么是中庸？','非常道是什么？','应该怎么学习？'])

mweb = lazyllm.WebModule(m)
mweb.update_server().eval()
print(m.eval_result)

import time
time.sleep(123439)
