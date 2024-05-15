import lazyllm
from lazyllm import pipeline, dpes, ID, launchers, Document, Retriever, deploy
from llama_index.core.node_parser import HierarchicalNodeParser, get_deeper_nodes
from lazyllm.llms.embedding.embed import LazyHuggingFaceEmbedding

template = (
    '<|im_start|>user\n作为国学大师，你将扮演一个人工智能国学问答助手的角色，完成一项对话任务。在这个任务中，你需要根据给定的已知国学篇章以及问题，给出你的结论。请注意，你的回答应基于给定的国学篇章，而非你的先验知识，且注意你回答的前后逻辑不要出现'
    '重复，且不需要提到具体文件名称。\n任务示例如下：\n示例国学篇章：《礼记 大学》大学之道，在明明德，在亲民，在止于至善。\n问题：什么是大学？\n回答：“大学”在《礼记》中代表的是一种理想的教育和社会实践过程，旨在通过个人的'
    '道德修养和社会实践达到最高的善治状态。\n注意以上仅为示例，禁止在下面任务中提取或使用上述示例已知国学篇章。\n现在，请对比以下给定的国学篇章和给出的问题。如果已知国学篇章中有该问题相关的原文，请提取相关原文出来。\n'
    '已知国学篇章：{context_str}\n问题: {query_str}\n回答：\n<|im_end|>\n<|im_start|>assistant\n')

base_model = '/mnt/lustrenew/share_data/sunxiaoye/Models/internlm2-chat-7b'
base_embed = '/mnt/lustrenew/share_data/sunxiaoye/Models/BAAI--bge-large-zh-v1.5'
launcher = launchers.slurm(partition='pat_rd', ngpus=1, sync=False)

llm = lazyllm.TrainableModule(None, base_model
        ).deploy((deploy.vllm, {'launcher': launcher, 'max-model-len': 12000})
        ).prompt(template, response_split='<|im_start|>assistant\n')

documents = Document('/mnt/lustre/share_data/sunxiaoye.vendor/MyWorks/LazyLLM/RAG/docs2', 
                    lazyllm.ServerModule(LazyHuggingFaceEmbedding(base_embed), launcher=launcher))
documents.add_parse(name='base', parser=HierarchicalNodeParser, chunk_sizes=[2048, 512, 128]
        ).add_parse(name='line', parser=get_deeper_nodes, depth=1, parent='base')

rma1 = Retriever(documents, algo='vector', parser='line', similarity_top_k=3)
rma2 = Retriever(documents, algo='bm25', parser='line', similarity_top_k=6)
rmf = lazyllm.ServerModule(pipeline(
        dpes(rma1, rma2),
        lambda x,y: x+y,
        lambda nodes: '《'+nodes[0].metadata["file_name"].split('.')[0] + '》 ' + nodes[0].get_content()))

m = lazyllm.ActionModule(lazyllm.pipeline(
        lazyllm.parallel(
            rmf,
            ID),
        lambda x,y: {'context_str':x, 'query_str':y},
        llm)
    )
m.evalset(['介绍五行。','什么是色？','什么是中庸？','非常道是什么？','应该怎么学习？'])

mweb = lazyllm.WebModule(m)
mweb.update_server().eval()
print(m.eval_result)

import time
time.sleep(123439)