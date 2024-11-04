# -*- coding: utf-8 -*-

import os
import lazyllm
import tempfile
from lazyllm.tools.rag import MilvusField, EMBED_DEFAULT_KEY

_, store_file = tempfile.mkstemp(suffix=".db")

fields = {
    EMBED_DEFAULT_KEY: MilvusField(data_type=MilvusField.DTYPE_FLOAT_VECTOR,
                                   index_type='HNSW', metric_type='COSINE'),
}

milvus_store_conf = {
    'type': 'milvus',
    'kwargs': {
        'uri': store_file,
        'group_fields': {
            'sentences': fields,
        },
    },
    'indices': {
        'map': {}
    },
}

documents = lazyllm.Document(dataset_path="rag_master",
                             embed=lazyllm.TrainableModule("bge-large-zh-v1.5"),
                             manager=False,
                             store_conf=milvus_store_conf)

documents.create_node_group(name="sentences",
                            transform=lambda s: '。'.split(s))

prompt = 'You will play the role of an AI Q&A assistant and complete a dialogue task.'\
    ' In this task, you need to provide your answer based on the given context and question.'

with lazyllm.pipeline() as ppl:
    with lazyllm.parallel().sum as ppl.prl:
        prl.retriever1 = lazyllm.Retriever(doc=documents,  # noqa F821
                                           group_name="CoarseChunk",
                                           similarity="bm25_chinese",
                                           topk=3)
        prl.retriever2 = lazyllm.Retriever(doc=documents,  # noqa F821
                                           group_name="sentences",
                                           similarity="cosine",
                                           topk=3)

    ppl.reranker = lazyllm.Reranker(name='ModuleReranker',
                                    model="bge-reranker-large",
                                    topk=1,
                                    output_format='content',
                                    join=True) | bind(query=ppl.input)  # noqa F821

    ppl.formatter = (
        lambda nodes, query: dict(context_str=nodes, query=query)
    ) | bind(query=ppl.input)  # noqa F821

    ppl.llm = lazyllm.TrainableModule('internlm2-chat-7b').prompt(
        lazyllm.ChatPrompter(instruction=prompt, extro_keys=['context_str']))

rag = lazyllm.ActionModule(ppl)
rag.start()

print("answer: ", rag('who are you?'))

os.remove(store_file)
