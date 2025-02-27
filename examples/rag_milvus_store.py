# -*- coding: utf-8 -*-

import os
import lazyllm
from lazyllm import bind, config
from lazyllm.tools.rag import DocField, DataType
import shutil
import traceback

def print_stack():
    stack = traceback.format_stack()
    for line in stack:
        print(line.strip())

class TmpDir:
    def __init__(self):
        self.root_dir = os.path.expanduser(os.path.join(config['home'], 'rag_for_example_ut'))
        self.rag_dir = os.path.join(self.root_dir, 'rag_master')
        os.makedirs(self.rag_dir, exist_ok=True)
        self.store_file = os.path.join(self.root_dir, "milvus.db")

tmp_dir = TmpDir()

milvus_store_conf = {
    'type': 'milvus',
    'kwargs': {
        'uri': tmp_dir.store_file,
        'index_kwargs': {
            'index_type': 'HNSW',
            'metric_type': 'COSINE',
        }
    },
}

doc_fields = {
    'comment': DocField(data_type=DataType.VARCHAR, max_size=65535, default_value=' '),
    'signature': DocField(data_type=DataType.VARCHAR, max_size=32, default_value=' '),
}

prompt = 'You will play the role of an AI Q&A assistant and complete a dialogue task.'\
    ' In this task, you need to provide your answer based on the given context and question.'

documents = lazyllm.Document(dataset_path=tmp_dir.rag_dir,
                             embed=lazyllm.TrainableModule("bge-large-zh-v1.5"),
                             manager=True,
                             store_conf=milvus_store_conf,
                             doc_fields=doc_fields)

documents.create_node_group(name="block", transform=lambda s: s.split("\n") if s else '')

with lazyllm.pipeline() as ppl:
    ppl.retriever = lazyllm.Retriever(doc=documents, group_name="block", topk=3)

    ppl.reranker = lazyllm.Reranker(name='ModuleReranker',
                                    model="bge-reranker-large",
                                    topk=1,
                                    output_format='content',
                                    join=True) | bind(query=ppl.input)

    ppl.formatter = (
        lambda nodes, query: dict(context_str=nodes, query=query)
    ) | bind(query=ppl.input)

    ppl.llm = lazyllm.TrainableModule('internlm2-chat-7b').prompt(
        lazyllm.ChatPrompter(instruction=prompt, extro_keys=['context_str']))

if __name__ == '__main__':
    try:
        rag = lazyllm.ActionModule(ppl)
        rag.start()
        res = rag('何为天道？')
        print(f'answer: {res}')
    finally:
        shutil.rmtree(tmp_dir.root_dir)
