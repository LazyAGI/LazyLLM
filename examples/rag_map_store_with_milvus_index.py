# -*- coding: utf-8 -*-

import os
import lazyllm
from lazyllm import bind
import tempfile

def run(query):
    fd, store_file = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        milvus_store_conf = {
            'type': 'map',
            'indices': {
                'smart_embedding_index': {
                    'backend': 'milvus',
                    'kwargs': {
                        'uri': store_file,
                        'index_kwargs': {
                            'index_type': 'FLAT',
                            'metric_type': 'COSINE',
                        }
                    },
                },
            },
        }

        documents = lazyllm.Document(dataset_path="rag_master",
                                     embed=lazyllm.TrainableModule("bge-large-zh-v1.5"),
                                     manager=False,
                                     store_conf=milvus_store_conf)

        documents.create_node_group(name="sentences",
                                    transform=lambda s: [x for x in s.split('。') if x.strip()])

        prompt = 'You will play the role of an AI Q&A assistant and complete a dialogue task.'\
            ' In this task, you need to provide your answer based on the given context and question.'

        with lazyllm.pipeline() as ppl:
            ppl.retriever = lazyllm.Retriever(doc=documents, group_name="sentences", topk=3)

            ppl.reranker = lazyllm.Reranker(name='ModuleReranker',
                                            model="bge-reranker-large",
                                            topk=1,
                                            output_format='content',
                                            join=True) | bind(query=ppl.input)

            ppl.formatter = (
                lambda nodes, query: dict(context_str=nodes, query=query)
            ) | bind(query=ppl.input)

            ppl.llm = lazyllm.TrainableModule('internlm2-chat-7b').prompt(
                lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))

        rag = lazyllm.ActionModule(ppl)
        rag.start()
        res = rag(query)
        return res
    finally:
        try:
            os.remove(store_file)
        except Exception:
            pass

if __name__ == '__main__':
    res = run('何为天道？')
    print(f'answer: {res}')
