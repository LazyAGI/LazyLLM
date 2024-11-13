# -*- coding: utf-8 -*-

import os
import lazyllm
from lazyllm import bind
import tempfile

class Runner:
    def __init__(self):
        _, self._store_file = tempfile.mkstemp(suffix=".db")

        milvus_store_conf = {
            'type': 'milvus',
            'kwargs': {
                'uri': self._store_file,
                'embedding_index_type': 'HNSW',
                'embedding_metric_type': 'COSINE',
            },
        }

        prompt = 'You will play the role of an AI Q&A assistant and complete a dialogue task.'\
            ' In this task, you need to provide your answer based on the given context and question.'

        self._documents = lazyllm.Document(dataset_path="rag_master",
                                           embed=lazyllm.TrainableModule("bge-large-zh-v1.5"),
                                           manager=True,
                                           store_conf=milvus_store_conf)

        self._documents.create_node_group(name="sentences",
                                          transform=lambda s: '。'.split(s))

        with lazyllm.pipeline() as self._ppl:
            self._ppl.retriever = lazyllm.Retriever(doc=self._documents, group_name="sentences", topk=3)

            self._ppl.reranker = lazyllm.Reranker(name='ModuleReranker',
                                                  model="bge-reranker-large",
                                                  topk=1,
                                                  output_format='content',
                                                  join=True) | bind(query=self._ppl.input)

            self._ppl.formatter = (
                lambda nodes, query: dict(context_str=nodes, query=query)
            ) | bind(query=self._ppl.input)

            self._ppl.llm = lazyllm.TrainableModule('internlm2-chat-7b').prompt(
                lazyllm.ChatPrompter(instruction=prompt, extro_keys=['context_str']))

    def __del__(self):
        os.remove(self._store_file)

    @property
    def pipeline(self):
        return self._ppl

    @property
    def doc_server_addr(self) -> str:
        url_pattern = r'(http://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+)'
        content = re.findall(url_pattern, self._documents.manager._url)
        return content[0]

if __name__ == '__main__':
    runner = Runner()
    rag = lazyllm.ActionModule(runner.pipeline)
    rag.start()
    res = rag('何为天道？')
    print(f'answer: {res}')
