# -*- coding: utf-8 -*-
# flake8: noqa: F821

import lazyllm
from lazyllm import pipeline, parallel, bind, _0, Document, Retriever, Reranker

# Three ways to specify the model:
#   1. Specify the model name (e.g. 'internlm2-chat-7b'):
#           the model will be automatically downloaded from the Internet;
#   2. Specify the model name (e.g. 'internlm2-chat-7b') ​​+ set
#      the environment variable `export LAZYLLM_MODEL_PATH="/path/to/modelzoo"`:
#           the model will be found in `path/to/modelazoo/internlm2-chat-7b/`
#   3. Directly pass the absolute path to TrainableModule:
#           `path/to/modelazoo/internlm2-chat-7b`

prompt = ('You will play the role of an AI Q&A assistant and complete a dialogue task. In this task, '
          'you need to provide your answer based on the given context and question.')

documents = Document(dataset_path='rag_master', embed=lazyllm.TrainableModule('bge-large-zh-v1.5'), create_ui=False)
with pipeline() as ppl:
    with parallel().sum as ppl.prl:
        prl.retriever1 = Retriever(documents, parser='CoarseChunk', similarity_top_k=6)
        prl.retriever2 = Retriever(documents, parser='SentenceDivider', similarity='chinese_bm25', similarity_top_k=6)
    ppl.reranker = Reranker(types='ModuleReranker', model='bge-reranker-large') | bind(ppl.input, _0)
    ppl.post_processer = lambda nodes: f'《{nodes[0].metadata["file_name"].split(".")[0]}》{nodes[0].get_content()}' \
        if len(nodes) > 0 else 'cannot find.'
    ppl.formatter = (lambda ctx, query: dict(context_str=ctx, query_str=query)) | bind(query=ppl.input)
    ppl.llm = lazyllm.TrainableModule('internlm2-chat-7b')\
        .prompt(lazyllm.ChatPrompter(prompt, extro_keys=['context_str']))

if __name__ == '__main__':
    lazyllm.WebModule(ppl, port=23466).start().wait()
