# -*- coding: utf-8 -*-
# flake8: noqa: F821

import lazyllm
from lazyllm import pipeline, parallel, bind, _0, Document, Retriever, Reranker

# Before running, set the environment variable:
#
# 1. `export LAZYLLM_GLM_API_KEY=xxxx`: the API key of Zhipu AI, default model "glm-4", `source="glm"`. 
#     You can apply for the API key at https://open.bigmodel.cn/
#     Also supports other API keys:
#       - LAZYLLM_OPENAI_API_KEY: the API key of OpenAI, default model "gpt-3.5-turbo", `source="openai"`.
#           You can apply for the API key at https://openai.com/index/openai-api/
#       - LAZYLLM_KIMI_API_KEY: the API key of Moonshot AI, default model "moonshot-v1-8k", `source="kimi"`.
#           You can apply for the API key at https://platform.moonshot.cn/console
#       - LAZYLLM_QWEN_API_KEY: the API key of Alibaba Cloud, default model "qwen-plus", `source="qwen"`.
#           You can apply for the API key at https://home.console.aliyun.com/
#       - LAZYLLM_SENSENOVA_API_KEY: the API key of SenseTime, default model "SenseChat-5", `source="sensenova"`.
#                                  You also have to set LAZYLLM_SENSENOVA_SECRET_KEY` togather.
#           You can apply for the API key at https://platform.sensenova.cn/home
#     * `source` needs to be specified for multiple API keys, but it does not need to be set for a single API key.
#
# 2. `export LAZYLLM_DATA_PATH=path/to/docs/folder/`: The parent folder of the document folder `rag_master`.
#                                                    Alternatively, you can set the `dataset_path` of the Document 
#                                                    to `path/to/docs/folder/rag_master` to replace 
#                                                    the setting of this environment variable.

prompt = ('You will play the role of an AI Q&A assistant and complete a dialogue task. In this task, '
          'you need to provide your answer based on the given context and question.')

documents = Document(dataset_path='rag_master', embed=lazyllm.OnlineEmbeddingModule(), create_ui=False)
with pipeline() as ppl:
    with parallel().sum as ppl.prl:
        prl.retriever1 = Retriever(documents, parser='CoarseChunk', similarity_top_k=6)
        prl.retriever2 = Retriever(documents, parser='SentenceDivider', similarity='chinese_bm25', similarity_top_k=6)
    ppl.reranker = Reranker(types='ModuleReranker', model='bge-reranker-large') | bind(ppl.input, _0)
    ppl.post_processer = lambda nodes: f'《{nodes[0].metadata["file_name"].split(".")[0]}》{nodes[0].get_content()}' \
        if len(nodes) > 0 else 'cannot find.'
    ppl.formatter = (lambda ctx, query: dict(context_str=ctx, query_str=query)) | bind(query=ppl.input)
    ppl.llm = lazyllm.OnlineChatModule(stream=False)\
        .prompt(lazyllm.ChatPrompter(prompt, extro_keys=['context_str']))

if __name__ == '__main__':
    lazyllm.WebModule(ppl, port=23466).start().wait()
