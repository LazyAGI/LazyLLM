import sys
import time
import os

# os.environ['PYTHONPATH'] = f"{os.environ.get('PYTHONPATH')}:{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}"
# sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
os.environ['LAZYLLM_DEBUG'] = '1'

import lazyllm
from lazyllm import pipeline, parallel, Identity, Retriever
from lazyllm.llms.embedding.embed import LazyHuggingFaceEmbedding
from lazyllm import Document
from lazyllm.launcher import EmptyLauncher

# os.environ['LAZYLLM_REDIS_URL']='redis://103.177.28.196:9997'

os.environ['LAZYLLM_REDIS_URL']='redis://:PEDGFVkfWo235rd@103.177.28.196:6379'

# embed = lazyllm.EmbeddingModule(source="openai", embed_url="https://gf.nekoapi.com/v1/embeddings")


class EmiEmbedding(lazyllm.ModuleBase):
    def __init__(self, *, return_trace=False):
        super().__init__(return_trace=return_trace)
        host = "101.230.144.233"
        port = "8330"
        from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
        tei_url = f"http://{host}:{port}"
        self._text_embedding_inference_model = TextEmbeddingsInference(
            base_url=tei_url,
            model_name="BAAI/bge-large-zh-v1.5",
            embed_batch_size= 128
        )
    
    def __call__(self, string):
        if type(string) is str:
            res = self._text_embedding_inference_model.get_text_embedding(string)
        else:
            res = self._text_embedding_inference_model.get_text_embedding_batch(string)
        return res


launcher = EmptyLauncher(sync=False)
# launcher = lazyllm.launchers.sco(partition='a100', ngpus=1, sync=False)
# embed=lazyllm.ServerModule(LazyHuggingFaceEmbedding(base_embed="/home/mnt/qitianlong/models/BAAI--bge-large-zh-v1.5"), launcher=launcher)
embed=EmiEmbedding()

documents = Document(
    dataset_path='/home/mnt2/sunshangbin/lazyllm/examples/dataset', 
    embed=embed
)

rma1 = Retriever(documents, parser='Hierarchy', similarity_top_k=3)
rma2 = Retriever(documents, algo='chinese_bm25', parser='Hierarchy', similarity_top_k=6)

m = lazyllm.ActionModule(
    pipeline(parallel.sequential(x=rma1,y=rma2), lambda x, y:x + y),
    lambda nodes: '《'+nodes[0].metadata["file_name"].split('.')[0] + '》 ' + nodes[0].get_content() if len(nodes)>0 else '未找到'
)

# m.evalset(['介绍五行。','什么是色？','什么是中庸？','非常道是什么？','应该怎么学习？'])
# m.update_server().eval()

mweb = lazyllm.WebModule(m)
mweb.start()
print(m("hi"))

import time
time.sleep(123439)