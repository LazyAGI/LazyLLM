import json
import lazyllm
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from lazyllm.module.utils.downloader import ModelDownloader


class LazyHuggingFaceEmbedding(object):
    def __init__(self, base_embed, source=lazyllm.config['model_source'], embed_batch_size=30, trust_remote_code=True):
        self.embed_batch_size = embed_batch_size
        self.trust_remote_code = trust_remote_code
        self.embed = None
        self.init_flag = lazyllm.once_flag()
        self.base_embed = ModelDownloader(source).download(base_embed)

    def load_embed(self):
        self.embed = HuggingFaceEmbedding(model_name=self.base_embed,
                                          embed_batch_size=self.embed_batch_size,
                                          trust_remote_code=self.trust_remote_code)

    def __call__(self, string):
        lazyllm.call_once(self.init_flag, self.load_embed)
        if type(string) is str:
            res = self.embed.get_text_embedding(string)
        else:
            res = self.embed.get_text_embedding_batch(string)
        return json.dumps(res)
