import os
import json
import lazyllm
from lazyllm import LOG


class LazyHuggingFaceEmbedding(object):
    def __init__(self, base_embed, source=None, embed_batch_size=30, trust_remote_code=True, init=False):
        from ..utils.downloader import ModelDownloader
        source = lazyllm.config['model_source'] if not source else source
        self.base_embed = ModelDownloader(source).download(base_embed)
        self.embed_batch_size = embed_batch_size
        self.trust_remote_code = trust_remote_code
        self.embed = None
        self.init_flag = lazyllm.once_flag()
        if init:
            lazyllm.call_once(self.init_flag, self.load_embed)

    def load_embed(self):
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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

    @classmethod
    def rebuild(cls, base_embed, embed_batch_size, trust_remote_code):
        assert os.environ['LAZYLLM_ON_CLOUDPICKLE'] == 'OFF'
        return cls(base_embed, embed_batch_size, trust_remote_code, True)

    def __reduce__(self):
        assert os.environ['LAZYLLM_ON_CLOUDPICKLE'] == 'ON'
        return LazyHuggingFaceEmbedding.rebuild, (self.base_embed, self.embed_batch_size, self.trust_remote_code)


class EmbeddingDeploy():
    message_format = None
    input_key_name = None
    default_headers = {'Content-Type': 'application/json'}

    def __init__(self, trust_remote_code=True, launcher=None):
        self.trust_remote_code = trust_remote_code
        self.launcher = launcher

    def __call__(self, finetuned_model=None, base_model=None):
        if not os.path.exists(finetuned_model) or \
            not any(filename.endswith('.bin') or filename.endswith('.safetensors')
                    for filename in os.listdir(finetuned_model)):
            if not finetuned_model:
                LOG.warning(f"Note! That finetuned_model({finetuned_model}) is an invalid path, "
                            f"base_model({base_model}) will be used")
            finetuned_model = base_model
        return lazyllm.deploy.RelayServer(func=LazyHuggingFaceEmbedding(
            finetuned_model, trust_remote_code=self.trust_remote_code), launcher=self.launcher)()
