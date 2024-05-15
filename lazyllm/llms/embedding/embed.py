import json

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class LazyHuggingFaceEmbedding(object):
    def __init__(self, base_embed, embed_batch_size=30, trust_remote_code=True):
        self.base_embed = base_embed
        self.embed_batch_size = embed_batch_size
        self.trust_remote_code = trust_remote_code
        self.embed = None

    def load_embed(self):  
        return HuggingFaceEmbedding(model_name=self.base_embed,
                                    embed_batch_size=self.embed_batch_size,
                                    trust_remote_code=self.trust_remote_code)
    
    def __call__(self, string):
        if not self.embed:
            self.embed = self.load_embed()
        if type(string) is str:
            res = self.embed.get_text_embedding(string)
        else:
            res = self.embed.get_text_embedding_batch(string)
        return json.dumps(res)
