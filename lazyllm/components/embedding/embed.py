import os
import json
import lazyllm
from lazyllm import LOG
from lazyllm.thirdparty import transformers as tf, torch, sentence_transformers, numpy as np


class LazyHuggingFaceEmbedding(object):
    def __init__(self, base_embed, source=None, init=False):
        from ..utils.downloader import ModelManager
        source = lazyllm.config['model_source'] if not source else source
        self.base_embed = ModelManager(source).download(base_embed)
        self.embed = None
        self.tokenizer = None
        self.device = "cpu"
        self.init_flag = lazyllm.once_flag()
        if init:
            lazyllm.call_once(self.init_flag, self.load_embed)

    def load_embed(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tf.AutoTokenizer.from_pretrained(self.base_embed)
        self.embed = tf.AutoModel.from_pretrained(self.base_embed).to(self.device)
        self.embed.eval()

    def __call__(self, string):
        lazyllm.call_once(self.init_flag, self.load_embed)
        encoded_input = self.tokenizer(string, padding=True, truncation=True, return_tensors='pt',
                                       max_length=512, add_special_tokens=True).to(self.device)
        with torch.no_grad():
            model_output = self.embed(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        res = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1).cpu().numpy().tolist()
        if type(string) is str:
            return json.dumps(res[0])
        else:
            return json.dumps(res)

    @classmethod
    def rebuild(cls, base_embed, init):
        return cls(base_embed, init)

    def __reduce__(self):
        init = bool(os.getenv('LAZYLLM_ON_CLOUDPICKLE', None) == 'ON' or self.init_flag)
        return LazyHuggingFaceEmbedding.rebuild, (self.base_embed, init)

class LazyHuggingFaceRerank(object):
    def __init__(self, base_rerank, source=None, init=False):
        from ..utils.downloader import ModelManager
        source = lazyllm.config['model_source'] if not source else source
        self.base_rerank = ModelManager(source).download(base_rerank)
        self.reranker = None
        self.init_flag = lazyllm.once_flag()
        if init:
            lazyllm.call_once(self.init_flag, self.load_reranker)

    def load_reranker(self):
        self.reranker = sentence_transformers.CrossEncoder(self.base_rerank)

    def __call__(self, inps):
        lazyllm.call_once(self.init_flag, self.load_reranker)
        query, documents, top_n = inps['query'], inps['documents'], inps['top_n']
        query_pairs = [(query, doc) for doc in documents]
        scores = self.reranker.predict(query_pairs)
        sorted_indices = np.argsort(scores)[::-1]
        if top_n > 0:
            sorted_indices = sorted_indices[:top_n]
        return sorted_indices.tolist()

    @classmethod
    def rebuild(cls, base_rerank, init):
        return cls(base_rerank, init)

    def __reduce__(self):
        init = bool(os.getenv('LAZYLLM_ON_CLOUDPICKLE', None) == 'ON' or self.init_flag)
        return LazyHuggingFaceRerank.rebuild, (self.base_rerank, init)

class EmbeddingDeploy():
    message_format = None
    keys_name_handle = None
    default_headers = {'Content-Type': 'application/json'}

    def __init__(self, launcher=None, model_type='embed'):
        self.launcher = launcher
        self._model_type = model_type

    def __call__(self, finetuned_model=None, base_model=None):
        if not os.path.exists(finetuned_model) or \
            not any(filename.endswith('.bin') or filename.endswith('.safetensors')
                    for filename in os.listdir(finetuned_model)):
            if not finetuned_model:
                LOG.warning(f"Note! That finetuned_model({finetuned_model}) is an invalid path, "
                            f"base_model({base_model}) will be used")
            finetuned_model = base_model
        if self._model_type == 'embed':
            return lazyllm.deploy.RelayServer(func=LazyHuggingFaceEmbedding(
                finetuned_model), launcher=self.launcher)()
        if self._model_type == 'reranker':
            return lazyllm.deploy.RelayServer(func=LazyHuggingFaceRerank(
                finetuned_model), launcher=self.launcher)()
        else:
            raise RuntimeError(f'Not support model type: {self._model_type}.')
