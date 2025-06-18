import os
import json
import lazyllm
from lazyllm import LOG
from lazyllm.thirdparty import transformers as tf, torch, sentence_transformers, numpy as np, FlagEmbedding as fe
from .base import LazyLLMDeployBase
from typing import Union, List, Dict, Any


class _EmbeddingModuleMeta(type):
    def __instancecheck__(self, __instance: Any) -> bool:
        if isinstance(__instance, LazyHuggingFaceEmbedding):
            return True
        return super().__instancecheck__(__instance)

    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        if name != 'LazyHuggingFaceEmbedding':
            # register sub class
            if not hasattr(cls, 'model_id'):
                raise ValueError(f"Class {name} must define 'model_id' class variable")
            LazyHuggingFaceEmbedding._models[cls.model_id] = cls
        return cls

class LazyHuggingFaceEmbedding(object, metaclass=_EmbeddingModuleMeta):
    # Child-Class must set this key, it shoule be the same with last part of model path
    model_id = "UNKNOWN"

    def __init__(self, base_embed, source=None, init=False):
        from ..utils.downloader import ModelManager
        source = lazyllm.config['model_source'] if not source else source
        self.base_embed = ModelManager(source).download(base_embed) or ''
        self.embed = None
        self.tokenizer = None
        self.device = "cpu"
        self.init_flag = lazyllm.once_flag()
        if init:
            lazyllm.call_once(self.init_flag, self.load_embed)

    _models = {}

    @classmethod
    def create(cls, model_id: str, base_embed: str, source: str = None, init: bool = False):
        """
        create real instance by model_id

        Args:
            model_id: model_id, it is usually the last segment of base_embed
            base_embed: model path
            source: model souce (huggingface, modelscope etc)
            init: if set true, load the model when initiating

        Returns:
            LazyHuggingFaceEmbedding: instance

        Raises:
            KeyError: failed to find class definition for model_id
        """
        if model_id in cls._models:
            return cls._models[model_id](base_embed, source=source, init=init)
        elif model_id == LazyHuggingFaceEmbedding.model_id:
            return LazyHuggingFaceEmbedding(base_embed, source=source, init=init)
        else:
            raise KeyError(f"Model ID '{model_id}' not found. Available models: {list(cls._models.keys())}")

    def load_embed(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tf.AutoTokenizer.from_pretrained(self.base_embed, trust_remote_code=True)
        self.embed = tf.AutoModel.from_pretrained(self.base_embed, trust_remote_code=True).to(self.device)
        self.embed.eval()

    def __call__(self, data: Dict[str, Union[str, List[str]]]):
        lazyllm.call_once(self.init_flag, self.load_embed)
        string, _ = data['text'], data['images']
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
    def rebuild(cls, base_embed, init, models):
        model_id = base_embed.split('/')[-1]
        LazyHuggingFaceEmbedding._models = models
        return cls.create(model_id, base_embed, init=init)

    def __reduce__(self):
        init = bool(os.getenv('LAZYLLM_ON_CLOUDPICKLE', None) == 'ON' or self.init_flag)
        return LazyHuggingFaceEmbedding.rebuild, (self.base_embed, init, LazyHuggingFaceEmbedding._models)


class LazyFlagEmbedding(object):
    def __init__(self, base_embed, sparse=False, source=None, init=False):
        from ..utils.downloader import ModelManager
        source = lazyllm.config['model_source'] if not source else source
        self.base_embed = ModelManager(source).download(base_embed) or ''
        self.embed = None
        self.device = "cpu"
        self.sparse = sparse
        self.init_flag = lazyllm.once_flag()
        if init:
            lazyllm.call_once(self.init_flag, self.load_embed)

    def load_embed(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embed = fe.FlagAutoModel.from_finetuned(self.base_embed, use_fp16=False, devices=[self.device])

    def __call__(self, data: Dict[str, Union[str, List[str]]]):
        lazyllm.call_once(self.init_flag, self.load_embed)
        string, _ = data['text'], data['images']
        with torch.no_grad():
            model_output = self.embed.encode(string, return_sparse=self.sparse)
        if self.sparse:
            embeddings = model_output['lexical_weights']
            if isinstance(string, list):
                res = [dict(embedding) for embedding in embeddings]
            else:
                res = dict(embeddings)
        else:
            res = model_output['dense_vecs'].tolist()

        if type(string) is list and type(res) is dict:
            return json.dumps([res], default=lambda x: float(x))
        else:
            return json.dumps(res, default=lambda x: float(x))

    @classmethod
    def rebuild(cls, base_embed, sparse, init):
        return cls(base_embed, sparse, init=init)

    def __reduce__(self):
        init = bool(os.getenv('LAZYLLM_ON_CLOUDPICKLE', None) == 'ON' or self.init_flag)
        return LazyFlagEmbedding.rebuild, (self.base_embed, self.sparse, init)

class LazyHuggingFaceRerank(object):
    def __init__(self, base_rerank, source=None, init=False):
        from ..utils.downloader import ModelManager
        source = lazyllm.config['model_source'] if not source else source
        self.base_rerank = ModelManager(source).download(base_rerank) or ''
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
        sorted_indices = [(index, scores[index]) for index in np.argsort(scores)[::-1]]
        if top_n > 0:
            sorted_indices = sorted_indices[:top_n]
        return sorted_indices

    @classmethod
    def rebuild(cls, base_rerank, init):
        return cls(base_rerank, init)

    def __reduce__(self):
        init = bool(os.getenv('LAZYLLM_ON_CLOUDPICKLE', None) == 'ON' or self.init_flag)
        return LazyHuggingFaceRerank.rebuild, (self.base_rerank, init)

class EmbeddingDeploy(LazyLLMDeployBase):
    message_format = {
        'text': 'text',  # str,
        'images': []  # Union[str, List[str]]
    }
    keys_name_handle = {
        'inputs': 'text',
        'image': 'images'
    }
    default_headers = {'Content-Type': 'application/json'}

    def __init__(self, launcher=None, model_type='embed', log_path=None, embed_type='dense'):
        self._launcher = launcher
        self._model_type = model_type
        self._log_path = log_path
        self._sparse_embed = True if embed_type == 'sparse' else False
        if self._model_type == "reranker":
            self._update_reranker_message()
        elif self._model_type == "cross_modal_embed":
            self._update_cross_codal_message()

    def _update_reranker_message(self):
        self.keys_name_handle = {
            'inputs': 'query',
        }
        self.message_format = {
            'query': 'who are you ?',
            'documents': ['string'],
            'top_n': 1,
        }
        self.default_headers = {'Content-Type': 'application/json'}

    def _update_cross_codal_message(self):
        # Disable those var as they just cause error
        self.keys_name_handle = {}
        self.message_format = {}
        self.default_headers = {'Content-Type': 'application/json'}

    def __call__(self, finetuned_model=None, base_model=None):
        if not os.path.exists(finetuned_model) or \
            not any(filename.endswith('.bin') or filename.endswith('.safetensors')
                    for filename in os.listdir(finetuned_model)):
            if not finetuned_model:
                LOG.warning(f"Note! That finetuned_model({finetuned_model}) is an invalid path, "
                            f"base_model({base_model}) will be used")
            finetuned_model = base_model
        if self._model_type == 'embed' or self._model_type == 'cross_modal_embed':
            if self._sparse_embed or lazyllm.config['default_embedding_engine'] == 'flagEmbedding':
                return lazyllm.deploy.RelayServer(func=LazyFlagEmbedding(
                    finetuned_model, sparse=self._sparse_embed),
                    launcher=self._launcher, log_path=self._log_path, cls='embedding')()
            else:
                model_id = finetuned_model.split('/')[-1]
                emb_obj = LazyHuggingFaceEmbedding.create(model_id, finetuned_model)
                return lazyllm.deploy.RelayServer(func=emb_obj, launcher=self.launcher,
                                                  log_path=self._log_path, cls='embedding')()
        if self._model_type == 'reranker':
            return lazyllm.deploy.RelayServer(func=LazyHuggingFaceRerank(
                finetuned_model), launcher=self._launcher, log_path=self._log_path, cls='embedding')()
        else:
            raise RuntimeError(f'Not support model type: {self._model_type}.')
