import os
import json
import lazyllm
from lazyllm.components.utils.file_operate import _base64_to_file, _is_base64_with_mime
from lazyllm import LOG, LazyLLMLaunchersBase
from lazyllm.thirdparty import transformers as tf, torch, sentence_transformers, numpy as np, FlagEmbedding as fe
from .base import LazyLLMDeployBase
from typing import Union, List, Dict, Optional
from abc import ABC, abstractmethod


class AbstractEmbedding(ABC):
    """Abstract embedding base class that provides unified interface and basic functionality for all embedding models. This class defines the standard interface for embedding models, including model loading, calling, and serialization capabilities.

Args:
    base_embed (str): The base path or identifier of the embedding model, used to specify which embedding model to load.
    source (str, optional): Model source, default to ``None``. If not specified, will use the default model source from LazyLLM configuration.
    init (bool): Whether to load the model immediately during initialization, default to ``False``. If ``True``, will call the ``load_embed()`` method immediately when the object is created.
"""
    def __init__(self, base_embed, source=None, init=False):
        from ..utils.downloader import ModelManager
        self._source = source or lazyllm.config['model_source']
        self._base_embed = ModelManager(self._source).download(base_embed) or ''
        self._embed = None
        self._init = lazyllm.once_flag()
        if init:
            lazyllm.call_once(self._init, self.load_embed)

    @abstractmethod
    def load_embed(self) -> None:
        """Abstract method for loading embedding models. This method is implemented by subclasses to perform specific model loading logic.

**Note**: This method is currently under development.
"""
        pass

    @abstractmethod
    def _call(self, data: Dict[str, Union[str, List[str]]]) -> str:
        pass

    def __call__(self, data: Dict[str, Union[str, List[str]]]) -> str:
        lazyllm.call_once(self._init, self.load_embed)
        return self._call(data)

    def __reduce__(self):
        init = bool(os.getenv('LAZYLLM_ON_CLOUDPICKLE', None) == 'ON' or self._init)
        return self.__class__, (self._base_embed, self._source, init)

class LazyHuggingFaceDefaultEmbedding(AbstractEmbedding):

    def load_embed(self):
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._tokenizer = tf.AutoTokenizer.from_pretrained(self._base_embed, trust_remote_code=True)
        self._embed = tf.AutoModel.from_pretrained(self._base_embed, trust_remote_code=True).to(self._device)
        self._embed.eval()

    def _call(self, data: Dict[str, Union[str, List[str]]]):
        string, _ = data['text'], data['images']
        encoded_input = self._tokenizer(string, padding=True, truncation=True, return_tensors='pt',
                                        max_length=512, add_special_tokens=True).to(self._device)
        with torch.no_grad():
            model_output = self._embed(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        res = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1).cpu().numpy().tolist()
        if type(string) is str:
            return json.dumps(res[0])
        else:
            return json.dumps(res)

class HuggingFaceEmbedding:
    """HuggingFace embedding model management class for managing and registering different embedding model implementations.

Attributes:
    _model_id_mapping (dict): Mapping dictionary from model IDs to implementation classes.

Args:
    base_embed (str): Path or name of the base embedding model.
    source (Optional[str]): Model source, defaults to None.
"""
    _model_id_mapping = {}

    @classmethod
    def get_emb_cls(cls, model_name: str):
        """Get the embedding implementation class for a model.

Args:
    model_name (str): Model name or path.

**Returns:**

- type: Returns corresponding embedding model implementation class, defaults to LazyHuggingFaceDefaultEmbedding if not found.
"""
        model_id = model_name.split('/')[-1].lower()
        return cls._model_id_mapping.get(model_id, LazyHuggingFaceDefaultEmbedding)

    @classmethod
    def register(cls, model_ids: List[str]):
        """Decorator for registering model IDs to specific implementation classes.

Args:
    model_ids (List[str]): List of model IDs to register.

**Returns:**

- Callable: Returns decorator function.
"""
        def decorator(target_class):
            for ele in model_ids:
                cls._model_id_mapping[ele.lower()] = target_class
            return target_class
        return decorator

    def __init__(self, base_embed, source=None):
        self._embed = self.__class__.get_emb_cls(base_embed)(base_embed, source)

    def load_embed(self):
        """Load the embedding model.

This method calls the load_embed method of the internal embedding implementation class to load the model.
"""
        self._embed.load_embed()

    def __call__(self, *args, **kwargs):
        try:
            args[0]['images'] = [_base64_to_file(image) if _is_base64_with_mime(image) else image
                                 for image in args[0]['images']]
        except Exception as e:
            LOG.error(f'Error converting base64 to image: {e}')
        return self._embed(*args, **kwargs)

class LazyFlagEmbedding(object):
    """A lazily loaded wrapper for the FlagEmbedding module.

This class encapsulates loading and usage of FlagEmbedding, with support for both sparse and dense embeddings. It leverages the lazyllm.once_flag() mechanism to initialize only once on demand, and integrates with LazyLLM's model downloading utilities.

Args:
    base_embed (str): The model name or path to be used as the embedding backend.
    sparse (bool): Whether to enable sparse embedding output. Defaults to False.
    source (str, optional): Source URL or identifier for model downloading. Defaults to global config.
    init (bool): Whether to initialize the model immediately upon construction. Defaults to False.
"""
    def __init__(self, base_embed, sparse=False, source=None, init=False):
        from ..utils.downloader import ModelManager
        source = lazyllm.config['model_source'] if not source else source
        self.base_embed = ModelManager(source).download(base_embed) or ''
        self.embed = None
        self.device = 'cpu'
        self.sparse = sparse
        self.init_flag = lazyllm.once_flag()
        if init:
            lazyllm.call_once(self.init_flag, self.load_embed)

    def load_embed(self):
        """Load the embedding model onto the appropriate device.

This method selects the available device (GPU or CPU) and initializes the pretrained FlagEmbedding model from the provided path or model hub.
"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        """Rebuild a LazyFlagEmbedding instance.

This class method reconstructs an instance of LazyFlagEmbedding, typically used during deserialization or multiprocessing scenarios.

Args:
    base_embed (str): The path or name of the embedding model.
    sparse (bool): Whether to enable sparse embedding mode.
    init (bool): Whether to load the model immediately during instantiation.

**Returns:**

- LazyFlagEmbedding: A newly constructed LazyFlagEmbedding instance.
"""
        return cls(base_embed, sparse, init=init)

    def __reduce__(self):
        init = bool(os.getenv('LAZYLLM_ON_CLOUDPICKLE', None) == 'ON' or self.init_flag)
        return LazyFlagEmbedding.rebuild, (self.base_embed, self.sparse, init)

class LazyHuggingFaceRerank(object):
    """Wrapper class for HuggingFace CrossEncoder-based reranking.  
Ranks candidate documents by relevance score with respect to a given query.  
Supports downloading and loading a specified rerank model at initialization, with optional lazy loading for faster startup.

Args:
    base_rerank (str): Name or local path of the rerank model. Supports HuggingFace Hub identifiers or local paths.
    source (Optional[str]): Source of the model, supports `huggingface` and `modelscope`. Defaults to global config `model_source`.
    init (bool): Whether to load the model immediately upon instantiation. If `False`, the model will be loaded lazily on first call.
"""
    def __init__(self, base_rerank, source=None, init=False):
        from ..utils.downloader import ModelManager
        source = lazyllm.config['model_source'] if not source else source
        self.base_rerank = ModelManager(source).download(base_rerank) or ''
        self.reranker = None
        self.init_flag = lazyllm.once_flag()
        if init:
            lazyllm.call_once(self.init_flag, self.load_reranker)

    def load_reranker(self):
        """Load the rerank model.  

This method initializes a `sentence_transformers.CrossEncoder` instance using `self.base_rerank`  
and assigns it to the class attribute `self.reranker` for subsequent reranking tasks.  
"""
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
        """Class method to rebuild a `LazyHuggingFaceRerank` instance.  
Used primarily for deserialization during pickle/cloudpickle operations,  
reinstantiating the object with the provided parameters.

Args:
    base_rerank (str): Model name or path.
    init (bool): Whether to load the model immediately upon rebuilding.

**Returns:**

- LazyHuggingFaceRerank: The rebuilt class instance.
"""
        return cls(base_rerank, init)

    def __reduce__(self):
        init = bool(os.getenv('LAZYLLM_ON_CLOUDPICKLE', None) == 'ON' or self.init_flag)
        return LazyHuggingFaceRerank.rebuild, (self.base_rerank, init)

class EmbeddingDeploy(LazyLLMDeployBase):
    """This class is a subclass of ``LazyLLMDeployBase``, designed for deploying text embedding services. It supports both dense and sparse embedding methods, compatible with HuggingFace models and FlagEmbedding models.

Args:
    launcher (Optional[lazyllm.launcher]): The launcher instance, defaults to ``None``.
    model_type (Optional[str]): Model type, defaults to ``'embed'``.
    log_path (Optional[str]): Path for log file, defaults to ``None``.
    embed_type (Optional[str]): Embedding type, either ``'dense'`` or ``'sparse'``, defaults to ``'dense'``.
    trust_remote_code (bool): Whether to trust remote code, defaults to ``True``.
    port (Optional[int]): Service port number, defaults to ``None``, in which case LazyLLM will generate a random port.

Call Arguments:
    finetuned_model (Optional[str]): Path or name of the fine-tuned model. 

    base_model (Optional[str]): Path or name of the base model, used when finetuned_model is invalid. 


Message Format:
    Input format is a dictionary containing text and images list.

    - text (str): Text content to be encoded

    - images (Union[str, List[str]]): List of images to be encoded (optional)



Examples:
    >>> from lazyllm import deploy
    >>> embed_service = deploy.EmbeddingDeploy(embed_type='dense')
    >>> embed_service('path/to/model')
    """
    message_format = {
        'text': 'text',  # str,
        'images': []  # Union[str, List[str]]
    }
    keys_name_handle = {
        'inputs': 'text',
        'image': 'images'
    }
    default_headers = {'Content-Type': 'application/json'}

    def __init__(self, launcher: LazyLLMLaunchersBase = None, model_type: str = 'embed', log_path: Optional[str] = None,
                 embed_type: Optional[str] = 'dense', trust_remote_code: bool = True, port: Optional[int] = None):
        super().__init__(launcher=launcher)
        self._launcher = launcher
        self._port = port
        self._model_type = model_type
        self._log_path = log_path
        self._sparse_embed = True if embed_type == 'sparse' else False
        self._trust_remote_code = trust_remote_code
        self._port = port

    def _get_model_path(self, finetuned_model=None, base_model=None):
        if not os.path.exists(finetuned_model) or \
            not any(filename.endswith('.bin') or filename.endswith('.safetensors')
                    for filename in os.listdir(finetuned_model)):
            if not finetuned_model:
                LOG.warning(f'Note! That finetuned_model({finetuned_model}) is an invalid path, '
                            f'base_model({base_model}) will be used')
            finetuned_model = base_model
        return finetuned_model

    def __call__(self, finetuned_model=None, base_model=None):
        finetuned_model = self._get_model_path(finetuned_model, base_model)
        if self._sparse_embed or lazyllm.config['default_embedding_engine'] == 'flagEmbedding':
            return lazyllm.deploy.RelayServer(port=self._port, func=LazyFlagEmbedding(
                finetuned_model, sparse=self._sparse_embed),
                launcher=self._launcher, log_path=self._log_path, cls='embedding')()
        else:
            return lazyllm.deploy.RelayServer(port=self._port, func=HuggingFaceEmbedding(finetuned_model),
                                              launcher=self._launcher, log_path=self._log_path, cls='embedding')()


@HuggingFaceEmbedding.register(model_ids=['BGE-VL-v1.5-mmeb'])
class _BGEVLEmbedding(AbstractEmbedding):

    def __init__(self, base_embed, source=None, init=False):
        super().__init__(base_embed, source, init)

    def load_embed(self):
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._embed = tf.AutoModel.from_pretrained(self._base_embed, trust_remote_code=True).to(self._device)
        self._embed.set_processor(self._base_embed)
        self._embed.processor.patch_size = self._embed.config.vision_config.patch_size
        self._embed.processor.vision_feature_select_strategy = self._embed.config.vision_feature_select_strategy
        self._embed.eval()

    def _call(self, data: Dict[str, Union[str, List[str]]]):
        DEFAULT_INSTRUCTION = 'Retrieve the target image that best meets the combined criteria by ' \
            'using both the provided image and the image retrieval instructions: '
        with torch.no_grad():
            # text='Make the background dark, as if the camera has taken the photo at night'
            # images='./cir_query.png'
            text, images = data['text'], data['images'][0] if isinstance(data['images'], list) else data['images']

            query_inputs = self._embed.data_process(
                text=text,
                images=images,
                q_or_c='q',
                task_instruction=DEFAULT_INSTRUCTION
            )
            query_embs = self._embed(**query_inputs, output_hidden_states=True)[:, -1, :]
            res = torch.nn.functional.normalize(query_embs, dim=-1).cpu().numpy().tolist()
            return json.dumps(res[0])


class RerankDeploy(EmbeddingDeploy):
    """This class is a subclass of ``EmbeddingDeploy``, designed for deploying reranking services. It supports text reranking using HuggingFace models.

Args:
    launcher (lazyllm.launcher): The launcher instance, defaults to ``None``.
    model_type (str): Model type, defaults to ``'embed'``.
    log_path (str): Path for log file, defaults to ``None``.
    trust_remote_code (bool): Whether to trust remote code, defaults to ``True``.
    port (int): Service port number, defaults to ``None``, in which case LazyLLM will generate a random port.

Call Arguments:
    finetuned_model: Path or name of the fine-tuned model. 

    base_model: Path or name of the base model, used when finetuned_model is invalid.


Message Format:
    Input format is a dictionary containing query (query text), documents (list of candidate documents), and top_n (number of documents to return).\n
    - query: Query text 

    - documents: List of candidate documents 

    - top_n: Number of documents to return, defaults to 1 



Examples:
    >>> from lazyllm import deploy
    >>> rerank_service = deploy.embed.RerankDeploy()
    >>> rerank_service('path/to/model')
    >>> input_data = {
    ...     "query": "What is machine learning?",
    ...     "documents": [
    ...         "Machine learning is a branch of AI.",
    ...         "Machine learning uses data to improve.",
    ...         "Deep learning is a subset of machine learning."
    ...     ],
    ...     "top_n": 2
    ... }
    >>> result = rerank_service(input_data)
    """
    message_format = {'query': 'query', 'documents': ['string'], 'top_n': 1}
    keys_name_handle = {'inputs': 'query', 'documents': 'documents', 'top_n': 'top_n'}
    default_headers = {'Content-Type': 'application/json'}

    def __call__(self, finetuned_model=None, base_model=None):
        finetuned_model = self._get_model_path(finetuned_model, base_model)
        return lazyllm.deploy.RelayServer(port=self._port, func=LazyHuggingFaceRerank(
            finetuned_model), launcher=self._launcher, log_path=self._log_path, cls='embedding')()
