from functools import lru_cache
from typing import Callable, List, Optional, Union, Dict, Any
import requests

import lazyllm
from lazyllm import ModuleBase, LOG
from .doc_node import DocNode, MetadataMode
from .retriever import _PostProcess


class Reranker(ModuleBase, _PostProcess):
    registered_reranker = dict()

    def __new__(cls, name: str = "ModuleReranker", *args, **kwargs):
        assert name in cls.registered_reranker, f"Reranker: {name} is not registered, please register first."
        item = cls.registered_reranker[name]
        if isinstance(item, type) and issubclass(item, Reranker):
            return super(Reranker, cls).__new__(item)
        else:
            return super(Reranker, cls).__new__(cls)

    def __init__(self, name: str = "ModuleReranker", target: Optional[str] = None,
                 output_format: Optional[str] = None, join: Union[bool, str] = False, **kwargs) -> None:
        super().__init__()
        self._name = name
        self._kwargs = kwargs
        _PostProcess.__init__(self, target, output_format, join)

    def forward(self, nodes: List[DocNode], query: str = "") -> List[DocNode]:
        results = self.registered_reranker[self._name](nodes, query=query, **self._kwargs)
        LOG.debug(f"Rerank use `{self._name}` and get nodes: {results}")
        return self._post_process(results)

    @classmethod
    def register_reranker(
        cls: "Reranker", func: Optional[Callable] = None, batch: bool = False
    ):
        def decorator(f):
            if isinstance(f, type):
                cls.registered_reranker[f.__name__] = f
                return f
            else:
                def wrapper(nodes, **kwargs):
                    if batch:
                        return f(nodes, **kwargs)
                    else:
                        results = [f(node, **kwargs) for node in nodes]
                        return [result for result in results if result]

                cls.registered_reranker[f.__name__] = wrapper
                return wrapper

        return decorator(func) if func else decorator


@lru_cache(maxsize=None)
def get_nlp_and_matchers(language):
    import spacy
    from spacy.matcher import PhraseMatcher

    nlp = spacy.blank(language)
    required_matcher = PhraseMatcher(nlp.vocab)
    exclude_matcher = PhraseMatcher(nlp.vocab)
    return nlp, required_matcher, exclude_matcher


@Reranker.register_reranker
def KeywordFilter(
    node: DocNode,
    required_keys: List[str] = [],
    exclude_keys: List[str] = [],
    language: str = "en",
    **kwargs,
) -> Optional[DocNode]:
    assert required_keys or exclude_keys, 'One of required_keys or exclude_keys should be provided'
    nlp, required_matcher, exclude_matcher = get_nlp_and_matchers(language)
    if required_keys:
        required_matcher.add("RequiredKeywords", list(nlp.pipe(required_keys)))
    if exclude_keys:
        exclude_matcher.add("ExcludeKeywords", list(nlp.pipe(exclude_keys)))

    doc = nlp(node.get_text())
    if required_keys and not required_matcher(doc):
        return None
    if exclude_keys and exclude_matcher(doc):
        return None
    return node

@Reranker.register_reranker()
class ModuleReranker(Reranker):

    def __init__(self, name: str = "ModuleReranker", target: Optional[str] = None,
                 output_format: Optional[str] = None, join: Union[bool, str] = False, **kwargs) -> None:
        super().__init__(name, target, output_format, join, **kwargs)
        assert 'model' in self._kwargs
        self._reranker = lazyllm.TrainableModule(self._kwargs['model'])

    def forward(self, nodes: List[DocNode], query: str = "") -> List[DocNode]:
        docs = [node.get_text(metadata_mode=MetadataMode.EMBED) for node in nodes]
        top_n = self._kwargs['topk'] if 'topk' in self._kwargs else len(docs)
        if self._reranker._deploy_type == lazyllm.deploy.Infinity:
            sorted_indices = self._reranker(query, documents=docs, top_n=top_n)
        else:
            inps = {'query': query, 'documents': docs, 'top_n': top_n}
            sorted_indices = self._reranker(inps)
        results = [nodes[i] for i in sorted_indices]
        LOG.debug(f"Rerank use `{self._name}` and get nodes: {results}")
        return self._post_process(results)


@Reranker.register_reranker()
class OnlineReranker(Reranker):
    MODELS = {
        "qwen": {
            "url": "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
            "model_name": "gte-rerank"
        }
    }
    def __init__(self,
                 name: str = "OnlineReranker",
                 source: Optional[str] = None,
                 api_key: Optional[str] = None,
                 rerank_url: Optional[str] = None,
                 rerank_model_name: Optional[str] = None,
                 return_trace: bool = False, **kwargs):
        super().__init__(name, **kwargs)
        
        if source is None:
            if "api_key" in kwargs and kwargs["api_key"]:
                raise ValueError("No source is given but an api_key is provided.")
                for source in OnlineReranker.MODELS.keys():
                    if lazyllm.config[f'{source}_api_key']: break
            else:
                raise KeyError(f"No api_key is configured for any of the models {OnlineReranker.MODELS.keys()}.")
        else:
            assert source in OnlineReranker.MODELS.keys(), f"Unsupported source: {source}"

        if rerank_url is None:
            rerank_url = OnlineReranker.MODELS[source]["url"]
        if rerank_model_name is None:
            rerank_model_name = OnlineReranker.MODELS[source]["model_name"]
        if api_key is None:
            api_key = lazyllm.config[f'{source}_api_key']
            
        self._rerank_url = rerank_url
        self._api_key = api_key
        self._rerank_model_name = rerank_model_name
        self._set_headers()
    
    def _set_headers(self) -> Dict[str, str]:
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }
    
    def forward(self, nodes: List[DocNode], query: str = "") -> List[DocNode]:
        docs = [node.get_text(metadata_mode=MetadataMode.EMBED) for node in nodes]
        top_n = self._kwargs['topk'] if 'topk' in self._kwargs else len(docs)
        inps = self._encapsulate_data(query, docs, top_n)
        with requests.post(self._rerank_url, json=inps, headers=self._headers) as r:
            if r.status_code == 200:
                sorted_indices = self._parse_response(r.json())
            else:
                raise requests.RequestException('\n'.join([c.decode('utf-8') for c in r.iter_content(None)]))
        
        results = [nodes[i] for i in sorted_indices]
        LOG.debug(f"Rerank use `{self._name}` and get nodes: {results}")
        return self._post_process(results)

    def _encapsulate_data(self, query: str, docs: List[str], top_n: int, **kwargs) -> Dict[str, str]:
        json_data = {
            "input": {
                "query": query,
                "documents": docs
            },
            "parameters": {
                "top_n": top_n,
                "return_documents": "true"
            },
            "model": self._rerank_model_name
        }

        return json_data
    
    def _parse_response(self, response: Dict[str, Any]) -> List[float]:
        results = response['output']['results']
        return [result["index"] for result in results]
    
    
# User-defined similarity decorator
def register_reranker(func=None, batch=False):
    return Reranker.register_reranker(func, batch)
