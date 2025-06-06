import importlib.util

from functools import lru_cache
from typing import Callable, List, Optional, Union

import lazyllm
from lazyllm.thirdparty import spacy
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
        lazyllm.deprecated(bool(target), '`target` parameter of reranker')
        _PostProcess.__init__(self, output_format, join)

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
    nlp = spacy.blank(language)

    spec = importlib.util.find_spec("spacy.matcher")
    if spec is None:
        raise ImportError(
            "Please install spacy to use spacy module. "
            "You can install it with `pip install spacy==3.7.5`"
        )
    matcher_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matcher_module)

    required_matcher = matcher_module.PhraseMatcher(nlp.vocab)
    exclude_matcher = matcher_module.PhraseMatcher(nlp.vocab)
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

    def __init__(self, name: str = "ModuleReranker", model: Union[Callable, str] = None, target: Optional[str] = None,
                 output_format: Optional[str] = None, join: Union[bool, str] = False, **kwargs) -> None:
        super().__init__(name, target, output_format, join, **kwargs)
        assert model is not None, "Reranker model must be specified as a model name or a callable."
        if isinstance(model, str):
            self._reranker = lazyllm.TrainableModule(model)
        else:
            self._reranker = model

    def forward(self, nodes: List[DocNode], query: str = "") -> List[DocNode]:
        if not nodes:
            return self._post_process([])

        docs = [node.get_text(metadata_mode=MetadataMode.EMBED) for node in nodes]
        top_n = self._kwargs['topk'] if 'topk' in self._kwargs else len(docs)
        sorted_indices = self._reranker(query, documents=docs, top_n=top_n)
        results = []
        for index, relevance_score in sorted_indices:
            results.append(nodes[index].with_score(relevance_score))
        LOG.debug(f"Rerank use `{self._name}` and get nodes: {results}")
        return self._post_process(results)

# User-defined similarity decorator
def register_reranker(func=None, batch=False):
    return Reranker.register_reranker(func, batch)
