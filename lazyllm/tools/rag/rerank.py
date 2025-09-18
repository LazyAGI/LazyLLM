import importlib.util

from functools import lru_cache
from typing import Callable, List, Optional, Union

import lazyllm
from lazyllm.thirdparty import spacy
from lazyllm import ModuleBase, LOG
from .doc_node import DocNode, MetadataMode
from .retriever import _PostProcess


class Reranker(ModuleBase, _PostProcess):
    """Initializes a Rerank module for postprocessing and reranking of nodes (documents).
This constructor initializes a Reranker module that configures a reranking process based on a specified reranking type. It allows for the dynamic selection and instantiation of reranking kernels (algorithms) based on the type and provided keyword arguments.

Args:
    name: The type of reranker used for the postprocessing and reranking process. Defaults to 'ModuleReranker'.
    target (str): **Deprecated** parameter, only used to notify users.
    output_format: Specifies the output format. Defaults to None. Optional values include 'content' and 'dict'. 
        - 'content' means the output is in string format.
        - 'dict' means the output is a dictionary.
    join: Determines whether to join the top-k output nodes.
        - When `output_format` is 'content':
            - If set to True, returns a single long string.
            - If set to False, returns a list of strings, each representing one node’s content.
        - When `output_format` is 'dict':
            - Joining is not supported; `join` defaults to False.
            - Returns a dictionary with three keys: 'content', 'embedding', and 'metadata'.
    kwargs: Additional keyword arguments passed to the reranker upon instantiation.

**Detailed explanation of reranker types**

- Reranker: Instantiates a `SentenceTransformerRerank` reranker with a list of document nodes and a query.

- KeywordFilter: This registered reranking function instantiates a KeywordNodePostprocessor with specified required and excluded keywords. It filters nodes based on the presence or absence of these keywords.


Examples:
    
    >>> import lazyllm
    >>> from lazyllm.tools import Document, Reranker, Retriever, DocNode
    >>> m = lazyllm.OnlineEmbeddingModule()
    >>> documents = Document(dataset_path='/path/to/user/data', embed=m, manager=False)
    >>> retriever = Retriever(documents, group_name='CoarseChunk', similarity='bm25', similarity_cut_off=0.01, topk=6)
    >>> reranker = Reranker(DocNode(text=user_data),query="user query")
    >>> ppl = lazyllm.ActionModule(retriever, reranker)
    >>> ppl.start()
    >>> print(ppl("user query"))
    """
    registered_reranker = dict()

    def __new__(cls, name: str = 'ModuleReranker', *args, **kwargs):
        assert name in cls.registered_reranker, f'Reranker: {name} is not registered, please register first.'
        item = cls.registered_reranker[name]
        if isinstance(item, type) and issubclass(item, Reranker):
            return super(Reranker, cls).__new__(item)
        else:
            return super(Reranker, cls).__new__(cls)

    def __init__(self, name: str = 'ModuleReranker', target: Optional[str] = None,
                 output_format: Optional[str] = None, join: Union[bool, str] = False, **kwargs) -> None:
        super().__init__()
        self._name = name
        self._kwargs = kwargs
        lazyllm.deprecated(bool(target), '`target` parameter of reranker')
        _PostProcess.__init__(self, output_format, join)

    def forward(self, nodes: List[DocNode], query: str = '') -> List[DocNode]:
        results = self.registered_reranker[self._name](nodes, query=query, **self._kwargs)
        LOG.debug(f'Rerank use `{self._name}` and get nodes: {results}')
        return self._post_process(results)

    @classmethod
    def register_reranker(
        cls: 'Reranker', func: Optional[Callable] = None, batch: bool = False
    ):
        """A class decorator factory method that provides a flexible mechanism for registering custom reranking algorithms to the `Reranker` class.

Args:
    func (Optional[Callable]): The reranking function or class to register. This can be omitted when using decorator syntax (@).
    batch (bool): Whether to process nodes in batches. Defaults to False, meaning nodes are processed individually.


Examples:
    
    @Reranker.register_reranker
    def my_reranker(node: DocNode, **kwargs):
        return node.score * 0.8  # 自定义分数计算
    """
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

    spec = importlib.util.find_spec('spacy.matcher')
    if spec is None:
        raise ImportError(
            'Please install spacy to use spacy module. '
            'You can install it with `pip install spacy==3.7.5`'
        )
    matcher_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matcher_module)

    required_matcher = matcher_module.PhraseMatcher(nlp.vocab)
    exclude_matcher = matcher_module.PhraseMatcher(nlp.vocab)
    return nlp, required_matcher, exclude_matcher


@Reranker.register_reranker
def KeywordFilter(node: DocNode, required_keys: Optional[List[str]] = None, exclude_keys: Optional[List[str]] = None,
                  language: str = 'en', **kwargs) -> Optional[DocNode]:
    assert required_keys or exclude_keys, 'One of required_keys or exclude_keys should be provided'
    nlp, required_matcher, exclude_matcher = get_nlp_and_matchers(language)
    if required_keys:
        required_matcher.add('RequiredKeywords', list(nlp.pipe(required_keys)))
    if exclude_keys:
        exclude_matcher.add('ExcludeKeywords', list(nlp.pipe(exclude_keys)))

    doc = nlp(node.get_text())
    if required_keys and not required_matcher(doc):
        return None
    if exclude_keys and exclude_matcher(doc):
        return None
    return node

@Reranker.register_reranker()
class ModuleReranker(Reranker):
    """A reranker that uses trainable modules to reorder documents based on relevance to a query.

ModuleReranker is a specialized reranker that leverages trainable models (such as BGE-reranker, Cohere rerank, etc.) to improve the relevance of retrieved documents. It takes a list of documents and a query, then returns the documents reordered by their relevance scores.

Args:
    name (str): The name of the reranker. Defaults to "ModuleReranker".
    model (Union[Callable, str]): The reranking model. Can be either a model name (string) or a callable function.
    target (Optional[str]): Defaults to None.
    output_format (Optional[str]): The format for output processing. Defaults to None.
    join (Union[bool, str]): Whether to join the results. Defaults to False.
    **kwargs: Additional keyword arguments passed to the reranker model.


Examples:
    >>> from lazyllm.tools.rag.rerank import ModuleReranker, DocNode
    >>> def simple_reranker(query, documents, top_n):
    ...     query_lower = query.lower()
    ...     scores = []
    ...     for i, doc in enumerate(documents):
    ...         score = sum(1 for word in query_lower.split() if word in doc)
    ...         scores.append((i, score))
    ...     scores.sort(key=lambda x: x[1], reverse=True)
    ...     return scores[:top_n]
    >>> reranker = ModuleReranker(
    ...     model=simple_reranker,
    ...     topk=2
    ... )
    >>> docs = [
    ...     DocNode(text="机器学习算法在数据分析中应用广泛"),
    ...     DocNode(text="深度学习模型需要大量训练数据"),
    ...     DocNode(text="自然语言处理技术发展迅速"),
    ...     DocNode(text="计算机视觉在自动驾驶中的应用")
    ... ]
    >>> query = "机器学习"
    >>> results = reranker.forward(docs, query)
    >>> for i, doc in enumerate(results):
    ...     print(f"  {i+1}. : {doc.text}")
    ...     print(f"     相关性分数: {doc.relevance_score:.4f}")        
    """

    def __init__(self, name: str = 'ModuleReranker', model: Union[Callable, str] = None, target: Optional[str] = None,
                 output_format: Optional[str] = None, join: Union[bool, str] = False, **kwargs) -> None:
        super().__init__(name, target, output_format, join, **kwargs)
        assert model is not None, 'Reranker model must be specified as a model name or a callable.'
        if isinstance(model, str):
            self._reranker = lazyllm.TrainableModule(model)
        else:
            self._reranker = model

    def forward(self, nodes: List[DocNode], query: str = '') -> List[DocNode]:
        """Forward pass of the reranker that reorders documents based on relevance to the query.

This method takes a list of documents and a query, then uses the underlying reranking model to score and reorder the documents by relevance. The documents are processed in MetadataMode.EMBED format to ensure compatibility with the reranking model.

Args:
    nodes (List[DocNode]): List of document nodes to be reranked.
    query (str): The query string to rank documents against. Defaults to "".

**Returns:**

- List[DocNode]: List of document nodes reordered by relevance score, with relevance_score attribute added.
"""
        if not nodes:
            return self._post_process([])

        docs = [node.get_text(metadata_mode=MetadataMode.EMBED) for node in nodes]
        top_n = self._kwargs['topk'] if 'topk' in self._kwargs else len(docs)
        sorted_indices = self._reranker(query, documents=docs, top_n=top_n)
        results = []
        for index, relevance_score in sorted_indices:
            results.append(nodes[index].with_score(relevance_score))
        LOG.debug(f'Rerank use `{self._name}` and get nodes: {results}')
        return self._post_process(results)

# User-defined similarity decorator
def register_reranker(func=None, batch=False):
    return Reranker.register_reranker(func, batch)
