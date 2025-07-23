from typing import List, Optional, Union, Dict, Set, Callable
from lazyllm import ModuleBase, once_wrapper

from .doc_node import DocNode
from .document import Document, UrlDocument, DocImpl
from .store import LAZY_ROOT_NAME
from .similarity import registered_similarities
import functools
import lazyllm


class _PostProcess(object):
    def __init__(self, output_format: Optional[str] = None,
                 join: Union[bool, str] = False) -> None:
        assert output_format in (None, 'content', 'dict'), 'output_format should be None, \'content\', or \'dict\''
        self._output_format = output_format
        if join is True: join = ''
        assert join is False or (isinstance(join, str) and output_format == 'content'), (
            'Only content output can be joined')
        self._join = join

    def _post_process(self, nodes):
        if self._output_format == 'content':
            nodes = [node.get_content() for node in nodes]
            if isinstance(self._join, str): nodes = self._join.join(nodes)
        elif self._output_format == 'dict':
            nodes = [node.to_dict() for node in nodes]
        return nodes

class Retriever(ModuleBase, _PostProcess):
    """
Create a retrieval module for document querying and retrieval. This constructor initializes a retrieval module that configures the document retrieval process based on the specified similarity metric.

Args:
    doc: An instance of the document module. The document module can be a single instance or a list of instances. If it is a single instance, it means searching for a single Document, and if it is a list of instances, it means searching for multiple Documents.
    group_name: The name of the node group on which to perform the retrieval.
    similarity: The similarity function to use for setting up document retrieval. Defaults to 'dummy'. Candidates include ["bm25", "bm25_chinese", "cosine"].
    similarity_cut_off: Discard the document when the similarity is below the specified value. In a multi-embedding scenario, if you need to specify different values for different embeddings, you need to specify them in a dictionary, where the key indicates which embedding is specified and the value indicates the corresponding threshold. If all embeddings use the same threshold, you only need to specify one value.
    index: The type of index to use for document retrieval. Currently, only 'default' is supported.
    topk: The number of documents to retrieve with the highest similarity.
    embed_keys: Indicates which embeddings are used for retrieval. If not specified, all embeddings are used for retrieval.
    similarity_kw: Additional parameters to pass to the similarity calculation function.
    output_format: Represents the output format, with a default value of None. Optional values include 'content' and 'dict', where 'content' corresponds to a string output format and 'dict' corresponds to a dictionary.
    join:  Determines whether to concatenate the output of k nodes - when output format is 'content', setting True returns a single concatenated string while False returns a list of strings (each corresponding to a node's text content); when output format is 'dict', joining is unsupported (join defaults to False) and the output will be a dictionary containing 'content', 'embedding' and 'metadata' keys.
                
The `group_name` has three built-in splitting strategies, all of which use `SentenceSplitter` for splitting, with the difference being in the chunk size:

- CoarseChunk: Chunk size is 1024, with an overlap length of 100
- MediumChunk: Chunk size is 256, with an overlap length of 25
- FineChunk: Chunk size is 128, with an overlap length of 12

Also, `Image` is available for `group_name` since LazyLLM supports image embedding and retrieval.


Examples:
    
    >>> import lazyllm
    >>> from lazyllm.tools import Retriever, Document, SentenceSplitter
    >>> m = lazyllm.OnlineEmbeddingModule()
    >>> documents = Document(dataset_path='/path/to/user/data', embed=m, manager=False)
    >>> rm = Retriever(documents, group_name='CoarseChunk', similarity='bm25', similarity_cut_off=0.01, topk=6)
    >>> rm.start()
    >>> print(rm("user query"))
    >>> m1 = lazyllm.TrainableModule('bge-large-zh-v1.5').start()
    >>> document1 = Document(dataset_path='/path/to/user/data', embed={'online':m , 'local': m1}, manager=False)
    >>> document1.create_node_group(name='sentences', transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
    >>> retriever = Retriever(document1, group_name='sentences', similarity='cosine', similarity_cut_off=0.4, embed_keys=['local'], topk=3)
    >>> print(retriever("user query"))
    >>> document2 = Document(dataset_path='/path/to/user/data', embed={'online':m , 'local': m1}, manager=False)
    >>> document2.create_node_group(name='sentences', transform=SentenceSplitter, chunk_size=512, chunk_overlap=50)
    >>> retriever2 = Retriever([document1, document2], group_name='sentences', similarity='cosine', similarity_cut_off=0.4, embed_keys=['local'], topk=3)
    >>> print(retriever2("user query"))
    >>>
    >>> filters = {
    >>>     "author": ["A", "B", "C"],
    >>>     "public_year": [2002, 2003, 2004],
    >>> }
    >>> document3 = Document(dataset_path='/path/to/user/data', embed={'online':m , 'local': m1}, manager=False)
    >>> document3.create_node_group(name='sentences', transform=SentenceSplitter, chunk_size=512, chunk_overlap=50)
    >>> retriever3 = Retriever([document1, document3], group_name='sentences', similarity='cosine', similarity_cut_off=0.4, embed_keys=['local'], topk=3)
    >>> print(retriever3(query="user query", filters=filters))
    >>> document4 = Document(dataset_path='/path/to/user/data', embed=lazyllm.TrainableModule('siglip'))
    >>> retriever4 = Retriever(document4, group_name='Image', similarity='cosine')
    >>> nodes = retriever4("user query")
    >>> print([node.get_content() for node in nodes])
    >>> document5 = Document(dataset_path='/path/to/user/data', embed=m, manager=False)
    >>> rm = Retriever(document5, group_name='CoarseChunk', similarity='bm25_chinese', similarity_cut_off=0.01, topk=3, output_format='content')
    >>> rm.start()
    >>> print(rm("user query"))
    >>> document6 = Document(dataset_path='/path/to/user/data', embed=m, manager=False)
    >>> rm = Retriever(document6, group_name='CoarseChunk', similarity='bm25_chinese', similarity_cut_off=0.01, topk=3, output_format='content', join=True)
    >>> rm.start()
    >>> print(rm("user query"))
    >>> document7 = Document(dataset_path='/path/to/user/data', embed=m, manager=False)
    >>> rm = Retriever(document7, group_name='CoarseChunk', similarity='bm25_chinese', similarity_cut_off=0.01, topk=3, output_format='dict')
    >>> rm.start()
    >>> print(rm("user query"))
    """
    def __init__(self, doc: object, group_name: str, similarity: Optional[str] = None,
                 similarity_cut_off: Union[float, Dict[str, float]] = float("-inf"), index: str = "default",
                 topk: int = 6, embed_keys: Optional[List[str]] = None, target: Optional[str] = None,
                 output_format: Optional[str] = None, join: Union[bool, str] = False, **kwargs):
        super().__init__()

        if similarity:
            _, mode, _ = registered_similarities[similarity]
        else:
            mode = 'embedding'  # TODO FIXME XXX should be removed after similarity args refactor
        group_name, target = str(group_name), (str(target) if target else None)

        self._docs: List[Document] = [doc] if isinstance(doc, Document) else doc
        for doc in self._docs:
            assert isinstance(doc, (Document, UrlDocument)), 'Only Document or List[Document] are supported'
            if isinstance(doc, UrlDocument): continue
            self._submodules.append(doc)
            if mode == 'embedding' and embed_keys is None:
                embed_keys = list(doc._impl.embed.keys())
            doc.activate_group(group_name, embed_keys)
            if target: doc.activate_group(target)

        self._group_name = group_name
        self._similarity = similarity  # similarity function str
        self._similarity_cut_off = similarity_cut_off
        self._index = index
        self._topk = topk
        self._similarity_kw = kwargs  # kw parameters
        self._embed_keys = embed_keys
        self._target = target
        _PostProcess.__init__(self, output_format, join)

    @once_wrapper
    def _lazy_init(self):
        docs = [doc for doc in self._docs if isinstance(doc, UrlDocument) or self._group_name in doc._impl.node_groups
                or self._group_name in DocImpl._builtin_node_groups or self._group_name in DocImpl._global_node_groups]
        if not docs: raise RuntimeError(f'Group {self._group_name} not found in document {self._docs}')
        self._docs = docs

    def forward(
            self, query: str, filters: Optional[Dict[str, Union[str, int, List, Set]]] = None,
            **kwargs
    ) -> Union[List[DocNode], str]:
        self._lazy_init()
        all_nodes: List[DocNode] = []
        for doc in self._docs:
            nodes = doc.forward(query=query, group_name=self._group_name, similarity=self._similarity,
                                similarity_cut_off=self._similarity_cut_off, index=self._index,
                                topk=self._topk, similarity_kws=self._similarity_kw, embed_keys=self._embed_keys,
                                filters=filters, **kwargs)
            if nodes and self._target and self._target != nodes[0]._group:
                nodes = doc.find(self._target)(nodes)
            all_nodes.extend(nodes)
        return self._post_process(all_nodes)


class TempDocRetriever(ModuleBase, _PostProcess):
    def __init__(self, embed: Callable = None, output_format: Optional[str] = None, join: Union[bool, str] = False):
        super().__init__()
        self._doc = Document(doc_files=[])
        self._embed = embed
        self._node_groups = []
        _PostProcess.__init__(self, output_format, join)

    def create_node_group(self, name: str = None, *, transform: Callable, parent: str = LAZY_ROOT_NAME,
                          trans_node: bool = None, num_workers: int = 0, **kwargs):
        self._doc.create_node_group(name, transform=transform, parent=parent,
                                    trans_node=trans_node, num_workers=num_workers, **kwargs)
        return self

    def add_subretriever(self, group: str, **kwargs):
        if 'similarity' not in kwargs: kwargs['similarity'] = ('cosine' if self._embed else 'bm25')
        self._node_groups.append((group, kwargs))
        return self

    @functools.lru_cache
    def _get_retrievers(self, doc_files: List[str]):
        active_node_groups = self._node_groups or [[Document.MediumChunk,
                                                    dict(similarity=('cosine' if self._embed else 'bm25'))]]
        doc = Document(embed=self._embed, doc_files=doc_files)
        doc._impl.node_groups = self._doc._impl.node_groups
        retrievers = [Retriever(doc, name, **kw) for (name, kw) in active_node_groups]
        return retrievers

    def forward(self, files: Union[str, List[str]], query: str):
        if isinstance(files, str): files = [files]
        retrievers = self._get_retrievers(doc_files=tuple(set(files)))
        r = lazyllm.parallel(*retrievers).sum
        return self._post_process(r(query))
