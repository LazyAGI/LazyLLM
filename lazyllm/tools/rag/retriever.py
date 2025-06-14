from lazyllm import ModuleBase, once_wrapper
from .doc_node import DocNode
from .document import Document, UrlDocument, DocImpl
from .store_base import LAZY_ROOT_NAME
from typing import List, Optional, Union, Dict, Set, Callable
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
            self, query: str, filters: Optional[Dict[str, Union[str, int, List, Set]]] = None
    ) -> Union[List[DocNode], str]:
        self._lazy_init()
        all_nodes: List[DocNode] = []
        for doc in self._docs:
            nodes = doc.forward(query=query, group_name=self._group_name, similarity=self._similarity,
                                similarity_cut_off=self._similarity_cut_off, index=self._index,
                                topk=self._topk, similarity_kws=self._similarity_kw, embed_keys=self._embed_keys,
                                filters=filters)
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
