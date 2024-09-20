from lazyllm import ModuleBase, pipeline
from .store import DocNode
from .document import Document, DocImpl
from typing import List, Optional, Union

class _PostProcess(object):
    def __init__(self, target: Optional[str] = None,
                 output_format: Optional[str] = None,
                 join: Union[bool, str] = False) -> None:
        self._target = target
        assert output_format in (None, 'content', 'dict'), 'output_format should be None, \'content\', or \'dict\''
        self._output_format = output_format
        if join is True: join = ''
        assert join is False or (isinstance(join, str) and output_format == 'content'), (
            'Only content output can be joined')
        self._join = join

    def _post_process(self, nodes):
        if self._target:
            # TODO(wangzhihong): search relationship and add find_child
            nodes = self._doc.find_parent(self._target)(nodes)
        if self._output_format == 'content':
            nodes = [node.get_content() for node in nodes]
            if isinstance(self._join, str): nodes = self._join.join(nodes)
        elif self._output_format == 'dict':
            nodes = [node.to_dict() for node in nodes]
        return nodes

class Retriever(ModuleBase, _PostProcess):
    __enable_request__ = False

    def __init__(
        self,
        doc: object,
        group_name: str,
        similarity: str = "dummy",
        similarity_cut_off: float = float("-inf"),
        index: str = "default",
        topk: int = 6,
        embed_keys: Optional[List[str]] = None,
        target: Optional[str] = None,
        output_format: Optional[str] = None,
        join: Union[bool, str] = False,
        **kwargs,
    ):
        super().__init__()

        self._docs = [doc] if isinstance(doc, Document) else doc
        for doc in self._docs:
            assert isinstance(doc, Document), 'Only Document or List[Document] are supported'
            self._submodules.append(doc)

        self._group_name = group_name
        self._similarity = similarity  # similarity function str
        self._similarity_cut_off = similarity_cut_off
        self._index = index
        self._topk = topk
        self._similarity_kw = kwargs  # kw parameters
        self._embed_keys = embed_keys
        _PostProcess.__init__(self, target, output_format, join)

    def _get_post_process_tasks(self):
        return pipeline(lambda *a: self('Test Query'))

    def forward(self, query: str) -> Union[List[DocNode], str]:
        nodes = []
        for doc in self._docs:
            if self._group_name not in doc._impl._impl.node_groups and \
                    self._group_name not in DocImpl._builtin_node_groups and \
                    self._group_name not in DocImpl._global_node_groups:
                if len(self._docs) > 1: continue
                raise RuntimeError(f'Group {self._group_name} not found in document {doc}')
            nodes.extend(doc.forward(func_name="retrieve", query=query, group_name=self._group_name,
                                     similarity=self._similarity, similarity_cut_off=self._similarity_cut_off,
                                     index=self._index, topk=self._topk, similarity_kws=self._similarity_kw,
                                     embed_keys=self._embed_keys))
        return self._post_process(nodes)
