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
创建一个用于文档查询和检索的检索模块。此构造函数初始化一个检索模块，该模块根据指定的相似度度量配置文档检索过程。

Args:
    doc: 文档模块实例。该文档模块可以是单个实例，也可以是一个实例的列表。如果是单个实例，表示对单个Document进行检索，如果是实例的列表，则表示对多个Document进行检索。
    group_name: 在哪个 node group 上进行检索。
    similarity: 用于设置文档检索的相似度函数。默认为 'dummy'。候选集包括 ["bm25", "bm25_chinese", "cosine"]。
    similarity_cut_off: 当相似度低于指定值时丢弃该文档。在多 embedding 场景下，如果需要对不同的 embedding 指定不同的值，则需要使用字典的方式指定，key 表示指定的是哪个 embedding，value 表示相应的阈值。如果所有的 embedding 使用同一个阈值，则只指定一个数值即可。
    index: 用于文档检索的索引类型。目前仅支持 'default'。
    topk: 表示取相似度最高的多少篇文档。
    embed_keys: 表示通过哪些 embedding 做检索，不指定表示用全部 embedding 进行检索。
    target：目标组名，将结果转换到目标组。
    output_format: 代表输出格式，默认为None，可选值有 'content' 和 'dict'，其中 content 对应输出格式为字符串，dict 对应字典。
    join: 是否联合输出的 k 个节点，当输出格式为 content 时，如果设置该值为 True，则输出一个长字符串，如果设置为 False 则输出一个字符串列表，其中每个字符串对应每个节点的文本内容。当输出格式是 dict 时，不能联合输出，此时join默认为False,，将输出一个字典，包括'content、'embedding'、'metadata'三个key。

其中 `group_name` 有三个内置的切分策略，都是使用 `SentenceSplitter` 做切分，区别在于块大小不同：

- CoarseChunk: 块大小为 1024，重合长度为 100
- MediumChunk: 块大小为 256，重合长度为 25
- FineChunk: 块大小为 128，重合长度为 12

此外，LazyLLM提供了内置的`Image`节点组存储了所有图像节点，支持图像嵌入和检索。


Examples:

    >>> import lazyllm
    >>> from lazyllm.tools import Retriever, Document, SentenceSplitter
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
    """
临时文档检索器，继承自 ModuleBase 和 _PostProcess，用于快速处理临时文件并执行检索任务。
Args:
    embed:嵌入函数。
    output_format:结果输出格式(如json),可选默认为None
    join:是否合并多段结果(True或用分隔符如"
")



Examples:

    >>> import lazyllm
    >>> from lazyllm.tools import TempDocRetriever, Document, SentenceSplitter
    >>> retriever = TempDocRetriever(output_format="text", join="
    ---------------
    ")
        retriever.create_node_group(transform=lambda text: [s.strip() for s in text.split("。") if s] )
        retriever.add_subretriever(group=Document.MediumChunk, topk=3)
        files = ["机器学习是AI的核心领域。深度学习是其重要分支。"]
        results = retriever.forward(files, "什么是机器学习?")
        print(results)
    """
    def __init__(self, embed: Callable = None, output_format: Optional[str] = None, join: Union[bool, str] = False):
        super().__init__()
        self._doc = Document(doc_files=[])
        self._embed = embed
        self._node_groups = []
        _PostProcess.__init__(self, output_format, join)

    def create_node_group(self, name: str = None, *, transform: Callable, parent: str = LAZY_ROOT_NAME,
                          trans_node: bool = None, num_workers: int = 0, **kwargs):
        """
创建具有特定处理流程的节点组。

Args:
    name (str): 节点组名称，None时自动生成。
    transform (Callable): 该组文档的处理函数。
    parent (str): 父组名称，默认为根组。
    trans_node (bool): 是否转换节点，None时继承父组设置。
    num_workers (int): 并行处理worker数，0表示串行。
    **kwargs: 其他组参数。

"""
        self._doc.create_node_group(name, transform=transform, parent=parent,
                                    trans_node=trans_node, num_workers=num_workers, **kwargs)
        return self

    def add_subretriever(self, group: str, **kwargs):
        """
添加带搜索配置的子检索器。

Args:
    group (str): 目标节点组名称。
    **kwargs: 检索器参数（如similarity='cosine'）。

Returns:
    self: 支持链式调用。
"""
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
