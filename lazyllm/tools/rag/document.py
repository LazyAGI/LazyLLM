import os

from typing import Callable, Optional, Dict, Union, List
import lazyllm
from lazyllm import ModuleBase, ServerModule, DynamicDescriptor, deprecated, OnlineChatModule, TrainableModule
from lazyllm.launcher import LazyLLMLaunchersBase as Launcher
from lazyllm.tools.sql.sql_manager import SqlManager, DBStatus
from lazyllm.common.bind import _MetaBind

from .doc_manager import DocManager
from .doc_impl import DocImpl, StorePlaceholder, EmbedPlaceholder, BuiltinGroups, DocumentProcessor, NodeGroupType
from .doc_node import DocNode
from .doc_to_db import DocInfoSchema, DocToDbProcessor, extract_db_schema_from_files
from .store import LAZY_ROOT_NAME, EMBED_DEFAULT_KEY
from .index_base import IndexBase
from .utils import DocListManager
from .global_metadata import GlobalMetadataDesc as DocField
from .web import DocWebModule
import copy
import functools


class CallableDict(dict):
    def __call__(self, cls, *args, **kw):
        return self[cls](*args, **kw)


class _MetaDocument(_MetaBind):
    def __instancecheck__(self, __instance):
        if isinstance(__instance, UrlDocument): return True
        return super().__instancecheck__(__instance)


class Document(ModuleBase, BuiltinGroups, metaclass=_MetaDocument):
    """初始化一个具有可选用户界面的文档模块。

此构造函数初始化一个可以有或没有用户界面的文档模块。如果启用了用户界面，它还会提供一个ui界面来管理文档操作接口，并提供一个用于用户界面交互的网页。

Args:
    dataset_path (str): 数据集目录的路径。此目录应包含要由文档模块管理的文档。
    embed (Optional[Union[Callable, Dict[str, Callable]]]): 用于生成文档 embedding 的对象。如果需要对文本生成多个 embedding，此处需要通过字典的方式指定多个 embedding 模型，key 标识 embedding 对应的名字, value 为对应的 embedding 模型。
    create_ui (bool):[已弃用] 是否创建用户界面。请改用'manager'参数
    manager (bool, optional): 指示是否为文档模块创建用户界面的标志。默认为 False。
    server (Union[bool, int]):服务器配置。True表示默认服务器，False表示已指定端口号作为自定义服务器
    name (Optional[str]):文档集合的名称标识符。云服务模式下必须提供
    launcher (optional): 负责启动服务器模块的对象或函数。如果未提供，则使用 `lazyllm.launchers` 中的默认异步启动器 (`sync=False`)。            
    doc_files (Optional[List[str]]):临时文档文件列表（dataset_path的替代方案）。使用时dataset_path必须为None且仅支持map存储类型
    doc_fields (optional): 配置需要存储和检索的字段继对应的类型（目前只有 Milvus 后端会用到）。
    store_conf (optional): 配置使用哪种存储后端和索引后端。


Examples:
    >>> import lazyllm
    >>> from lazyllm.tools import Document
    >>> m = lazyllm.OnlineEmbeddingModule(source="glm")
    >>> documents = Document(dataset_path='your_doc_path', embed=m, manager=False)  # or documents = Document(dataset_path='your_doc_path', embed={"key": m}, manager=False)
    >>> m1 = lazyllm.TrainableModule("bge-large-zh-v1.5").start()
    >>> document1 = Document(dataset_path='your_doc_path', embed={"online": m, "local": m1}, manager=False)

    >>> store_conf = {
    >>>     'type': 'chroma',
    >>>     'indices': {
    >>>         'smart_embedding_index': {
    >>>             'backend': 'milvus',
    >>>             'kwargs': {
    >>>                 'uri': '/tmp/tmp.db',
    >>>                 'index_kwargs': {
    >>>                     'index_type': 'HNSW',
    >>>                     'metric_type': 'COSINE'
    >>>                  }
    >>>             },
    >>>         },
    >>>     },
    >>> }
    >>> doc_fields = {
    >>>     'author': DocField(data_type=DataType.VARCHAR, max_size=128, default_value=' '),
    >>>     'public_year': DocField(data_type=DataType.INT32),
    >>> }
    >>> document2 = Document(dataset_path='your_doc_path', embed={"online": m, "local": m1}, store_conf=store_conf, doc_fields=doc_fields)
    """
    class _Manager(ModuleBase):
        def __init__(self, dataset_path: Optional[str], embed: Optional[Union[Callable, Dict[str, Callable]]] = None,
                     manager: Union[bool, str] = False, server: Union[bool, int] = False, name: Optional[str] = None,
                     launcher: Optional[Launcher] = None, store_conf: Optional[Dict] = None,
                     doc_fields: Optional[Dict[str, DocField]] = None, cloud: bool = False,
                     doc_files: Optional[List[str]] = None, processor: Optional[DocumentProcessor] = None):
            super().__init__()
            self._origin_path, self._doc_files, self._cloud = dataset_path, doc_files, cloud

            if dataset_path and not os.path.exists(dataset_path):
                defatult_path = os.path.join(lazyllm.config["data_path"], dataset_path)
                if os.path.exists(defatult_path):
                    dataset_path = defatult_path
            elif dataset_path:
                dataset_path = os.path.join(os.getcwd(), dataset_path)

            self._launcher: Launcher = launcher if launcher else lazyllm.launchers.remote(sync=False)
            self._dataset_path = dataset_path
            self._embed = self._get_embeds(embed)
            self._processor = processor

            self._dlm = None if (self._cloud or self._doc_files is not None) else DocListManager(
                dataset_path, name, enable_path_monitoring=False if manager else True)
            self._kbs = CallableDict({DocListManager.DEFAULT_GROUP_NAME: DocImpl(
                embed=self._embed, dlm=self._dlm, doc_files=doc_files, global_metadata_desc=doc_fields,
                store_conf=store_conf, processor=processor, algo_name=name)})

            if manager: self._manager = ServerModule(DocManager(self._dlm), launcher=self._launcher)
            if manager == 'ui': self._docweb = DocWebModule(doc_server=self._manager)
            if server: self._kbs = ServerModule(self._kbs, port=(None if isinstance(server, bool) else int(server)))
            self._global_metadata_desc = doc_fields

        @property
        def url(self):
            if hasattr(self, '_manager'): return self._manager._url
            return None

        @property
        @deprecated('Document.manager.url')
        def _url(self):
            return self.url

        @property
        def web_url(self):
            if hasattr(self, '_docweb'): return self._docweb.url
            return None

        def _get_embeds(self, embed):
            embeds = embed if isinstance(embed, dict) else {EMBED_DEFAULT_KEY: embed} if embed else {}
            for embed in embeds.values():
                if isinstance(embed, ModuleBase):
                    self._submodules.append(embed)
            return embeds

        def add_kb_group(self, name, doc_fields: Optional[Dict[str, DocField]] = None,
                         store_conf: Optional[Dict] = None,
                         embed: Optional[Union[Callable, Dict[str, Callable]]] = None):
            embed = self._get_embeds(embed) if embed else self._embed
            if isinstance(self._kbs, ServerModule):
                self._kbs._impl._m[name] = DocImpl(dlm=self._dlm, embed=embed, kb_group_name=name,
                                                   global_metadata_desc=doc_fields, store_conf=store_conf)
            else:
                self._kbs[name] = DocImpl(dlm=self._dlm, embed=self._embed, kb_group_name=name,
                                          global_metadata_desc=doc_fields, store_conf=store_conf)
            self._dlm.add_kb_group(name=name)

        def get_doc_by_kb_group(self, name):
            return self._kbs._impl._m[name] if isinstance(self._kbs, ServerModule) else self._kbs[name]

        def stop(self):
            if hasattr(self, '_docweb'):
                self._docweb.stop()
            self._launcher.cleanup()

        def __call__(self, *args, **kw):
            return self._kbs(*args, **kw)

    def __new__(cls, *args, **kw):
        if url := kw.pop('url', None):
            name = kw.pop('name', None)
            assert name, 'Document name must be provided with `url`'
            assert not args and not kw, 'Only `name` is supported with `url`'
            return UrlDocument(url, name)
        else:
            return super().__new__(cls)

    def __init__(self, dataset_path: Optional[str] = None, embed: Optional[Union[Callable, Dict[str, Callable]]] = None,
                 create_ui: bool = False, manager: Union[bool, str, "Document._Manager", DocumentProcessor] = False,
                 server: Union[bool, int] = False, name: Optional[str] = None, launcher: Optional[Launcher] = None,
                 doc_files: Optional[List[str]] = None, doc_fields: Dict[str, DocField] = None,
                 store_conf: Optional[Dict] = None):
        super().__init__()
        if create_ui:
            lazyllm.LOG.warning('`create_ui` for Document is deprecated, use `manager` instead')
            manager = create_ui
        if isinstance(dataset_path, (tuple, list)):
            doc_fields = dataset_path
            dataset_path = None
        if doc_files is not None:
            assert dataset_path is None and not manager, (
                'Manager and dataset_path are not supported for Document with temp-files')
            assert store_conf is None or store_conf['type'] == 'map', (
                'Only map store is supported for Document with temp-files')

        if isinstance(manager, Document._Manager):
            assert not server, 'Server infomation is already set to by manager'
            assert not launcher, 'Launcher infomation is already set to by manager'
            assert not manager._cloud, 'manager is not allowed to share in cloud mode'
            assert manager._doc_files is None, 'manager is not allowed to share with temp files'
            if dataset_path != manager._dataset_path and dataset_path != manager._origin_path:
                raise RuntimeError(f'Document path mismatch, expected `{manager._dataset_path}`'
                                   f'while received `{dataset_path}`')
            manager.add_kb_group(name=name, doc_fields=doc_fields, store_conf=store_conf, embed=embed)
            self._manager = manager
            self._curr_group = name
        else:
            if isinstance(manager, DocumentProcessor):
                processor, cloud = manager, True
                processor._impl.start()
                manager = False
                assert name, '`Name` of Document is necessary when using cloud service'
                assert store_conf['type'] != 'map', 'Cloud manager is not supported when using map store'
                assert not dataset_path, 'Cloud manager is not supported with local dataset path'
            else:
                cloud, processor = False, None
            self._manager = Document._Manager(dataset_path, embed, manager, server, name, launcher, store_conf,
                                              doc_fields, cloud=cloud, doc_files=doc_files, processor=processor)
            self._curr_group = DocListManager.DEFAULT_GROUP_NAME
        self._doc_to_db_processor: DocToDbProcessor = None

    def _list_all_files_in_dataset(self) -> List[str]:
        files_list = []
        for root, _, files in os.walk(self._manager._dataset_path):
            files = [os.path.join(root, file_path) for file_path in files]
            files_list.extend(files)
        return files_list

    def connect_sql_manager(
        self,
        sql_manager: SqlManager,
        schma: Optional[DocInfoSchema] = None,
        force_refresh: bool = True,
    ):
        def format_schema_to_dict(schema: DocInfoSchema):
            if schema is None:
                return None, None
            desc_dict = {ele["key"]: ele["desc"] for ele in schema}
            type_dict = {ele["key"]: ele["type"] for ele in schema}
            return desc_dict, type_dict

        def compare_schema(old_schema: DocInfoSchema, new_schema: DocInfoSchema):
            old_desc_dict, old_type_dict = format_schema_to_dict(old_schema)
            new_desc_dict, new_type_dict = format_schema_to_dict(new_schema)
            return old_desc_dict == new_desc_dict and old_type_dict == new_type_dict

        # 1. Check valid arguments
        if sql_manager.check_connection().status != DBStatus.SUCCESS:
            raise RuntimeError(f'Failed to connect to sql manager: {sql_manager._gen_conn_url()}')
        pre_doc_table_schema = None
        if self._doc_to_db_processor:
            pre_doc_table_schema = self._doc_to_db_processor.doc_info_schema
        assert pre_doc_table_schema or schma, "doc_table_schma must be given"

        schema_equal = compare_schema(pre_doc_table_schema, schma)
        assert (
            schema_equal or force_refresh is True
        ), "When changing doc_table_schema, force_refresh should be set to True"

        # 2. Init handler if needed
        need_init_processor = False
        if self._doc_to_db_processor is None:
            need_init_processor = True
        else:
            # avoid reinit for the same db
            if sql_manager != self._doc_to_db_processor.sql_manager:
                need_init_processor = True
        if need_init_processor:
            self._doc_to_db_processor = DocToDbProcessor(sql_manager)

        # 3. Reset doc_table_schema if needed
        if schma and not schema_equal:
            # This api call will clear existing db table "lazyllm_doc_elements"
            self._doc_to_db_processor._reset_doc_info_schema(schma)

    def get_sql_manager(self):
        if self._doc_to_db_processor is None:
            raise None
        return self._doc_to_db_processor.sql_manager

    def extract_db_schema(
        self, llm: Union[OnlineChatModule, TrainableModule], print_schema: bool = False
    ) -> DocInfoSchema:
        file_paths = self._list_all_files_in_dataset()
        schema = extract_db_schema_from_files(file_paths, llm)
        if print_schema:
            lazyllm.LOG.info(f"Extracted Schema:\n\t{schema}\n")
        return schema

    def update_database(self, llm: Union[OnlineChatModule, TrainableModule]):
        assert self._doc_to_db_processor, "Please call connect_db to init handler first"
        file_paths = self._list_all_files_in_dataset()
        info_dicts = self._doc_to_db_processor.extract_info_from_docs(llm, file_paths)
        self._doc_to_db_processor.export_info_to_db(info_dicts)

    @deprecated('Document(dataset_path, manager=doc.manager, name=xx, doc_fields=xx, store_conf=xx)')
    def create_kb_group(self, name: str, doc_fields: Optional[Dict[str, DocField]] = None,
                        store_conf: Optional[Dict] = None) -> "Document":
        self._manager.add_kb_group(name=name, doc_fields=doc_fields, store_conf=store_conf)
        doc = copy.copy(self)
        doc._curr_group = name
        return doc

    @property
    @deprecated('Document._manager')
    def _impls(self): return self._manager

    @property
    def _impl(self) -> DocImpl: return self._manager.get_doc_by_kb_group(self._curr_group)

    @property
    def manager(self): return self._manager._processor or self._manager

    def activate_group(self, group_name: str, embed_keys: Optional[Union[str, List[str]]] = None):
        if isinstance(embed_keys, str): embed_keys = [embed_keys]
        elif embed_keys is None: embed_keys = []
        self._impl.activate_group(group_name, embed_keys)

    def activate_groups(self, groups: Union[str, List[str]]):
        if isinstance(groups, str): groups = [groups]
        for group in groups:
            self.activate_group(group)

    @DynamicDescriptor
    def create_node_group(self, name: str = None, *, transform: Callable, parent: str = LAZY_ROOT_NAME,
                          trans_node: bool = None, num_workers: int = 0, display_name: str = None,
                          group_type: NodeGroupType = NodeGroupType.CHUNK, **kwargs) -> None:
        """
创建一个由指定规则生成的 node group。

Args:
    name (str): node group 的名称。
    transform (Callable): 将 node 转换成 node group 的转换规则，函数原型是 `(DocNode, group_name, **kwargs) -> List[DocNode]`。目前内置的有 [SentenceSplitter][lazyllm.tools.SentenceSplitter]。用户也可以自定义转换规则。
    trans_node (bool): 决定了transform的输入和输出是 `DocNode` 还是 `str` ，默认为None。只有在 `transform` 为 `Callable` 时才可以设置为true。
    num_workers (int): Transform时所用的新线程数量，默认为0
    parent (str): 需要进一步转换的节点。转换之后得到的一系列新的节点将会作为该父节点的子节点。如果不指定则从根节点开始转换。
    kwargs: 和具体实现相关的参数。


Examples:

    >>> import lazyllm
    >>> from lazyllm.tools import Document, SentenceSplitter
    >>> m = lazyllm.OnlineEmbeddingModule(source="glm")
    >>> documents = Document(dataset_path='your_doc_path', embed=m, manager=False)
    >>> documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
    """
        if isinstance(self, type):
            DocImpl.create_global_node_group(name, transform=transform, parent=parent, trans_node=trans_node,
                                             num_workers=num_workers, display_name=display_name,
                                             group_type=group_type, **kwargs)
        else:
            self._impl.create_node_group(name, transform=transform, parent=parent, trans_node=trans_node,
                                         num_workers=num_workers, display_name=display_name, group_type=group_type,
                                         **kwargs)

    @DynamicDescriptor
    def add_reader(self, pattern: str, func: Optional[Callable] = None):
        """
用于实例指定文件读取器，作用范围仅对注册的 Document 对象可见。注册的文件读取器必须是 Callable 对象。只能通过函数调用的方式进行注册。并且通过实例注册的文件读取器的优先级高于通过类注册的文件读取器，并且实例和类注册的文件读取器的优先级高于系统默认的文件读取器。即优先级的顺序是：实例文件读取器 > 类文件读取器 > 系统默认文件读取器。

Args:
    pattern (str): 文件读取器适用的匹配规则
    func (Callable): 文件读取器，必须是Callable的对象


Examples:

    >>> from lazyllm.tools.rag import Document, DocNode
    >>> from lazyllm.tools.rag.readers import ReaderBase
    >>> class YmlReader(ReaderBase):
    ...     def _load_data(self, file, fs=None):
    ...         try:
    ...             import yaml
    ...         except ImportError:
    ...             raise ImportError("yaml is required to read YAML file: `pip install pyyaml`")
    ...         with open(file, 'r') as f:
    ...             data = yaml.safe_load(f)
    ...         print("Call the class YmlReader.")
    ...         return [DocNode(text=data)]
    ...
    >>> def processYml(file):
    ...     with open(file, 'r') as f:
    ...         data = f.read()
    ...     print("Call the function processYml.")
    ...     return [DocNode(text=data)]
    ...
    >>> doc1 = Document(dataset_path="your_files_path", create_ui=False)
    >>> doc2 = Document(dataset_path="your_files_path", create_ui=False)
    >>> doc1.add_reader("**/*.yml", YmlReader)
    >>> print(doc1._impl._local_file_reader)
    {'**/*.yml': <class '__main__.YmlReader'>}
    >>> print(doc2._impl._local_file_reader)
    {}
    >>> files = ["your_yml_files"]
    >>> Document.register_global_reader("**/*.yml", processYml)
    >>> doc1._impl._reader.load_data(input_files=files)
    Call the class YmlReader.
    >>> doc2._impl._reader.load_data(input_files=files)
    Call the function processYml.
    """
        if isinstance(self, type):
            return DocImpl.register_global_reader(pattern=pattern, func=func)
        else:
            self._impl.add_reader(pattern, func)

    @classmethod
    def register_global_reader(cls, pattern: str, func: Optional[Callable] = None):
        """
用于指定文件读取器，作用范围对于所有的 Document 对象都可见。注册的文件读取器必须是 Callable 对象。可以使用装饰器的方式进行注册，也可以通过函数调用的方式进行注册。

Args:
    pattern (str): 文件读取器适用的匹配规则
    func (Callable): 文件读取器，必须是Callable的对象


Examples:

    >>> from lazyllm.tools.rag import Document, DocNode
    >>> @Document.register_global_reader("**/*.yml")
    >>> def processYml(file):
    ...     with open(file, 'r') as f:
    ...         data = f.read()
    ...     return [DocNode(text=data)]
    ...
    >>> doc1 = Document(dataset_path="your_files_path", create_ui=False)
    >>> doc2 = Document(dataset_path="your_files_path", create_ui=False)
    >>> files = ["your_yml_files"]
    >>> docs1 = doc1._impl._reader.load_data(input_files=files)
    >>> docs2 = doc2._impl._reader.load_data(input_files=files)
    >>> print(docs1[0].text == docs2[0].text)
    # True
    """
        return cls.add_reader(pattern, func)

    def get_store(self):
        return StorePlaceholder()

    def get_embed(self):
        return EmbedPlaceholder()

    def register_index(self, index_type: str, index_cls: IndexBase, *args, **kwargs) -> None:
        self._impl.register_index(index_type, index_cls, *args, **kwargs)

    def _forward(self, func_name: str, *args, **kw):
        return self._manager(self._curr_group, func_name, *args, **kw)

    def find_parent(self, target) -> Callable:
        """
查找指定节点的父节点。

Args:
    group (str): 需要查找的节点名称


Examples:

    >>> import lazyllm
    >>> from lazyllm.tools import Document, SentenceSplitter
    >>> m = lazyllm.OnlineEmbeddingModule(source="glm")
    >>> documents = Document(dataset_path='your_doc_path', embed=m, manager=False)
    >>> documents.create_node_group(name="parent", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
    >>> documents.create_node_group(name="children", transform=SentenceSplitter, parent="parent", chunk_size=1024, chunk_overlap=100)
    >>> documents.find_parent('children')
    """
        return functools.partial(self._forward, 'find_parent', group=target)

    def find_children(self, target) -> Callable:
        """
查找指定节点的子节点。

Args:
    group (str): 需要查找的名称


Examples:

    >>> import lazyllm
    >>> from lazyllm.tools import Document, SentenceSplitter
    >>> m = lazyllm.OnlineEmbeddingModule(source="glm")
    >>> documents = Document(dataset_path='your_doc_path', embed=m, manager=False)
    >>> documents.create_node_group(name="parent", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
    >>> documents.create_node_group(name="children", transform=SentenceSplitter, parent="parent", chunk_size=1024, chunk_overlap=100)
    >>> documents.find_children('parent')
    """
        return functools.partial(self._forward, 'find_children', group=target)

    def find(self, target) -> Callable:
        return functools.partial(self._forward, 'find', group=target)

    def forward(self, *args, **kw) -> List[DocNode]:
        return self._forward('retrieve', *args, **kw)

    def clear_cache(self, group_names: Optional[List[str]]) -> None:
        return self._forward('clear_cache', group_names)

    def _get_post_process_tasks(self):
        return lazyllm.pipeline(lambda *a: self._forward('_lazy_init'))

    def __repr__(self):
        return lazyllm.make_repr("Module", "Document", manager=hasattr(self._manager, '_manager'),
                                 server=isinstance(self._manager._kbs, ServerModule))

class UrlDocument(ModuleBase):
    def __init__(self, url: str, name: str):
        super().__init__()
        self._missing_keys = set(dir(Document)) - set(dir(UrlDocument))
        self._manager = lazyllm.UrlModule(url=url)
        self._curr_group = name

    def _forward(self, func_name: str, *args, **kwargs):
        args = (self._curr_group, func_name, *args)
        args, kwargs = lazyllm.dump_obj(args), lazyllm.dump_obj(kwargs)
        return self._manager("__call__", args, kwargs)

    def find(self, target) -> Callable:
        return functools.partial(self._forward, 'find', group=target)

    def forward(self, *args, **kw):
        return self._forward('retrieve', *args, **kw)

    @functools.lru_cache
    def active_node_groups(self):
        return self._forward('active_node_groups')

    def __getattr__(self, name):
        if name in self._missing_keys:
            raise RuntimeError(f'Document generated with url and name has no attribute `{name}`')
