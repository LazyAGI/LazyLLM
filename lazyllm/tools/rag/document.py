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
    """Initialize a document module with an optional user interface.

This constructor initializes a document module that can have an optional user interface. If the user interface is enabled, it also provides a UI to manage the document operation interface and offers a web page for user interface interaction.

Args:
    dataset_path (str): The path to the dataset directory. This directory should contain the documents to be managed by the document module.
    embed (Optional[Union[Callable, Dict[str, Callable]]]): The object used to generate document embeddings. If you need to generate multiple embeddings for the text, you need to specify multiple embedding models in a dictionary format. The key identifies the name corresponding to the embedding, and the value is the corresponding embedding model.
    manager (bool, optional): A flag indicating whether to create a user interface for the document module. Defaults to False.
    launcher (optional): An object or function responsible for launching the server module. If not provided, the default asynchronous launcher from `lazyllm.launchers` is used (`sync=False`).
    store_conf (optional): Configure which storage backend and index backend to use.
    doc_fields (optional): Configure the fields that need to be stored and retrieved along with their corresponding types (currently only used by the Milvus backend).


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
Generate a node group produced by the specified rule.

Args:
    name (str): The name of the node group.
    transform (Callable): The transformation rule that converts a node into a node group. The function prototype is `(DocNode, group_name, **kwargs) -> List[DocNode]`. Currently built-in options include [SentenceSplitter][lazyllm.tools.SentenceSplitter], and users can define their own transformation rules.
    trans_node (bool): Determines whether the input and output of transform are `DocNode` or `str`, default is None. Can only be set to true when `transform` is `Callable`.
    num_workers (int): number of new threads used for transform. default: 0
    parent (str): The node that needs further transformation. The series of new nodes obtained after transformation will be child nodes of this parent node. If not specified, the transformation starts from the root node.
    kwargs: Parameters related to the specific implementation.


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
Used to specify the file reader for an instance. The scope of action is visible only to the registered Document object. The registered file reader must be a Callable object. It can only be registered by calling a function. The priority of the file reader registered by the instance is higher than that of the file reader registered by the class, and the priority of the file reader registered by the instance and class is higher than the system default file reader. That is, the order of priority is: instance file reader > class file reader > system default file reader.

Args:
    pattern (str): Matching rules applied by the file reader.
    func (Callable): File reader, must be a Callable object.


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
Used to specify a file reader, which is visible to all Document objects. The registered file reader must be a Callable object. It can be registered using a decorator or by a function call.

Args:
    pattern (str): Matching rules applied by the file reader.
    func (Callable): File reader, must be a Callable object.


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
Find the parent node of the specified node.

Args:
    group (str): The name of the node for which to find the parent.


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
Find the child nodes of the specified node.

Args:
    group (str): The name of the node for which to find the children.


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
