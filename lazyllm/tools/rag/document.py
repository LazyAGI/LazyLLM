import os

from typing import Callable, Optional, Dict, Union, List
import lazyllm
from lazyllm import ModuleBase, ServerModule, DynamicDescriptor, deprecated, OnlineChatModule, TrainableModule
from lazyllm.launcher import LazyLLMLaunchersBase as Launcher
from lazyllm.tools.sql.sql_manager import SqlManager, DBStatus
from lazyllm.common.bind import _MetaBind

from .doc_manager import DocManager
from .doc_impl import DocImpl, StorePlaceholder, EmbedPlaceholder, BuiltinGroups, DocumentProcessor
from .doc_node import DocNode
from .doc_to_db import DocInfoSchema, DocToDbProcessor, extract_db_schema_from_files
from .index_base import IndexBase
from .store_base import LAZY_ROOT_NAME, EMBED_DEFAULT_KEY
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
    class _Manager(ModuleBase):
        def __init__(self, dataset_path: Optional[str], embed: Optional[Union[Callable, Dict[str, Callable]]] = None,
                     manager: Union[bool, str] = False, server: bool = False, name: Optional[str] = None,
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
            if server: self._kbs = ServerModule(self._kbs)
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
                 server: bool = False, name: Optional[str] = None, launcher: Optional[Launcher] = None,
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
                          trans_node: bool = None, num_workers: int = 0, **kwargs) -> None:
        if isinstance(self, type):
            DocImpl.create_global_node_group(name, transform=transform, parent=parent, trans_node=trans_node,
                                             num_workers=num_workers, **kwargs)
        else:
            self._impl.create_node_group(name, transform=transform, parent=parent, trans_node=trans_node,
                                         num_workers=num_workers, **kwargs)

    @DynamicDescriptor
    def add_reader(self, pattern: str, func: Optional[Callable] = None):
        if isinstance(self, type):
            return DocImpl.register_global_reader(pattern=pattern, func=func)
        else:
            self._impl.add_reader(pattern, func)

    @classmethod
    def register_global_reader(cls, pattern: str, func: Optional[Callable] = None):
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
        # TODO: Currently, when a DocNode is returned from the server, it will carry all parent nodes and child nodes.
        # So the query of parent and child nodes can be performed locally, and there is no need to search the
        # document service through the server for the time being. When this item is optimized, the code will become:
        # return functools.partial(self._forward, 'find_parent', group=target)
        return functools.partial(DocImpl.find_parent, group=target)

    def find_children(self, target) -> Callable:
        # TODO: Currently, when a DocNode is returned from the server, it will carry all parent nodes and child nodes.
        # So the query of parent and child nodes can be performed locally, and there is no need to search the
        # document service through the server for the time being. When this item is optimized, the code will become:
        # return functools.partial(self._forward, 'find_children', group=target)
        return functools.partial(DocImpl.find_children, group=target)

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

    def _forward(self, func_name: str, *args, **kw):
        return self._manager(self._curr_group, func_name, *args, **kw)

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
