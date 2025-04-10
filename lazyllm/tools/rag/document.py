import os

from typing import Callable, Optional, Dict, Union, List
import lazyllm
from lazyllm import ModuleBase, ServerModule, DynamicDescriptor, deprecated
from lazyllm.launcher import LazyLLMLaunchersBase as Launcher

from .doc_manager import DocManager
from lazyllm.tools.sql.sql_manager import SqlManager, DBStatus
from .doc_kws_tool import DocKWSManager, DocKwDesc
from .doc_impl import DocImpl, StorePlaceholder, EmbedPlaceholder
from .doc_node import DocNode
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

class Document(ModuleBase):
    class _Manager(ModuleBase):
        def __init__(self, dataset_path: str, embed: Optional[Union[Callable, Dict[str, Callable]]] = None,
                     manager: Union[bool, str] = False, server: bool = False, name: Optional[str] = None,
                     launcher: Optional[Launcher] = None, store_conf: Optional[Dict] = None,
                     doc_fields: Optional[Dict[str, DocField]] = None):
            super().__init__()
            self._origin_path = dataset_path
            if not os.path.exists(dataset_path):
                defatult_path = os.path.join(lazyllm.config["data_path"], dataset_path)
                if os.path.exists(defatult_path):
                    dataset_path = defatult_path
            else:
                dataset_path = os.path.join(os.getcwd(), dataset_path)
            self._launcher: Launcher = launcher if launcher else lazyllm.launchers.remote(sync=False)
            self._dataset_path = dataset_path
            self._embed = self._get_embeds(embed)
            self.name = name
            self._dlm = DocListManager(dataset_path, name, enable_path_monitoring=False if manager else True)
            self._kbs = CallableDict({DocListManager.DEFAULT_GROUP_NAME:
                                      DocImpl(embed=self._embed, dlm=self._dlm,
                                              global_metadata_desc=doc_fields,
                                              store_conf=store_conf)})
            if manager: self._manager = ServerModule(DocManager(self._dlm))
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

    def __init__(self, dataset_path: str, embed: Optional[Union[Callable, Dict[str, Callable]]] = None,
                 create_ui: bool = False, manager: Union[bool, str] = False, server: bool = False,
                 name: Optional[str] = None, launcher: Optional[Launcher] = None,
                 doc_fields: Dict[str, DocField] = None, store_conf: Optional[Dict] = None):
        super().__init__()
        if create_ui:
            lazyllm.LOG.warning('`create_ui` for Document is deprecated, use `manager` instead')
        if isinstance(manager, Document._Manager):
            assert not server, 'Server infomation is already set to by manager'
            assert not launcher, 'Launcher infomation is already set to by manager'
            if dataset_path != manager._dataset_path and dataset_path != manager._origin_path:
                raise RuntimeError(f'Document path mismatch, expected `{manager._dataset_path}`'
                                   f'while received `{dataset_path}`')
            manager.add_kb_group(name=name, doc_fields=doc_fields, store_conf=store_conf, embed=embed)
            self._manager = manager
            self._curr_group = name
        else:
            self._manager = Document._Manager(dataset_path, embed, create_ui or manager, server, name,
                                              launcher, store_conf, doc_fields)
            self._curr_group = DocListManager.DEFAULT_GROUP_NAME
        self._doc_kws_manager = None

    def _list_all_files_in_dataset(self) -> List[str]:
        files_list = []
        for root, _, files in os.walk(self._manager._dataset_path):
            files = [os.path.join(root, file_path) for file_path in files]
            files_list.extend(files)
        return files_list

    def kws_tool_init(self, llm: callable, sql_manager: SqlManager, kws_desc: List[DocKwDesc] = []):
        self._doc_kws_manager = DocKWSManager(llm=llm, sql_manager=sql_manager)
        if not kws_desc:
            files_list = self._list_all_files_in_dataset()
            if len(files_list) == 0:
                lazyllm.LOG.warning(f"Failed to find any files in {self._manager._dataset_path}")
            else:
                self._doc_kws_manager.analyse_and_init_kws_desc(files_list)
        else:
            self._doc_kws_manager.set_kws_desc(kws_desc)

    def kws_tool_reset_schema(self, kws_desc: List[DocKwDesc]):
        # Alert, set_kws_table_schema will drop old result in db
        assert self._doc_kws_manager, "Please call prepare_kws_table_schema first"
        self._doc_kws_manager.set_kws_desc(kws_desc)

    def kws_tool_extract_to_db(self):
        assert self._doc_kws_manager, "Please call prepare_kws_table_schema first"
        assert self._doc_kws_manager._kws_desc, "Please call prepare_kws_table_schema or reset_kws_table_schema first"
        files_list = self._list_all_files_in_dataset()

        db_result = self._doc_kws_manager.extract_and_record_kws(files_list)
        return db_result.status == DBStatus.SUCCESS

    @deprecated('Document(dataset_path, manager=doc.manager, name=xx, doc_fields=xx, store_conf=xx)')
    def create_kb_group(self, name: str, doc_fields: Optional[Dict[str, DocField]] = None,
                        store_conf: Optional[Dict] = None) -> "Document":
        self._manager.add_kb_group(name=name, doc_fields=doc_fields, store_conf=store_conf)
        doc = copy.copy(self)
        doc._curr_group = name
        return doc

    @property
    @deprecated('Document._manager')
    def _impls(self):
        return self._manager

    @property
    def _impl(self):
        return self._manager.get_doc_by_kb_group(self._curr_group)

    @property
    def manager(self):
        return self._manager

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
        return functools.partial(Document.find_parent, group=target)

    def find_children(self, target) -> Callable:
        # TODO: Currently, when a DocNode is returned from the server, it will carry all parent nodes and child nodes.
        # So the query of parent and child nodes can be performed locally, and there is no need to search the
        # document service through the server for the time being. When this item is optimized, the code will become:
        # return functools.partial(self._forward, 'find_children', group=target)
        return functools.partial(Document.find_children, group=target)

    def forward(self, *args, **kw) -> List[DocNode]:
        return self._forward('retrieve', *args, **kw)

    def __repr__(self):
        return lazyllm.make_repr("Module", "Document", manager=hasattr(self._manager, '_manager'),
                                 server=isinstance(self._manager._kbs, ServerModule))
