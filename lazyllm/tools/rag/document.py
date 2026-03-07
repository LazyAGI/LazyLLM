import os
from typing import Callable, Optional, Dict, Union, List, Type, Set
from functools import cached_property
from pydantic import BaseModel
import lazyllm
from lazyllm import ModuleBase, ServerModule, DynamicDescriptor, deprecated, OnlineChatModule, TrainableModule
from lazyllm.module import LLMBase
from lazyllm.launcher import LazyLLMLaunchersBase as Launcher
from lazyllm.tools.sql.sql_manager import SqlManager, DBStatus
from lazyllm.common.bind import _MetaBind

from .doc_manager import DocManager
from .doc_impl import DocImpl, StorePlaceholder, EmbedPlaceholder, BuiltinGroups, DocumentProcessor, NodeGroupType
from .doc_node import DocNode
from .doc_to_db import SchemaExtractor
from lazyllm.tools.rag.doc_to_db.model import SchemaSetInfo, Table_ALGO_KB_SCHEMA
from .store import LAZY_ROOT_NAME, EMBED_DEFAULT_KEY
from .store.store_base import DEFAULT_KB_ID
from .index_base import IndexBase
from .utils import DocListManager, ensure_call_endpoint, _get_default_db_config
from .global_metadata import GlobalMetadataDesc as DocField
from .web import DocWebModule
import copy
import functools
import weakref


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
                     manager: Union[bool, str] = False, server: Union[bool, int] = False, name: Optional[str] = None,
                     launcher: Optional[Launcher] = None, store_conf: Optional[Dict] = None,
                     doc_fields: Optional[Dict[str, DocField]] = None, cloud: bool = False,
                     doc_files: Optional[List[str]] = None, processor: Optional[DocumentProcessor] = None,
                     display_name: Optional[str] = '', description: Optional[str] = 'algorithm description',
                     schema_extractor: Optional[Union[LLMBase, SchemaExtractor]] = None):
            super().__init__()
            self._origin_path, self._doc_files, self._cloud = dataset_path, doc_files, cloud

            if dataset_path and not os.path.exists(dataset_path):
                defatult_path = os.path.join(lazyllm.config['data_path'], dataset_path)
                if os.path.exists(defatult_path):
                    dataset_path = defatult_path
            elif dataset_path:
                dataset_path = os.path.join(os.getcwd(), dataset_path)

            self._launcher: Launcher = launcher if launcher else lazyllm.launchers.remote(sync=False)
            self._dataset_path = dataset_path
            self._embed = self._get_embeds(embed)
            self._processor = processor
            name = name or DocListManager.DEFAULT_GROUP_NAME
            if not display_name: display_name = name

            self._dlm = None if (self._cloud or self._doc_files is not None) else DocListManager(
                dataset_path, name, enable_path_monitoring=False if manager else True)
            self._kbs = CallableDict({name: DocImpl(
                embed=self._embed, dlm=self._dlm, doc_files=doc_files, global_metadata_desc=doc_fields,
                store=store_conf, processor=processor, algo_name=name, display_name=display_name,
                description=description, schema_extractor=schema_extractor)})

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
                                                   global_metadata_desc=doc_fields, store=store_conf)
            else:
                self._kbs[name] = DocImpl(dlm=self._dlm, embed=self._embed, kb_group_name=name,
                                          global_metadata_desc=doc_fields, store=store_conf)
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
            if args or kw:
                raise TypeError(
                    f"When 'url' is provided, only 'name' is allowed. "
                    f'Got args={args}, extra kwargs={kw}'
                )
            return UrlDocument(url, name)
        else:
            return super().__new__(cls)

    def __init__(self, dataset_path: Optional[str] = None, embed: Optional[Union[Callable, Dict[str, Callable]]] = None,
                 create_ui: bool = False, manager: Union[bool, str, 'Document._Manager', DocumentProcessor] = False,
                 server: Union[bool, int] = False, name: Optional[str] = None, launcher: Optional[Launcher] = None,
                 doc_files: Optional[List[str]] = None, doc_fields: Dict[str, DocField] = None,
                 store_conf: Optional[Dict] = None, display_name: Optional[str] = '',
                 description: Optional[str] = 'algorithm description',
                 schema_extractor: Optional[Union[LLMBase, SchemaExtractor]] = None):
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

        name = name or DocListManager.DEFAULT_GROUP_NAME
        if schema_extractor is not None and not isinstance(schema_extractor, SchemaExtractor):
            if isinstance(schema_extractor, LLMBase):
                metadata_store_config = None
                if isinstance(store_conf, dict):
                    metadata_store_config = store_conf.get('metadata_store')
                metadata_store_config = metadata_store_config or _get_default_db_config(
                    db_name=f'{name}_metadata'
                )
                schema_extractor = SchemaExtractor(db_config=metadata_store_config, llm=schema_extractor)
            else:
                raise ValueError(f'Invalid type for schema extractor: {type(schema_extractor)}')
        self._schema_extractor: SchemaExtractor = schema_extractor

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
                processor.start()
                manager = False
                assert name, '`Name` of Document is necessary when using cloud service'
                assert store_conf.get('type') != 'map', 'Cloud manager is not supported when using map store'
                assert not dataset_path, 'Cloud manager is not supported with local dataset path'
            else:
                cloud, processor = False, None
            self._manager = Document._Manager(dataset_path, embed, manager, server, name, launcher, store_conf,
                                              doc_fields, cloud=cloud, doc_files=doc_files, processor=processor,
                                              display_name=display_name, description=description,
                                              schema_extractor=self._schema_extractor)
            self._curr_group = name
        self._graph_document: weakref.ref = None

    @staticmethod
    def list_all_files_in_directory(
        dataset_path: str, skip_hidden_path: bool = True, recursive: bool = True
    ) -> List[str]:
        files_list = []

        if not os.path.exists(dataset_path):
            return files_list

        if not os.path.isdir(dataset_path):
            return [dataset_path] if os.path.isfile(dataset_path) else files_list

        if recursive:
            for root, dirs, files in os.walk(os.path.abspath(dataset_path)):
                # Skip hidden directories
                if skip_hidden_path:
                    path_parts = root.split(os.sep)
                    if any(part.startswith('.') for part in path_parts if part):
                        continue
                    # Filter out hidden directories
                    dirs[:] = [d for d in dirs if not d.startswith('.')]

                # Skip hidden files
                if skip_hidden_path:
                    files = [file_path for file_path in files if not file_path.startswith('.')]

                files = [os.path.join(root, file_path) for file_path in files]
                files_list.extend(files)
        else:
            items = os.listdir(dataset_path)
            for item in items:
                item_path = os.path.join(dataset_path, item)
                # Skip hidden files/directories
                if skip_hidden_path and item.startswith('.'):
                    continue
                # Only add files, not directories
                if os.path.isfile(item_path):
                    files_list.append(item_path)

        return files_list

    def _list_all_files_in_dataset(self, skip_hidden_path: bool = True) -> List[str]:
        return self.list_all_files_in_directory(self._manager._dataset_path, skip_hidden_path)

    @property
    def url(self):
        assert isinstance(self._manager._kbs, ServerModule), 'Document is not a service, please set `manager` to `True`'
        return self._manager._kbs._url

    def _resolve_schema_extractor(self, sql_manager: SqlManager) -> SchemaExtractor:
        if self._schema_extractor is None:
            raise ValueError('schema_extractor is required to connect sql manager')
        if not isinstance(self._schema_extractor, SchemaExtractor):
            raise ValueError(f'Invalid type for schema extractor: {type(self._schema_extractor)}')
        if sql_manager == self._schema_extractor.sql_manager:
            return self._schema_extractor
        return SchemaExtractor(sql_manager=sql_manager, llm=self._schema_extractor._llm)

    @staticmethod
    def _compare_schema_rows(rows: Union[List, object, None], schma: BaseModel,
                             extractor: SchemaExtractor) -> bool:
        if schma is None:
            return False
        sid = extractor.register_schema_set(schma)
        if not rows:
            return False
        if not isinstance(rows, (list, tuple, set)):
            rows = [rows]
        return any(getattr(row, 'schema_set_id', row) == sid for row in rows)

    def _get_schema_bind_rows(self, extractor: SchemaExtractor) -> List:
        mgr = extractor.sql_manager
        bind_cls = mgr.get_table_orm_class(Table_ALGO_KB_SCHEMA['name'])
        if bind_cls is None:
            return []
        with mgr.get_session() as s:
            return s.query(bind_cls).filter_by(algo_id=self._impl.algo_name).all()

    def connect_sql_manager(
        self, sql_manager: SqlManager, schma: Optional[BaseModel] = None,
        force_refresh: bool = True,
    ):
        # 1. Check valid arguments
        if sql_manager.check_connection().status != DBStatus.SUCCESS:
            raise RuntimeError(f'Failed to connect to sql manager: {sql_manager._gen_conn_url()}')

        extractor = self._resolve_schema_extractor(sql_manager)
        rows = self._get_schema_bind_rows(extractor)
        assert rows or schma, 'doc_table_schma must be given'

        schema_equal = self._compare_schema_rows(rows, schma, extractor)
        assert (
            schema_equal or force_refresh is True
        ), 'When changing doc_table_schema, force_refresh should be set to True'

        # 2. Init handler if needed
        if extractor is not self._schema_extractor:
            # reuse the extractor instance used for schema comparison/registration
            self._schema_extractor = extractor
            self._impl._schema_extractor = extractor

        # 3. Reset doc_table_schema if needed
        if schma and not schema_equal:
            # This api call will clear existing db table 'lazyllm_doc_elements'
            schema_set_id = self._schema_extractor.register_schema_set(schma)
            self._schema_extractor.register_schema_set_to_kb(
                algo_id=self._impl._algo_name, schema_set_id=schema_set_id, force_refresh=True)

    def get_sql_manager(self):
        if self._schema_extractor is None:
            raise ValueError('Please call connect_sql_manager to init handler first')
        return self._schema_extractor.sql_manager

    def extract_db_schema(
        self, llm: Union[OnlineChatModule, TrainableModule], print_schema: bool = False
    ) -> SchemaSetInfo:

        schema = self._forward('_analyze_schema_by_llm')
        if print_schema:
            lazyllm.LOG.info(f'Extracted Schema:\n\t{schema}\n')
        return schema

    def update_database(self, llm: Union[OnlineChatModule, TrainableModule]):
        assert self._doc_to_db_processor, 'Please call connect_db to init handler first'
        file_paths = self._list_all_files_in_dataset()
        info_dicts = self._doc_to_db_processor.extract_info_from_docs(llm, file_paths)
        self._doc_to_db_processor.export_info_to_db(info_dicts)

    @deprecated('Document(dataset_path, manager=doc.manager, name=xx, doc_fields=xx, store_conf=xx)')
    def create_kb_group(self, name: str, doc_fields: Optional[Dict[str, DocField]] = None,
                        store_conf: Optional[Dict] = None) -> 'Document':
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

    def activate_group(self, group_name: str, embed_keys: Optional[Union[str, List[str]]] = None,
                       enable_embed: bool = True):
        if embed_keys and not enable_embed:
            raise ValueError('`enable_embed` must be set to True when `embed_keys` is provided')
        # if embed_keys is None, use default embed keys
        if (enable_embed and not embed_keys) and self._manager._embed:
            embed_keys = self._manager._embed.keys()
        if isinstance(embed_keys, str): embed_keys = [embed_keys]
        self._impl.activate_group(group_name, embed_keys, enable_embed)

    def activate_groups(self, groups: Union[str, List[str]], **kwargs):
        if isinstance(groups, str): groups = [groups]
        for group in groups:
            self.activate_group(group, **kwargs)

    @DynamicDescriptor
    def create_node_group(self, name: str = None, *, transform: Callable, parent: str = LAZY_ROOT_NAME,
                          trans_node: bool = None, num_workers: int = 0, display_name: str = None,
                          ref: str = None, group_type: NodeGroupType = NodeGroupType.CHUNK, **kwargs) -> None:
        assert ref is None or parent != ref, 'parent and ref must be different'
        if isinstance(self, type):
            DocImpl.create_global_node_group(name, transform=transform, parent=parent, trans_node=trans_node,
                                             num_workers=num_workers, display_name=display_name,
                                             group_type=group_type, ref=ref, **kwargs)
        else:
            self._impl.create_node_group(name, transform=transform, parent=parent, trans_node=trans_node,
                                         num_workers=num_workers, display_name=display_name, group_type=group_type,
                                         ref=ref, **kwargs)

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
        return functools.partial(self._forward, 'find_parent', group=target)

    def find_children(self, target) -> Callable:
        return functools.partial(self._forward, 'find_children', group=target)

    def find(self, target) -> Callable:
        return functools.partial(self._forward, 'find', group=target)

    def forward(self, *args, **kw) -> List[DocNode]:
        return self._forward('retrieve', *args, **kw)

    def clear_cache(self, group_names: Optional[List[str]] = None) -> None:
        return self._forward('clear_cache', group_names)

    def drop_algorithm(self):
        return self._forward('drop_algorithm')

    def analyze_schema_by_llm(self, kb_id: Optional[str] = None, doc_ids: Optional[List[str]] = None):
        return self._forward('_analyze_schema_by_llm', kb_id, doc_ids)

    def register_schema_set(self, schema_set: Type[BaseModel], kb_id: Optional[str] = DEFAULT_KB_ID,
                            force_refresh: bool = False) -> str:
        return self._forward('_register_schema_set', schema_set, kb_id, force_refresh)

    def get_nodes(self, uids: Optional[List[str]] = None, doc_ids: Optional[Set] = None,
                  group: Optional[str] = None, kb_id: Optional[str] = None, numbers: Optional[Set] = None
                  ) -> List[DocNode]:
        return self._forward('_get_nodes', uids, doc_ids, group, kb_id, numbers)

    def get_window_nodes(self, node: DocNode, span: tuple[int, int] = (-5, 5),
                         merge: bool = False) -> Union[List[DocNode], DocNode]:
        return self._forward('_get_window_nodes', node, span, merge)

    def _get_post_process_tasks(self):
        return lazyllm.pipeline(lambda *a: self._forward('_lazy_init'))

    def __repr__(self):
        return lazyllm.make_repr('Module', 'Document', manager=hasattr(self._manager, '_manager'),
                                 server=isinstance(self._manager._kbs, ServerModule))

class UrlDocument(ModuleBase):
    def __init__(self, url: str, name: str = None):
        super().__init__()
        self._missing_keys = set(dir(Document)) - set(dir(UrlDocument))
        self._manager = lazyllm.UrlModule(url=ensure_call_endpoint(url))
        self._curr_group = name or DocListManager.DEFAULT_GROUP_NAME

    def _forward(self, func_name: str, *args, **kwargs):
        args = (self._curr_group, func_name, *args)
        return self._manager._call('__call__', *args, **kwargs)

    def find(self, target) -> Callable:
        return functools.partial(self._forward, 'find', group=target)

    def forward(self, *args, **kw):
        return self._forward('retrieve', *args, **kw)

    def get_nodes(self, uids: Optional[List[str]] = None, doc_ids: Optional[Set] = None,
                  group: Optional[str] = None, kb_id: Optional[str] = None, numbers: Optional[Set] = None
                  ) -> List[DocNode]:
        return self._forward('_get_nodes', uids, doc_ids, group, kb_id, numbers)

    def get_window_nodes(self, node: DocNode, span: tuple[int, int] = (-5, 5),
                         merge: bool = False) -> Union[List[DocNode], DocNode]:
        return self._forward('_get_window_nodes', node, span, merge)

    @cached_property
    def active_node_groups(self):
        return self._forward('active_node_groups')

    def __getattr__(self, name):
        if name in self._missing_keys:
            raise RuntimeError(f'Document generated with url and name has no attribute `{name}`')
