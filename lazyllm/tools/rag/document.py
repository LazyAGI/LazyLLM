import copy
import functools
import os
import warnings
import weakref
from functools import cached_property
from typing import Callable, Optional, Dict, Union, List, Type, Set, Tuple

from pydantic import BaseModel
import lazyllm
from lazyllm import ModuleBase, ServerModule, DynamicDescriptor, deprecated, OnlineChatModule, TrainableModule
from lazyllm.module import LLMBase
from lazyllm.launcher import LazyLLMLaunchersBase as Launcher
from lazyllm.tools.sql.sql_manager import SqlManager, DBStatus
from lazyllm.common.bind import _MetaBind

from ._store_config import is_local_map_store, is_persistent_store, iter_embedded_store_endpoints
from .doc_impl import DocImpl, StorePlaceholder, EmbedPlaceholder, BuiltinGroups, DocumentProcessor, NodeGroupType
from .doc_node import DocNode
from .doc_to_db import DocInfoSchema, DocToDbProcessor, extract_db_schema_from_files, SchemaExtractor
from .store import LAZY_ROOT_NAME, EMBED_DEFAULT_KEY
from .store.store_base import DEFAULT_KB_ID
from .index_base import IndexBase
from .utils import RAG_DEFAULT_GROUP_NAME, ensure_call_endpoint
from .global_metadata import GlobalMetadataDesc as DocField
from .doc_service import DocServer
from .web import DocWebModule

_LOCAL_PYTHONPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


class CallableDict(dict):
    def __call__(self, cls, *args, **kw):
        return self[cls](*args, **kw)


class _MetaDocument(_MetaBind):
    def __instancecheck__(self, __instance):
        if isinstance(__instance, UrlDocument): return True
        return super().__instancecheck__(__instance)


class Document(ModuleBase, BuiltinGroups, metaclass=_MetaDocument):
    class _Manager(ModuleBase):
        @staticmethod
        def _resolve_dataset_path(dataset_path: Optional[str]) -> Optional[str]:
            if not dataset_path:
                return dataset_path
            if os.path.exists(dataset_path):
                return os.path.join(os.getcwd(), dataset_path)
            default_path = os.path.join(lazyllm.config['data_path'], dataset_path)
            return default_path if os.path.exists(default_path) else dataset_path

        @staticmethod
        def _decide_service_mode(manager, store_conf, processor, dataset_path) -> Tuple[bool, bool]:
            '''Returns ``(spawn_doc_server, connect_doc_server)``.'''
            if isinstance(manager, str) and manager != 'ui':
                raise ValueError(f'Unsupported manager value: {manager}')
            spawn = bool(manager) and not isinstance(manager, DocServer)
            connect = isinstance(manager, DocServer)
            if (not spawn and not connect and not processor and is_persistent_store(store_conf)
                    and dataset_path and not os.path.isfile(dataset_path)):
                lazyllm.LOG.info(f'Persistent store detected (type={store_conf.get("type")}),'
                                 f' auto-enabling DocServer for production-grade file tracking and scan.')
                spawn = True
            return spawn, connect

        @staticmethod
        def _reject_embedded_store_with_service_mode(store_conf, *, spawn, connect, processor):
            '''Service-mode RAG + an embedded single-process backend races the subprocesses on
            shared on-disk state; force the user to point at a networked endpoint instead.'''
            if not (spawn or connect or processor is not None):
                return
            embedded = list(iter_embedded_store_endpoints(store_conf))
            if not embedded:
                return
            raise ValueError(
                'Document with `manager=True` / `manager=DocServer(...)` / `manager=DocumentProcessor(...)`'
                ' does not support embedded (filesystem-bound) vector stores. Point the store config at a'
                ' remote service (http/https/tcp/grpc/unix scheme), e.g.'
                " Milvus: {'type': 'milvus', 'kwargs': {'uri': os.getenv('MILVUS_URI', 'http://<host>:19530')}}"
                " or Chroma: {'type': 'chroma', 'kwargs': {'uri': 'http://<host>:8000'}}."
                f' Offending store(s): {embedded!r}.')

        def _iter_kbs(self):
            return self._kbs._impl._m if isinstance(self._kbs, ServerModule) else self._kbs

        def __init__(self, dataset_path: Optional[str],
                     embed: Optional[Union[Callable, Dict[str, Callable]]] = None,
                     manager: Union[bool, str, DocServer] = False, server: Union[bool, int] = False,
                     name: Optional[str] = None, launcher: Optional[Launcher] = None,
                     store_conf: Optional[Dict] = None, doc_fields: Optional[Dict[str, DocField]] = None,
                     cloud: bool = False, doc_files: Optional[List[str]] = None,
                     processor: Optional[DocumentProcessor] = None, display_name: Optional[str] = '',
                     description: Optional[str] = 'algorithm description',
                     schema_extractor: Optional[Union[LLMBase, SchemaExtractor]] = None,
                     create_ui: bool = False):
            super().__init__()
            self._origin_path, self._doc_files, self._cloud = dataset_path, doc_files, cloud
            self._dataset_path = self._resolve_dataset_path(dataset_path)
            self._embed = self._get_embeds(embed)
            self._processor = processor
            self._create_ui = create_ui
            self._spawn_doc_server = False
            self._doc_processor_started = False

            spawn_doc_server, connect_doc_server = self._decide_service_mode(
                manager, store_conf, processor, self._dataset_path)
            self._reject_embedded_store_with_service_mode(
                store_conf, spawn=spawn_doc_server, connect=connect_doc_server, processor=processor)

            self._launcher: Launcher = launcher if launcher else (
                lazyllm.launchers.empty(sync=False) if spawn_doc_server else lazyllm.launchers.remote(sync=False))
            self._doc_impl_dataset_path = self._dataset_path if not (spawn_doc_server or connect_doc_server) else None
            self._doc_processor = None
            if spawn_doc_server:
                self._spawn_doc_server = True
                self._doc_processor = DocumentProcessor(launcher=self._launcher, pythonpath=_LOCAL_PYTHONPATH)
                self._submodules.remove(self._doc_processor)
            elif connect_doc_server:
                self._manager = manager
                parser_url = getattr(getattr(manager, '_raw_impl', None), '_parser_url', None) or manager.parser_url
                if parser_url:
                    self._doc_processor = DocumentProcessor(url=parser_url)
            self._schema_extractor = schema_extractor
            self._store_conf = store_conf
            self._display_name = display_name
            self._description = description
            name = name or RAG_DEFAULT_GROUP_NAME
            if not display_name: display_name = name
            doc_processor = self._doc_processor or processor
            self._kbs = CallableDict({name: DocImpl(
                embed=self._embed, dataset_path=self._doc_impl_dataset_path, doc_files=doc_files,
                global_metadata_desc=doc_fields, store=store_conf, processor=doc_processor,
                algo_name=name, display_name=display_name, description=description,
                schema_extractor=schema_extractor)})

            if create_ui and not self._spawn_doc_server:
                self.ensure_doc_web()
            if server:
                self._kbs = ServerModule(self._kbs, port=(None if isinstance(server, bool) else int(server)))
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

        def ensure_doc_web(self):
            if hasattr(self, '_docweb'):
                return self._docweb
            if self._spawn_doc_server and not hasattr(self, '_manager'):
                raise ValueError('`create_ui=True` with `manager=True` requires `Document.start()` before using the UI')
            if not hasattr(self, '_manager') or not isinstance(self._manager, DocServer):
                raise ValueError(
                    '`create_ui=True` requires an available DocServer. '
                    'Set `manager=True` or pass `manager=DocServer(...)`.'
                )
            self._docweb = DocWebModule(doc_server=self._manager)
            return self._docweb

        def _ensure_doc_processor_started(self):
            if self._doc_processor and not self._doc_processor_started:
                self._doc_processor.start()
                self._doc_processor_started = True

        def _ensure_managed_services_started(self):
            if self._spawn_doc_server:
                self._ensure_doc_processor_started()
                if not hasattr(self, '_manager'):
                    # Start DocServer with scanning disabled; enable only after
                    # all KBs + parser algorithms are registered so the first
                    # scan sees a consistent routing table.
                    self._manager = DocServer(
                        launcher=self._launcher,
                        storage_dir=self._dataset_path,
                        parser_url=self._doc_processor.url,
                        pythonpath=_LOCAL_PYTHONPATH,
                        enable_scan=bool(self._dataset_path),
                    )
                    self._manager.start()
                    kbs = self._iter_kbs()
                    for kb_name in kbs:
                        self._manager.ensure_kb_registered(kb_name)
                    for impl in kbs.values():
                        impl._lazy_init()
                    self._manager.enable_scanning()
                if self._create_ui and not hasattr(self, '_docweb'):
                    self.ensure_doc_web()
                    self._docweb.start()

        def _get_deploy_tasks(self):
            if self._spawn_doc_server and not hasattr(self, '_manager'):
                return lazyllm.pipeline(self._ensure_managed_services_started)
            return None

        def _get_embeds(self, embed):
            embeds = embed if isinstance(embed, dict) else {EMBED_DEFAULT_KEY: embed} if embed else {}
            for index, module in enumerate(embeds.values()):
                if isinstance(module, ModuleBase):
                    setattr(self, f'_embed_module_{index}', module)
            return embeds

        def add_kb_group(self, name, doc_fields: Optional[Dict[str, DocField]] = None,
                         store_conf: Optional[Dict] = None, embed: Optional[Union[Callable, Dict[str, Callable]]] = None,
                         schema_extractor: Optional[Union[LLMBase, SchemaExtractor]] = None):
            embed = self._get_embeds(embed) if embed else self._embed
            schema_extractor = schema_extractor or self._schema_extractor
            if isinstance(schema_extractor, ModuleBase):
                setattr(self, f'_schema_extractor_{name}', schema_extractor)
            impl = DocImpl(
                dataset_path=self._doc_impl_dataset_path, embed=embed, kb_group_name=name,
                global_metadata_desc=doc_fields,
                store=store_conf or self._store_conf, processor=self._doc_processor or self._processor,
                algo_name=name, display_name=name, description=self._description,
                schema_extractor=schema_extractor,
            )
            self._iter_kbs()[name] = impl
            # Register KB with DocServer if it's already running so the next scan cycle picks it up.
            if hasattr(self, '_manager') and isinstance(self._manager, DocServer):
                self._manager.ensure_kb_registered(name)
                impl._lazy_init()

        def get_doc_by_kb_group(self, name):
            return self._iter_kbs()[name]

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

    @staticmethod
    def _coerce_document_processor_manager(manager, store_conf, dataset_path):
        '''Validate the ``manager=DocumentProcessor(...)`` combination.

        Returns ``(processor, manager)``: when ``manager`` is a ``DocumentProcessor``
        the returned ``processor`` is the original instance and ``manager`` becomes
        ``False``; otherwise ``processor`` is ``None`` and ``manager`` passes through.
        '''
        if not isinstance(manager, DocumentProcessor):
            return None, manager
        if store_conf is None:
            raise ValueError('`store_conf` is required when `manager` is a DocumentProcessor')
        if is_local_map_store(store_conf):
            raise ValueError('`manager=DocumentProcessor(...)` does not support pure local map store')
        if dataset_path is not None:
            raise ValueError(
                '`manager=DocumentProcessor(...)` does not accept a local `dataset_path`: the external'
                ' parsing service does not own directory scanning / lifecycle management. Use'
                ' `manager=True` or `manager=DocServer(...)` for scan-based ingestion, or drop'
                ' `dataset_path` and upload documents via explicit API calls.')
        return manager, False

    def __init__(self, dataset_path: Optional[str] = None, embed: Optional[Union[Callable, Dict[str, Callable]]] = None,
                 create_ui: bool = False,
                 manager: Union[bool, str, DocServer, 'Document._Manager', DocumentProcessor] = False,
                 server: Union[bool, int] = False, name: Optional[str] = None,
                 launcher: Optional[Launcher] = None, doc_files: Optional[List[str]] = None,
                 doc_fields: Dict[str, DocField] = None,
                 store_conf: Optional[Dict] = None, display_name: Optional[str] = '',
                 description: Optional[str] = 'algorithm description',
                 schema_extractor: Optional[Union[LLMBase, SchemaExtractor]] = None,
                 enable_path_monitoring: Optional[bool] = None):
        super().__init__()
        if create_ui:
            warnings.warn('`create_ui=True` (and the legacy `manager="ui"` alias) is deprecated and will be removed'
                          ' in a future release. Prefer `manager=True` and interact with DocServer via its HTTP API'
                          ' / SDK instead.', DeprecationWarning, stacklevel=2)
        if isinstance(manager, str):
            if manager != 'ui': raise ValueError(f'Unsupported manager value: {manager}')
            create_ui = manager = True
        if enable_path_monitoring is not None:
            warnings.warn('`enable_path_monitoring` is deprecated: DocImpl no longer polls the dataset '
                          'directory. Persistent-store setups auto-upgrade to DocServer which owns scanning; '
                          'map-store setups get a one-time ingest at `_lazy_init`. The parameter is accepted '
                          'for backward compatibility but has no effect.', DeprecationWarning, stacklevel=2)
        if isinstance(dataset_path, (tuple, list)):
            doc_fields = dataset_path
            dataset_path = None
        if doc_files is not None:
            assert dataset_path is None and not manager, (
                'Manager and dataset_path are not supported for Document with temp-files')
            assert store_conf is None or store_conf['type'] == 'map', (
                'Only map store is supported for Document with temp-files')

        name = name or RAG_DEFAULT_GROUP_NAME

        if isinstance(manager, Document._Manager):
            assert not server, 'Server information is already set by manager'
            assert not launcher, 'Launcher information is already set by manager'
            assert not manager._cloud, 'manager is not allowed to share in cloud mode'
            assert manager._doc_files is None, 'manager is not allowed to share with temp files'
            if dataset_path != manager._dataset_path and dataset_path != manager._origin_path:
                raise RuntimeError(f'Document path mismatch, expected `{manager._dataset_path}`'
                                   f'while received `{dataset_path}`')
            manager.add_kb_group(name=name, doc_fields=doc_fields, store_conf=store_conf, embed=embed,
                                 schema_extractor=schema_extractor)
            if create_ui:
                manager.ensure_doc_web()
            self._manager = manager
            self._curr_group = name
        else:
            processor, manager = self._coerce_document_processor_manager(manager, store_conf, dataset_path)
            cloud = processor is not None
            self._manager = Document._Manager(dataset_path, embed, manager, server, name, launcher, store_conf,
                                              doc_fields, cloud=cloud, doc_files=doc_files, processor=processor,
                                              display_name=display_name, description=description,
                                              schema_extractor=schema_extractor, create_ui=create_ui)
            self._curr_group = name
        self._doc_to_db_processor: DocToDbProcessor = None
        self._graph_document: weakref.ref = None

    @staticmethod
    def list_all_files_in_directory(dataset_path: str, skip_hidden_path: bool = True,
                                    recursive: bool = True) -> List[str]:
        if not os.path.exists(dataset_path):
            return []
        if not os.path.isdir(dataset_path):
            return [dataset_path] if os.path.isfile(dataset_path) else []
        files_list = []
        if recursive:
            for root, dirs, files in os.walk(os.path.abspath(dataset_path)):
                if skip_hidden_path:
                    if any(part.startswith('.') for part in root.split(os.sep) if part):
                        continue
                    dirs[:] = [d for d in dirs if not d.startswith('.')]
                    files = [f for f in files if not f.startswith('.')]
                files_list.extend(os.path.join(root, f) for f in files)
        else:
            for item in os.listdir(dataset_path):
                if skip_hidden_path and item.startswith('.'):
                    continue
                item_path = os.path.join(dataset_path, item)
                if os.path.isfile(item_path):
                    files_list.append(item_path)
        return files_list

    def _list_all_files_in_dataset(self, skip_hidden_path: bool = True) -> List[str]:
        return self.list_all_files_in_directory(self._manager._dataset_path, skip_hidden_path)

    @property
    def url(self):
        assert isinstance(self._manager._kbs, ServerModule), 'Document is not a service, please set `manager` to `True`'
        return self._manager._kbs._url

    def connect_sql_manager(self, sql_manager: SqlManager, schma: Optional[DocInfoSchema] = None,
                            force_refresh: bool = True):
        def schema_as_dicts(schema: DocInfoSchema):
            if schema is None:
                return None, None
            return ({e['key']: e['desc'] for e in schema}, {e['key']: e['type'] for e in schema})

        if sql_manager.check_connection().status != DBStatus.SUCCESS:
            raise RuntimeError(f'Failed to connect to sql manager: {sql_manager._gen_conn_url()}')

        pre_schema = self._doc_to_db_processor.doc_info_schema if self._doc_to_db_processor else None
        assert pre_schema or schma, 'doc_table_schma must be given'
        schema_equal = schema_as_dicts(pre_schema) == schema_as_dicts(schma)
        assert schema_equal or force_refresh is True, \
            'When changing doc_table_schema, force_refresh should be set to True'

        if self._doc_to_db_processor is None or sql_manager != self._doc_to_db_processor.sql_manager:
            self._doc_to_db_processor = DocToDbProcessor(sql_manager)

        if schma and not schema_equal:
            # Clears existing lazyllm_doc_elements table.
            self._doc_to_db_processor._reset_doc_info_schema(schma)

    def get_sql_manager(self):
        if self._doc_to_db_processor is None:
            raise ValueError('Please call connect_sql_manager to init handler first')
        return self._doc_to_db_processor.sql_manager

    def extract_db_schema(
        self, llm: Union[OnlineChatModule, TrainableModule], print_schema: bool = False
    ) -> DocInfoSchema:
        file_paths = self._list_all_files_in_dataset()
        schema = extract_db_schema_from_files(file_paths, llm)
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
    def _schema_extractor(self):
        # Compat shim for pre-refactor callers (e.g. ``SqlCall.create_from_document``);
        # read through the active DocImpl so shared-manager KBs keep per-group values.
        impl = self._manager.get_doc_by_kb_group(self._curr_group)
        return getattr(impl, '_schema_extractor', None)

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

    def start(self):
        return super().start()

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
                  group: Optional[str] = None, kb_id: Optional[str] = None, numbers: Optional[Set] = None,
                  limit: Optional[int] = None, offset: int = 0, return_total: bool = False,
                  sort_by_number: bool = False) -> Union[List[DocNode], Tuple[List[DocNode], int]]:
        return self._forward(
            '_get_nodes', uids, doc_ids, group, kb_id, numbers, limit, offset, return_total, sort_by_number,
        )

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
        self._curr_group = name or RAG_DEFAULT_GROUP_NAME

    def _forward(self, func_name: str, *args, **kwargs):
        args = (self._curr_group, func_name, *args)
        return self._manager._call('__call__', *args, **kwargs)

    def find(self, target) -> Callable:
        return functools.partial(self._forward, 'find', group=target)

    def forward(self, *args, **kw):
        return self._forward('retrieve', *args, **kw)

    def get_nodes(self, uids: Optional[List[str]] = None, doc_ids: Optional[Set] = None,
                  group: Optional[str] = None, kb_id: Optional[str] = None, numbers: Optional[Set] = None,
                  limit: Optional[int] = None, offset: int = 0, return_total: bool = False,
                  sort_by_number: bool = False) -> Union[List[DocNode], Tuple[List[DocNode], int]]:
        return self._forward(
            '_get_nodes', uids, doc_ids, group, kb_id, numbers, limit, offset, return_total, sort_by_number,
        )

    def get_window_nodes(self, node: DocNode, span: tuple[int, int] = (-5, 5),
                         merge: bool = False) -> Union[List[DocNode], DocNode]:
        return self._forward('_get_window_nodes', node, span, merge)

    @cached_property
    def active_node_groups(self):
        return self._forward('active_node_groups')

    def __getattr__(self, name):
        if name in self._missing_keys:
            raise RuntimeError(f'Document generated with url and name has no attribute `{name}`')
