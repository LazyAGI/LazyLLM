import os

from typing import Callable, Optional, Dict, Union, List
import lazyllm
from lazyllm import ModuleBase, ServerModule, DynamicDescriptor
from lazyllm.launcher import LazyLLMLaunchersBase as Launcher

from .doc_manager import DocManager
from .doc_impl import DocImpl
from .store import LAZY_ROOT_NAME, EMBED_DEFAULT_KEY, DocNode
from .utils import DocListManager
from .web import DocWebModule
import copy
import functools


class CallableDict(dict):
    def __call__(self, cls, *args, **kw):
        return self[cls](*args, **kw)

class Document(ModuleBase):
    class _Impl(ModuleBase):
        def __init__(self, dataset_path: str, embed: Optional[Union[Callable, Dict[str, Callable]]] = None,
                     manager: Union[bool, str] = False, server: bool = False, name: Optional[str] = None,
                     launcher: Launcher = None):
            super().__init__()
            if not os.path.exists(dataset_path):
                defatult_path = os.path.join(lazyllm.config["data_path"], dataset_path)
                if os.path.exists(defatult_path):
                    dataset_path = defatult_path
            self._launcher: Launcher = launcher if launcher else lazyllm.launchers.remote(sync=False)
            self._dataset_path = dataset_path
            self._embed = embed if isinstance(embed, dict) else {EMBED_DEFAULT_KEY: embed} if embed else {}
            self.name = name
            for embed in self._embed.values():
                if isinstance(embed, ModuleBase):
                    self._submodules.append(embed)
            self._dlm = DocListManager(dataset_path, name).init_tables()
            self._kbs = CallableDict({DocListManager.DEDAULT_GROUP_NAME: DocImpl(embed=self._embed, dlm=self._dlm)})
            if manager: self._manager = ServerModule(DocManager(self._dlm))
            if manager == 'ui': self._docweb = DocWebModule(doc_server=self._manager)
            if server: self._kbs = ServerModule(self._kbs)

        def add_kb_group(self, name):
            if isinstance(self._kbs, ServerModule):
                self._kbs._impl._m[name] = DocImpl(dlm=self._dlm, embed=self._embed, kb_group_name=name)
            else:
                self._kbs[name] = DocImpl(dlm=self._dlm, embed=self._embed, kb_group_name=name)
            self._dlm.add_kb_group(name)

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
                 name: Optional[str] = None, launcher=None):
        super().__init__()
        if create_ui:
            lazyllm.LOG.warning('`create_ui` for Document is deprecated, use `manager` instead')
        self._impls = Document._Impl(dataset_path, embed, create_ui or manager, server, name, launcher)
        self._curr_group = DocListManager.DEDAULT_GROUP_NAME

    def create_kb_group(self, name: str) -> "Document":
        self._impls.add_kb_group(name)
        doc = copy.copy(self)
        doc._curr_group = name
        return doc

    @property
    def _impl(self): return self._impls.get_doc_by_kb_group(self._curr_group)

    @property
    def manager(self): return getattr(self._impls, '_manager', None)

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

    def _forward(self, func_name: str, *args, **kw):
        return self._impls(self._curr_group, func_name, *args, **kw)

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
        return lazyllm.make_repr("Module", "Document", manager=hasattr(self._impl, '_manager'),
                                 server=isinstance(self._impls._kbs, ServerModule))
