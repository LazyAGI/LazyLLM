from functools import partial
import os

from typing import Callable, Optional, Dict, Union, List
import lazyllm
from lazyllm import ModuleBase, ServerModule, DynamicDescriptor

from .doc_manager import DocManager
from .doc_impl import DocImpl
from .store import LAZY_ROOT_NAME, EMBED_DEFAULT_KEY, DocNode
import copy


class Document(ModuleBase):
    class _Impl(ModuleBase):
        def __init__(self, dataset_path: str, embed: Optional[Union[Callable, Dict[str, Callable]]] = None,
                     manager: bool = False, server: bool = False, launcher=None):
            super().__init__()
            if not os.path.exists(dataset_path):
                defatult_path = os.path.join(lazyllm.config["data_path"], dataset_path)
                if os.path.exists(defatult_path):
                    dataset_path = defatult_path
            launcher = launcher if launcher else lazyllm.launchers.remote(sync=False)
            self._dataset_path = dataset_path
            self._embed = embed if isinstance(embed, dict) else {EMBED_DEFAULT_KEY: embed}
            for embed in self._embed.values():
                if isinstance(embed, ModuleBase):
                    self._submodules.append(embed)
            self._kbs = {DocImpl.DEDAULT_GROUP_NAME: DocImpl(dataset_path=dataset_path, embed=self._embed)}
            if manager: self._manager = DocManager(dataset_path)
            if server: self._doc = ServerModule(self._doc)

        def forward(self, func_name: str, *args, **kwargs):
            return getattr(self._impl, func_name)(*args, **kwargs)

        def add_kb_group(self, name): self._kbs[name] = DocImpl(dataset_path=self.dataset_path, embed=self._embed)
        def get_doc_by_kb_group(self, name): return self._kbs[name]

    def __init__(self, dataset_path: str, embed: Optional[Union[Callable, Dict[str, Callable]]] = None,
                 create_ui: bool = False, manager: bool = False, server: bool = False, launcher=None):
        super().__init__()
        if create_ui:
            lazyllm.LOG.warning('`create_ui` for Document is deprecated, use `manager` instead')
        self._impls = Document._Impl(dataset_path, embed, create_ui or manager, server, launcher)
        self._curr_group = DocImpl.DEDAULT_GROUP_NAME

    def create_kb_group(self, name: str) -> "Document":
        self._impls.add_kb_group(name)
        doc = copy.copy(self)
        doc._curr_group = name
        return doc

    @property
    def _impl(self): return self._impls.get_doc_by_kb_group(self._curr_group)

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

    def find_parent(self) -> Callable:
        return partial(self._impl.forward, "find_parent")

    def find_children(self) -> Callable:
        return partial(self._impl.forward, "find_children")

    def forward(self, *args, **kw) -> List[DocNode]:
        return self._impl.forward("retrieve", *args, **kw)

    def __repr__(self):
        return lazyllm.make_repr("Module", "Document", manager=bool(self._manager))
