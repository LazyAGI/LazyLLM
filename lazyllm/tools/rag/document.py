from functools import partial
import os

from typing import Callable, Optional, Dict
import lazyllm
from lazyllm import ModuleBase, ServerModule, TrainableModule

from .web import DocWebModule
from .doc_manager import DocManager
from .group_doc import DocGroupImpl, DocImpl
from .store import LAZY_ROOT_NAME


class _DynamicDescriptor:
    class Impl:
        def __init__(self, func, instance, owner):
            self.func = func
            self.instance = instance
            self.owner = owner

        def __call__(self, *args, **kw):
            return self.func(self.instance, *args, **kw) if self.instance else self.func(self.owner, *args, **kw)

        @property
        def __doc__(self): return self.func.__doc__

        @__doc__.setter
        def __doc__(self, value): self.func.__doc__ = value

        def __repr__(self): return repr(self.func)

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        return _DynamicDescriptor.Impl(self.func, instance, owner)


class Document(ModuleBase):
    _registered_file_reader: Dict[str, Callable] = {}

    def __init__(self, dataset_path: str, embed: Optional[TrainableModule] = None,
                 create_ui: bool = False, manager: bool = False, launcher=None):
        super().__init__()
        if not os.path.exists(dataset_path):
            defatult_path = os.path.join(lazyllm.config["data_path"], dataset_path)
            if os.path.exists(defatult_path):
                dataset_path = defatult_path
        if create_ui:
            lazyllm.LOG.warning('`create_ui` for Document is deprecated, use `manager` instead')
        self._manager = create_ui or manager
        launcher = launcher if launcher else lazyllm.launchers.remote(sync=False)
        self._local_file_reader: Dict[str, Callable] = {}

        self._impl = DocGroupImpl(dataset_path=dataset_path, embed=embed, local_readers=self._local_file_reader,
                                  global_readers=self._registered_file_reader)
        if self._manager:
            doc_manager = DocManager(self._impl)
            self.doc_server = ServerModule(doc_manager, launcher=launcher)
            self.web = DocWebModule(doc_server=self.doc_server)

    def forward(self, func_name: str, *args, **kwargs):
        if self._manager:
            kwargs["func_name"] = func_name
            return self.doc_server.forward(*args, **kwargs)
        else:
            return getattr(self._impl, func_name)(*args, **kwargs)

    def find_parent(self, group: str) -> Callable:
        return partial(self.forward, "find_parent", group=group)

    def find_children(self, group: str) -> Callable:
        return partial(self.forward, "find_children", group=group)

    def __repr__(self):
        return lazyllm.make_repr("Module", "Document", manager=self._manager)

    @_DynamicDescriptor
    def create_node_group(self, name: str = None, *, transform: Callable, parent: str = LAZY_ROOT_NAME,
                          trans_node: bool = None, num_workers: int = 0, **kwargs) -> None:
        if isinstance(self, type):
            DocImpl.create_global_node_group(name, transform=transform, parent=parent, trans_node=trans_node,
                                             num_workers=num_workers, **kwargs)
        else:
            self._impl.create_node_group(name, transform=transform, parent=parent, trans_node=trans_node,
                                         num_workers=num_workers, **kwargs)

    def add_reader(self, pattern: str, func: Callable):
        self._local_file_reader[pattern] = func

    @classmethod
    def register_global_reader(cls, pattern: str, func: Optional[Callable] = None):
        if func is not None:
            cls._registered_file_reader[pattern] = func

        def decorator(klass):
            if callable(klass): cls._registered_file_reader[pattern] = klass
            else: raise TypeError(f"The registered object {klass} is not a callable object.")
            return klass
        return decorator
