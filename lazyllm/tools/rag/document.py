from functools import partial
import os

from typing import Callable, Optional, Dict, Union
import lazyllm
from lazyllm import ModuleBase, ServerModule

from .web import DocWebModule
from .doc_manager import DocManager
from .group_doc import DocGroupImpl
from .store import LAZY_ROOT_NAME


class Document(ModuleBase):
    _registered_file_reader: Dict[str, Callable] = {}

    def __init__(self, dataset_path: str, embed: Optional[Union[Callable, Dict[str, Callable]]] = None,
                 create_ui: bool = True, launcher=None):
        super().__init__()
        if not os.path.exists(dataset_path):
            defatult_path = os.path.join(lazyllm.config["data_path"], dataset_path)
            if os.path.exists(defatult_path):
                dataset_path = defatult_path
        self._create_ui = create_ui
        launcher = launcher if launcher else lazyllm.launchers.remote(sync=False)
        self._local_file_reader: Dict[str, Callable] = {}
        self._embed = embed if isinstance(embed, dict) else {"default": embed}
        for embed in self._embed.values():
            if isinstance(embed, ModuleBase):
                self._submodules.append(embed)

        self._impl = DocGroupImpl(dataset_path=dataset_path, embed=self._embed, local_readers=self._local_file_reader,
                                  global_readers=self._registered_file_reader)
        if create_ui:
            doc_manager = DocManager(self._impl)
            self.doc_server = ServerModule(doc_manager, launcher=launcher)
            self.web = DocWebModule(doc_server=self.doc_server)

    def forward(self, func_name: str, *args, **kwargs):
        if self._create_ui:
            kwargs["func_name"] = func_name
            return self.doc_server.forward(*args, **kwargs)
        else:
            return getattr(self._impl, func_name)(*args, **kwargs)

    def find_parent(self, group: str) -> Callable:
        return partial(self.forward, "find_parent", group=group)

    def find_children(self, group: str) -> Callable:
        return partial(self.forward, "find_children", group=group)

    def __repr__(self):
        return lazyllm.make_repr("Module", "Document", create_ui=self._create_ui)

    def create_node_group(
        self, name: str, transform: Callable, parent: str = LAZY_ROOT_NAME, **kwargs
    ) -> None:
        self._impl.create_node_group(name, transform, parent, **kwargs)

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
