from functools import partial
import os

from typing import Callable, Optional, Dict
import lazyllm
from lazyllm import ModuleBase, ServerModule, TrainableModule

from .web import DocWebModule
from .doc_manager import DocManager
from .group_doc import DocGroupImpl
from .store import LAZY_ROOT_NAME


class Document(ModuleBase):
    _registered_file_reader: Dict[str, Callable] = {}

    def __init__(self, dataset_path: str, embed: Optional[TrainableModule] = None,
                 create_ui: bool = True, launcher=None):
        super().__init__()
        if not os.path.exists(dataset_path):
            defatult_path = os.path.join(lazyllm.config["data_path"], dataset_path)
            if os.path.exists(defatult_path):
                dataset_path = defatult_path
        self._create_ui = create_ui
        launcher = launcher if launcher else lazyllm.launchers.remote(sync=False)
        self._instance_registered_file_reader: Dict[str, Callable] = {}

        self._impl = DocGroupImpl(dataset_path=dataset_path, embed=embed, readers=self._registered_file_reader)
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

    def register_instance_file_reader(self, readers: Dict[str, Callable]):
        self._instance_registered_file_reader.update(readers)
        self._impl._impl.directory_reader.register_file_reader({**self._instance_registered_file_reader,
                                                                **self._registered_file_reader})

    @classmethod
    def register_cls_file_reader(cls, match_key: str):
        if isinstance(match_key, type):
            raise TypeError("Document.register_file_reader() missing 1 required positional argument: 'match_key'")

        def decorator(klass):
            if isinstance(klass, type): cls._registered_file_reader[match_key] = klass()
            elif callable(klass): cls._registered_file_reader[match_key] = klass
            else: raise TypeError(f"The registered object {klass} is not a callable object.")
            return klass
        return decorator

    @classmethod
    def get_registry(cls):
        return cls._registered_file_reader
