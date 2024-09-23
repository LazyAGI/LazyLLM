from functools import partial

from typing import Callable, Optional, Dict
import lazyllm
from lazyllm import ModuleBase, TrainableModule, DynamicDescriptor

from .doc_impl import DocImpl
from .store import LAZY_ROOT_NAME
from .db import KBFileRecord, FileState
from .doc_polling import DocPolling


class Document(ModuleBase):
    _registered_file_reader: Dict[str, Callable] = {}

    def __init__(self, doc_name: str, embed: Optional[TrainableModule] = None, clear_cache: bool = False):
        super().__init__()

        self._local_file_reader: Dict[str, Callable] = {}
        self._doc_name = doc_name

        if clear_cache:
            KBFileRecord.del_node(kb_name=doc_name)

        files = KBFileRecord.get_file_path_by_kb_name(kb_name=doc_name, file_state=FileState.PARSED)
        self._impl = DocImpl(
            doc_files=files, 
            embed=embed, 
            local_readers=self._local_file_reader,
            global_readers=self._registered_file_reader
        )
        DocPolling.start_polling(kb_name=doc_name, doc_impl=self._impl)

    @property
    def doc_name(self):
        return self._doc_name

    def forward(self, func_name: str, *args, **kwargs):
        return getattr(self._impl, func_name)(*args, **kwargs)

    def find_parent(self, group: str) -> Callable:
        return partial(self.forward, "find_parent", group=group)

    def find_children(self, group: str) -> Callable:
        return partial(self.forward, "find_children", group=group)

    def __repr__(self):
        return lazyllm.make_repr("Module", "Document", manager=self._manager)

    @DynamicDescriptor
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
