import fsspec
from fsspec.implementations.local import LocalFileSystem
from typing import Iterable, List

from lazyllm.thirdparty import torch

from ....common import LazyLLMRegisterMetaClass
from ..doc_node import DocNode
from lazyllm.module import ModuleBase

class LazyLLMReaderBase(ModuleBase, metaclass=LazyLLMRegisterMetaClass):
    def __init__(self, *args, return_trace: bool = True, **kwargs):
        super().__init__(return_trace=return_trace)

    def _lazy_load_data(self, *args, **load_kwargs) -> Iterable[DocNode]:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement lazy_load_data method.")

    def _load_data(self, *args, **load_kwargs) -> List[DocNode]:
        return list(self._lazy_load_data(*args, **load_kwargs))

    def forward(self, *args, **kwargs) -> List[DocNode]:
        return self._load_data(*args, **kwargs)


def get_default_fs():
    return LocalFileSystem()

def is_default_fs(fs: fsspec.AbstractFileSystem) -> bool:
    return isinstance(fs, LocalFileSystem) or not fs.auto_mkdir

def infer_torch_device() -> str:
    try:
        has_cuda = torch.cuda.is_available()
    except NameError:
        has_cuda = torch.cuda.is_available()

    if has_cuda: return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"
