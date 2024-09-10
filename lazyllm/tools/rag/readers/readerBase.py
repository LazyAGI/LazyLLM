import fsspec
from fsspec.implementations.local import LocalFileSystem
from typing import Iterable, List

from ....common import LazyLLMRegisterMetaClass
from ..store import DocNode

class LazyLLMReaderBase(metaclass=LazyLLMRegisterMetaClass):
    def lazy_load_data(self, *args, **load_kwargs) -> Iterable[DocNode]:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement lazy_load_data method.")

    def load_data(self, *args, **load_kwargs) -> List[DocNode]:
        return list(self.lazy_load_data(*args, **load_kwargs))


def get_default_fs():
    return LocalFileSystem()

def is_default_fs(fs: fsspec.AbstractFileSystem) -> bool:
    return isinstance(fs, LocalFileSystem) or not fs.auto_mkdir

def infer_torch_device() -> str:
    try:
        has_cuda = torch.cuda.is_available()
    except NameError:
        import torch
        has_cuda = torch.cuda.is_available()

    if has_cuda: return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"
