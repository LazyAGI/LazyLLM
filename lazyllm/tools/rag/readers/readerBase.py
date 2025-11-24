from lazyllm.thirdparty import fsspec
from typing import Iterable, List, Optional

from lazyllm.thirdparty import torch
from lazyllm import LOG

from ....common import LazyLLMRegisterMetaClass
from ..doc_node import DocNode
from lazyllm.module import ModuleBase
from pathlib import Path

class LazyLLMReaderBase(ModuleBase, metaclass=LazyLLMRegisterMetaClass):
    post_action = None

    def __init__(self, *args, return_trace: bool = True, **kwargs):
        super().__init__(return_trace=return_trace)

    def _lazy_load_data(self, *args, **load_kwargs) -> Iterable[DocNode]:
        raise NotImplementedError(f'{self.__class__.__name__} does not implement lazy_load_data method.')

    def _load_data(self, *args, **load_kwargs) -> List[DocNode]:
        return list(self._lazy_load_data(*args, **load_kwargs))

    def forward(self, *args, **kwargs) -> List[DocNode]:
        r = self._load_data(*args, **kwargs)
        r = [r] if isinstance(r, DocNode) else [] if r is None else r
        if r and self.post_action:
            r = [x for sub in [self.post_action(n) for n in r] for x in (sub if isinstance(sub, list) else [sub])]
        return r


def get_default_fs():
    return fsspec.implementations.local.LocalFileSystem()

def is_default_fs(fs: 'fsspec.AbstractFileSystem') -> bool:
    return isinstance(fs, fsspec.implementations.local.LocalFileSystem) or not fs.auto_mkdir

def infer_torch_device() -> str:
    try:
        has_cuda = torch.cuda.is_available()
    except NameError:
        has_cuda = torch.cuda.is_available()

    if has_cuda: return 'cuda'
    if torch.backends.mps.is_available(): return 'mps'
    return 'cpu'

class TxtReader(LazyLLMReaderBase):
    def __init__(self, encoding: str = 'utf-8', return_trace: bool = True) -> None:
        super().__init__(return_trace=return_trace)
        self._encoding = encoding

    def _load_data(self, file: Path, fs: Optional['fsspec.AbstractFileSystem'] = None) -> List[DocNode]:
        if not isinstance(file, Path): file = Path(file)
        with (fs or get_default_fs()).open(file, encoding=self._encoding) as f:
            return [DocNode(text=f.read().decode(self._encoding))]

class DefaultReader(TxtReader):
    def _load_data(self, file: Path, fs: Optional['fsspec.AbstractFileSystem'] = None) -> List[DocNode]:
        try:
            return super()._load_data(file, fs)
        except Exception:
            LOG.error(f'no pattern found for {file} and it is not {self._encoding}, skip it!')
            return []
