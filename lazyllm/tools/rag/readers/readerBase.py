from lazyllm.thirdparty import fsspec
from typing import Iterable, List

from lazyllm.thirdparty import torch

from ....common import LazyLLMRegisterMetaClass
from ..doc_node import DocNode
from lazyllm.module import ModuleBase

class LazyLLMReaderBase(ModuleBase, metaclass=LazyLLMRegisterMetaClass):
    """
Base document reader class that provides fundamental interfaces for document loading. Inherits from ModuleBase and uses LazyLLMRegisterMetaClass as metaclass.

Args:
    *args: Positional arguments, reserved for parent or subclass use.
    return_trace (bool): Whether to return processing trace information. Defaults to True.
    **kwargs: Keyword arguments, reserved for parent or subclass use.


Examples:
    
    from lazyllm.tools.rag.readers.readerBase import LazyLLMReaderBase
    from lazyllm.tools.rag.doc_node import DocNode
    from typing import Iterable
    
    class CustomReader(LazyLLMReaderBase):
        def _lazy_load_data(self, file_paths: list, **kwargs) -> Iterable[DocNode]:
            for file_path in file_paths:
                # Process each file and yield DocNode
                content = self._read_file(file_path)
                yield DocNode(
                    text=content,
                    metadata={"source": file_path}
                )
    
    # Create reader instance
    reader = CustomReader(return_trace=True)
    
    # Load documents
    documents = reader.forward(file_paths=["doc1.txt", "doc2.txt"])
    """
    def __init__(self, *args, return_trace: bool = True, **kwargs):
        super().__init__(return_trace=return_trace)

    def _lazy_load_data(self, *args, **load_kwargs) -> Iterable[DocNode]:
        raise NotImplementedError(f'{self.__class__.__name__} does not implement lazy_load_data method.')

    def _load_data(self, *args, **load_kwargs) -> List[DocNode]:
        return list(self._lazy_load_data(*args, **load_kwargs))

    def forward(self, *args, **kwargs) -> List[DocNode]:
        return self._load_data(*args, **kwargs)


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
