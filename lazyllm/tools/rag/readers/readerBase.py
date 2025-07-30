import fsspec
from fsspec.implementations.local import LocalFileSystem
from typing import Iterable, List

from lazyllm.thirdparty import torch

from ....common import LazyLLMRegisterMetaClass
from ..doc_node import DocNode
from lazyllm.module import ModuleBase

class LazyLLMReaderBase(ModuleBase, metaclass=LazyLLMRegisterMetaClass):
    """
The base class of file readers, which inherits from the ModuleBase base class and has Callable capabilities. Subclasses that inherit from this class only need to implement the _load_data function, and its return parameter type is List[DocNode]. Generally, the input parameters of the _load_data function are file (Path) and fs (AbstractFileSystem).

Args:
    args (Any): Pass the corresponding position parameters as needed.
    return_trace (bool): Set whether to record trace logs.
    kwargs (Dict): Pass the corresponding keyword arguments as needed.


Examples:
    
    >>> from lazyllm.tools.rag.readers import ReaderBase
    >>> from lazyllm.tools.rag import DocNode, Document
    >>> from typing import Dict, Optional, List
    >>> from pathlib import Path
    >>> from fsspec import AbstractFileSystem
    >>> @Document.register_global_reader("**/*.yml")
    >>> class YmlReader(ReaderBase):
    ...     def _load_data(self, file: Path, fs: Optional[AbstractFileSystem] = None) -> List[DocNode]:
    ...         try:
    ...             import yaml
    ...         except ImportError:
    ...             raise ImportError("yaml is required to read YAML file: `pip install pyyaml`")
    ...         with open(file, 'r') as f:
    ...             data = yaml.safe_load(f)
    ...         print("Call the class YmlReader.")
    ...         return [DocNode(text=data)]
    ...
    >>> files = ["your_yml_files"]
    >>> doc = Document(dataset_path="your_files_path", create_ui=False)
    >>> reader = doc._impl._reader.load_data(input_files=files)
    # Call the class YmlReader.
    """
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
