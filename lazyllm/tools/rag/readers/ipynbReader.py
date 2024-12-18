import re
from pathlib import Path
from typing import Dict, List, Optional
from fsspec import AbstractFileSystem

from .readerBase import LazyLLMReaderBase
from ..doc_node import DocNode

class IPYNBReader(LazyLLMReaderBase):
    def __init__(self, parser_config: Optional[Dict] = None, concatenate: bool = False, return_trace: bool = True):
        super().__init__(return_trace=return_trace)
        self._parser_config = parser_config
        self._concatenate = concatenate

    def _load_data(self, file: Path, extra_info: Optional[Dict] = None,
                   fs: Optional[AbstractFileSystem] = None) -> List[DocNode]:
        if not isinstance(file, Path): file = Path(file)

        if file.name.endswith(".ipynb"):
            try:
                import nbconvert
            except ImportError:
                raise ImportError("Please install nbconvert `pip install nbconvert`")

        if fs:
            with fs.open(file, encoding='utf-8') as f:
                doc_str = nbconvert.exporters.ScriptExporter().from_file(f)[0]
        else:
            doc_str = nbconvert.exporters.ScriptExporter().from_file(file)[0]

        splits = re.split(r"In\[\d+\]:", doc_str)
        splits.pop(0)

        if self._concatenate: docs = [DocNode(text="\n\n".join(splits), metadata=extra_info or {})]
        else: docs = [DocNode(text=s, global_metadata=extra_info) for s in splits]

        return docs
