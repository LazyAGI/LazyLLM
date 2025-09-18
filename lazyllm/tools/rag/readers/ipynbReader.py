import re
from pathlib import Path
from typing import Dict, List, Optional
from lazyllm.thirdparty import fsspec

from .readerBase import LazyLLMReaderBase
from ..doc_node import DocNode

class IPYNBReader(LazyLLMReaderBase):
    """Module for reading and parsing Jupyter Notebook (.ipynb) files. Converts the notebook to script text, then splits it by code cells into multiple document nodes or concatenates into a single text node.

Args:
    parser_config (Optional[Dict]): Reserved parser configuration parameter, currently unused. Defaults to None.
    concatenate (bool): Whether to concatenate all code cells into one text node. Defaults to False (split into multiple nodes).
    return_trace (bool): Whether to record processing trace. Default is True.
"""
    def __init__(self, parser_config: Optional[Dict] = None, concatenate: bool = False, return_trace: bool = True):
        super().__init__(return_trace=return_trace)
        self._parser_config = parser_config
        self._concatenate = concatenate

    def _load_data(self, file: Path, fs: Optional['fsspec.AbstractFileSystem'] = None) -> List[DocNode]:
        if not isinstance(file, Path): file = Path(file)

        if file.name.endswith('.ipynb'):
            try:
                import nbconvert
            except ImportError:
                raise ImportError('Please install nbconvert `pip install nbconvert`')

        if fs:
            with fs.open(file, encoding='utf-8') as f:
                doc_str = nbconvert.exporters.ScriptExporter().from_file(f)[0]
        else:
            doc_str = nbconvert.exporters.ScriptExporter().from_file(file)[0]

        splits = re.split(r'In\[\d+\]:', doc_str)
        splits.pop(0)

        if self._concatenate: docs = [DocNode(text='\n\n'.join(splits))]
        else: docs = [DocNode(text=s) for s in splits]

        return docs
