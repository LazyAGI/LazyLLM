from pathlib import Path
from typing import Dict, List, Optional
from lazyllm.thirdparty import fsspec

from .readerBase import LazyLLMReaderBase
from ..doc_node import DocNode

class IPYNBReader(LazyLLMReaderBase):
    def __init__(self, parser_config: Optional[Dict] = None, concatenate: bool = False, return_trace: bool = True):
        super().__init__(return_trace=return_trace)
        self._parser_config = parser_config
        self._concatenate = concatenate

    def _load_data(self, file: Path, fs: Optional['fsspec.AbstractFileSystem'] = None,
                   use_cache: bool = True, **kwargs) -> List[DocNode]:
        if not isinstance(file, Path): file = Path(file)

        try:
            import nbformat
        except ImportError:
            raise ImportError('Please install nbformat: pip install nbformat')

        if fs:
            with fs.open(file, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)
        else:
            with open(file, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)

        cell_texts = []
        for cell in notebook.cells:
            source = getattr(cell, 'source', '')
            if isinstance(source, list):
                source = ''.join(source)
            if not source or not source.strip():
                continue

            cell_texts.append(source)

        if not cell_texts:
            return []

        if self._concatenate:
            return [DocNode(text=('\n\n' + '-' * 80 + '\n\n').join(cell_texts))]
        return [DocNode(text=text) for text in cell_texts]
