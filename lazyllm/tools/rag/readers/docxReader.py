from pathlib import Path
from fsspec import AbstractFileSystem
from typing import Dict, Optional, List

from .readerBase import LazyLLMReaderBase
from ..doc_node import DocNode

class DocxReader(LazyLLMReaderBase):
    def _load_data(self, file: Path, extra_info: Optional[Dict] = None,
                   fs: Optional[AbstractFileSystem] = None) -> List[DocNode]:
        if not isinstance(file, Path): file = Path(file)
        try:
            import docx2txt
        except ImportError:
            raise ImportError("docx2txt is required to read Microsoft Word files: `pip install docx2txt`")

        if fs:
            with fs.open(file) as f:
                text = docx2txt.process(f)
        else:
            text = docx2txt.process(file)

        return [DocNode(text=text, global_metadata=extra_info)]
