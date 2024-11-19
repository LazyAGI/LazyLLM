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
        metadata = {"file_name": file.name}
        if extra_info is not None: metadata.update(extra_info)

        return [DocNode(text=text, metadata=metadata)]
