from pathlib import Path
from fsspec import AbstractFileSystem
from typing import Optional, List

from lazyllm.thirdparty import docx2txt
from .readerBase import LazyLLMReaderBase
from ..doc_node import DocNode

class DocxReader(LazyLLMReaderBase):
    def _load_data(self, file: Path, fs: Optional[AbstractFileSystem] = None) -> List[DocNode]:
        if not isinstance(file, Path): file = Path(file)

        if fs:
            with fs.open(file) as f:
                text = docx2txt.process(f)
        else:
            text = docx2txt.process(file)

        return [DocNode(text=text)]
