from pathlib import Path
from lazyllm.thirdparty import fsspec
from typing import Optional, List

from lazyllm.thirdparty import docx2txt
from .readerBase import LazyLLMReaderBase
from ..doc_node import DocNode

class DocxReader(LazyLLMReaderBase):
    """A docx format file parser, reading text content from a `.docx` file and return a list of `DocNode` objects.

Args:
    file (Path): Path to the `.docx` file.
    fs (Optional[AbstractFileSystem]): Optional file system object for custom reading.

**Returns:**

- List[DocNode]: A list containing the extracted text content as `DocNode` instances.
"""
    def _load_data(self, file: Path, fs: Optional['fsspec.AbstractFileSystem'] = None) -> List[DocNode]:
        if not isinstance(file, Path): file = Path(file)

        if fs:
            with fs.open(file) as f:
                text = docx2txt.process(f)
        else:
            text = docx2txt.process(file)

        return [DocNode(text=text)]
