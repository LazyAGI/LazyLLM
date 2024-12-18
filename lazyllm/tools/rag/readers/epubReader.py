from pathlib import Path
from typing import Dict, List, Optional
from fsspec import AbstractFileSystem

from .readerBase import LazyLLMReaderBase
from ..doc_node import DocNode
from lazyllm import LOG

class EpubReader(LazyLLMReaderBase):
    def _load_data(self, file: Path, extra_info: Optional[Dict] = None,
                   fs: Optional[AbstractFileSystem] = None) -> List[DocNode]:
        try:
            import ebooklib
            import html2text
            from ebooklib import epub
        except ImportError:
            raise ImportError("Please install extra dependencies that are required "
                              "for the EpubReader: `pip install EbookLib html2text`")

        if not isinstance(file, Path): file = Path(file)

        if fs:
            LOG.warning("fs was specified but EpubReader doesn't support loading from "
                        "fsspec filesystems. Will load from local filesystem instead.")

        text_list = []
        book = epub.read_epub(file, options={"ignore_ncs": True})

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                text_list.append(html2text.html2text(item.get_content().decode("utf-8")))
        text = "\n".join(text_list)
        return [DocNode(text=text, global_metadata=extra_info)]
