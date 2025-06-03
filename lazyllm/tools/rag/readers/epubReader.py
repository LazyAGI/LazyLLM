import importlib.util

from pathlib import Path
from typing import List, Optional
from fsspec import AbstractFileSystem

from .readerBase import LazyLLMReaderBase
from ..doc_node import DocNode
from lazyllm.thirdparty import html2text, ebooklib
from lazyllm import LOG

class EpubReader(LazyLLMReaderBase):
    def _load_data(self, file: Path, fs: Optional[AbstractFileSystem] = None) -> List[DocNode]:
        if not isinstance(file, Path): file = Path(file)

        if fs:
            LOG.warning("fs was specified but EpubReader doesn't support loading from "
                        "fsspec filesystems. Will load from local filesystem instead.")

        text_list = []

        spec = importlib.util.find_spec("ebooklib.epub")
        if spec is None:
            raise ImportError(
                "Please install ebooklib to use ebooklib module. "
                "You can install it with `pip install ebooklib`"
            )
        epub_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(epub_module)

        book = epub_module.read_epub(file, options={"ignore_ncs": True})

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                text_list.append(html2text.html2text(item.get_content().decode("utf-8")))
        text = "\n".join(text_list)
        return [DocNode(text=text)]
