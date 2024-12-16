import io
from tenacity import retry, stop_after_attempt
from pathlib import Path
from typing import Dict, List, Optional
from fsspec import AbstractFileSystem

from .readerBase import LazyLLMReaderBase, get_default_fs, is_default_fs
from ..doc_node import DocNode

RETRY_TIMES = 3

class PDFReader(LazyLLMReaderBase):
    def __init__(self, return_full_document: bool = False, return_trace: bool = True) -> None:
        super().__init__(return_trace=return_trace)
        self._return_full_document = return_full_document

    @retry(stop=stop_after_attempt(RETRY_TIMES))
    def _load_data(self, file: Path, extra_info: Optional[Dict] = None,
                   fs: Optional[AbstractFileSystem] = None) -> List[DocNode]:
        if not isinstance(file, Path): file = Path(file)

        try:
            import pypdf
        except ImportError:
            raise ImportError("pypdf is required to read PDF files: `pip install pypdf`")

        fs = fs or get_default_fs()
        with fs.open(file, 'rb') as fp:
            stream = fp if is_default_fs(fs) else io.BytesIO(fp.read())
            pdf = pypdf.PdfReader(stream)
            num_pages = len(pdf.pages)
            docs = []
            if self._return_full_document:
                text = "\n".join(pdf.pages[page].extract_text() for page in range(num_pages))
                docs.append(DocNode(text=text, global_metadata=extra_info))
            else:
                for page in range(num_pages):
                    page_text = pdf.pages[page].extract_text()
                    page_label = pdf.page_labels[page]
                    metadata = {"page_label": page_label}
                    docs.append(DocNode(text=page_text, metadata=metadata, global_metadata=extra_info))
            return docs
