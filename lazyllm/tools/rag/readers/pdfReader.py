import io
from pathlib import Path
from typing import List, Optional, Callable
from lazyllm.common import retry
from lazyllm.thirdparty import fsspec, pypdf

from .readerBase import LazyLLMReaderBase, get_default_fs, is_default_fs
from ..doc_node import DocNode, RichDocNode

RETRY_TIMES = 3


class _RichPDFReader(LazyLLMReaderBase):
    def __init__(self, post_func: Optional[Callable] = None,
                 return_trace: bool = True):
        super().__init__(return_trace=return_trace)
        self.post_func = post_func

    def forward(self, *args, **kwargs) -> List[DocNode]:
        r = super().forward(*args, **kwargs)
        if self.post_func:
            r = self.post_func(r)
            assert isinstance(r, list), f'Expected list, got {type(r)}, please check your post function'
            for n in r:
                assert isinstance(n, DocNode), f'Expected DocNode, got {type(n)}, \
                    please check your post function'
                n.global_metadata = kwargs.get('extra_info')
        return [RichDocNode(r)] if len(r) > 1 else r

class PDFReader(_RichPDFReader):
    def __init__(self, return_full_document: bool = False,
                 post_func: Optional[Callable[[List[DocNode]], List[DocNode]]] = None,
                 return_trace: bool = True) -> None:
        super().__init__(post_func=post_func, return_trace=return_trace)
        self._return_full_document = return_full_document

    @retry(stop_after_attempt=3)
    def _load_data(self, file: Path, fs: Optional['fsspec.AbstractFileSystem'] = None) -> List[DocNode]:
        if not isinstance(file, Path): file = Path(file)

        fs = fs or get_default_fs()
        with fs.open(file, 'rb') as fp:
            stream = fp if is_default_fs(fs) else io.BytesIO(fp.read())
            pdf = pypdf.PdfReader(stream)
            num_pages = len(pdf.pages)
            docs = []
            if self._return_full_document:
                text = '\n'.join(pdf.pages[page].extract_text() for page in range(num_pages))
                docs.append(DocNode(text=text))
            else:
                for page in range(num_pages):
                    page_text = pdf.pages[page].extract_text()
                    page_label = pdf.page_labels[page]
                    metadata = {'page_label': page_label}
                    docs.append(DocNode(text=page_text, metadata=metadata))
            return docs
