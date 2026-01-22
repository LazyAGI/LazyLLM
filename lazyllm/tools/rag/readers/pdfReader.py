import io
from pathlib import Path
from typing import List, Optional, Callable
from lazyllm.common import retry
from lazyllm.thirdparty import fsspec, pypdf
from lazyllm import LOG

from .readerBase import LazyLLMReaderBase, get_default_fs, is_default_fs
from ..doc_node import DocNode, RichDocNode

RETRY_TIMES = 3


class _RichPDFReader(LazyLLMReaderBase):
    def __init__(self, post_func: Optional[Callable] = None, split_doc: bool = True,
                 return_trace: bool = True):
        super().__init__(return_trace=return_trace)
        self._post_func = post_func
        self._split_doc = split_doc

    def forward(self, *args, **kwargs) -> List[DocNode]:
        r = super().forward(*args, **kwargs)
        if self._post_func:
            r = self._post_func(r)
            assert isinstance(r, list), f'Expected list, got {type(r)}, please check your post function'
            for n in r:
                assert isinstance(n, DocNode), f'Expected DocNode, got {type(n)}, \
                    please check your post function'
                n.global_metadata = kwargs.get('extra_info')
        return [RichDocNode(r)] if self._split_doc else r

class PDFReader(_RichPDFReader):
    def __init__(self, split_doc: bool = True,
                 post_func: Optional[Callable[[List[DocNode]], List[DocNode]]] = None,
                 return_trace: bool = True, *, return_full_document=None) -> None:
        if return_full_document is not None:
            LOG.warning('return_full_document is deprecated, please use split_doc instead')
            assert split_doc ^ return_full_document, \
                'split_doc and return_full_document cannot be both True or False'
            split_doc = not return_full_document
        super().__init__(post_func=post_func, split_doc=split_doc, return_trace=return_trace)

    @retry(stop_after_attempt=3)
    def _load_data(self, file: Path, fs: Optional['fsspec.AbstractFileSystem'] = None) -> List[DocNode]:
        if not isinstance(file, Path): file = Path(file)

        fs = fs or get_default_fs()
        with fs.open(file, 'rb') as fp:
            stream = fp if is_default_fs(fs) else io.BytesIO(fp.read())
            pdf = pypdf.PdfReader(stream)
            num_pages = len(pdf.pages)
            docs = []
            if self._split_doc:
                for page in range(num_pages):
                    page_text = pdf.pages[page].extract_text()
                    page_label = pdf.page_labels[page]
                    metadata = {'page_label': page_label}
                    docs.append(DocNode(text=page_text, metadata=metadata))
            else:
                text = '\n'.join(pdf.pages[page].extract_text() for page in range(num_pages))
                docs.append(DocNode(text=text))
            return docs
