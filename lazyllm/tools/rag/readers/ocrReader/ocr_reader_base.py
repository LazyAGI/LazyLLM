from typing import Optional, Callable, Set
from enum import Enum

from ..readerBase import _RichReader


class ServiceVariant(str, Enum):
    ONLINE = 'online'
    OFFLINE = 'offline'

    @classmethod
    def _missing_(cls, value):
        supported = ', '.join(f'{m.value!r}' for m in cls)
        raise ValueError(f'Invalid backend: {value!r}, only support: {supported}')


class _OcrReaderBase(_RichReader):
    def __init__(self,
            url,
            api_key: Optional[str] = None,
            droped_types: Set[str] = Set(),
            split_doc: bool = True,
            post_func: Optional[Callable] = None,
            return_trace: bool = True):
        super().__init__(post_func=post_func, split_doc=split_doc, return_trace=return_trace)
        self._url = url
        self._api_key = api_key
        self._droped_types = droped_types
    
    def _fetch_response(self) -> str:
