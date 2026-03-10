from typing import List, Dict, Any

from lazyllm.module import ModuleBase


_TITLE_KEY = 'title'
_URL_KEY = 'url'
_SNIPPET_KEY = 'snippet'
_SOURCE_KEY = 'source'
_EXTRA_KEY = 'extra'


def _make_result(title: str, url: str, snippet: str = '', source: str = '', **extra: Any) -> Dict[str, Any]:
    item = {
        _TITLE_KEY: title,
        _URL_KEY: url,
        _SNIPPET_KEY: snippet,
        _SOURCE_KEY: source,
    }
    if extra:
        item[_EXTRA_KEY] = extra
    return item


class SearchBase(ModuleBase):

    def __init__(self, source_name: str = '', **kwargs):
        super().__init__(**kwargs)
        self._source_name = source_name or self.__class__.__name__.replace('Search', '').lower()

    @property
    def source_name(self) -> str:
        return self._source_name

    def search(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
        raise NotImplementedError('Subclass must implement search')

    def forward(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        try:
            return self.search(query, **kwargs)
        except Exception as err:
            import lazyllm
            lazyllm.LOG.error('Search request failed: %s', err)
            return []
