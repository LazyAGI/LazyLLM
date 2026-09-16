import re
from html import unescape
from typing import List, Dict, Any, Optional

from lazyllm import globals as lazyllm_globals
from lazyllm.common import (
    AuthStrategy, BearerTokenStrategy, Credential, CredentialMixin, KeyAuthError,
)
from lazyllm.module import ModuleBase
from lazyllm.module.module import ModuleExecutionError
from lazyllm.thirdparty import httpx


_TITLE_KEY = 'title'
_URL_KEY = 'url'
_SNIPPET_KEY = 'snippet'
_SOURCE_KEY = 'source'
_EXTRA_KEY = 'extra'


class _SearchResult(dict):
    # Keep fetched text available to a host's result middleware without putting
    # it into the serialized preview. No instance/session cache is involved.
    def __init__(self, value):
        super().__init__(value)
        self.full_result = {**value, 'extra': dict(value.get('extra') or {})}


def _html_to_text(html: str) -> str:
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<[^>]+>', ' ', html)
    html = html.replace('&nbsp;', ' ').replace('\n', ' ')
    return unescape(re.sub(r'\s+', ' ', html).strip())


def _make_result(title: str, url: str, snippet: str = '', source: str = '',
                 *, snippet_limit: Optional[int] = 700, **extra: Any) -> Dict[str, Any]:
    full_snippet = snippet = str(snippet or '')
    if snippet_limit is not None and len(snippet) > snippet_limit:
        snippet = snippet[:snippet_limit]
        extra['truncated'] = True
    item = {
        _TITLE_KEY: title,
        _URL_KEY: url,
        _SNIPPET_KEY: snippet,
        _SOURCE_KEY: source,
    }
    if extra:
        item[_EXTRA_KEY] = extra
    result = _SearchResult(item)
    result.full_result['snippet'] = full_snippet
    for key in ('raw_content', 'content'):
        if isinstance(extra.get(key), str):
            result.setdefault('extra', {})[key] = extra[key][:700]
    return result


def _make_content_result(item: Dict[str, Any], content: str, *,
                         content_type: Optional[str] = None, fallback: bool = False) -> Dict[str, Any]:
    extra = item.get(_EXTRA_KEY)
    extra = dict(extra) if isinstance(extra, dict) else {}
    if content_type is not None:
        extra['content_read'] = {'content_type': content_type, 'fallback': fallback}
    return _SearchResult({
        _TITLE_KEY: str(item.get(_TITLE_KEY) or ''),
        _URL_KEY: str(item.get(_URL_KEY) or item.get('link') or ''),
        _SNIPPET_KEY: str(item.get(_SNIPPET_KEY) or ''),
        _SOURCE_KEY: str(item.get(_SOURCE_KEY) or ''),
        _EXTRA_KEY: extra,
        'content': str(content or ''),
    })


# TODO: add tests after key is ready
class SearchBase(ModuleBase, CredentialMixin):
    __public_apis__ = ['search', 'get_content', 'get_contents']

    def __init__(
        self,
        source_name: str = '',
        api_key: Optional[str] = None,
        auth_strategy: Optional[AuthStrategy] = None,
        dynamic_auth: bool = False,
        skip_auth: bool = False,
        **kwargs,
    ):
        ModuleBase.__init__(self, **kwargs)
        self._source_name = source_name or self.__class__.__name__.replace('Search', '').lower()
        if dynamic_auth:
            credential = Credential(kind='dynamic')
        else:
            credential = Credential(kind='static', secret_key=api_key or '')
        self.__init_credential__(
            credential,
            strategy=auth_strategy or BearerTokenStrategy(),
            skip_auth=skip_auth,
        )

    def _resolve_dynamic_token(self) -> str:
        mapping = lazyllm_globals.config['dynamic_tool_auth'] or {}
        return mapping.get(self._source_name, '')

    def _http_execute(self, method: str, url: str, **kwargs) -> Any:
        resp = httpx.request(method, url, **kwargs)
        if self._is_key_auth_error(resp):
            raise KeyAuthError(f'{resp.status_code} for {url}')
        resp.raise_for_status()
        return resp

    @property
    def source_name(self) -> str:
        return self._source_name

    def search(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
        raise NotImplementedError('Subclass must implement search')

    def _handle_error(self, err: Exception, *, raise_on_error: bool) -> List[Dict[str, Any]]:
        if raise_on_error:
            raise err
        import lazyllm
        lazyllm.LOG.error('Search request failed: %s', type(err).__name__)
        return []

    def __call__(self, *args, **kwargs) -> List[Dict[str, Any]]:
        raise_on_error = bool(kwargs.pop('raise_on_error', False))
        try:
            return super().__call__(*args, **kwargs)
        except Exception as err:
            if isinstance(err, ModuleExecutionError) and err.__context__:
                err = err.__context__
            return self._handle_error(err, raise_on_error=raise_on_error)

    def forward(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        return self.search(query, **kwargs)

    def _fetch_content_text(self, item: Dict[str, Any]) -> str:
        url = item.get('url') or item.get('link') or ''
        if not url:
            return ''
        timeout = getattr(self, '_timeout', 15)
        try:
            resp = httpx.get(url, timeout=timeout, follow_redirects=True)
            resp.raise_for_status()
            return _html_to_text(resp.text)
        except Exception:
            return ''

    def _fetch_content_result(self, item: Dict[str, Any]) -> Dict[str, Any]:
        content = self._fetch_content_text(item)
        return _make_content_result(item, content, content_type='webpage', fallback=not bool(content))

    def get_content(self, item: Dict[str, Any], offset: int = 0, limit: int = 700) -> Dict[str, Any]:
        offset, limit = max(0, int(offset)), max(1, min(int(limit), 700))
        result = self._fetch_content_result(item)
        content = result['content']
        extra = result['extra']
        read = dict(extra.get('content_read') or {})
        fallback = read.get('fallback', False)
        start = 0 if fallback else offset
        result['content'] = content[start:start + limit]
        result['snippet'] = result['snippet'][:700]
        extra.pop('content', None)
        extra.pop('raw_content', None)
        read.update(offset=None if fallback else offset, limit=limit, truncated=len(content) > start + limit)
        if not fallback:
            next_offset = offset + len(result['content'])
            read.update(more=next_offset < len(content), next_offset=next_offset)
        extra['content_read'] = read
        return result

    def get_contents(self, items: List[Dict[str, Any]], offset: int = 0, limit: int = 700) -> List[Dict[str, Any]]:
        return [self.get_content(it, offset=offset, limit=limit) for it in items]
