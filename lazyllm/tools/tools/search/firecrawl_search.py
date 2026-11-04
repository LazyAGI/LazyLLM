from typing import Any, Dict, List, Optional

from lazyllm.common import AllKeysExhaustedError, BearerTokenStrategy, KeyAuthError
from lazyllm.tools.agent.toolsManager import fc_register
from .base import SearchBase, _make_result


# Firecrawl buckets its search response by source type. Only the text-bearing buckets map
# onto the framework's title/url/snippet contract; images and tool contracts are not
# retrieval evidence for an agent, so they are rejected instead of being silently dropped.
_TEXT_SOURCES = ('web', 'news')

# Firecrawl rejects limit outside 1..100 per source type; failing locally keeps a typo from
# costing an authenticated round trip.
_MAX_LIMIT = 100


class FirecrawlSearch(SearchBase):

    def __init__(self, api_key: Optional[str] = None,
                 base_url: str = 'https://api.firecrawl.dev/v2',
                 timeout: int = 60, source_name: str = 'firecrawl'):
        super().__init__(
            source_name=source_name, api_key=api_key,
            auth_strategy=BearerTokenStrategy(),
            dynamic_auth=(api_key is None),
        )
        self._base_url = base_url.rstrip('/')
        self._timeout = timeout

    @staticmethod
    def _normalize_sources(sources: Optional[List[str]]) -> List[str]:
        selected = list(dict.fromkeys(sources or ['web']))
        unsupported = [name for name in selected if name not in _TEXT_SOURCES]
        if unsupported:
            raise ValueError(f'unsupported source(s) {unsupported}; FirecrawlSearch supports {list(_TEXT_SOURCES)}')
        return selected

    @fc_register(host_file='NONE')
    def search(self, query: str,
               limit: int = 10,
               sources: Optional[List[str]] = None,
               include_domains: Optional[List[str]] = None,
               exclude_domains: Optional[List[str]] = None,
               tbs: Optional[str] = None,
               country: Optional[str] = None,
               safe: Optional[bool] = None,
               scrape_markdown: bool = False,
               ) -> List[Dict[str, Any]]:
        if type(limit) is not int or not 1 <= limit <= _MAX_LIMIT:
            raise ValueError(f'limit must be an integer between 1 and {_MAX_LIMIT}')
        # Firecrawl answers 400 when both filters are present; raising here keeps the error
        # attached to the caller's arguments instead of an HTTP status.
        if include_domains and exclude_domains:
            raise ValueError('include_domains and exclude_domains are mutually exclusive')

        selected = self._normalize_sources(sources)
        body: Dict[str, Any] = {
            'query': query,
            'limit': limit,
            'sources': [{'type': name} for name in selected],
        }
        if include_domains:
            body['includeDomains'] = list(include_domains)
        if exclude_domains:
            body['excludeDomains'] = list(exclude_domains)
        if tbs:
            body['tbs'] = tbs
        if country:
            body['country'] = country
        if safe is not None:
            body['safe'] = safe
        if scrape_markdown:
            body['scrapeOptions'] = {'formats': [{'type': 'markdown'}], 'onlyMainContent': True}

        resp = self._request('POST', f'{self._base_url}/search', json=body, timeout=self._timeout)
        payload = resp.json()
        # Firecrawl can answer 200 with success=false and a human-readable error, which would
        # otherwise surface as an unexplained empty result list.
        if not isinstance(payload, dict) or payload.get('success') is False:
            detail = None
            if isinstance(payload, dict):
                detail = payload.get('error') or payload.get('warning')
            raise RuntimeError(f'Firecrawl search failed: {detail or "unknown error"}')
        return self._parse_results(payload.get('data'), selected)

    def _parse_results(self, data: Any, sources: List[str]) -> List[Dict[str, Any]]:
        buckets: List[Any] = []
        if isinstance(data, dict):
            # Preserve the caller's source order so the result ordering stays predictable.
            for name in sources:
                entries = data.get(name)
                if isinstance(entries, list):
                    buckets.append((name, entries))
        elif isinstance(data, list):
            # v1 answered with a single flat array; keeping it working costs one isinstance
            # branch and avoids a hard break for callers pinned to an older base_url.
            buckets.append(('web', data))

        out: List[Dict[str, Any]] = []
        for source_type, entries in buckets:
            for item in entries:
                if isinstance(item, dict):
                    out.append(self._map_result(item, source_type))
        return out

    def _map_result(self, item: Dict[str, Any], source_type: str) -> Dict[str, Any]:
        extra: Dict[str, Any] = {}
        markdown = item.get('markdown')
        if isinstance(markdown, str) and markdown:
            extra['raw_content'] = markdown
        published = item.get('date') or item.get('publishedDate')
        if published:
            extra['published_at'] = published
        if source_type != 'web':
            extra['result_type'] = source_type
        return _make_result(
            title=item.get('title') or '',
            url=item.get('url') or '',
            snippet=item.get('description') or item.get('snippet') or '',
            source=self.source_name,
            **extra,
        )

    def _fetch_content_text(self, item: Dict[str, Any]) -> str:
        url = item.get('url') or item.get('link') or ''
        if not url:
            return ''
        try:
            resp = self._request(
                'POST', f'{self._base_url}/scrape',
                json={'url': url, 'formats': [{'type': 'markdown'}], 'onlyMainContent': True},
                timeout=self._timeout,
            )
            data = resp.json()
            markdown = data.get('data', {}).get('markdown') if isinstance(data, dict) else None
            if isinstance(markdown, str) and markdown.strip():
                return markdown
        except (KeyAuthError, AllKeysExhaustedError):
            # A rejected key stays visible: degrading to the shared HTML reader here would
            # turn a misconfiguration into quietly worse content. The pool error is raised by
            # _request() instead of KeyAuthError when every key of a pool was rejected, so it
            # has to be listed explicitly.
            raise
        except Exception:
            pass
        # Rendering is the reason to route reads through Firecrawl, but a single page may be
        # rejected or throttled; falling back keeps get_content no worse than the shared reader.
        return super()._fetch_content_text(item)
