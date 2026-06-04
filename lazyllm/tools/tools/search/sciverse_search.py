from typing import Any, Dict, List, Literal, Optional

from lazyllm.thirdparty import httpx

from .base import SearchBase, _make_result


_DEFAULT_META_FIELDS = [
    'title',
    'doi',
    'doc_id',
    'abstract',
    'author',
    'publication_published_year',
    'publication_venue_name_unified',
]


def _items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = payload.get('data')
    source = data if isinstance(data, dict) else payload
    items = source.get('hits') or source.get('results') or source.get('items') or []
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _authors(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        return []
    names = []
    for item in value:
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, dict):
            name = str(item.get('name') or item.get('display_name') or item.get('full_name') or '').strip()
        else:
            name = ''
        if name:
            names.append(name)
    return names


class SciverseSearch(SearchBase):
    __public_apis__ = SearchBase.__public_apis__ + ['meta_search', 'meta_catalog']

    def __init__(self, api_key: Optional[str] = None,
                 base_url: str = 'https://api.sciverse.space',
                 timeout: int = 15, source_name: str = 'sciverse'):
        super().__init__(source_name=source_name, api_key=api_key, dynamic_auth=(api_key is None))
        self._base_url = base_url.rstrip('/')
        self._timeout = timeout

    def get_content(self, item: Dict[str, Any], offset: Optional[int] = None, limit: int = 700) -> str:
        extra = item.get('extra') or {}
        doc_id = item.get('doc_id') or extra.get('doc_id')
        if doc_id:
            params: Dict[str, Any] = {'doc_id': doc_id}
            if offset is not None:
                params.update({'offset': max(0, int(offset)), 'limit': max(1, int(limit))})
            try:
                resp = httpx.get(
                    f'{self._base_url}/content',
                    headers=self.inject_auth_header(),
                    params=params,
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict) and data.get('text'):
                    return data['text']
            except Exception:
                pass
        return extra.get('content') or item.get('snippet') or super().get_content(item)

    def search(self, query: str, topk: int = 5, include_content: bool = True,
               search_type: Literal['agentic', 'meta'] = 'agentic',
               year_from: Optional[int] = None,
               year_to: Optional[int] = None) -> List[dict]:
        normalized_type = str(search_type or 'agentic').lower()
        if normalized_type not in ('agentic', 'meta'):
            raise ValueError("search_type must be one of 'agentic' or 'meta'")

        limit = max(1, min(int(topk), 10))
        if normalized_type == 'meta':
            return self.meta_search(
                query=query,
                page_size=limit,
                include_content=include_content,
                year_from=year_from,
                year_to=year_to,
            )['items']

        payload: Dict[str, Any] = {'query': query, 'top_k': limit}
        headers = self.inject_auth_header({'Content-Type': 'application/json'})
        resp = httpx.post(f'{self._base_url}/agentic-search', headers=headers, json=payload, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            return []

        return self._normalize_items(_items(data)[:limit], include_content=include_content, search_type='agentic')

    def meta_search(
        self,
        query: str = '',
        filters: Optional[List[Dict[str, Any]]] = None,
        sort: Optional[List[Dict[str, Any]]] = None,
        fields: Optional[List[str]] = None,
        page: int = 1,
        page_size: int = 25,
        cursor: Optional[str] = None,
        freshness_boost: Literal['NONE', 'MILD', 'STRONG'] = 'NONE',
        include_content: bool = True,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
    ) -> Dict[str, Any]:
        if query and sort:
            raise ValueError('meta_search sort cannot be used together with query')
        if cursor and page > 1:
            raise ValueError('meta_search cursor cannot be used together with page > 1')
        normalized_freshness = str(freshness_boost or 'NONE').upper()
        if normalized_freshness not in ('NONE', 'MILD', 'STRONG'):
            raise ValueError("freshness_boost must be one of 'NONE', 'MILD', or 'STRONG'")

        merged_filters = list(filters or [])
        if year_from is not None:
            merged_filters.append({
                'field': 'publication_published_year',
                'operator': 'FILTER_OP_GTE',
                'value': int(year_from),
            })
        if year_to is not None:
            merged_filters.append({
                'field': 'publication_published_year',
                'operator': 'FILTER_OP_LTE',
                'value': int(year_to),
            })

        payload: Dict[str, Any] = {
            'fields': list(fields or _DEFAULT_META_FIELDS),
            'page_size': max(1, min(int(page_size), 200)),
        }
        if query:
            payload['query'] = query
        if merged_filters:
            payload['filters'] = merged_filters
        if sort:
            payload['sort'] = sort
        if cursor:
            payload['cursor'] = cursor
        else:
            payload['page'] = max(1, int(page))
        if normalized_freshness != 'NONE':
            payload['freshness_boost'] = normalized_freshness

        resp = httpx.post(
            f'{self._base_url}/meta-search',
            headers=self.inject_auth_header({'Content-Type': 'application/json'}),
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            data = {}

        return {
            'items': self._normalize_items(
                _items(data),
                include_content=include_content,
                search_type='meta',
            ),
            'total_count': data.get('total_count'),
            'total_pages': data.get('total_pages'),
            'page': data.get('page'),
            'page_size': data.get('page_size'),
            'next_cursor': data.get('next_cursor'),
            'search_time_ms': data.get('search_time_ms'),
        }

    def meta_catalog(self, include_sample_values: bool = False) -> Dict[str, Any]:
        resp = httpx.get(
            f'{self._base_url}/meta-catalog',
            headers=self.inject_auth_header(),
            params={'include_sample_values': str(bool(include_sample_values)).lower()},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else {}

    def _normalize_items(
        self,
        items: List[Dict[str, Any]],
        *,
        include_content: bool,
        search_type: str,
    ) -> List[dict]:
        out: List[dict] = []
        for item in items:
            title = str(item.get('title') or item.get('paper_title') or item.get('name') or '').strip()
            doi = str(item.get('doi') or item.get('publication_doi') or '').strip()
            url = str(item.get('url') or item.get('paper_url') or item.get('source_url') or '').strip()
            if not url and doi and search_type == 'meta':
                url = f'https://doi.org/{doi}'

            abstract = str(item.get('abstract') or item.get('summary') or item.get('description') or '').strip()
            chunk = str(item.get('chunk') or item.get('text') or item.get('content') or '').strip()
            content = chunk or abstract
            extra = {
                'doc_id': item.get('doc_id') or item.get('document_id') or item.get('id'),
                'doi': doi,
                'year': item.get('publication_published_year') or item.get('year') or item.get('published_year'),
                'venue': item.get('publication_venue_name_unified') or item.get('publication_venue_name')
                or item.get('venue') or item.get('journal'),
                'authors': _authors(item.get('author') or item.get('authors') or item.get('author_names')),
                'score': item.get('score') or item.get('relevance_score'),
                'chunk_id': item.get('chunk_id'),
                'page_no': item.get('page_no') or item.get('page'),
                'offset': item.get('offset'),
            }
            if include_content and content:
                extra['content'] = content
            out.append(_make_result(
                title=title,
                url=url,
                snippet='\n'.join(part for part in (abstract, chunk) if part),
                source=self.source_name,
                **{key: value for key, value in extra.items() if value not in (None, '', [])},
            ))
        return out
