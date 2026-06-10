from typing import Dict, Any, List, Optional

from lazyllm.common import BearerTokenStrategy
from .base import SearchBase, _make_result


class TavilySearch(SearchBase):

    def __init__(self, api_key: Optional[str] = None,
                 base_url: str = 'https://api.tavily.com',
                 timeout: int = 30, source_name: str = 'tavily'):
        super().__init__(
            source_name=source_name, api_key=api_key,
            auth_strategy=BearerTokenStrategy(),
            dynamic_auth=(api_key is None),
        )
        self._base_url = base_url.rstrip('/')
        self._timeout = timeout

    def search(self, query: str,
               search_depth: str = 'basic',
               topic: str = 'general',
               days: int = 3,
               max_results: int = 10,
               include_domains: Optional[List[str]] = None,
               exclude_domains: Optional[List[str]] = None,
               include_answer: bool = False,
               include_raw_content: bool = False,
               include_images: bool = False,
               ) -> List[Dict[str, Any]]:
        url = f'{self._base_url}/search'
        body: Dict[str, Any] = {
            'query': query,
            'search_depth': search_depth,
            'topic': topic,
            'days': days,
            'max_results': max_results,
            'include_answer': include_answer,
            'include_raw_content': include_raw_content,
            'include_images': include_images,
        }
        if include_domains:
            body['include_domains'] = include_domains
        if exclude_domains:
            body['exclude_domains'] = exclude_domains

        resp = self._request('POST', url, json=body, timeout=self._timeout)
        data = resp.json()
        results = data.get('results') or []
        out: List[Dict[str, Any]] = []
        for it in results:
            extra = {}
            score = it.get('score')
            if score is not None:
                extra['score'] = score
            raw_content = it.get('raw_content')
            if raw_content:
                extra['raw_content'] = raw_content
            out.append(_make_result(
                title=it.get('title') or '',
                url=it.get('url') or '',
                snippet=it.get('content') or '',
                source=self.source_name,
                **extra,
            ))
        return out
