import re
from typing import Any, Dict, List, Optional

from lazyllm.thirdparty import httpx

from .base import SearchBase, _html_to_text, _make_result


class StackOverflowSearch(SearchBase):

    def __init__(self, site: str = 'stackoverflow', key: Optional[str] = None,
                 timeout: int = 10, source_name: str = 'stackoverflow'):
        super().__init__(source_name=source_name)
        self._site = site
        self._key = key
        self._timeout = timeout

    def get_content(self, item: Dict[str, Any]) -> str:
        url = item.get('url') or ''
        m = re.search(r'/questions/(\d+)', url) if url else None
        if not m:
            return super().get_content(item)
        qid = m.group(1)
        api_url = f'https://api.stackexchange.com/2.3/questions/{qid}'
        params = {'site': self._site, 'filter': 'withbody'}
        if self._key:
            params['key'] = self._key
        try:
            resp = httpx.get(api_url, params=params, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return super().get_content(item)
        items = data.get('items') or []
        if not items:
            return super().get_content(item)
        q = items[0]
        raw = (q.get('body') or '').strip()
        if not raw:
            return super().get_content(item)
        body = _html_to_text(raw)
        accepted_id = q.get('accepted_answer_id')
        if not accepted_id:
            return body
        ans_url = f'https://api.stackexchange.com/2.3/answers/{accepted_id}'
        try:
            ar = httpx.get(ans_url, params=params, timeout=self._timeout)
            ar.raise_for_status()
            ans_items = ar.json().get('items') or []
            if ans_items and ans_items[0].get('body'):
                body = body + '\n\n--- Accepted Answer ---\n\n' + _html_to_text(ans_items[0]['body'])
        except Exception:
            pass
        return body

    def search(self, query: str, count: int = 10,
               sort: str = 'relevance') -> List[dict]:
        url = 'https://api.stackexchange.com/2.3/search/advanced'
        params = {
            'order': 'desc',
            'sort': sort,
            'q': query,
            'site': self._site,
            'pagesize': min(count, 100),
        }
        if self._key:
            params['key'] = self._key
        try:
            resp = httpx.get(url, params=params, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []
        items = data.get('items') or []
        out: List[dict] = []
        for it in items:
            link = it.get('link', '')
            title = it.get('title', '')
            snippet = it.get('body', '')[:500] if it.get('body') else ''
            if snippet and len(it.get('body', '')) > 500:
                snippet = snippet + '...'
            out.append(_make_result(
                title=title,
                url=link,
                snippet=snippet,
                source=self.source_name,
                score=it.get('score'),
                answer_count=it.get('answer_count'),
                is_answered=it.get('is_answered'),
            ))
        return out
