import re
from typing import Any, Dict, List

from lazyllm.thirdparty import httpx, xml

from .base import SearchBase, _make_result


class ArxivSearch(SearchBase):

    def __init__(self, timeout: int = 15, source_name: str = 'arxiv'):
        super().__init__(source_name=source_name)
        self._timeout = timeout
        self._url = 'https://export.arxiv.org/api/query'

    def get_content(self, item: Dict[str, Any]) -> str:
        url = item.get('url') or ''
        m = re.search(r'/abs/([\d.]+(?:v\d+)?)', url) if url else None
        if not m:
            return super().get_content(item)
        arxiv_id = m.group(1)
        try:
            resp = httpx.get(
                self._url,
                params={'id_list': arxiv_id},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            text = resp.text
        except Exception:
            return super().get_content(item)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        try:
            root = xml.etree.ElementTree.fromstring(text)
        except xml.etree.ElementTree.ParseError:
            return super().get_content(item)
        for entry in root.findall('atom:entry', ns):
            summary_el = entry.find('atom:summary', ns)
            if summary_el is not None and summary_el.text:
                return summary_el.text.strip().replace('\n', ' ')
        return super().get_content(item)

    def search(self, query: str, max_results: int = 10,
               sort_by: str = 'relevance') -> List[dict]:
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': min(max_results, 2000),
            'sortBy': sort_by,
            'sortOrder': 'descending',
        }
        try:
            resp = httpx.get(self._url, params=params, timeout=self._timeout)
            resp.raise_for_status()
            text = resp.text
        except Exception:
            return []
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        try:
            root = xml.etree.ElementTree.fromstring(text)
        except xml.etree.ElementTree.ParseError:
            return []
        out: List[dict] = []
        for entry in root.findall('atom:entry', ns):
            title_el = entry.find('atom:title', ns)
            title = (title_el.text or '').strip().replace('\n', ' ')
            summary_el = entry.find('atom:summary', ns)
            snippet = (summary_el.text or '').strip().replace('\n', ' ')[:500]
            if len((summary_el.text or '')) > 500:
                snippet = snippet + '...'
            url = ''
            for link in entry.findall('atom:link', ns):
                if link.get('type') == 'text/html':
                    url = link.get('href', '')
                    break
            if not url:
                id_el = entry.find('atom:id', ns)
                url = id_el.text if id_el is not None else ''
            authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)
                       if a.find('atom:name', ns) is not None and a.find('atom:name', ns).text]
            published = entry.find('atom:published', ns)
            published_text = published.text if published is not None else None
            out.append(_make_result(
                title=title,
                url=url,
                snippet=snippet,
                source=self.source_name,
                authors=authors,
                published=published_text,
            ))
        return out
