from typing import List

from lazyllm.thirdparty import httpx

from .base import SearchBase, _make_result


class ArxivSearch(SearchBase):

    def __init__(self, timeout: int = 15, source_name: str = 'arxiv'):
        super().__init__(source_name=source_name)
        self._timeout = timeout
        self._url = 'http://export.arxiv.org/api/query'

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
        import xml.etree.ElementTree as ET
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
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
